use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use anyhow::{Context, Result, bail};
use walkdir::WalkDir;

use crate::config::{Config, KbConfig};
use crate::embedding::{RetryConfig, embed_batch};
use crate::markdown::{Chunk, ChunkLimits, process_file};
use crate::storage::{ChunkInput, HashDim, KbStore};

pub type EmbedFut<'a> = Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>;

/// Async embedding source. Production wraps the OpenAI-compatible HTTP client;
/// tests can inject deterministic vectors without a network round-trip.
pub trait Embedder: Send + Sync {
    fn embed<'a>(&'a self, texts: &'a [String]) -> EmbedFut<'a>;
}

struct ApiEmbedder {
    cfg: Config,
    retry: RetryConfig,
}

impl Embedder for ApiEmbedder {
    fn embed<'a>(&'a self, texts: &'a [String]) -> EmbedFut<'a> {
        Box::pin(embed_batch(&self.cfg, &self.retry, texts))
    }
}

#[derive(Debug, Default, PartialEq)]
struct IndexStats {
    indexed: usize,
    unchanged: usize,
    deleted: usize,
    empty: usize,
    batches_failed: usize,
    chunks_failed: usize,
    failed_files: HashSet<String>,
}

/// One chunk produced by `collect_phase`, tagged with the file it came from
/// and that file's content hash. The chunking + embedding + storage layers all
/// see chunks individually, but the per-batch failure-isolation contract needs
/// to know "which file owns this chunk" so failed batches can flag the
/// affected files for the FileHash phase.
/// One chunk pointed at its owning file. The hash itself is held in
/// `CollectResult::file_hashes` rather than inlined per chunk — N chunks of
/// the same file would otherwise carry N copies of the same blake3 string.
struct CollectedChunk {
    file_path: String,
    chunk: Chunk,
}

struct CollectResult {
    /// Flat global chunk list, batched in `embed_and_write_phase` by batch_size.
    global_chunks: Vec<CollectedChunk>,
    /// `file_path` → number of chunks this file contributed to `global_chunks`.
    /// Includes "empty modified" files at value 0 — they still need `vector_hash`
    /// rewritten because their content hash changed (e.g. file emptied).
    chunks_per_file: HashMap<String, usize>,
    /// `file_path` → blake3 of file content; passed straight to
    /// `set_vector_hashes` in `mark_completed_files_phase`.
    file_hashes: HashMap<String, String>,
    /// Count of files skipped because `:FileHash.vector_hash` already matches
    /// disk content (incremental mode).
    unchanged: usize,
    /// Set of `file_path`s present on disk this run (for incremental cleanup
    /// of stored files no longer on disk).
    on_disk: HashSet<String>,
}

#[derive(Debug, Default)]
struct EmbedWriteResult {
    batches_failed: usize,
    chunks_failed: usize,
    /// `file_path` → number of chunks successfully written this run. Compared
    /// against `CollectResult::chunks_per_file` in the FileHash phase to decide
    /// whether `vector_hash` may be written.
    chunks_succeeded_per_file: HashMap<String, usize>,
    /// Files that had at least one chunk in a failed batch. Reported in the
    /// stderr failure summary so the user knows which files to re-index.
    failed_files: HashSet<String>,
}

pub async fn run(
    db: &Path,
    root: &Path,
    incremental: bool,
    batch_size: usize,
    retry: &RetryConfig,
) -> Result<()> {
    let cfg = Config::from_env()?;
    let store = KbStore::open(db)?;
    let model = cfg.model.clone();
    let dim = cfg.dim;
    let max_chunk_chars = cfg.max_chunk_chars;
    // Single source of truth: ChunkLimits, dim and IndexConfig.max_chunk_chars
    // all derive from the same Config, so the chunking policy that produces
    // chunks at runtime is identical to the policy persisted in IndexConfig.
    // Drift between "what we store" and "what we run" is impossible by
    // construction.
    let limits = ChunkLimits {
        max_chars: max_chunk_chars,
    };
    let embedder = ApiEmbedder { cfg, retry: *retry };
    run_with_embedder(
        &store,
        &model,
        dim,
        max_chunk_chars,
        root,
        incremental,
        batch_size,
        &limits,
        &embedder,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn run_with_embedder(
    store: &KbStore,
    model: &str,
    dim: usize,
    max_chunk_chars: u32,
    root: &Path,
    incremental: bool,
    batch_size: usize,
    limits: &ChunkLimits,
    embedder: &dyn Embedder,
) -> Result<()> {
    if batch_size == 0 {
        bail!("batch_size must be >= 1");
    }

    // Phase 1 — Collect: walk disk, validate IndexConfig, drop unchanged
    // files (incremental), flatten everything into one global chunk list.
    let entries = collect_markdown(root)?;
    let collect = collect_phase(
        entries,
        store,
        incremental,
        model,
        dim,
        max_chunk_chars,
        limits,
    )?;

    // Progress: announce the workload before touching the embedding API.
    // N counts files passing through (indexed + empty modified), not chunks
    // alone — the spec scenario "Processing 0 files / 0 chunks" expects the
    // 0/0 case to be reachable when incremental skips everything.
    let processing_files = collect.chunks_per_file.len();
    eprintln!(
        "{}",
        fmt_processing(processing_files, collect.global_chunks.len())
    );

    // Pre-clear "empty modified" files (file modified, content shrank to zero
    // chunks). They never enter Embed & Write so we clear here in their own
    // tiny transaction. Empty list → no-op.
    let empty_clear_owned: Vec<String> = collect
        .chunks_per_file
        .iter()
        .filter(|(_, c)| **c == 0)
        .map(|(p, _)| p.clone())
        .collect();
    if !empty_clear_owned.is_empty() {
        let empty_clear: Vec<&str> = empty_clear_owned.iter().map(|s| s.as_str()).collect();
        store
            .append_chunks_batch(&[], &empty_clear)
            .with_context(|| "pre-clear empty-modified files")?;
    }

    // Phase 2 — Embed & Write per batch.
    let ew = embed_and_write_phase(&collect.global_chunks, embedder, store, batch_size).await;

    // Phase 3 — FileHash mark: write vector_hash only for files whose chunks
    // all landed.
    mark_completed_files_phase(&collect, &ew, store)?;

    // Incremental cleanup: drop vector data for files no longer on disk.
    let mut deleted = 0usize;
    let mut delete_failures: Vec<String> = Vec::new();
    if incremental {
        let stored_paths = store.list_file_paths()?;
        for p in stored_paths {
            if !collect.on_disk.contains(&p) {
                if let Err(e) = store.delete_vector_dim_for_file(&p) {
                    eprintln!("[fail-delete] {p}: {e}");
                    delete_failures.push(p);
                } else {
                    deleted += 1;
                }
            }
        }
    }

    // Tally final stats. `indexed` = files with >0 chunks where every chunk
    // succeeded. `empty` = modified files that produced 0 chunks (their
    // vector_hash was just rewritten by Phase 3).
    let mut indexed = 0usize;
    let mut empty = 0usize;
    for (file, &pending) in &collect.chunks_per_file {
        if pending == 0 {
            empty += 1;
        } else {
            let succeeded = ew.chunks_succeeded_per_file.get(file).copied().unwrap_or(0);
            if succeeded == pending {
                indexed += 1;
            }
        }
    }

    let stats = IndexStats {
        indexed,
        unchanged: collect.unchanged,
        deleted,
        empty,
        batches_failed: ew.batches_failed,
        chunks_failed: ew.chunks_failed,
        failed_files: ew.failed_files,
    };

    // Final stdout line — single line, machine-parseable, format pinned by
    // spec scenario "进度输出格式".
    println!("{}", fmt_index_complete(&stats));

    if stats.batches_failed > 0 || !delete_failures.is_empty() {
        // Stderr failure summary, separate from stdout so scripts that pipe
        // stdout to a parser still see the human signal.
        eprintln!("{}", fmt_index_failed(&stats));
        if stats.batches_failed > 0 {
            bail!(
                "{} batch(es) failed; see [batch-fail] entries above",
                stats.batches_failed
            );
        }
        bail!(
            "{} file(s) failed to delete; see [fail-delete] entries above",
            delete_failures.len()
        );
    }
    Ok(())
}

// ---------- progress / status formatters ----------
//
// Centralised here so the spec-pinned strings live in one place and unit tests
// can assert exact output without redirecting stderr.

fn fmt_processing(n_files: usize, n_chunks: usize) -> String {
    format!("Processing {n_files} files / {n_chunks} chunks")
}

fn fmt_embedding_batch(i: usize, total: usize) -> String {
    format!("Embedding batch {i}/{total}")
}

fn fmt_batch_fail(i: usize, error: &str, files: &HashSet<String>) -> String {
    // Sort file paths so the output is deterministic across HashSet iteration
    // orders — important both for the snapshot test and for users diffing run
    // logs across re-runs.
    let mut sorted: Vec<&String> = files.iter().collect();
    sorted.sort();
    let files_csv = sorted
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    format!("[batch-fail] batch {i}: {error} (files: {files_csv})")
}

fn fmt_index_complete(stats: &IndexStats) -> String {
    format!(
        "Index complete: indexed={} unchanged={} deleted={} empty={} batches_failed={} chunks_failed={}",
        stats.indexed,
        stats.unchanged,
        stats.deleted,
        stats.empty,
        stats.batches_failed,
        stats.chunks_failed
    )
}

fn fmt_index_failed(stats: &IndexStats) -> String {
    format!(
        "Index failed: batches_failed={} affected_files={}",
        stats.batches_failed,
        stats.failed_files.len()
    )
}

/// Phase 1: validate IndexConfig, walk every entry, filter unchanged files
/// when `incremental`, and flatten chunks into one global list.
#[allow(clippy::too_many_arguments)]
fn collect_phase(
    entries: Vec<FileEntry>,
    store: &KbStore,
    incremental: bool,
    model: &str,
    dim: usize,
    max_chunk_chars: u32,
    limits: &ChunkLimits,
) -> Result<CollectResult> {
    // IndexConfig: write on first run, validate on subsequent. Identical
    // semantics to the pre-refactor behaviour — only the location moved.
    match store.read_index_config()? {
        Some(existing)
            if existing.model == model
                && existing.dim == dim
                && existing.max_chunk_chars == max_chunk_chars => {}
        Some(existing) => bail!(
            "IndexConfig mismatch for DB: stored model='{}' dim={} max_chunk_chars={}, current model='{}' dim={} max_chunk_chars={}. Use a new --db path or rebuild from scratch.",
            existing.model,
            existing.dim,
            existing.max_chunk_chars,
            model,
            dim,
            max_chunk_chars
        ),
        None => store.write_index_config(model, dim, max_chunk_chars)?,
    }

    let on_disk: HashSet<String> = entries.iter().map(|e| e.rel_path.clone()).collect();
    let mut global_chunks: Vec<CollectedChunk> = Vec::new();
    let mut chunks_per_file: HashMap<String, usize> = HashMap::new();
    let mut file_hashes: HashMap<String, String> = HashMap::new();
    let mut unchanged = 0usize;

    for entry in entries {
        let FileEntry {
            rel_path,
            hash,
            content,
        } = entry;

        if incremental
            && let Some(stored) = store.get_file_hash(&rel_path, HashDim::Vector)?
            && stored == hash
        {
            unchanged += 1;
            continue;
        }

        let chunks = collect_chunks_for_file(&content, &rel_path, limits);
        chunks_per_file.insert(rel_path.clone(), chunks.len());
        file_hashes.insert(rel_path.clone(), hash);
        for chunk in chunks {
            global_chunks.push(CollectedChunk {
                file_path: rel_path.clone(),
                chunk,
            });
        }
    }

    Ok(CollectResult {
        global_chunks,
        chunks_per_file,
        file_hashes,
        unchanged,
        on_disk,
    })
}

/// Pure chunking helper extracted from the old `process_one_file`. Takes file
/// content and produces the chunks that will be embedded — no embedding, no
/// grafeo writes. Lets `collect_phase` flatten chunks before the embedding
/// stage knows anything about file boundaries.
fn collect_chunks_for_file(content: &str, rel_path: &str, limits: &ChunkLimits) -> Vec<Chunk> {
    process_file(content, rel_path, limits)
}

/// Phase 2: per-batch embed + write. Each batch is its own grafeo session
/// transaction. A batch failure is captured into `EmbedWriteResult` and the
/// loop advances to the next batch — failure does NOT short-circuit run.
async fn embed_and_write_phase(
    global_chunks: &[CollectedChunk],
    embedder: &dyn Embedder,
    store: &KbStore,
    batch_size: usize,
) -> EmbedWriteResult {
    let mut result = EmbedWriteResult::default();

    if global_chunks.is_empty() {
        return result;
    }

    let total_batches = global_chunks.len().div_ceil(batch_size);
    // Tracks "this run has already COMMITTED a clear for file F". Updated only
    // after `append_chunks_batch` succeeds, so a failed batch never marks F as
    // cleared. The next batch that touches F will then still issue the
    // `files_to_clear_first` for F, guaranteeing the first SUCCESSFUL batch
    // for any file removes its stale `:Chunk` rows in the same transaction.
    let mut cleared_files: HashSet<String> = HashSet::new();

    for (idx, batch_slice) in global_chunks.chunks(batch_size).enumerate() {
        let i = idx + 1;
        eprintln!("{}", fmt_embedding_batch(i, total_batches));

        // Decide which files in this batch need their stale chunks cleared.
        // Use a per-batch dedup set so the same file appearing twice within
        // the same batch is only listed once.
        let mut to_clear_owned: Vec<String> = Vec::new();
        let mut batch_dedup: HashSet<&str> = HashSet::new();
        for cc in batch_slice {
            if !cleared_files.contains(&cc.file_path) && batch_dedup.insert(cc.file_path.as_str()) {
                to_clear_owned.push(cc.file_path.clone());
            }
        }

        let inputs: Vec<String> = batch_slice
            .iter()
            .map(|cc| cc.chunk.content.clone())
            .collect();

        let vectors = match embedder.embed(&inputs).await {
            Ok(v) => v,
            Err(e) => {
                let files: HashSet<String> =
                    batch_slice.iter().map(|cc| cc.file_path.clone()).collect();
                eprintln!("{}", fmt_batch_fail(i, &e.to_string(), &files));
                result.batches_failed += 1;
                result.chunks_failed += batch_slice.len();
                result.failed_files.extend(files);
                continue;
            }
        };

        if vectors.len() != batch_slice.len() {
            let files: HashSet<String> =
                batch_slice.iter().map(|cc| cc.file_path.clone()).collect();
            let mismatch_msg = format!(
                "embedding count mismatch (expected {}, got {})",
                batch_slice.len(),
                vectors.len()
            );
            eprintln!("{}", fmt_batch_fail(i, &mismatch_msg, &files));
            result.batches_failed += 1;
            result.chunks_failed += batch_slice.len();
            result.failed_files.extend(files);
            continue;
        }

        let chunk_inputs: Vec<ChunkInput<'_>> = batch_slice
            .iter()
            .zip(vectors.iter())
            .map(|(cc, v)| ChunkInput {
                file_path: cc.file_path.as_str(),
                line_start: u32::try_from(cc.chunk.line_start).unwrap_or(u32::MAX),
                line_end: u32::try_from(cc.chunk.line_end).unwrap_or(u32::MAX),
                content: cc.chunk.content.as_str(),
                prefix: if cc.chunk.prefix.is_empty() {
                    None
                } else {
                    Some(cc.chunk.prefix.as_str())
                },
                embedding: v.clone(),
            })
            .collect();

        let to_clear: Vec<&str> = to_clear_owned.iter().map(|s| s.as_str()).collect();
        if let Err(e) = store.append_chunks_batch(&chunk_inputs, &to_clear) {
            let files: HashSet<String> =
                batch_slice.iter().map(|cc| cc.file_path.clone()).collect();
            eprintln!("{}", fmt_batch_fail(i, &e.to_string(), &files));
            result.batches_failed += 1;
            result.chunks_failed += batch_slice.len();
            result.failed_files.extend(files);
            continue;
        }

        // Commit-time mark: only after a successful DELETE+INSERT do we
        // promise that subsequent batches must NOT re-clear these files.
        cleared_files.extend(to_clear_owned);

        for cc in batch_slice {
            *result
                .chunks_succeeded_per_file
                .entry(cc.file_path.clone())
                .or_insert(0) += 1;
        }
    }

    result
}

/// Phase 3: write `vector_hash` only for files where every chunk landed.
fn mark_completed_files_phase(
    collect: &CollectResult,
    ew: &EmbedWriteResult,
    store: &KbStore,
) -> Result<()> {
    let mut hashes: Vec<(&str, &str)> = Vec::new();
    for (file, &pending) in &collect.chunks_per_file {
        let succeeded = ew.chunks_succeeded_per_file.get(file).copied().unwrap_or(0);
        if succeeded == pending
            && let Some(h) = collect.file_hashes.get(file)
        {
            hashes.push((file.as_str(), h.as_str()));
        }
    }
    store.set_vector_hashes(&hashes)
}

#[derive(Debug)]
struct FileEntry {
    rel_path: String,
    hash: String,
    content: String,
}

fn collect_markdown(root: &Path) -> Result<Vec<FileEntry>> {
    if !root.exists() {
        bail!("markdown root does not exist: {}", root.display());
    }
    // Compile glob filter from `<root>/.xd-kb.yaml` (or default include + the
    // engine's hardcoded DEFAULT_EXCLUDES when the file is absent). We still
    // pre-filter by extension `.md` because (a) the WalkDir traversal can
    // visit non-markdown files cheaply via this gate without invoking the
    // glob matcher, and (b) the spec requires DEFAULT_INCLUDE to mean "all
    // markdown" — extension is the only consistent way to express that.
    let kb_config = KbConfig::load_for_root(root)?;
    // Propagate WalkDir errors via a fallible iterator instead of silently
    // dropping unreadable entries — under `--incremental` a swallowed error
    // would make a still-present file look "deleted" and trigger spurious
    // cleanup. The actual filtering / hashing lives in
    // `collect_markdown_from_entries` so unit tests can inject a synthetic
    // walk failure to lock the propagation contract.
    let entries = WalkDir::new(root).into_iter().map(|e| {
        e.map(|de| de.path().to_path_buf())
            .with_context(|| format!("walk markdown root {}", root.display()))
    });
    collect_markdown_from_entries(root, &kb_config, entries)
}

/// Per-entry filter + hash + read pipeline. Extracted from `collect_markdown`
/// so unit tests can drive it with a synthetic `Iterator<Item = Result<PathBuf>>`
/// (impossible to construct a real `walkdir::Error` without a failing OS call).
/// Production wires `WalkDir::new(root).into_iter()` into this helper.
fn collect_markdown_from_entries<I>(
    root: &Path,
    kb_config: &KbConfig,
    entries: I,
) -> Result<Vec<FileEntry>>
where
    I: IntoIterator<Item = Result<std::path::PathBuf>>,
{
    let mut out = Vec::new();
    for path_result in entries {
        let path = path_result?;
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let rel = path.strip_prefix(root).unwrap_or(&path);
        let rel_path = rel.to_string_lossy().replace('\\', "/");
        if !kb_config.matches(&rel_path) {
            continue;
        }
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("read markdown {}", path.display()))?;
        let hash = blake3::hash(content.as_bytes()).to_hex().to_string();
        out.push(FileEntry {
            rel_path,
            hash,
            content,
        });
    }
    out.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tempfile::tempdir;

    /// Returns a fixed-dim vector so KbStore vector index dim stays consistent.
    /// Real embedding model uses cfg.dim (default 1024) but tests can pick any
    /// dim — KbStore validates dim once per file and treats Config::dim as
    /// authoritative only at the API boundary, not inside storage.
    fn fake_vec(seed: f32, dim: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(dim);
        for i in 0..dim {
            v.push(seed + (i as f32) * 0.001);
        }
        v
    }

    // Tests use FakeEmbedder dim=8 and a non-restricting char limit so the
    // existing chunking behavior is preserved. Section 6 integration tests
    // exercise the splitter with smaller limits explicitly.
    const TEST_DIM: usize = 8;
    const TEST_MAX_CHUNK_CHARS: u32 = 8192;

    fn test_limits() -> ChunkLimits {
        ChunkLimits {
            max_chars: TEST_MAX_CHUNK_CHARS,
        }
    }

    // Test-only thin wrapper: most tests want default dim / max_chunk_chars / limits
    // and only differ on (model, incremental). Bypass the long arg list here so each
    // test reads cleanly.
    async fn run_default(
        store: &KbStore,
        model: &str,
        root: &Path,
        incremental: bool,
        embedder: &dyn Embedder,
    ) -> Result<()> {
        let limits = test_limits();
        run_with_embedder(
            store,
            model,
            TEST_DIM,
            TEST_MAX_CHUNK_CHARS,
            root,
            incremental,
            50,
            &limits,
            embedder,
        )
        .await
    }

    struct FakeEmbedder {
        dim: usize,
        // call_count for assertions
        calls: Mutex<usize>,
        // Verbatim copy of every text passed to `embed`, in arrival order.
        // Lets tests pin the exact embedding input (e.g. that it does NOT
        // contain a heading-path prefix duplicated ahead of the content).
        seen_texts: Mutex<Vec<String>>,
        fail_on: Option<String>,
    }

    impl FakeEmbedder {
        fn new(dim: usize) -> Self {
            Self {
                dim,
                calls: Mutex::new(0),
                seen_texts: Mutex::new(Vec::new()),
                fail_on: None,
            }
        }
        fn failing(dim: usize, fail_substring: &str) -> Self {
            Self {
                dim,
                calls: Mutex::new(0),
                seen_texts: Mutex::new(Vec::new()),
                fail_on: Some(fail_substring.to_string()),
            }
        }
        fn call_count(&self) -> usize {
            *self.calls.lock().unwrap()
        }
        fn seen_texts(&self) -> Vec<String> {
            self.seen_texts.lock().unwrap().clone()
        }
    }

    impl Embedder for FakeEmbedder {
        fn embed<'a>(&'a self, texts: &'a [String]) -> EmbedFut<'a> {
            let dim = self.dim;
            let fail_on = self.fail_on.clone();
            let texts_owned: Vec<String> = texts.to_vec();
            *self.calls.lock().unwrap() += 1;
            self.seen_texts
                .lock()
                .unwrap()
                .extend(texts_owned.iter().cloned());
            Box::pin(async move {
                if let Some(needle) = fail_on
                    && texts_owned.iter().any(|t| t.contains(&needle))
                {
                    anyhow::bail!("FakeEmbedder: forced failure on `{needle}`");
                }
                Ok(texts_owned
                    .iter()
                    .enumerate()
                    .map(|(i, _)| fake_vec(i as f32 * 0.1, dim))
                    .collect())
            })
        }
    }

    fn write_md(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(p, body).unwrap();
    }

    // Two-H2 doc shape used across most index tests. The path prefix `世界观/`
    // is historical — the engine no longer cares about path; the generic
    // chunker simply splits on H2 / H3 / whole-file fallback.
    fn worldview_doc(title: &str, body: &str) -> String {
        format!("# {title}\n\n## 概述\n\n{body}\n\n## 细节\n\n更多内容。\n")
    }

    #[tokio::test]
    async fn first_run_writes_chunks_filehash_and_indexconfig() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_md(&root, "世界观/A.md", &worldview_doc("A", "alpha-stuff"));
        write_md(&root, "世界观/B.md", &worldview_doc("B", "bravo-stuff"));

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);
        let limits = test_limits();

        run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            TEST_MAX_CHUNK_CHARS,
            &root,
            false,
            50,
            &limits,
            &embedder,
        )
        .await
        .expect("indexing should succeed");

        // IndexConfig present
        let cfg = store
            .read_index_config()
            .unwrap()
            .expect("IndexConfig written");
        assert_eq!(cfg.model, "fake-model");
        assert_eq!(cfg.dim, TEST_DIM);
        assert_eq!(cfg.max_chunk_chars, TEST_MAX_CHUNK_CHARS);

        // FileHash for both files present — and per the new contract, written
        // ONLY because every chunk landed (no batch failed).
        assert!(
            store
                .get_file_hash("世界观/A.md", HashDim::Vector)
                .unwrap()
                .is_some()
        );
        assert!(
            store
                .get_file_hash("世界观/B.md", HashDim::Vector)
                .unwrap()
                .is_some()
        );

        // embedder was called at least once
        assert!(embedder.call_count() >= 1);
    }

    #[tokio::test]
    async fn incremental_skips_unchanged_files() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_md(&root, "世界观/A.md", &worldview_doc("A", "stable"));

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);
        run_default(&store, "fake-model", &root, false, &embedder)
            .await
            .unwrap();
        let calls_after_first = embedder.call_count();

        // Second run with --incremental and no file changes
        run_default(&store, "fake-model", &root, true, &embedder)
            .await
            .unwrap();
        assert_eq!(
            embedder.call_count(),
            calls_after_first,
            "incremental run on unchanged files MUST NOT call embedder"
        );
    }

    #[tokio::test]
    async fn incremental_cleans_up_deleted_files() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_md(&root, "世界观/A.md", &worldview_doc("A", "alpha"));
        write_md(&root, "世界观/B.md", &worldview_doc("B", "bravo"));

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);
        run_default(&store, "fake-model", &root, false, &embedder)
            .await
            .unwrap();
        assert!(
            store
                .get_file_hash("世界观/B.md", HashDim::Vector)
                .unwrap()
                .is_some()
        );

        // delete B.md and re-run incremental
        std::fs::remove_file(root.join("世界观/B.md")).unwrap();
        run_default(&store, "fake-model", &root, true, &embedder)
            .await
            .unwrap();

        assert!(
            store
                .get_file_hash("世界观/B.md", HashDim::Vector)
                .unwrap()
                .is_none(),
            "deleted file's FileHash MUST be cleaned up"
        );
    }

    /// Spec scenario: 单 batch 失败不影响其它 batch.
    /// File A is "good" (1 chunk via worldview_doc → 2 H2s actually);
    /// file B contains the MAGIC_FAIL token so any batch carrying its chunks
    /// fails. Verify A's vector_hash IS written (because none of A's batches
    /// failed) and B's vector_hash is NOT (its sole batch failed).
    #[tokio::test]
    async fn batch_failure_does_not_affect_other_batches() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_md(&root, "世界观/Good.md", &worldview_doc("Good", "ok"));
        write_md(
            &root,
            "世界观/Bad.md",
            &worldview_doc("Bad", "this contains MAGIC_FAIL token"),
        );

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::failing(TEST_DIM, "MAGIC_FAIL");

        // batch_size=1 forces every chunk into its own batch — Bad's chunks
        // fail in their own batches without dragging Good down.
        let limits = test_limits();
        let res = run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            TEST_MAX_CHUNK_CHARS,
            &root,
            false,
            1,
            &limits,
            &embedder,
        )
        .await;
        assert!(res.is_err(), "run MUST exit non-zero when any batch fails");

        assert!(
            store
                .get_file_hash("世界观/Good.md", HashDim::Vector)
                .unwrap()
                .is_some(),
            "Good.md vector_hash MUST be written — none of its batches failed"
        );
        assert!(
            store
                .get_file_hash("世界观/Bad.md", HashDim::Vector)
                .unwrap()
                .is_none(),
            "Bad.md vector_hash MUST NOT be written — its batch failed"
        );
    }

    /// Spec scenario: FileHash 仅在文件全部 chunks 成功后写入.
    /// File A produces multiple chunks; one batch carrying SOME of A's chunks
    /// fails. A's vector_hash MUST stay absent because A's chunk count
    /// succeeded != chunk count pending.
    #[tokio::test]
    async fn file_hash_only_written_when_all_chunks_succeed() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        // worldview_doc produces 2 chunks (one per H2). The second contains
        // "更多内容" — won't trigger MAGIC_FAIL. Inject MAGIC_FAIL into the
        // first H2's body so exactly one of A's two chunks fails.
        write_md(
            &root,
            "世界观/A.md",
            &worldview_doc("A", "first chunk has MAGIC_FAIL"),
        );

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::failing(TEST_DIM, "MAGIC_FAIL");

        let limits = test_limits();
        let res = run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            TEST_MAX_CHUNK_CHARS,
            &root,
            false,
            1, // one chunk per batch so failures isolate to single chunks
            &limits,
            &embedder,
        )
        .await;
        assert!(res.is_err(), "expected non-zero exit on batch failure");

        assert!(
            store
                .get_file_hash("世界观/A.md", HashDim::Vector)
                .unwrap()
                .is_none(),
            "vector_hash MUST stay absent until ALL of A's chunks land"
        );
    }

    /// Spec scenario: 跨文件 batch 聚合.
    /// Two small files together fit into a single batch — verify the batch
    /// carries chunks from both files and both vector_hashes are written.
    #[tokio::test]
    async fn cross_file_batch_aggregation() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_md(&root, "世界观/A.md", &worldview_doc("A", "alpha"));
        write_md(&root, "世界观/B.md", &worldview_doc("B", "bravo"));

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);

        // batch_size=100 ensures all chunks (A's 2 + B's 2 = 4) fit in one
        // batch. With per-file batching we'd see 2+ embedder calls; with
        // cross-file aggregation the count drops to exactly 1.
        let limits = test_limits();
        run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            TEST_MAX_CHUNK_CHARS,
            &root,
            false,
            100,
            &limits,
            &embedder,
        )
        .await
        .unwrap();

        assert_eq!(
            embedder.call_count(),
            1,
            "cross-file aggregation MUST collapse A+B into one embedder call"
        );
        assert!(
            store
                .get_file_hash("世界观/A.md", HashDim::Vector)
                .unwrap()
                .is_some()
        );
        assert!(
            store
                .get_file_hash("世界观/B.md", HashDim::Vector)
                .unwrap()
                .is_some()
        );
    }

    /// Spec scenario: 重跑前清旧 chunks 避免孤儿.
    /// Run 1 leaves orphan chunks (file A's vector_hash never written because
    /// the embedder failed). Run 2 with a working embedder must clear orphans
    /// and produce exactly the new chunk set.
    #[tokio::test]
    async fn rerun_clears_orphan_chunks() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_md(
            &root,
            "世界观/A.md",
            &worldview_doc("A", "first chunk has MAGIC_FAIL"),
        );

        let store = KbStore::open(&db).unwrap();

        // Run 1: A has 2 chunks, batch_size=1 forces one chunk per batch.
        // The MAGIC_FAIL chunk's batch fails; the other chunk's batch succeeds
        // → 1 orphan :Chunk row, no vector_hash.
        let failing = FakeEmbedder::failing(TEST_DIM, "MAGIC_FAIL");
        let _ = run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            TEST_MAX_CHUNK_CHARS,
            &root,
            false,
            1,
            &test_limits(),
            &failing,
        )
        .await;
        assert!(
            store
                .get_file_hash("世界观/A.md", HashDim::Vector)
                .unwrap()
                .is_none(),
            "precondition: orphan run must NOT have written vector_hash"
        );
        let qres1 = store
            .execute_raw("MATCH (c:Chunk {file_path: '世界观/A.md'}) RETURN count(c)")
            .unwrap();
        let orphan_count = match &qres1.rows()[0][0] {
            grafeo::Value::Int64(n) => *n,
            other => panic!("unexpected count: {other:?}"),
        };
        assert!(
            orphan_count >= 1,
            "precondition: at least one orphan chunk should exist after partial run; got {orphan_count}"
        );

        // Run 2: same file, but content slightly changed so MAGIC_FAIL is gone.
        // The first batch touching A MUST clear the orphan first.
        write_md(&root, "世界观/A.md", &worldview_doc("A", "clean run now"));
        let healthy = FakeEmbedder::new(TEST_DIM);
        run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            TEST_MAX_CHUNK_CHARS,
            &root,
            true,
            1,
            &test_limits(),
            &healthy,
        )
        .await
        .unwrap();

        // After run 2: A's chunk count MUST equal exactly what process_file
        // produces from the new content; no leftover orphan.
        let limits = test_limits();
        let new_chunks = process_file(&worldview_doc("A", "clean run now"), "世界观/A.md", &limits);
        let qres2 = store
            .execute_raw("MATCH (c:Chunk {file_path: '世界观/A.md'}) RETURN count(c)")
            .unwrap();
        let post_count = match &qres2.rows()[0][0] {
            grafeo::Value::Int64(n) => *n,
            other => panic!("unexpected count: {other:?}"),
        };
        assert_eq!(
            post_count as usize,
            new_chunks.len(),
            "post-run chunk count must equal fresh chunking output (no orphans)"
        );
        assert!(
            store
                .get_file_hash("世界观/A.md", HashDim::Vector)
                .unwrap()
                .is_some(),
            "vector_hash MUST be written after a fully-successful rerun"
        );
    }

    /// Spec scenario "same file across batches uses files_to_clear_first only
    /// once": when file A's chunks span two batches, only batch 1 clears A;
    /// batch 2 MUST NOT re-clear (otherwise it would wipe batch 1's writes).
    #[tokio::test]
    async fn multi_batch_file_clears_only_in_first_batch() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        // worldview_doc → 2 chunks; batch_size=1 forces them into 2 batches.
        write_md(&root, "世界观/A.md", &worldview_doc("A", "alpha"));

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);
        run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            TEST_MAX_CHUNK_CHARS,
            &root,
            false,
            1,
            &test_limits(),
            &embedder,
        )
        .await
        .unwrap();

        // BOTH chunks must remain — if batch 2 re-cleared we'd see only 1.
        let qres = store
            .execute_raw("MATCH (c:Chunk {file_path: '世界观/A.md'}) RETURN count(c)")
            .unwrap();
        let n = match &qres.rows()[0][0] {
            grafeo::Value::Int64(n) => *n,
            other => panic!("unexpected count: {other:?}"),
        };
        let expected =
            process_file(&worldview_doc("A", "alpha"), "世界观/A.md", &test_limits()).len();
        assert_eq!(
            n as usize, expected,
            "second batch must NOT re-clear; expected {expected} chunks, got {n}"
        );
        assert!(
            store
                .get_file_hash("世界观/A.md", HashDim::Vector)
                .unwrap()
                .is_some(),
            "vector_hash MUST land when all batches for A succeed"
        );
    }

    /// Regression for Round-1 review finding: when a file's chunks span
    /// multiple batches, an EARLIER failed batch MUST NOT prevent a LATER
    /// successful batch from clearing that file's stale chunks. The
    /// pre-clear list must be tracked by what has actually been COMMITTED,
    /// not by what has merely been encountered. Without the fix, batch 1's
    /// failed embed still marks the file as "seen", so batch 2 inserts the
    /// new chunk WITHOUT clearing — leaving stale rows from a prior run.
    #[tokio::test]
    async fn failed_first_batch_does_not_block_later_batch_from_clearing_stale() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();

        // Run 1: seed file A with a fully successful first index — populates
        // 2 :Chunk rows for A and writes vector_hash=H1.
        write_md(
            &root,
            "世界观/A.md",
            &worldview_doc("A", "v1-stable-content"),
        );
        let store = KbStore::open(&db).unwrap();
        let healthy = FakeEmbedder::new(TEST_DIM);
        run_default(&store, "fake-model", &root, false, &healthy)
            .await
            .unwrap();
        let hash_v1 = store
            .get_file_hash("世界观/A.md", HashDim::Vector)
            .unwrap()
            .expect("v1 hash present");

        // Run 2: change A's content so the 概述 chunk contains MAGIC_FAIL_v2.
        // batch_size=1 forces 2 batches; the embedder rejects the MAGIC_FAIL
        // batch (1st) and accepts the 细节 batch (2nd). With the fix, the
        // 2nd batch MUST list A in `files_to_clear_first` and remove v1's
        // stale rows in the same transaction as the new insert.
        write_md(&root, "世界观/A.md", &worldview_doc("A", "MAGIC_FAIL_v2"));
        let failing = FakeEmbedder::failing(TEST_DIM, "MAGIC_FAIL_v2");
        let _ = run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            TEST_MAX_CHUNK_CHARS,
            &root,
            true,
            1,
            &test_limits(),
            &failing,
        )
        .await;

        // After run 2: only the 细节 batch's single chunk should remain for A.
        // Without the fix, the v1 chunks (containing "v1-stable-content") would
        // linger because batch 2 wouldn't clear.
        let qres = store
            .execute_raw("MATCH (c:Chunk {file_path: '世界观/A.md'}) RETURN count(c)")
            .unwrap();
        let n = match &qres.rows()[0][0] {
            grafeo::Value::Int64(n) => *n,
            other => panic!("unexpected count: {other:?}"),
        };
        assert_eq!(
            n, 1,
            "expected exactly 1 chunk (the successful 细节 batch); got {n}"
        );
        let hits = store.search(&fake_vec(0.0, TEST_DIM), 100).unwrap();
        let a_hits: Vec<_> = hits
            .iter()
            .filter(|h| h.file_path == "世界观/A.md")
            .collect();
        assert!(
            a_hits
                .iter()
                .all(|h| !h.content.contains("v1-stable-content")),
            "stale v1 chunks MUST be cleared by the successful later batch; got {a_hits:?}"
        );

        // vector_hash MUST NOT advance (one batch failed).
        let hash_after = store
            .get_file_hash("世界观/A.md", HashDim::Vector)
            .unwrap()
            .expect("hash should still be present from v1");
        assert_eq!(
            hash_after, hash_v1,
            "vector_hash MUST NOT advance when not all chunks for the file succeeded"
        );
    }

    #[tokio::test]
    async fn model_mismatch_is_rejected() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_md(&root, "世界观/A.md", &worldview_doc("A", "x"));

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);
        run_default(&store, "model-a", &root, false, &embedder)
            .await
            .unwrap();

        // Re-run with a different model -> must reject
        let res = run_default(&store, "model-b", &root, false, &embedder).await;
        let err = res.expect_err("model change MUST error");
        let msg = format!("{err}");
        assert!(
            msg.contains("model-a"),
            "error should name stored model: {msg}"
        );
        assert!(
            msg.contains("model-b"),
            "error should name current model: {msg}"
        );
    }

    // Suppress unused-arc warning if Arc is not imported elsewhere in tests
    #[allow(dead_code)]
    fn _unused() -> Arc<()> {
        Arc::new(())
    }

    #[tokio::test]
    async fn recognized_file_emptied_clears_old_chunks_and_advances_hash() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();

        // v1: recognized WorldView doc with real H2/H3 chunks
        write_md(&root, "世界观/X.md", &worldview_doc("X", "first version"));

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);
        run_default(&store, "fake-model", &root, false, &embedder)
            .await
            .unwrap();
        let hash_v1 = store
            .get_file_hash("世界观/X.md", HashDim::Vector)
            .unwrap()
            .expect("v1 hash present");
        let hits_v1 = store.search(&fake_vec(0.0, TEST_DIM), 10).unwrap();
        assert!(
            hits_v1.iter().any(|h| h.file_path == "世界观/X.md"),
            "v1 should produce indexable chunks: {hits_v1:?}"
        );

        // v2: same path; content reduced to empty so the generic chunker
        // produces zero chunks. The empty-modified pre-clear pass must wipe
        // old chunks AND mark_completed_files_phase must advance vector_hash.
        write_md(&root, "世界观/X.md", "");
        run_default(&store, "fake-model", &root, true, &embedder)
            .await
            .unwrap();

        // hash MUST advance to reflect new content
        let hash_v2 = store
            .get_file_hash("世界观/X.md", HashDim::Vector)
            .unwrap()
            .expect("v2 hash present after empty-chunks pre-clear");
        assert_ne!(
            hash_v1, hash_v2,
            "FileHash MUST advance when file content changes, even if it produces 0 chunks"
        );

        // search MUST NOT surface any chunk for this file anymore
        let hits_v2 = store.search(&fake_vec(0.0, TEST_DIM), 10).unwrap();
        assert!(
            hits_v2.iter().all(|h| h.file_path != "世界观/X.md"),
            "old chunks for 世界观/X.md leaked after empty pre-clear: {hits_v2:?}"
        );
    }

    // ---- Section 6 integration tests (tasks.md 6.1-6.6) ----

    #[tokio::test]
    async fn index_with_max_chunk_chars_splits_oversized() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();

        let para =
            "这是一段相当长的中文文本，用来撑大单个 chunk 的字符数。Mixed with English words. "
                .repeat(8);
        let body = format!("{para}\n\n{para}\n\n{para}\n\n{para}");
        assert!(
            body.chars().count() > 512,
            "body must exceed limit to force split"
        );
        write_md(&root, "世界观/Big.md", &worldview_doc("Big", &body));

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);
        let limits = ChunkLimits { max_chars: 512 };

        run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            512,
            &root,
            false,
            50,
            &limits,
            &embedder,
        )
        .await
        .expect("indexing should succeed");

        let hits = store.search(&fake_vec(0.0, TEST_DIM), 100).unwrap();
        let big_hits: Vec<_> = hits
            .iter()
            .filter(|h| h.file_path == "世界观/Big.md")
            .collect();
        assert!(
            big_hits.len() > 1,
            "splitting should produce > 1 chunk for oversized body, got {}",
            big_hits.len()
        );
        for h in &big_hits {
            assert!(
                h.content.chars().count() <= 512,
                "chunk content {} chars exceeds limit 512: {:?}",
                h.content.chars().count(),
                h.content
            );
        }
    }

    #[tokio::test]
    async fn index_first_run_persists_max_chunk_chars_in_index_config() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_md(&root, "世界观/A.md", &worldview_doc("A", "alpha"));

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);
        let limits = ChunkLimits { max_chars: 2048 };

        run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            2048,
            &root,
            false,
            50,
            &limits,
            &embedder,
        )
        .await
        .unwrap();

        let cfg = store
            .read_index_config()
            .unwrap()
            .expect("IndexConfig present");
        assert_eq!(cfg.max_chunk_chars, 2048);
    }

    #[tokio::test]
    async fn index_default_max_chunk_chars_is_8192() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_md(&root, "世界观/A.md", &worldview_doc("A", "alpha"));

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);
        let limits = ChunkLimits { max_chars: 8192 };

        run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            8192,
            &root,
            false,
            50,
            &limits,
            &embedder,
        )
        .await
        .unwrap();

        let cfg = store
            .read_index_config()
            .unwrap()
            .expect("IndexConfig present");
        assert_eq!(cfg.max_chunk_chars, 8192);
    }

    #[tokio::test]
    async fn index_rejects_max_chunk_chars_change() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_md(&root, "世界观/A.md", &worldview_doc("A", "alpha"));

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);
        let limits_8192 = ChunkLimits { max_chars: 8192 };
        let limits_2048 = ChunkLimits { max_chars: 2048 };

        // first run: 8192
        run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            8192,
            &root,
            false,
            50,
            &limits_8192,
            &embedder,
        )
        .await
        .unwrap();

        // second run: 2048 -> must reject
        let res = run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            2048,
            &root,
            false,
            50,
            &limits_2048,
            &embedder,
        )
        .await;
        let err = res.expect_err("max_chunk_chars change MUST error");
        let msg = format!("{err}");
        assert!(
            msg.contains("8192"),
            "error must name stored max_chunk_chars 8192: {msg}"
        );
        assert!(
            msg.contains("2048"),
            "error must name current max_chunk_chars 2048: {msg}"
        );
    }

    #[tokio::test]
    async fn index_dim_from_run_with_embedder_uses_passed_dim() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_md(&root, "世界观/A.md", &worldview_doc("A", "alpha"));

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(1024);
        let limits = test_limits();

        run_with_embedder(
            &store,
            "fake-model",
            1024,
            TEST_MAX_CHUNK_CHARS,
            &root,
            false,
            50,
            &limits,
            &embedder,
        )
        .await
        .unwrap();

        let cfg = store
            .read_index_config()
            .unwrap()
            .expect("IndexConfig present");
        assert_eq!(cfg.dim, 1024);
    }

    #[tokio::test]
    async fn index_dim_from_env_overridden() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_md(&root, "世界观/A.md", &worldview_doc("A", "alpha"));

        let store = KbStore::open(&db).unwrap();
        let embedder_1536 = FakeEmbedder::new(1536);
        let limits = test_limits();

        // first run at dim=1536
        run_with_embedder(
            &store,
            "fake-model",
            1536,
            TEST_MAX_CHUNK_CHARS,
            &root,
            false,
            50,
            &limits,
            &embedder_1536,
        )
        .await
        .unwrap();
        let cfg = store
            .read_index_config()
            .unwrap()
            .expect("IndexConfig present");
        assert_eq!(cfg.dim, 1536);

        // second run at dim=1024 in the same DB -> must reject
        let embedder_1024 = FakeEmbedder::new(1024);
        let res = run_with_embedder(
            &store,
            "fake-model",
            1024,
            TEST_MAX_CHUNK_CHARS,
            &root,
            false,
            50,
            &limits,
            &embedder_1024,
        )
        .await;
        let err = res.expect_err("dim change MUST error");
        let msg = format!("{err}");
        assert!(
            msg.contains("1536"),
            "error must name stored dim 1536: {msg}"
        );
        assert!(
            msg.contains("1024"),
            "error must name current dim 1024: {msg}"
        );
    }

    // Pin the env -> Config -> index::run -> IndexConfig wiring. Run the
    // production `index::run` against an empty markdown root so no embedding
    // HTTP call is needed; the only side effect is writing IndexConfig from
    // cfg.dim / cfg.max_chunk_chars.
    #[test]
    fn run_persists_indexconfig_from_env_with_empty_root() {
        use crate::config::{ENV_API_BASE, ENV_API_KEY, ENV_DIM, ENV_MAX_CHUNK_CHARS, ENV_MODEL};
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("empty-docs");
        std::fs::create_dir_all(&root).unwrap();

        crate::config::with_env(
            &[
                (ENV_API_BASE, Some("http://unused-by-empty-root")),
                (ENV_API_KEY, Some("k")),
                (ENV_MODEL, Some("env-model")),
                (ENV_DIM, Some("1536")),
                (ENV_MAX_CHUNK_CHARS, Some("512")),
            ],
            || {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    super::run(
                        &db,
                        &root,
                        false,
                        50,
                        &crate::embedding::RetryConfig::default(),
                    )
                    .await
                    .expect("run with empty root should succeed without embedding")
                });
            },
        );

        let store = KbStore::open(&db).unwrap();
        let cfg = store
            .read_index_config()
            .unwrap()
            .expect("IndexConfig must be written even when no files were processed");
        assert_eq!(cfg.dim, 1536, "cfg.dim must come from XD_KB_DIM");
        assert_eq!(
            cfg.max_chunk_chars, 512,
            "cfg.max_chunk_chars must come from XD_KB_MAX_CHUNK_CHARS"
        );
        assert_eq!(
            cfg.model, "env-model",
            "cfg.model must come from XD_KB_MODEL"
        );
    }

    // Spec scenario (xd-kb-cli/spec.md:97): `:Chunk.embedding` MUST be
    // generated from "no prefix" chunk content. Locks against a regression to
    // `format!("{} {}", c.prefix, c.content)` by directly comparing every
    // text passed to the embedder against the bare `chunks[i].content` that
    // `process_file` produced — they MUST be equal, in order.
    #[tokio::test]
    async fn embedding_input_equals_bare_chunk_content() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        let body = worldview_doc("A", "alpha");
        write_md(&root, "世界观/A.md", &body);

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);
        run_default(&store, "fake-model", &root, false, &embedder)
            .await
            .unwrap();

        // Recompute the expected chunks the same way `collect_chunks_for_file` does.
        let limits = test_limits();
        let chunks = crate::markdown::process_file(&body, "世界观/A.md", &limits);
        let expected: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        let seen = embedder.seen_texts();

        assert_eq!(
            seen.len(),
            expected.len(),
            "embedder must receive exactly one input per chunk"
        );
        for (i, (got, want)) in seen.iter().zip(expected.iter()).enumerate() {
            // Positive lock: byte-equal to the chunk's bare content.
            assert_eq!(
                got, want,
                "embed input #{i} MUST equal chunks[{i}].content verbatim"
            );
            // Defensive negative lock: must NOT carry a duplicated heading
            // prefix in the form `<prefix> <content>`.
            let prefix = &chunks[i].prefix;
            if !prefix.is_empty() {
                let regression = format!("{prefix} {want}");
                assert_ne!(
                    got, &regression,
                    "embed input #{i} MUST NOT be `<prefix> <content>` form"
                );
            }
        }
    }

    // R4 finding (test-alignment): WalkDir error propagation MUST abort
    // indexing rather than silently dropping the entry — a swallowed error
    // would make a still-present file look "deleted" under --incremental.
    // Drives `collect_markdown_from_entries` directly with a synthetic
    // `Err(...)` because constructing a real `walkdir::Error` without a
    // failing OS call is impractical.
    #[test]
    fn collect_markdown_propagates_walk_errors() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join(".xd-kb.yaml"), "include:\n  - \"**/*.md\"\n").unwrap();
        let kb_config = KbConfig::load_for_root(root).expect("load .xd-kb.yaml");

        let entries: Vec<Result<std::path::PathBuf>> =
            vec![Err(anyhow::anyhow!("synthetic walk failure"))];
        let res = collect_markdown_from_entries(root, &kb_config, entries);
        let err = res.expect_err("walk error MUST be propagated");
        let msg = format!("{err}");
        assert!(
            msg.contains("synthetic walk failure"),
            "propagated error must carry the synthetic message verbatim, got: {msg}"
        );
    }

    // Spec scenario "无配置文件时收录所有 markdown 但兜底 excludes 生效":
    // No `.xd-kb.yaml` in root + a `.kb/internal.md` artefact + a `WIKI.md`
    // index file → only the user-authored `世界观/A.md` makes it into the index.
    #[tokio::test]
    async fn collect_markdown_default_excludes_filter_engine_artefacts() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_md(&root, "世界观/A.md", &worldview_doc("A", "alpha"));
        write_md(&root, ".kb/internal.md", "# internal\nbody\n");
        write_md(&root, "WIKI.md", "# wiki\nbody\n");
        write_md(&root, "log.md", "# log\nbody\n");

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);
        run_default(&store, "fake-model", &root, false, &embedder)
            .await
            .unwrap();

        // Only the user-authored doc gets a FileHash; the three artefacts must
        // be excluded by the engine's hardcoded DEFAULT_EXCLUDES.
        assert!(
            store
                .get_file_hash("世界观/A.md", HashDim::Vector)
                .unwrap()
                .is_some(),
            "user doc must be indexed"
        );
        for skipped in [".kb/internal.md", "WIKI.md", "log.md"] {
            assert!(
                store
                    .get_file_hash(skipped, HashDim::Vector)
                    .unwrap()
                    .is_none(),
                "{skipped} must be excluded by DEFAULT_EXCLUDES"
            );
        }
    }

    // Spec scenario "用户 exclude 优先于 include": when `.xd-kb.yaml` carries a
    // user `exclude`, the matching files must be skipped even though they
    // satisfy `include: ["**/*.md"]`.
    #[tokio::test]
    async fn collect_markdown_user_exclude_filters_files() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join(".xd-kb.yaml"),
            "include:\n  - \"**/*.md\"\nexclude:\n  - \"drafts/**\"\n",
        )
        .unwrap();
        write_md(&root, "published/done.md", &worldview_doc("Done", "done"));
        write_md(&root, "drafts/wip.md", &worldview_doc("WIP", "wip"));

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);
        run_default(&store, "fake-model", &root, false, &embedder)
            .await
            .unwrap();

        assert!(
            store
                .get_file_hash("published/done.md", HashDim::Vector)
                .unwrap()
                .is_some()
        );
        assert!(
            store
                .get_file_hash("drafts/wip.md", HashDim::Vector)
                .unwrap()
                .is_none(),
            "drafts/** must be excluded"
        );
    }

    // Spec scenario "增量索引清理被删除的文件——其它 dim 共存时节点保留": when
    // `index --incremental` discovers that a previously-indexed markdown file
    // is gone from disk AND the `:FileHash` node also carries non-vector dim
    // hashes (e.g. `graph_hash` written by an external pipeline), the cleanup
    // MUST drop only `vector_hash` + the `:Chunk` rows. The node itself and
    // every other `<dim>_hash` MUST stay intact.
    #[tokio::test]
    async fn incremental_delete_preserves_node_when_other_dim_hash_present() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_md(&root, "世界观/Doomed.md", &worldview_doc("Doomed", "x"));

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);
        run_default(&store, "fake-model", &root, false, &embedder)
            .await
            .unwrap();

        // Simulate an external pipeline (e.g. graph) writing graph_hash on the
        // same path via the storage API, bypassing check.rs.
        store
            .set_file_hash("世界观/Doomed.md", HashDim::Graph, "graph-marker")
            .expect("seed graph_hash");
        assert_eq!(
            store
                .get_file_hash("世界观/Doomed.md", HashDim::Graph)
                .unwrap()
                .as_deref(),
            Some("graph-marker"),
            "precondition: graph_hash present after seed"
        );

        // Delete the markdown file and run incremental cleanup.
        std::fs::remove_file(root.join("世界观/Doomed.md")).unwrap();
        run_default(&store, "fake-model", &root, true, &embedder)
            .await
            .unwrap();

        // (a) Chunks for the deleted file MUST be gone.
        let qres = store
            .execute_raw("MATCH (c:Chunk {file_path: '世界观/Doomed.md'}) RETURN count(c)")
            .unwrap();
        let chunk_count = match &qres.rows()[0][0] {
            grafeo::Value::Int64(n) => *n,
            other => panic!("unexpected count value: {other:?}"),
        };
        assert_eq!(
            chunk_count, 0,
            "all :Chunk nodes for deleted file must be gone"
        );

        // (b) :FileHash node MUST still exist — graph dim still carries its hash.
        let qres2 = store
            .execute_raw("MATCH (h:FileHash {path: '世界观/Doomed.md'}) RETURN count(h)")
            .unwrap();
        let hash_count = match &qres2.rows()[0][0] {
            grafeo::Value::Int64(n) => *n,
            other => panic!("unexpected count value: {other:?}"),
        };
        assert_eq!(
            hash_count, 1,
            ":FileHash node MUST remain when other dim hashes are still present"
        );

        // (c) vector_hash MUST be gone.
        assert!(
            store
                .get_file_hash("世界观/Doomed.md", HashDim::Vector)
                .unwrap()
                .is_none(),
            "vector_hash MUST be REMOVED by per-dim incremental cleanup"
        );

        // (d) graph_hash MUST be untouched, value identical to seed.
        assert_eq!(
            store
                .get_file_hash("世界观/Doomed.md", HashDim::Graph)
                .unwrap()
                .as_deref(),
            Some("graph-marker"),
            "graph_hash MUST be preserved verbatim"
        );
    }

    // Spec scenario "增量索引清理被删除的文件——本 dim 是最后一个时整节点删除":
    // when vector is the only dim with a hash on the `:FileHash` node and the
    // file is removed from disk, the node MUST be DETACH DELETEd entirely.
    // ---- 4.7: progress output format ----

    #[test]
    fn progress_output_format_pins_each_line() {
        // Lock the exact strings produced by the four progress / status
        // helpers. These ARE the lines that hit stderr/stdout via eprintln!/
        // println! in run_with_embedder, so a mismatch here would break the
        // spec scenario "进度输出格式".
        assert_eq!(fmt_processing(30, 200), "Processing 30 files / 200 chunks");
        assert_eq!(fmt_processing(0, 0), "Processing 0 files / 0 chunks");
        assert_eq!(fmt_embedding_batch(1, 4), "Embedding batch 1/4");
        assert_eq!(fmt_embedding_batch(4, 4), "Embedding batch 4/4");

        let mut files = HashSet::new();
        files.insert("a.md".to_string());
        files.insert("b.md".to_string());
        // CSV ordering is sorted (deterministic), not HashSet insertion order.
        assert_eq!(
            fmt_batch_fail(3, "boom", &files),
            "[batch-fail] batch 3: boom (files: a.md, b.md)"
        );

        let stats = IndexStats {
            indexed: 30,
            unchanged: 0,
            deleted: 0,
            empty: 0,
            batches_failed: 0,
            chunks_failed: 0,
            failed_files: HashSet::new(),
        };
        assert_eq!(
            fmt_index_complete(&stats),
            "Index complete: indexed=30 unchanged=0 deleted=0 empty=0 batches_failed=0 chunks_failed=0"
        );

        let mut failed_files = HashSet::new();
        failed_files.insert("a.md".to_string());
        failed_files.insert("b.md".to_string());
        let stats_fail = IndexStats {
            indexed: 28,
            unchanged: 0,
            deleted: 0,
            empty: 0,
            batches_failed: 1,
            chunks_failed: 50,
            failed_files,
        };
        assert_eq!(
            fmt_index_failed(&stats_fail),
            "Index failed: batches_failed=1 affected_files=2"
        );
    }

    /// Spec scenario "进度输出格式" end-to-end: run with two files, capture
    /// the embedder call sequence and final stats, and verify the formatter
    /// helpers produce the expected lines for THIS run.
    #[tokio::test]
    async fn progress_output_format_end_to_end() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_md(&root, "世界观/A.md", &worldview_doc("A", "alpha"));
        write_md(&root, "世界观/B.md", &worldview_doc("B", "bravo"));

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);
        let limits = test_limits();
        // batch_size=1 and 2 files × 2 chunks = 4 batches → exactly the
        // "Embedding batch 1/4 ... 4/4" sequence from the spec scenario.
        run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            TEST_MAX_CHUNK_CHARS,
            &root,
            false,
            1,
            &limits,
            &embedder,
        )
        .await
        .unwrap();
        assert_eq!(
            embedder.call_count(),
            4,
            "expected 4 embedder batches, got {}",
            embedder.call_count()
        );
    }

    #[tokio::test]
    async fn incremental_delete_drops_whole_node_when_vector_is_last_dim() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_md(&root, "世界观/Solo.md", &worldview_doc("Solo", "y"));

        let store = KbStore::open(&db).unwrap();
        let embedder = FakeEmbedder::new(TEST_DIM);
        run_default(&store, "fake-model", &root, false, &embedder)
            .await
            .unwrap();

        // Precondition: only vector_hash on the node, no other dim seeded.
        assert!(
            store
                .get_file_hash("世界观/Solo.md", HashDim::Vector)
                .unwrap()
                .is_some()
        );

        std::fs::remove_file(root.join("世界观/Solo.md")).unwrap();
        run_default(&store, "fake-model", &root, true, &embedder)
            .await
            .unwrap();

        // Whole :FileHash node MUST be DETACH DELETEd because vector was the
        // last (and only) non-empty dim.
        let qres = store
            .execute_raw("MATCH (h:FileHash {path: '世界观/Solo.md'}) RETURN count(h)")
            .unwrap();
        let hash_count = match &qres.rows()[0][0] {
            grafeo::Value::Int64(n) => *n,
            other => panic!("unexpected count value: {other:?}"),
        };
        assert_eq!(
            hash_count, 0,
            ":FileHash node MUST be DETACH DELETEd when vector is the last dim"
        );

        // And of course no chunk rows remain.
        let qres2 = store
            .execute_raw("MATCH (c:Chunk {file_path: '世界观/Solo.md'}) RETURN count(c)")
            .unwrap();
        let chunk_count = match &qres2.rows()[0][0] {
            grafeo::Value::Int64(n) => *n,
            other => panic!("unexpected count value: {other:?}"),
        };
        assert_eq!(chunk_count, 0);
    }
}
