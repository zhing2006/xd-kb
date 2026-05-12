use std::collections::HashSet;
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
    failed: usize,
    empty: usize,
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

    // 1. IndexConfig: write on first run, validate on subsequent.
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

    // 2. Enumerate disk markdown files
    let entries = collect_markdown(root)?;
    let on_disk: HashSet<String> = entries.iter().map(|e| e.rel_path.clone()).collect();

    let mut stats = IndexStats::default();

    // 3. Per-file processing (each upsert is its own grafeo transaction)
    for FileEntry {
        rel_path,
        hash,
        content,
    } in &entries
    {
        if incremental
            && let Some(stored) = store.get_file_hash(rel_path, HashDim::Vector)?
            && &stored == hash
        {
            stats.unchanged += 1;
            continue;
        }

        match process_one_file(store, rel_path, hash, content, embedder, batch_size, limits).await {
            Ok(0) => {
                stats.empty += 1;
            }
            Ok(_) => {
                stats.indexed += 1;
            }
            Err(e) => {
                eprintln!("[fail] {rel_path}: {e}");
                stats.failed += 1;
            }
        }
    }

    // 4. Cleanup files removed from disk (incremental mode only).
    if incremental {
        let stored_paths = store.list_file_paths()?;
        for p in stored_paths {
            if !on_disk.contains(&p) {
                // Per-dim cleanup: only the vector pipeline's data
                // (`:Chunk` rows + `vector_hash` property) is removed; other
                // dims' hashes on the same `:FileHash` node stay intact.
                if let Err(e) = store.delete_vector_dim_for_file(&p) {
                    eprintln!("[fail-delete] {p}: {e}");
                    stats.failed += 1;
                } else {
                    stats.deleted += 1;
                }
            }
        }
    }

    println!(
        "Index complete: indexed={} unchanged={} deleted={} empty={} failed={}",
        stats.indexed, stats.unchanged, stats.deleted, stats.empty, stats.failed
    );
    if stats.failed > 0 {
        bail!("{} file(s) failed; see [fail] entries above", stats.failed);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn process_one_file(
    store: &KbStore,
    rel_path: &str,
    file_hash: &str,
    content: &str,
    embedder: &dyn Embedder,
    batch_size: usize,
    limits: &ChunkLimits,
) -> Result<usize> {
    let chunks: Vec<Chunk> = process_file(content, rel_path, limits);
    // Empty chunk list (e.g. an empty markdown file): still treat as a valid upsert
    // with zero chunks so storage deletes any prior chunks AND refreshes the
    // FileHash atomically. Without this, a file that previously produced chunks
    // but no longer does would leave stale chunks in grafeo and incremental mode
    // would re-process it every run (because the hash never advances).
    if chunks.is_empty() {
        store.upsert_chunks_for_file(rel_path, file_hash, &[])?;
        return Ok(0);
    }

    // Embedding inputs are the raw chunk content (per spec: `:Chunk.embedding`
    // MUST be generated from "no prefix" chunk content). The heading-path
    // metadata is still surfaced as `:Chunk.prefix` below for downstream
    // consumers, but it never enters the embedding model's input.
    let inputs: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();

    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
    for batch in inputs.chunks(batch_size) {
        let v = embedder.embed(batch).await?;
        if v.len() != batch.len() {
            bail!(
                "embedding count mismatch in batch (expected {}, got {})",
                batch.len(),
                v.len()
            );
        }
        vectors.extend(v);
    }

    let chunk_inputs: Vec<ChunkInput<'_>> = chunks
        .iter()
        .zip(vectors.iter())
        .map(|(c, v)| ChunkInput {
            file_path: rel_path,
            line_start: u32::try_from(c.line_start).unwrap_or(u32::MAX),
            line_end: u32::try_from(c.line_end).unwrap_or(u32::MAX),
            content: &c.content,
            prefix: if c.prefix.is_empty() {
                None
            } else {
                Some(c.prefix.as_str())
            },
            embedding: v.clone(),
        })
        .collect();

    store.upsert_chunks_for_file(rel_path, file_hash, &chunk_inputs)?;
    Ok(chunk_inputs.len())
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

        // FileHash for both files present
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

    #[tokio::test]
    async fn embedding_failure_rolls_back_just_one_file() {
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

        // run returns Err because some files failed, but Good.md should still be indexed
        let _ = run_default(&store, "fake-model", &root, false, &embedder).await;

        assert!(
            store
                .get_file_hash("世界观/Good.md", HashDim::Vector)
                .unwrap()
                .is_some(),
            "successful file MUST be indexed despite sibling failure"
        );
        assert!(
            store
                .get_file_hash("世界观/Bad.md", HashDim::Vector)
                .unwrap()
                .is_none(),
            "failed file MUST NOT leave any FileHash record (single-file rollback)"
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
        // produces zero chunks. Without the empty-chunks upsert path the
        // file would leak: old chunks stayed in storage, hash never advanced.
        write_md(&root, "世界观/X.md", "");
        run_default(&store, "fake-model", &root, true, &embedder)
            .await
            .unwrap();

        // hash MUST advance to reflect new content
        let hash_v2 = store
            .get_file_hash("世界观/X.md", HashDim::Vector)
            .unwrap()
            .expect("v2 hash present after empty-chunks upsert");
        assert_ne!(
            hash_v1, hash_v2,
            "FileHash MUST advance when file content changes, even if it produces 0 chunks"
        );

        // search MUST NOT surface any chunk for this file anymore
        let hits_v2 = store.search(&fake_vec(0.0, TEST_DIM), 10).unwrap();
        assert!(
            hits_v2.iter().all(|h| h.file_path != "世界观/X.md"),
            "old chunks for 世界观/X.md leaked after empty-chunks upsert: {hits_v2:?}"
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

        // Recompute the expected chunks the same way `process_one_file` does.
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
