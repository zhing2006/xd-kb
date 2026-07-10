use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use anyhow::{Result, bail};

use crate::config::Config;
use crate::embedding::{RetryConfig, embed_one};
use crate::storage::{KbStore, SearchResult};

pub type EmbedOneFut<'a> = Pin<Box<dyn Future<Output = Result<Vec<f32>>> + Send + 'a>>;

pub trait QueryEmbedder: Send + Sync {
    fn embed<'a>(&'a self, text: &'a str) -> EmbedOneFut<'a>;
}

struct ApiQueryEmbedder {
    cfg: Config,
    retry: RetryConfig,
}

impl QueryEmbedder for ApiQueryEmbedder {
    fn embed<'a>(&'a self, text: &'a str) -> EmbedOneFut<'a> {
        Box::pin(embed_one(&self.cfg, &self.retry, text))
    }
}

pub async fn run(
    db: &Path,
    text: &str,
    top: usize,
    lambda: f32,
    fetch_pool: usize,
    full: bool,
    retry: &RetryConfig,
) -> Result<()> {
    let cfg = Config::from_env()?;
    let model = cfg.model.clone();
    let dim = cfg.dim;
    let max_chunk_chars = cfg.max_chunk_chars;
    let embedder = ApiQueryEmbedder { cfg, retry: *retry };
    let store = KbStore::open(db)?;
    let results = run_with_embedder(
        &store,
        &model,
        dim,
        max_chunk_chars,
        text,
        top,
        lambda,
        fetch_pool,
        &embedder,
    )
    .await?;
    println!("{}", format_results(&results, full));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_with_embedder(
    store: &KbStore,
    model: &str,
    dim: usize,
    max_chunk_chars: u32,
    text: &str,
    top: usize,
    lambda: f32,
    fetch_pool: usize,
    embedder: &dyn QueryEmbedder,
) -> Result<Vec<SearchResult>> {
    if top == 0 {
        bail!("--top must be >= 1");
    }

    // Refuse to run if the DB was never indexed (or the stored config is incompatible).
    // MUST NOT touch the KNN index when validation fails — error before any vector op.
    match store.read_index_config()? {
        None => bail!("DB has no IndexConfig — run `xd-kb --db <path> index <root>` first"),
        Some(existing)
            if existing.model != model
                || existing.dim != dim
                || existing.max_chunk_chars != max_chunk_chars =>
        {
            bail!(
                "IndexConfig mismatch: stored model='{}' dim={} max_chunk_chars={}, current model='{model}' dim={dim} max_chunk_chars={max_chunk_chars}. Use a new --db path or rebuild.",
                existing.model,
                existing.dim,
                existing.max_chunk_chars,
            )
        }
        Some(_) => {}
    }

    // grafeo (still true on 0.5.42, re-verified 2026-07-10) does not restore the
    // HNSW vector index across process open/close for directory-format DBs — the
    // procedure errors with "No vector index found for :Chunk(embedding)" on a
    // fresh process. Rebuild from current node state once per query process; this
    // is the ONLY place the index gets built (the index pipeline deliberately
    // skips it — see `append_chunks_batch`). ~1.15ms per chunk at dim 2048
    // (m=32, ef_construction=400): ~5s on a 4.6K-chunk KB.
    store.ensure_chunk_vector_index(dim)?;

    let q_vec = embedder.embed(text).await?;
    // lambda >= 1.0 = pure cosine — skip MMR's vector-reload + pairwise pass.
    if lambda >= 1.0 {
        store.search(&q_vec, top)
    } else {
        store.search_mmr(&q_vec, top, fetch_pool, lambda)
    }
}

fn format_results(results: &[SearchResult], full: bool) -> String {
    if results.is_empty() {
        return "(no results)".to_string();
    }
    let mut out = String::new();
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            out.push_str(if full { "\n\n" } else { "\n" });
        }
        let header = format!(
            "{:>2}. [{:.4}] {}:{}-{}",
            i + 1,
            r.score,
            r.file_path,
            r.line_start,
            r.line_end,
        );
        out.push_str(&header);
        out.push('\n');
        if full {
            for (j, line) in r.content.lines().enumerate() {
                if j > 0 {
                    out.push('\n');
                }
                out.push_str("    ");
                out.push_str(line);
            }
        } else {
            out.push_str("    ");
            out.push_str(
                &r.content
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(120)
                    .collect::<String>(),
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::ChunkInput;
    use tempfile::tempdir;

    fn sample_results() -> Vec<SearchResult> {
        let long_first_line: String = "x".repeat(150);
        let multi_line = format!("{long_first_line}\nsecond line\nthird line");
        vec![
            SearchResult {
                file_path: "doc-a.md".to_string(),
                line_start: 1,
                line_end: 5,
                content: multi_line,
                score: 0.9876,
            },
            SearchResult {
                file_path: "doc-b.md".to_string(),
                line_start: 10,
                line_end: 12,
                content: "single line content".to_string(),
                score: 0.5,
            },
        ]
    }

    #[test]
    fn format_results_compact_truncates_first_line() {
        let results = sample_results();
        let truncated_first: String = "x".repeat(120);
        let expected = format!(
            " 1. [0.9876] doc-a.md:1-5\n    {truncated_first}\n 2. [0.5000] doc-b.md:10-12\n    single line content"
        );
        assert_eq!(format_results(&results, false), expected);
    }

    #[test]
    fn format_results_full_preserves_content() {
        let results = sample_results();
        let long_first_line: String = "x".repeat(150);
        let expected = format!(
            " 1. [0.9876] doc-a.md:1-5\n    {long_first_line}\n    second line\n    third line\n\n 2. [0.5000] doc-b.md:10-12\n    single line content"
        );
        assert_eq!(format_results(&results, true), expected);
    }

    #[test]
    fn format_results_empty_returns_no_results_marker() {
        assert_eq!(format_results(&[], false), "(no results)");
        assert_eq!(format_results(&[], true), "(no results)");
    }

    struct FixedQueryEmbedder {
        vec: Vec<f32>,
    }

    impl QueryEmbedder for FixedQueryEmbedder {
        fn embed<'a>(&'a self, _text: &'a str) -> EmbedOneFut<'a> {
            let v = self.vec.clone();
            Box::pin(async move { Ok(v) })
        }
    }

    fn seed_chunks(store: &KbStore, dim: usize, max_chunk_chars: u32) {
        store
            .write_index_config("fake-model", dim, max_chunk_chars)
            .unwrap();
        let chunks = [
            ChunkInput {
                file_path: "doc-a.md",
                line_start: 1,
                line_end: 10,
                content: "alpha content here",
                prefix: None,
                embedding: vec![1.0; dim],
            },
            ChunkInput {
                file_path: "doc-a.md",
                line_start: 11,
                line_end: 20,
                content: "second chunk content",
                prefix: None,
                embedding: vec![0.5; dim],
            },
            ChunkInput {
                file_path: "doc-b.md",
                line_start: 1,
                line_end: 5,
                content: "different topic entirely",
                prefix: None,
                embedding: vec![-1.0; dim],
            },
        ];
        // upsert wants a single file per call; split:
        let a: Vec<ChunkInput<'_>> = chunks.iter().take(2).map(clone_chunk).collect();
        let b: Vec<ChunkInput<'_>> = chunks.iter().skip(2).map(clone_chunk).collect();
        store
            .upsert_chunks_for_file("doc-a.md", "hash-a", &a)
            .unwrap();
        store
            .upsert_chunks_for_file("doc-b.md", "hash-b", &b)
            .unwrap();
    }

    fn clone_chunk<'a>(c: &'a ChunkInput<'a>) -> ChunkInput<'a> {
        ChunkInput {
            file_path: c.file_path,
            line_start: c.line_start,
            line_end: c.line_end,
            content: c.content,
            prefix: c.prefix,
            embedding: c.embedding.clone(),
        }
    }

    const TEST_DIM: usize = 8;
    const TEST_MAX_CHUNK_CHARS: u32 = 8192;

    #[tokio::test]
    async fn query_returns_top_n_hits() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let store = KbStore::open(&db).unwrap();
        seed_chunks(&store, TEST_DIM, TEST_MAX_CHUNK_CHARS);

        let embedder = FixedQueryEmbedder {
            vec: vec![1.0; TEST_DIM],
        };

        let hits = run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            TEST_MAX_CHUNK_CHARS,
            "anything",
            2,
            1.0,
            8,
            &embedder,
        )
        .await
        .expect("query should succeed");

        assert!(hits.len() <= 2, "top=2 should bound result count");
        assert!(!hits.is_empty(), "should return at least one hit");
        for h in &hits {
            assert!(!h.file_path.is_empty());
            assert!(h.line_start >= 1);
            assert!(h.line_end >= h.line_start);
            assert!(!h.content.is_empty());
        }
    }

    #[tokio::test]
    async fn query_without_index_config_errors() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let store = KbStore::open(&db).unwrap();

        let embedder = FixedQueryEmbedder {
            vec: vec![1.0; TEST_DIM],
        };

        let err = run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            TEST_MAX_CHUNK_CHARS,
            "x",
            5,
            1.0,
            20,
            &embedder,
        )
        .await
        .expect_err("query without IndexConfig MUST fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("IndexConfig"),
            "error should mention IndexConfig: {msg}"
        );
    }

    #[tokio::test]
    async fn query_top_zero_is_rejected() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let store = KbStore::open(&db).unwrap();
        seed_chunks(&store, TEST_DIM, TEST_MAX_CHUNK_CHARS);

        let embedder = FixedQueryEmbedder {
            vec: vec![1.0; TEST_DIM],
        };

        let err = run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            TEST_MAX_CHUNK_CHARS,
            "x",
            0,
            1.0,
            0,
            &embedder,
        )
        .await
        .expect_err("top=0 MUST fail");
        assert!(format!("{err}").contains("top"));
    }

    #[tokio::test]
    async fn query_rejects_max_chunk_chars_change() {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let store = KbStore::open(&db).unwrap();
        seed_chunks(&store, TEST_DIM, 8192);

        let embedder = FixedQueryEmbedder {
            vec: vec![1.0; TEST_DIM],
        };

        let res = run_with_embedder(
            &store,
            "fake-model",
            TEST_DIM,
            2048,
            "anything",
            5,
            1.0,
            20,
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
        assert!(
            msg.contains("IndexConfig"),
            "error should mention IndexConfig: {msg}"
        );
    }

    /// Seed a homogeneous corpus where the cosine top-3 collapses onto one file
    /// and MMR pulls in the outlier files. Reused by both dispatch tests.
    fn seed_homogeneous(store: &KbStore) {
        store
            .write_index_config("fake-model", 4, TEST_MAX_CHUNK_CHARS)
            .unwrap();
        let a: Vec<ChunkInput<'_>> = (0..5)
            .map(|i| ChunkInput {
                file_path: "doc-a.md",
                line_start: i + 1,
                line_end: i + 1,
                content: "a-cluster",
                prefix: None,
                embedding: vec![1.0, 0.001 * (i as f32), 0.0, 0.0],
            })
            .collect();
        store.upsert_chunks_for_file("doc-a.md", "ha", &a).unwrap();
        store
            .upsert_chunks_for_file(
                "doc-b.md",
                "hb",
                &[ChunkInput {
                    file_path: "doc-b.md",
                    line_start: 1,
                    line_end: 1,
                    content: "b-topic",
                    prefix: None,
                    embedding: vec![0.7, 0.7, 0.0, 0.0],
                }],
            )
            .unwrap();
        store
            .upsert_chunks_for_file(
                "doc-c.md",
                "hc",
                &[ChunkInput {
                    file_path: "doc-c.md",
                    line_start: 1,
                    line_end: 1,
                    content: "c-topic",
                    prefix: None,
                    embedding: vec![0.6, 0.0, 0.8, 0.0],
                }],
            )
            .unwrap();
    }

    #[tokio::test]
    async fn query_lambda_default_uses_vector_path() {
        // lambda=1.0 → store.search → cosine top-N. We pin fetch_pool=0 so
        // an accidental detour into store.search_mmr would observe an empty
        // candidate pool and return no hits — the assertion below would then
        // fail. With the correct dispatch, fetch_pool is ignored and the
        // cosine path returns the expected cluster-dominated top-N.
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let store = KbStore::open(&db).unwrap();
        seed_homogeneous(&store);

        let embedder = FixedQueryEmbedder {
            vec: vec![1.0, 0.0, 0.0, 0.0],
        };

        let hits = run_with_embedder(
            &store,
            "fake-model",
            4,
            TEST_MAX_CHUNK_CHARS,
            "x",
            3,
            1.0,
            0,
            &embedder,
        )
        .await
        .unwrap();
        assert!(
            !hits.is_empty(),
            "lambda=1.0 path must call store.search and return hits even with fetch_pool=0"
        );
        let files: std::collections::HashSet<_> =
            hits.iter().map(|h| h.file_path.clone()).collect();
        assert!(
            files.len() <= 2,
            "cosine path should keep cluster dominance, got files {files:?}"
        );
    }

    #[tokio::test]
    async fn query_lambda_below_one_uses_mmr_path() {
        // lambda=0.3 → store.search_mmr → diversity wins. On the same
        // homogeneous corpus we should now reach all 3 files.
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        let store = KbStore::open(&db).unwrap();
        seed_homogeneous(&store);

        let embedder = FixedQueryEmbedder {
            vec: vec![1.0, 0.0, 0.0, 0.0],
        };

        let hits = run_with_embedder(
            &store,
            "fake-model",
            4,
            TEST_MAX_CHUNK_CHARS,
            "x",
            3,
            0.3,
            12,
            &embedder,
        )
        .await
        .unwrap();
        let files: std::collections::HashSet<_> =
            hits.iter().map(|h| h.file_path.clone()).collect();
        assert_eq!(
            files.len(),
            3,
            "mmr path with lambda=0.3 should cover all 3 files, got {files:?}"
        );
    }
}
