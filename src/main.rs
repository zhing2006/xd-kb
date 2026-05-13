mod check;
mod config;
mod embedding;
mod gql;
mod index;
mod markdown;
mod query;
mod storage;

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

use crate::embedding::RetryConfig;
use crate::storage::HashDim;

fn parse_positive_f64(s: &str) -> Result<f64, String> {
    let v: f64 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;
    if !v.is_finite() || v <= 0.0 {
        return Err(format!("value must be > 0 and finite, got {v}"));
    }
    Ok(v)
}

fn parse_positive_usize(s: &str) -> Result<usize, String> {
    let v: usize = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid integer"))?;
    if v == 0 {
        return Err("value must be >= 1".to_string());
    }
    Ok(v)
}

fn parse_lambda(s: &str) -> Result<f32, String> {
    let v: f32 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;
    if !v.is_finite() || !(0.0..=1.0).contains(&v) {
        return Err("--lambda must be in [0.0, 1.0]".to_string());
    }
    Ok(v)
}

/// Resolve `--fetch-pool` for `query`: if user omitted it, default to
/// `top * 4` (grafeo's MMR default heuristic, frozen at the CLI layer);
/// otherwise enforce `>= top`. Extracted from `main` so it can be unit-tested.
fn resolve_fetch_pool(top: usize, fetch_pool: Option<usize>) -> Result<usize> {
    let v = fetch_pool.unwrap_or(top.saturating_mul(4));
    if v < top {
        bail!("--fetch-pool must be >= --top");
    }
    Ok(v)
}

#[derive(Parser, Debug)]
#[command(
    name = "xd-kb",
    version,
    about = "Local knowledge-base engine: semantic search + GQL + file-hash tracking, backed by grafeo. One DB path = one KB."
)]
struct Cli {
    /// Path to the grafeo database directory (created if missing). Each DB is one KB;
    /// run separate `--db` paths for separate knowledge bases.
    #[arg(long)]
    db: PathBuf,

    /// Backoff initial delay in seconds for embedding HTTP retries
    #[arg(long, default_value = "2", value_parser = parse_positive_f64)]
    backoff_base: f64,

    /// Backoff growth factor for embedding HTTP retries
    #[arg(long, default_value = "2", value_parser = parse_positive_f64)]
    backoff_exp: f64,

    /// Maximum embedding HTTP retry attempts
    #[arg(long, default_value = "3")]
    max_retries: u32,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Build or refresh the KB's vector index from a markdown root
    Index {
        /// Markdown root directory to scan recursively
        root: PathBuf,
        /// Only process new or modified files (compared against stored FileHash nodes)
        #[arg(long)]
        incremental: bool,
        /// Number of chunks per embedding batch
        #[arg(long, default_value = "50", value_parser = parse_positive_usize)]
        batch_size: usize,
    },
    /// Semantic query against the KB's vector index
    Query {
        /// Query text
        text: String,
        /// Number of results to return
        #[arg(long, default_value = "5", value_parser = parse_positive_usize)]
        top: usize,
        /// Relevance vs. diversity weight in [0.0, 1.0]. 1.0 (default) =
        /// pure cosine top-N (vector_search). < 1.0 = MMR diversity
        /// (mmr_search); 0.0 favours diversity, 1.0 favours relevance.
        // allow_negative_numbers so `--lambda -0.1` reaches parse_lambda for
        // the proper range error (without it clap would treat -0.1 as a flag).
        #[arg(long, default_value = "1.0", value_parser = parse_lambda, allow_negative_numbers = true)]
        lambda: f32,
        /// Candidate pool size handed to MMR before re-ranking. Defaults
        /// to `top * 4` if omitted. Must be >= --top. Ignored when
        /// --lambda >= 1.0.
        #[arg(long)]
        fetch_pool: Option<usize>,
        /// Render each hit's content in full (preserve all newlines, indent
        /// each line by 4 spaces, separate hits by one blank line) instead of
        /// the default compact rendering (first line, truncated to 120 chars).
        /// Intended for LLM Agent / RAG callers that want the complete chunk
        /// without an extra Read round-trip.
        #[arg(long)]
        full: bool,
    },
    /// Execute a raw GQL statement against the DB. The CLI does NOT auto-inject any predicate.
    Gql {
        /// GQL statement to execute
        statement: String,
    },
    /// Check / mark a file's content hash against grafeo FileHash nodes
    Check {
        /// File path. With explicit `--root <dir>`, this is interpreted
        /// relative to that root (`disk = <root>/<file>`, store key =
        /// `<file>` normalised). With implicit `.xd-kb.yaml` discovery, this
        /// is interpreted relative to **cwd** (`disk = <cwd>/<file>`, store
        /// key = `<cwd-rel-to-root>/<file>` normalised). Without any root,
        /// the path is used verbatim for both disk read and storage write.
        file: PathBuf,
        /// Update the stored hash (or remove the node if file is gone)
        #[arg(long)]
        mark: bool,
        /// Pipeline dim to operate on. Selects which `<dim>_hash` property of
        /// the `:FileHash` node is read / written. CLI takes kebab-case values
        /// (vector / vector-2 / custom-2 / ...); the property name is the
        /// snake-case `<dim>_hash` (e.g. vector-2 → vector_2_hash).
        #[arg(long = "type", value_enum)]
        dim: HashDim,
        /// Explicit KB root. `<file>` is then resolved relative to this dir
        /// for disk reads, and `:FileHash.path` is written as the
        /// forward-slash-normalized relative form so it matches what
        /// `xd-kb index <root>` writes for the same file. Omitting this flag
        /// triggers an upward search for `.xd-kb.yaml` from cwd; on a hit
        /// the discovered dir becomes the implicit root and `<file>` is
        /// interpreted **relative to cwd** (NOT to the implicit root) — the
        /// store key is then `cwd-rel-to-root + <file>`. With no `--root`
        /// and no ancestor `.xd-kb.yaml`, falls back to the legacy verbatim
        /// path behaviour.
        #[arg(long)]
        root: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let cli = Cli::parse();

    let retry_config = RetryConfig {
        backoff_base: cli.backoff_base,
        backoff_exp: cli.backoff_exp,
        max_retries: cli.max_retries,
    };

    match cli.command {
        Commands::Index {
            root,
            incremental,
            batch_size,
        } => index::run(&cli.db, &root, incremental, batch_size, &retry_config).await,
        Commands::Query {
            text,
            top,
            lambda,
            fetch_pool,
            full,
        } => {
            let fetch_pool = resolve_fetch_pool(top, fetch_pool)?;
            query::run(&cli.db, &text, top, lambda, fetch_pool, full, &retry_config).await
        }
        Commands::Gql { statement } => gql::run(&cli.db, &statement),
        Commands::Check {
            file,
            mark,
            dim,
            root,
        } => check::run(&cli.db, &file, dim, mark, root.as_deref()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(args)
    }

    #[test]
    fn parses_query_with_db() {
        let cli = try_parse(&["xd-kb", "--db", "./kb.db", "query", "测试"]).unwrap();
        assert_eq!(cli.db, PathBuf::from("./kb.db"));
        match cli.command {
            Commands::Query {
                text,
                top,
                lambda,
                fetch_pool,
                ..
            } => {
                assert_eq!(text, "测试");
                assert_eq!(top, 5);
                assert_eq!(lambda, 1.0);
                assert!(fetch_pool.is_none());
            }
            _ => panic!("expected Query"),
        }
    }

    #[test]
    fn parses_query_with_lambda() {
        let cli =
            try_parse(&["xd-kb", "--db", "./kb.db", "query", "x", "--lambda", "0.3"]).unwrap();
        match cli.command {
            Commands::Query { lambda, .. } => assert!((lambda - 0.3).abs() < 1e-6),
            _ => panic!("expected Query"),
        }
    }

    #[test]
    fn parses_query_with_fetch_pool() {
        let cli = try_parse(&[
            "xd-kb",
            "--db",
            "./kb.db",
            "query",
            "x",
            "--top",
            "5",
            "--fetch-pool",
            "20",
        ])
        .unwrap();
        match cli.command {
            Commands::Query {
                top, fetch_pool, ..
            } => {
                assert_eq!(top, 5);
                assert_eq!(fetch_pool, Some(20));
            }
            _ => panic!("expected Query"),
        }
    }

    #[test]
    fn lambda_out_of_range_rejected() {
        // Cover both `--lambda=...` and space-separated `--lambda -0.1`
        // (the latter only reaches parse_lambda thanks to allow_negative_numbers).
        let cases: &[&[&str]] = &[
            &["xd-kb", "--db", "./kb.db", "query", "x", "--lambda=1.5"],
            &["xd-kb", "--db", "./kb.db", "query", "x", "--lambda=-0.1"],
            &["xd-kb", "--db", "./kb.db", "query", "x", "--lambda", "-0.1"],
        ];
        for args in cases {
            let res = try_parse(args);
            let err = res.expect_err(&format!("{args:?} MUST be rejected"));
            let msg = err.to_string();
            assert!(
                msg.contains("[0.0, 1.0]"),
                "error must mention valid range, got: {msg}"
            );
        }
    }

    #[test]
    fn fetch_pool_below_top_rejected() {
        let err =
            resolve_fetch_pool(5, Some(3)).expect_err("--fetch-pool < --top MUST be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("--fetch-pool must be >= --top"),
            "error must explain constraint, got: {msg}"
        );
    }

    #[test]
    fn fetch_pool_default_is_top_times_four() {
        let v = resolve_fetch_pool(5, None).unwrap();
        assert_eq!(v, 20);
    }

    #[test]
    fn missing_db_is_rejected() {
        let result = try_parse(&["xd-kb", "query", "测试"]);
        let err = result.expect_err("should fail without --db");
        let msg = err.to_string();
        assert!(
            msg.contains("--db") || msg.contains(" db "),
            "error should mention db: {msg}"
        );
    }

    #[test]
    fn parses_index_with_incremental_and_batch_size() {
        let cli = try_parse(&[
            "xd-kb",
            "--db",
            "./kb.db",
            "index",
            "./docs",
            "--incremental",
            "--batch-size",
            "20",
        ])
        .unwrap();
        match cli.command {
            Commands::Index {
                root,
                incremental,
                batch_size,
            } => {
                assert_eq!(root, PathBuf::from("./docs"));
                assert!(incremental);
                assert_eq!(batch_size, 20);
            }
            _ => panic!("expected Index"),
        }
    }

    #[test]
    fn parses_gql() {
        let cli = try_parse(&["xd-kb", "--db", "./kb.db", "gql", "MATCH (n) RETURN n"]).unwrap();
        match cli.command {
            Commands::Gql { statement } => assert_eq!(statement, "MATCH (n) RETURN n"),
            _ => panic!("expected Gql"),
        }
    }

    #[test]
    fn parses_check_with_mark() {
        let cli = try_parse(&[
            "xd-kb", "--db", "./kb.db", "check", "x.md", "--mark", "--type", "vector",
        ])
        .unwrap();
        match cli.command {
            Commands::Check {
                file,
                mark,
                dim,
                root,
            } => {
                assert_eq!(file, PathBuf::from("x.md"));
                assert!(mark);
                assert_eq!(dim, HashDim::Vector);
                assert!(root.is_none(), "no --root passed → expect None");
            }
            _ => panic!("expected Check"),
        }
    }

    #[test]
    fn parses_check_with_root() {
        let cli = try_parse(&[
            "xd-kb",
            "--db",
            "./kb.db",
            "check",
            "故事/X.md",
            "--type",
            "graph",
            "--root",
            "Wiki/example-kb",
        ])
        .unwrap();
        match cli.command {
            Commands::Check {
                file, root, dim, ..
            } => {
                assert_eq!(file, PathBuf::from("故事/X.md"));
                assert_eq!(root, Some(PathBuf::from("Wiki/example-kb")));
                assert_eq!(dim, HashDim::Graph);
            }
            _ => panic!("expected Check"),
        }
    }

    #[test]
    fn check_without_type_is_rejected() {
        let result = try_parse(&["xd-kb", "--db", "./kb.db", "check", "x.md"]);
        let err = result.expect_err("clap MUST reject check without --type");
        let msg = err.to_string();
        assert!(
            msg.contains("--type") || msg.contains(" type "),
            "error must mention --type: {msg}"
        );
    }

    #[test]
    fn check_with_invalid_type_is_rejected() {
        let result = try_parse(&[
            "xd-kb",
            "--db",
            "./kb.db",
            "check",
            "x.md",
            "--type",
            "unknown_dim",
        ]);
        assert!(
            result.is_err(),
            "clap MUST reject --type unknown_dim (not in enum)"
        );
    }

    #[test]
    fn check_accepts_all_nine_dims() {
        let cases = [
            ("vector", HashDim::Vector),
            ("graph", HashDim::Graph),
            ("llm", HashDim::Llm),
            ("fulltext", HashDim::Fulltext),
            ("summary", HashDim::Summary),
            ("entity", HashDim::Entity),
            ("vector-2", HashDim::Vector2),
            ("custom", HashDim::Custom),
            ("custom-2", HashDim::Custom2),
        ];
        for (cli_value, expected) in cases {
            let cli = try_parse(&[
                "xd-kb", "--db", "./kb.db", "check", "x.md", "--type", cli_value,
            ])
            .unwrap_or_else(|e| panic!("--type {cli_value} should parse: {e}"));
            match cli.command {
                Commands::Check { dim, .. } => assert_eq!(
                    dim, expected,
                    "--type {cli_value} should resolve to {expected:?}"
                ),
                _ => panic!("expected Check"),
            }
        }

        // snake_case form (e.g. vector_2) MUST be rejected — clap default render is kebab.
        let snake_rejected = try_parse(&[
            "xd-kb", "--db", "./kb.db", "check", "x.md", "--type", "vector_2",
        ]);
        assert!(
            snake_rejected.is_err(),
            "snake_case --type vector_2 MUST be rejected; only kebab-case is accepted"
        );
    }

    #[test]
    fn batch_size_zero_is_rejected() {
        let result = try_parse(&[
            "xd-kb",
            "--db",
            "./kb.db",
            "index",
            "./docs",
            "--batch-size",
            "0",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn backoff_base_zero_rejected() {
        let result = try_parse(&[
            "xd-kb",
            "--db",
            "./kb.db",
            "--backoff-base",
            "0",
            "query",
            "x",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn defaults_for_query() {
        let cli = try_parse(&["xd-kb", "--db", "./kb.db", "query", "abc"]).unwrap();
        assert_eq!(cli.backoff_base, 2.0);
        assert_eq!(cli.backoff_exp, 2.0);
        assert_eq!(cli.max_retries, 3);
    }

    #[test]
    fn parses_query_with_full() {
        let cli = try_parse(&["xd-kb", "--db", "./kb.db", "query", "x", "--full"]).unwrap();
        match cli.command {
            Commands::Query { full, .. } => assert!(full),
            _ => panic!("expected Query"),
        }
    }

    #[test]
    fn parses_query_full_default_false() {
        let cli = try_parse(&["xd-kb", "--db", "./kb.db", "query", "x"]).unwrap();
        match cli.command {
            Commands::Query { full, .. } => assert!(!full),
            _ => panic!("expected Query"),
        }
    }
}
