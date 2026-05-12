use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use globset::{Glob, GlobSet, GlobSetBuilder};
use serde::Deserialize;

pub const ENV_API_BASE: &str = "XD_KB_API_BASE";
pub const ENV_API_KEY: &str = "XD_KB_API_KEY";
pub const ENV_MODEL: &str = "XD_KB_MODEL";
pub const ENV_DIM: &str = "XD_KB_DIM";
pub const ENV_MAX_CHUNK_CHARS: &str = "XD_KB_MAX_CHUNK_CHARS";

pub const DEFAULT_DIM: usize = 1024;
pub const DEFAULT_MAX_CHUNK_CHARS: u32 = 8192;

#[derive(Debug, Clone)]
pub struct Config {
    pub api_base: String,
    pub api_key: String,
    pub model: String,
    pub dim: usize,
    pub max_chunk_chars: u32,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let api_base = std::env::var(ENV_API_BASE).ok();
        let api_key = std::env::var(ENV_API_KEY).ok();
        let model = std::env::var(ENV_MODEL).ok();

        let mut missing = Vec::new();
        if api_base.is_none() {
            missing.push(ENV_API_BASE);
        }
        if api_key.is_none() {
            missing.push(ENV_API_KEY);
        }
        if model.is_none() {
            missing.push(ENV_MODEL);
        }

        if !missing.is_empty() {
            bail!(
                "Missing required environment variables: {}",
                missing.join(", ")
            );
        }

        let dim = parse_positive_env::<usize>(ENV_DIM, DEFAULT_DIM)?;
        let max_chunk_chars =
            parse_positive_env::<u32>(ENV_MAX_CHUNK_CHARS, DEFAULT_MAX_CHUNK_CHARS)?;

        Ok(Self {
            api_base: api_base.unwrap(),
            api_key: api_key.unwrap(),
            model: model.unwrap(),
            dim,
            max_chunk_chars,
        })
    }
}

fn parse_positive_env<T>(name: &str, default: T) -> Result<T>
where
    T: std::str::FromStr + PartialOrd + Copy + From<u8> + std::fmt::Display,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    let raw = match std::env::var(name) {
        Ok(v) => v,
        Err(_) => return Ok(default),
    };
    let trimmed = raw.trim();
    let parsed: T = trimmed.parse().map_err(|e| {
        anyhow::anyhow!("环境变量 {name} 非法（无法解析为正整数 {trimmed:?}）: {e}")
    })?;
    let zero: T = T::from(0u8);
    if parsed <= zero {
        bail!("环境变量 {name} 必须为正整数（当前值 {trimmed:?}）");
    }
    Ok(parsed)
}

// =========================================================================
// .xd-kb.yaml — per-KB include/exclude glob filtering for `index`.
// =========================================================================

/// Filename xd-kb looks for in `<markdown-root>` to decide which files to index.
pub const KB_CONFIG_FILE: &str = ".xd-kb.yaml";

/// Engine-side default excludes — combined (set-union) with user `exclude`
/// entries. Cannot be cancelled by user `include`. Covers:
/// - `.kb/**` — engine output dir conventions
/// - `.xd-kb.yaml` — the config file itself (would otherwise be indexed)
/// - `WIKI.md` / `log.md` — common KB index/journal artefacts
pub const DEFAULT_EXCLUDES: &[&str] = &[".kb/**", ".xd-kb.yaml", "WIKI.md", "log.md"];

/// Default include when user provides none — collect every markdown.
pub const DEFAULT_INCLUDE: &str = "**/*.md";

#[derive(Debug, Default, Deserialize)]
struct KbConfigRaw {
    #[serde(default)]
    include: Option<Vec<String>>,
    #[serde(default)]
    exclude: Option<Vec<String>>,
}

/// Compiled glob filter for a single KB's `index` scope.
///
/// Construction is the only place yaml/glob errors surface — once built, calls
/// to [`KbConfig::matches`] are infallible and cheap. The filter applies to
/// **relative** paths inside the markdown root with `/` as separator (callers
/// must normalize before passing in).
#[derive(Debug)]
pub struct KbConfig {
    include: GlobSet,
    exclude: GlobSet,
}

impl KbConfig {
    /// Load filter for `<root>/.xd-kb.yaml`. When the file is absent, returns a
    /// filter built from `DEFAULT_INCLUDE` + `DEFAULT_EXCLUDES` only.
    pub fn load_for_root(root: &Path) -> Result<Self> {
        let yaml_path = root.join(KB_CONFIG_FILE);
        if yaml_path.exists() {
            let raw_text = std::fs::read_to_string(&yaml_path)
                .with_context(|| format!("read {}", yaml_path.display()))?;
            Self::from_yaml_str(&raw_text, &yaml_path)
        } else {
            Self::from_lists(&[DEFAULT_INCLUDE.to_string()], &[], &yaml_path)
        }
    }

    fn from_yaml_str(text: &str, source: &Path) -> Result<Self> {
        let raw: KbConfigRaw = serde_yaml_ng::from_str(text)
            .map_err(|e| anyhow!("parse {} failed: {e}", source.display()))?;
        let include = raw
            .include
            .unwrap_or_else(|| vec![DEFAULT_INCLUDE.to_string()]);
        let exclude = raw.exclude.unwrap_or_default();
        Self::from_lists(&include, &exclude, source)
    }

    fn from_lists(include: &[String], exclude: &[String], source: &Path) -> Result<Self> {
        let include = build_glob_set(include, source, "include")?;
        // Default excludes are merged set-union with user excludes — user
        // `include` cannot cancel them (spec hard requirement).
        let mut exclude_patterns: Vec<String> =
            DEFAULT_EXCLUDES.iter().map(|s| (*s).to_string()).collect();
        exclude_patterns.extend(exclude.iter().cloned());
        let exclude = build_glob_set(&exclude_patterns, source, "exclude")?;
        Ok(Self { include, exclude })
    }

    /// True iff `rel_path` (forward-slash, relative to the KB root) is included
    /// AND not excluded. `exclude` always wins.
    pub fn matches(&self, rel_path: &str) -> bool {
        if self.exclude.is_match(rel_path) {
            return false;
        }
        self.include.is_match(rel_path)
    }
}

fn build_glob_set(patterns: &[String], source: &Path, field: &str) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for p in patterns {
        let g = Glob::new(p).map_err(|e| {
            anyhow!(
                "{} glob pattern {p:?} (from {}) failed to compile: {e}",
                field,
                source.display()
            )
        })?;
        builder.add(g);
    }
    builder
        .build()
        .map_err(|e| anyhow!("{} glob set (from {}) failed to build: {e}", field, source.display()))
}

/// Walk upwards from `start` until a directory containing `.xd-kb.yaml` is
/// found; return that directory. Returns `None` if no ancestor (including
/// `start` itself) carries the file. Used by `check` to infer an implicit
/// `--root` when the user omitted it.
pub fn find_xd_kb_yaml_root(start: &Path) -> Option<PathBuf> {
    let mut cur = start.to_path_buf();
    loop {
        if cur.join(KB_CONFIG_FILE).exists() {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}

// env vars are process-global; tests across modules share the same lock so
// they serialise against each other rather than racing.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn with_env<F: FnOnce()>(vars: &[(&str, Option<&str>)], f: F) {
    let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let prev: Vec<(String, Option<String>)> = vars
        .iter()
        .map(|(k, _)| ((*k).to_string(), std::env::var(*k).ok()))
        .collect();
    for (k, v) in vars {
        match v {
            Some(val) => unsafe { std::env::set_var(k, val) },
            None => unsafe { std::env::remove_var(k) },
        }
    }
    f();
    for (k, v) in prev {
        match v {
            Some(val) => unsafe { std::env::set_var(&k, val) },
            None => unsafe { std::env::remove_var(&k) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_returns_all_three_when_present() {
        with_env(
            &[
                (ENV_API_BASE, Some("http://localhost/v1")),
                (ENV_API_KEY, Some("k")),
                (ENV_MODEL, Some("m")),
                (ENV_DIM, None),
                (ENV_MAX_CHUNK_CHARS, None),
            ],
            || {
                let cfg = Config::from_env().expect("config should load");
                assert_eq!(cfg.api_base, "http://localhost/v1");
                assert_eq!(cfg.api_key, "k");
                assert_eq!(cfg.model, "m");
            },
        );
    }

    #[test]
    fn from_env_returns_all_five_when_present() {
        with_env(
            &[
                (ENV_API_BASE, Some("http://localhost/v1")),
                (ENV_API_KEY, Some("k")),
                (ENV_MODEL, Some("m")),
                (ENV_DIM, Some("1536")),
                (ENV_MAX_CHUNK_CHARS, Some("4096")),
            ],
            || {
                let cfg = Config::from_env().expect("config should load");
                assert_eq!(cfg.api_base, "http://localhost/v1");
                assert_eq!(cfg.api_key, "k");
                assert_eq!(cfg.model, "m");
                assert_eq!(cfg.dim, 1536);
                assert_eq!(cfg.max_chunk_chars, 4096);
            },
        );
    }

    #[test]
    fn from_env_errors_when_api_key_missing() {
        with_env(
            &[
                (ENV_API_BASE, Some("http://x")),
                (ENV_API_KEY, None),
                (ENV_MODEL, Some("m")),
                (ENV_DIM, None),
                (ENV_MAX_CHUNK_CHARS, None),
            ],
            || {
                let err = Config::from_env().expect_err("should fail");
                let msg = format!("{err}");
                assert!(
                    msg.contains(ENV_API_KEY),
                    "error should name missing var: {msg}"
                );
            },
        );
    }

    #[test]
    fn from_env_errors_when_all_missing() {
        with_env(
            &[
                (ENV_API_BASE, None),
                (ENV_API_KEY, None),
                (ENV_MODEL, None),
                (ENV_DIM, None),
                (ENV_MAX_CHUNK_CHARS, None),
            ],
            || {
                let err = Config::from_env().expect_err("should fail");
                let msg = format!("{err}");
                assert!(msg.contains(ENV_API_BASE));
                assert!(msg.contains(ENV_API_KEY));
                assert!(msg.contains(ENV_MODEL));
            },
        );
    }

    #[test]
    fn from_env_dim_and_max_chunk_chars_default_when_unset() {
        with_env(
            &[
                (ENV_API_BASE, Some("http://x")),
                (ENV_API_KEY, Some("k")),
                (ENV_MODEL, Some("m")),
                (ENV_DIM, None),
                (ENV_MAX_CHUNK_CHARS, None),
            ],
            || {
                let cfg = Config::from_env().expect("config should load");
                assert_eq!(cfg.dim, DEFAULT_DIM);
                assert_eq!(cfg.dim, 1024);
                assert_eq!(cfg.max_chunk_chars, DEFAULT_MAX_CHUNK_CHARS);
                assert_eq!(cfg.max_chunk_chars, 8192);
            },
        );
    }

    #[test]
    fn from_env_dim_zero_errors() {
        with_env(
            &[
                (ENV_API_BASE, Some("http://x")),
                (ENV_API_KEY, Some("k")),
                (ENV_MODEL, Some("m")),
                (ENV_DIM, Some("0")),
                (ENV_MAX_CHUNK_CHARS, None),
            ],
            || {
                let err = Config::from_env().expect_err("should fail");
                let msg = format!("{err}");
                assert!(msg.contains(ENV_DIM), "error should name {ENV_DIM}: {msg}");
            },
        );
    }

    #[test]
    fn from_env_dim_non_numeric_errors() {
        with_env(
            &[
                (ENV_API_BASE, Some("http://x")),
                (ENV_API_KEY, Some("k")),
                (ENV_MODEL, Some("m")),
                (ENV_DIM, Some("abc")),
                (ENV_MAX_CHUNK_CHARS, None),
            ],
            || {
                let err = Config::from_env().expect_err("should fail");
                let msg = format!("{err}");
                assert!(msg.contains(ENV_DIM), "error should name {ENV_DIM}: {msg}");
            },
        );
    }

    #[test]
    fn from_env_max_chunk_chars_zero_errors() {
        with_env(
            &[
                (ENV_API_BASE, Some("http://x")),
                (ENV_API_KEY, Some("k")),
                (ENV_MODEL, Some("m")),
                (ENV_DIM, None),
                (ENV_MAX_CHUNK_CHARS, Some("0")),
            ],
            || {
                let err = Config::from_env().expect_err("should fail");
                let msg = format!("{err}");
                assert!(
                    msg.contains(ENV_MAX_CHUNK_CHARS),
                    "error should name {ENV_MAX_CHUNK_CHARS}: {msg}"
                );
            },
        );
    }

    #[test]
    fn from_env_max_chunk_chars_non_numeric_errors() {
        with_env(
            &[
                (ENV_API_BASE, Some("http://x")),
                (ENV_API_KEY, Some("k")),
                (ENV_MODEL, Some("m")),
                (ENV_DIM, None),
                (ENV_MAX_CHUNK_CHARS, Some("abc")),
            ],
            || {
                let err = Config::from_env().expect_err("should fail");
                let msg = format!("{err}");
                assert!(
                    msg.contains(ENV_MAX_CHUNK_CHARS),
                    "error should name {ENV_MAX_CHUNK_CHARS}: {msg}"
                );
            },
        );
    }

    #[test]
    fn from_env_dim_set_overrides_default() {
        with_env(
            &[
                (ENV_API_BASE, Some("http://x")),
                (ENV_API_KEY, Some("k")),
                (ENV_MODEL, Some("m")),
                (ENV_DIM, Some("1536")),
                (ENV_MAX_CHUNK_CHARS, None),
            ],
            || {
                let cfg = Config::from_env().expect("config should load");
                assert_eq!(cfg.dim, 1536);
                assert_eq!(cfg.max_chunk_chars, DEFAULT_MAX_CHUNK_CHARS);
            },
        );
    }

    #[test]
    fn from_env_max_chunk_chars_set_overrides_default() {
        with_env(
            &[
                (ENV_API_BASE, Some("http://x")),
                (ENV_API_KEY, Some("k")),
                (ENV_MODEL, Some("m")),
                (ENV_DIM, None),
                (ENV_MAX_CHUNK_CHARS, Some("2048")),
            ],
            || {
                let cfg = Config::from_env().expect("config should load");
                assert_eq!(cfg.dim, DEFAULT_DIM);
                assert_eq!(cfg.max_chunk_chars, 2048);
            },
        );
    }

    // -----------------------------------------------------------------
    // KbConfig (.xd-kb.yaml) tests — cover the 4 spec scenarios:
    //   - "无配置文件时收录所有 markdown 但兜底 excludes 生效"
    //   - "配置驱动收录 + 默认 excludes 不可被 include 抵消"
    //   - "用户 exclude 优先于 include"
    //   - "yaml 语法错误时报错"
    // -----------------------------------------------------------------

    use std::path::PathBuf;
    use tempfile::tempdir;

    fn yaml_path_in(dir: &Path) -> PathBuf {
        dir.join(KB_CONFIG_FILE)
    }

    #[test]
    fn kb_config_no_yaml_collects_all_md_with_default_excludes() {
        // Scenario: 无配置文件时收录所有 markdown 但兜底 excludes 生效
        let tmp = tempdir().unwrap();
        let cfg = KbConfig::load_for_root(tmp.path()).expect("load with no yaml");
        // Plain markdown files are included.
        assert!(cfg.matches("a.md"));
        assert!(cfg.matches("b.md"));
        // Default excludes still bite: `.kb/**`, `.xd-kb.yaml`, `WIKI.md`, `log.md`.
        assert!(!cfg.matches(".kb/internal.md"));
        assert!(!cfg.matches("WIKI.md"));
        assert!(!cfg.matches("log.md"));
        assert!(!cfg.matches(".xd-kb.yaml"));
    }

    #[test]
    fn kb_config_user_include_cannot_cancel_default_excludes() {
        // Scenario: 配置驱动收录 + 默认 excludes 不可被 include 抵消
        let tmp = tempdir().unwrap();
        std::fs::write(
            yaml_path_in(tmp.path()),
            "include:\n  - \"**/*.md\"\n  - \".kb/**/*.md\"\n",
        )
        .unwrap();
        let cfg = KbConfig::load_for_root(tmp.path()).expect("load yaml");
        assert!(cfg.matches("notes/x.md"));
        assert!(
            !cfg.matches(".kb/y.md"),
            ".kb/** default exclude must override user include"
        );
    }

    #[test]
    fn kb_config_user_exclude_overrides_include() {
        // Scenario: 用户 exclude 优先于 include
        let tmp = tempdir().unwrap();
        std::fs::write(
            yaml_path_in(tmp.path()),
            "include:\n  - \"**/*.md\"\nexclude:\n  - \"drafts/**\"\n",
        )
        .unwrap();
        let cfg = KbConfig::load_for_root(tmp.path()).expect("load yaml");
        assert!(cfg.matches("published/done.md"));
        assert!(!cfg.matches("drafts/wip.md"));
    }

    #[test]
    fn kb_config_yaml_syntax_error_reports_path_and_error() {
        // Scenario: yaml 语法错误时报错
        let tmp = tempdir().unwrap();
        std::fs::write(
            yaml_path_in(tmp.path()),
            "include:\n  - **/*.md\n  not-yaml: [unterminated\n",
        )
        .unwrap();
        let err = KbConfig::load_for_root(tmp.path()).expect_err("syntax error must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains(KB_CONFIG_FILE),
            "error must point at .xd-kb.yaml: {msg}"
        );
    }

    #[test]
    fn kb_config_invalid_glob_pattern_errors() {
        let tmp = tempdir().unwrap();
        std::fs::write(
            yaml_path_in(tmp.path()),
            "include:\n  - \"[unbalanced\"\n",
        )
        .unwrap();
        let err = KbConfig::load_for_root(tmp.path()).expect_err("bad glob must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains(KB_CONFIG_FILE) || msg.contains("[unbalanced"),
            "error must reference the source or pattern: {msg}"
        );
    }

    #[test]
    fn kb_config_missing_include_defaults_to_all_md() {
        let tmp = tempdir().unwrap();
        std::fs::write(yaml_path_in(tmp.path()), "exclude:\n  - \"x.md\"\n").unwrap();
        let cfg = KbConfig::load_for_root(tmp.path()).expect("load yaml");
        assert!(cfg.matches("a.md"));
        assert!(!cfg.matches("x.md"));
    }

    #[test]
    fn find_xd_kb_yaml_root_walks_upward_and_returns_none_when_absent() {
        let tmp = tempdir().unwrap();
        let inner = tmp.path().join("a/b/c");
        std::fs::create_dir_all(&inner).unwrap();
        std::fs::write(yaml_path_in(tmp.path()), "include:\n  - \"**/*.md\"\n").unwrap();
        let found = find_xd_kb_yaml_root(&inner).expect("walk should hit tmp root");
        assert_eq!(found, tmp.path());

        // After removing the marker, lookup must return None.
        std::fs::remove_file(yaml_path_in(tmp.path())).unwrap();
        // We cannot guarantee no .xd-kb.yaml exists higher up the real fs, so
        // we only assert that *if* something was found, it is not under tmp.
        if let Some(p) = find_xd_kb_yaml_root(&inner) {
            assert!(!p.starts_with(tmp.path()));
        }
    }

    #[test]
    fn config_drives_chunk_limits_and_dim_from_env() {
        // Documents the production path index::run uses: env vars feed into
        // Config::from_env, and ChunkLimits + dim + IndexConfig.max_chunk_chars
        // all derive from that single Config so they cannot drift. Without this
        // test, only the parsing side is covered (config tests) and only the
        // engine side is covered (index/query tests with explicit dim/limits);
        // this round-trip ties them together.
        use crate::markdown::ChunkLimits;
        with_env(
            &[
                (ENV_API_BASE, Some("http://x")),
                (ENV_API_KEY, Some("k")),
                (ENV_MODEL, Some("m")),
                (ENV_DIM, Some("1536")),
                (ENV_MAX_CHUNK_CHARS, Some("512")),
            ],
            || {
                let cfg = Config::from_env().expect("config should load");
                let limits = ChunkLimits {
                    max_chars: cfg.max_chunk_chars,
                };
                assert_eq!(cfg.dim, 1536);
                assert_eq!(cfg.max_chunk_chars, 512);
                assert_eq!(limits.max_chars, cfg.max_chunk_chars);
            },
        );
    }
}
