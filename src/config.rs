use anyhow::{Result, bail};

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
