use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::config::Config;

/// Exponential backoff retry configuration.
///
/// Delay formula: `delay(i) = backoff_base * backoff_exp^i`, where `i` starts at 0.
/// For 429 responses, actual delay is `max(formula_delay, Retry-After header)`.
#[derive(Debug, Clone, Copy)]
pub struct RetryConfig {
    pub backoff_base: f64,
    pub backoff_exp: f64,
    pub max_retries: u32,
}

impl RetryConfig {
    pub fn delay_secs(&self, attempt: u32) -> f64 {
        self.backoff_base * self.backoff_exp.powi(attempt as i32)
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            backoff_base: 2.0,
            backoff_exp: 2.0,
            max_retries: 3,
        }
    }
}

pub struct EmbeddingClient {
    client: reqwest::Client,
    api_base: String,
    api_key: String,
    model: String,
    dim: usize,
    retry_config: RetryConfig,
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

enum SendOutcome {
    Success(Vec<Vec<f32>>),
    RetryableError(String),
    RetryableHttpError {
        description: String,
        retry_after: Option<f64>,
    },
    FatalError(String),
}

fn parse_retry_after(response: &reqwest::Response) -> Option<f64> {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(|s| s as f64)
}

fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status == reqwest::StatusCode::REQUEST_TIMEOUT
        || status.is_server_error()
}

/// Decide the actual delay for a retryable HTTP error: prefer the larger of
/// the backoff formula and any Retry-After hint from the server.
fn compute_retry_delay(formula_delay: f64, retry_after: Option<f64>) -> f64 {
    match retry_after {
        Some(ra) => formula_delay.max(ra),
        None => formula_delay,
    }
}

impl EmbeddingClient {
    pub fn new(config: &Config, retry_config: RetryConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_base: config.api_base.clone(),
            api_key: config.api_key.clone(),
            model: config.model.clone(),
            dim: config.dim,
            retry_config,
        }
    }

    async fn try_send(&self, texts: &[String]) -> SendOutcome {
        let url = format!("{}/embeddings", self.api_base);
        let request = EmbeddingRequest {
            model: &self.model,
            input: texts,
        };

        let response = match self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                return SendOutcome::RetryableError(format!("network error: {e}"));
            }
        };

        let status = response.status();
        if status.is_success() {
            // Split body download from JSON parsing so transport errors are retryable.
            let body_bytes = match response.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    return SendOutcome::RetryableError(format!("body transfer error on 2xx: {e}"));
                }
            };
            match serde_json::from_slice::<EmbeddingResponse>(&body_bytes) {
                Ok(resp) => {
                    let vectors: Vec<Vec<f32>> =
                        resp.data.into_iter().map(|d| d.embedding).collect();
                    if vectors.len() != texts.len() {
                        return SendOutcome::FatalError(format!(
                            "Embedding count mismatch: expected {}, got {}",
                            texts.len(),
                            vectors.len()
                        ));
                    }
                    let expected_dim = self.dim;
                    for (i, v) in vectors.iter().enumerate() {
                        if v.len() != expected_dim {
                            return SendOutcome::FatalError(format!(
                                "Embedding dimension mismatch at index {i}: expected {expected_dim}, got {}",
                                v.len()
                            ));
                        }
                    }
                    SendOutcome::Success(vectors)
                }
                Err(e) => {
                    SendOutcome::FatalError(format!("Failed to parse embedding response: {e}"))
                }
            }
        } else if is_retryable_status(status) {
            let retry_after = parse_retry_after(&response);
            let body = response.text().await.unwrap_or_default();
            SendOutcome::RetryableHttpError {
                description: format!("status {status}: {body}"),
                retry_after,
            }
        } else {
            let body = response.text().await.unwrap_or_default();
            SendOutcome::FatalError(format!("Embedding API error (status {status}): {body}"))
        }
    }

    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let max = self.retry_config.max_retries;

        for attempt in 0..=max {
            match self.try_send(texts).await {
                SendOutcome::Success(vectors) => return Ok(vectors),
                SendOutcome::FatalError(msg) => bail!("{msg}"),
                SendOutcome::RetryableError(desc) => {
                    if attempt == max {
                        bail!("{desc} (retries exhausted)");
                    }
                    let delay = self.retry_config.delay_secs(attempt);
                    eprintln!(
                        "  \u{26a0} Request failed ({desc}), retrying in {delay:.0}s ({}/{max})...",
                        attempt + 1
                    );
                    tokio::time::sleep(std::time::Duration::from_secs_f64(delay)).await;
                }
                SendOutcome::RetryableHttpError {
                    description: desc,
                    retry_after,
                } => {
                    if attempt == max {
                        bail!("{desc} (retries exhausted)");
                    }
                    let formula_delay = self.retry_config.delay_secs(attempt);
                    let delay = compute_retry_delay(formula_delay, retry_after);
                    eprintln!(
                        "  \u{26a0} Request failed ({desc}), retrying in {delay:.0}s ({}/{max})...",
                        attempt + 1
                    );
                    tokio::time::sleep(std::time::Duration::from_secs_f64(delay)).await;
                }
            }
        }

        unreachable!("retry loop should have returned or bailed")
    }
}

/// Embed a single text using the configured remote OpenAI-compatible endpoint.
pub async fn embed_one(cfg: &Config, retry: &RetryConfig, text: &str) -> Result<Vec<f32>> {
    let client = EmbeddingClient::new(cfg, *retry);
    let mut vectors = client.embed_batch(&[text.to_string()]).await?;
    vectors
        .pop()
        .ok_or_else(|| anyhow::anyhow!("embedding response was empty"))
}

/// Embed a batch of texts in a single API call.
pub async fn embed_batch(
    cfg: &Config,
    retry: &RetryConfig,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    let client = EmbeddingClient::new(cfg, *retry);
    client.embed_batch(texts).await
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- RetryConfig delay formula tests ---

    #[test]
    fn test_delay_formula_default() {
        let cfg = RetryConfig::default();
        assert_eq!(cfg.delay_secs(0), 2.0); // 2 * 2^0 = 2
        assert_eq!(cfg.delay_secs(1), 4.0); // 2 * 2^1 = 4
        assert_eq!(cfg.delay_secs(2), 8.0); // 2 * 2^2 = 8
    }

    #[test]
    fn test_delay_formula_custom() {
        let cfg = RetryConfig {
            backoff_base: 1.0,
            backoff_exp: 3.0,
            max_retries: 4,
        };
        assert_eq!(cfg.delay_secs(0), 1.0);
        assert_eq!(cfg.delay_secs(1), 3.0);
        assert_eq!(cfg.delay_secs(2), 9.0);
        assert_eq!(cfg.delay_secs(3), 27.0);
    }

    #[test]
    fn test_delay_formula_fixed_interval() {
        let cfg = RetryConfig {
            backoff_base: 5.0,
            backoff_exp: 1.0,
            max_retries: 3,
        };
        assert_eq!(cfg.delay_secs(0), 5.0);
        assert_eq!(cfg.delay_secs(1), 5.0);
        assert_eq!(cfg.delay_secs(2), 5.0);
    }

    // --- Retry decision: which statuses are retryable ---

    #[test]
    fn test_retryable_statuses() {
        assert!(is_retryable_status(reqwest::StatusCode::TOO_MANY_REQUESTS));
        assert!(is_retryable_status(reqwest::StatusCode::REQUEST_TIMEOUT));
        assert!(is_retryable_status(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        ));
        assert!(is_retryable_status(reqwest::StatusCode::BAD_GATEWAY));
        assert!(is_retryable_status(
            reqwest::StatusCode::SERVICE_UNAVAILABLE
        ));
    }

    #[test]
    fn test_non_retryable_statuses() {
        assert!(!is_retryable_status(reqwest::StatusCode::UNAUTHORIZED));
        assert!(!is_retryable_status(reqwest::StatusCode::FORBIDDEN));
        assert!(!is_retryable_status(reqwest::StatusCode::BAD_REQUEST));
        assert!(!is_retryable_status(reqwest::StatusCode::NOT_FOUND));
    }

    // --- Retry-After header parsing ---

    #[test]
    fn test_parse_retry_after_valid_integer() {
        let resp = http::Response::builder()
            .status(429)
            .header("Retry-After", "10")
            .body("")
            .unwrap();
        let reqwest_resp = reqwest::Response::from(resp);
        assert_eq!(parse_retry_after(&reqwest_resp), Some(10.0));
    }

    #[test]
    fn test_parse_retry_after_missing_header() {
        let resp = http::Response::builder().status(429).body("").unwrap();
        let reqwest_resp = reqwest::Response::from(resp);
        assert_eq!(parse_retry_after(&reqwest_resp), None);
    }

    #[test]
    fn test_parse_retry_after_non_integer_ignored() {
        let resp = http::Response::builder()
            .status(429)
            .header("Retry-After", "Wed, 21 Oct 2025 07:28:00 GMT")
            .body("")
            .unwrap();
        let reqwest_resp = reqwest::Response::from(resp);
        assert_eq!(parse_retry_after(&reqwest_resp), None);
    }

    // --- max(formula, Retry-After) selection ---

    #[test]
    fn test_compute_delay_retry_after_larger() {
        assert_eq!(compute_retry_delay(2.0, Some(10.0)), 10.0);
    }

    #[test]
    fn test_compute_delay_formula_larger() {
        assert_eq!(compute_retry_delay(4.0, Some(1.0)), 4.0);
    }

    #[test]
    fn test_compute_delay_no_retry_after() {
        assert_eq!(compute_retry_delay(2.0, None), 2.0);
    }

    // --- Response parsing ---

    #[test]
    fn test_parse_embedding_response() {
        let json = r#"{
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1}
            ],
            "model": "test",
            "usage": {"prompt_tokens": 10}
        }"#;

        let resp: EmbeddingResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].embedding, vec![0.1, 0.2, 0.3]);
        assert_eq!(resp.data[1].embedding, vec![0.4, 0.5, 0.6]);
    }

    // Config-from-env gating is covered exhaustively in `config::tests`
    // (which serialises env-var mutations under a Mutex). Replicating that
    // here with bare `std::env::set_var` would race against the config
    // tests when cargo runs them in parallel and produced flaky failures
    // — the duplicate test was removed.
}
