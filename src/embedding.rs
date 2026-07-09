//! OpenAI-compatible embedding HTTP client.
//!
//! Two request shapes share one endpoint (`{api_base}/embeddings`):
//! - Flat: `input: [String]` — ordinary embedding models. Each text is
//!   encoded independently.
//! - Contextualized (model name contains `voyage-context`): `input` is a
//!   nested list-of-lists where each inner list holds one document's chunks
//!   in document order, plus an `input_type` hint (`document` / `query`).
//!   The gateway routes these to Voyage's contextualized-chunk-embeddings
//!   API and returns a two-level response (per-document objects each
//!   carrying per-chunk vectors) which this module flattens back to one
//!   vector per chunk.
//!
//! `embed_batch` / `embed_document_chunks` return `Err(...)` when the retry
//! budget is exhausted. The caller is responsible for translating that `Err`
//! into a per-batch failure counter rather than propagating it with `?` and
//! aborting the whole index run — the failure-isolation contract lives in
//! `crate::index`, NOT in this module.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::config::Config;

/// Contextualized chunk-embedding models need the nested request shape and
/// per-document grouping. Routing is by model name so switching models is a
/// pure `XD_KB_MODEL` change with no extra configuration.
pub fn is_contextualized_model(model: &str) -> bool {
    model.contains("voyage-context")
}

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
#[serde(untagged)]
enum EmbeddingInput<'a> {
    Flat(&'a [String]),
    /// One inner list per document; chunks of the same document are encoded
    /// together so each chunk vector carries whole-document context.
    Nested(&'a [&'a [String]]),
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: EmbeddingInput<'a>,
    dimensions: usize,
    /// Voyage retrieval hint: "document" at index time, "query" at query
    /// time. Omitted entirely for ordinary models — unknown fields are not
    /// universally tolerated by OpenAI-compatible backends.
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<&'a str>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

/// One `data[]` item. Ordinary models return flat items carrying `embedding`
/// directly; the contextualized endpoint returns one item per document whose
/// `data` holds that document's per-chunk embeddings. Untagged so both shapes
/// parse regardless of which backend variant answered.
#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingData {
    Flat { embedding: Vec<f32> },
    Nested { data: Vec<NestedEmbedding> },
}

#[derive(Deserialize)]
struct NestedEmbedding {
    embedding: Vec<f32>,
}

/// Flatten a response into one vector per input chunk, preserving document
/// order then in-document chunk order.
fn flatten_response(resp: EmbeddingResponse) -> Vec<Vec<f32>> {
    let mut vectors = Vec::new();
    for item in resp.data {
        match item {
            EmbeddingData::Flat { embedding } => vectors.push(embedding),
            EmbeddingData::Nested { data } => {
                vectors.extend(data.into_iter().map(|n| n.embedding));
            }
        }
    }
    vectors
}

/// Expected response shape, decided by the request we sent.
#[derive(Clone, Copy, PartialEq)]
enum ResponseShape {
    /// Accept flat items, and tolerate nested ones by flattening — gateways
    /// answer contextualized *query* requests with a nested single-vector
    /// response, so the lenient shape covers both ordinary batches and
    /// queries.
    Flat,
    /// Single-document contextualized request: the response MUST be exactly
    /// one nested document item. A flat response here means the backend fell
    /// back to ordinary (context-free) embeddings — indexing those silently
    /// would degrade every vector of the document, so it is a fatal error.
    NestedDocument,
}

/// Parse a 2xx body and validate vector count + dimensions against what the
/// request promised. Returns `Err(message)` for any mismatch — the caller
/// maps it to `SendOutcome::FatalError` (mismatches are not retryable).
fn parse_and_validate(
    body: &[u8],
    shape: ResponseShape,
    expected: usize,
    dim: usize,
) -> std::result::Result<Vec<Vec<f32>>, String> {
    let resp: EmbeddingResponse = serde_json::from_slice(body)
        .map_err(|e| format!("Failed to parse embedding response: {e}"))?;
    let vectors = match shape {
        ResponseShape::Flat => flatten_response(resp),
        ResponseShape::NestedDocument => {
            if resp.data.len() != 1 {
                return Err(format!(
                    "Contextualized document response must contain exactly 1 document, got {}",
                    resp.data.len()
                ));
            }
            match resp.data.into_iter().next().expect("len checked above") {
                EmbeddingData::Nested { data } => data.into_iter().map(|n| n.embedding).collect(),
                EmbeddingData::Flat { .. } => {
                    return Err(
                        "Contextualized document request received a flat (context-free) \
                         embedding response — refusing to index silently degraded vectors"
                            .to_string(),
                    );
                }
            }
        }
    };
    if vectors.len() != expected {
        return Err(format!(
            "Embedding count mismatch: expected {expected}, got {}",
            vectors.len()
        ));
    }
    for (i, v) in vectors.iter().enumerate() {
        if v.len() != dim {
            return Err(format!(
                "Embedding dimension mismatch at index {i}: expected {dim}, got {}",
                v.len()
            ));
        }
    }
    Ok(vectors)
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

    fn build_request<'a>(&'a self, texts: &'a [String]) -> EmbeddingRequest<'a> {
        EmbeddingRequest {
            model: &self.model,
            input: EmbeddingInput::Flat(texts),
            dimensions: self.dim,
            input_type: None,
        }
    }

    async fn try_send(
        &self,
        request: &EmbeddingRequest<'_>,
        shape: ResponseShape,
        expected: usize,
    ) -> SendOutcome {
        let url = format!("{}/embeddings", self.api_base);

        let response = match self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request)
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
            match parse_and_validate(&body_bytes, shape, expected, self.dim) {
                Ok(vectors) => SendOutcome::Success(vectors),
                Err(msg) => SendOutcome::FatalError(msg),
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

    /// Nested single-document request for contextualized models.
    fn build_document_request<'a>(&'a self, nested: &'a [&'a [String]]) -> EmbeddingRequest<'a> {
        EmbeddingRequest {
            model: &self.model,
            input: EmbeddingInput::Nested(nested),
            dimensions: self.dim,
            input_type: Some("document"),
        }
    }

    /// Flat query request for contextualized models: the contextualized
    /// endpoint accepts flat input only when the `query` hint is present.
    fn build_query_request<'a>(&'a self, texts: &'a [String]) -> EmbeddingRequest<'a> {
        EmbeddingRequest {
            model: &self.model,
            input: EmbeddingInput::Flat(texts),
            dimensions: self.dim,
            input_type: Some("query"),
        }
    }

    /// Flat request without `input_type` — the ordinary embedding path.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let request = self.build_request(texts);
        self.send_with_retry(&request, ResponseShape::Flat, texts.len())
            .await
    }

    /// One document's chunks embedded together (contextualized models only):
    /// nested `input` with a single inner list, `input_type: "document"`.
    /// Returns one vector per chunk, in chunk order. The response MUST come
    /// back in the nested per-document shape (see `ResponseShape`).
    pub async fn embed_document_chunks(&self, chunks: &[String]) -> Result<Vec<Vec<f32>>> {
        let nested = [chunks];
        let request = self.build_document_request(&nested);
        self.send_with_retry(&request, ResponseShape::NestedDocument, chunks.len())
            .await
    }

    /// Query-side embedding for contextualized models.
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let texts = [text.to_string()];
        let request = self.build_query_request(&texts);
        let mut vectors = self
            .send_with_retry(&request, ResponseShape::Flat, 1)
            .await?;
        vectors
            .pop()
            .ok_or_else(|| anyhow::anyhow!("embedding response was empty"))
    }

    async fn send_with_retry(
        &self,
        request: &EmbeddingRequest<'_>,
        shape: ResponseShape,
        expected: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let max = self.retry_config.max_retries;

        for attempt in 0..=max {
            match self.try_send(request, shape, expected).await {
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
/// Contextualized models require the `query` input_type hint; ordinary models
/// keep the plain flat request.
pub async fn embed_one(cfg: &Config, retry: &RetryConfig, text: &str) -> Result<Vec<f32>> {
    let client = EmbeddingClient::new(cfg, *retry);
    if is_contextualized_model(&cfg.model) {
        return client.embed_query(text).await;
    }
    let mut vectors = client.embed_batch(&[text.to_string()]).await?;
    vectors
        .pop()
        .ok_or_else(|| anyhow::anyhow!("embedding response was empty"))
}

/// Embed a batch of texts in a single API call (flat, ordinary models).
pub async fn embed_batch(
    cfg: &Config,
    retry: &RetryConfig,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    let client = EmbeddingClient::new(cfg, *retry);
    client.embed_batch(texts).await
}

/// Embed one document's chunks together via the contextualized request shape.
pub async fn embed_document(
    cfg: &Config,
    retry: &RetryConfig,
    chunks: &[String],
) -> Result<Vec<Vec<f32>>> {
    let client = EmbeddingClient::new(cfg, *retry);
    client.embed_document_chunks(chunks).await
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
    fn build_request_dimensions_follow_env_xd_kb_dim() {
        // Reproduces the bug we hit with voyage-4-large: when XD_KB_DIM is set
        // to a non-default value, the outgoing request body must carry that
        // dim as `dimensions` (OpenAI-compat name). Without this test, the
        // env var silently failed to reach the wire and we got the model
        // default back instead.
        crate::config::with_env(
            &[
                (crate::config::ENV_API_BASE, Some("http://x/v1")),
                (crate::config::ENV_API_KEY, Some("k")),
                (crate::config::ENV_MODEL, Some("voyage-4-large")),
                (crate::config::ENV_DIM, Some("2048")),
                (crate::config::ENV_MAX_CHUNK_CHARS, None),
            ],
            || {
                let cfg = Config::from_env().expect("config should load");
                let client = EmbeddingClient::new(&cfg, RetryConfig::default());
                let texts = vec!["hi".to_string()];
                let req = client.build_request(&texts);
                let v = serde_json::to_value(&req).expect("serialize request");
                assert_eq!(v["model"], "voyage-4-large");
                assert_eq!(v["dimensions"], 2048);
                assert_eq!(v["input"], serde_json::json!(["hi"]));
            },
        );
    }

    #[test]
    fn contextualized_model_detection() {
        assert!(is_contextualized_model("voyage/voyage-context-4"));
        assert!(is_contextualized_model("voyage-context-3"));
        assert!(!is_contextualized_model("voyage/voyage-4-large"));
        assert!(!is_contextualized_model("text-embedding-3-large"));
    }

    #[test]
    fn nested_document_request_serializes_list_of_lists_with_input_type() {
        // Locks the PRODUCTION builder (`build_document_request`) — a
        // hand-rolled request here would keep passing even if the builder
        // regressed to flat input or dropped the document hint.
        crate::config::with_env(
            &[
                (crate::config::ENV_API_BASE, Some("http://x/v1")),
                (crate::config::ENV_API_KEY, Some("k")),
                (crate::config::ENV_MODEL, Some("voyage/voyage-context-4")),
                (crate::config::ENV_DIM, Some("2048")),
                (crate::config::ENV_MAX_CHUNK_CHARS, None),
            ],
            || {
                let cfg = Config::from_env().expect("config should load");
                let client = EmbeddingClient::new(&cfg, RetryConfig::default());
                let chunks = vec!["c1".to_string(), "c2".to_string()];
                let nested = [chunks.as_slice()];
                let req = client.build_document_request(&nested);
                let v = serde_json::to_value(&req).expect("serialize request");
                assert_eq!(v["model"], "voyage/voyage-context-4");
                assert_eq!(v["input"], serde_json::json!([["c1", "c2"]]));
                assert_eq!(v["input_type"], "document");
                assert_eq!(v["dimensions"], 2048);
            },
        );
    }

    #[test]
    fn flat_request_omits_input_type_key_entirely() {
        let texts = vec!["hi".to_string()];
        let req = EmbeddingRequest {
            model: "voyage-4-large",
            input: EmbeddingInput::Flat(&texts),
            dimensions: 1024,
            input_type: None,
        };
        let v = serde_json::to_value(&req).unwrap();
        assert!(
            !v.as_object().unwrap().contains_key("input_type"),
            "ordinary requests must not carry input_type at all"
        );
        assert_eq!(v["input"], serde_json::json!(["hi"]));
    }

    /// Shape captured from the live gateway (LiteLLM → voyage-context-4):
    /// `data` holds one object per document, each with its own `data` array
    /// of per-chunk embeddings. Flattening preserves doc order then chunk
    /// order.
    #[test]
    fn parse_contextualized_nested_response() {
        let json = r#"{
            "model": "voyage/voyage-context-4",
            "data": [
                {"object": "list", "index": 0, "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]}
                ]},
                {"object": "list", "index": 1, "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.5, 0.6]}
                ]}
            ],
            "usage": {"prompt_tokens": 20, "total_tokens": 20}
        }"#;
        let resp: EmbeddingResponse = serde_json::from_str(json).unwrap();
        let vectors = flatten_response(resp);
        assert_eq!(
            vectors,
            vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]]
        );
    }

    #[test]
    fn query_request_is_flat_single_input_with_query_hint() {
        crate::config::with_env(
            &[
                (crate::config::ENV_API_BASE, Some("http://x/v1")),
                (crate::config::ENV_API_KEY, Some("k")),
                (crate::config::ENV_MODEL, Some("voyage/voyage-context-4")),
                (crate::config::ENV_DIM, Some("2048")),
                (crate::config::ENV_MAX_CHUNK_CHARS, None),
            ],
            || {
                let cfg = Config::from_env().expect("config should load");
                let client = EmbeddingClient::new(&cfg, RetryConfig::default());
                let texts = ["法国的首都是哪里？".to_string()];
                let req = client.build_query_request(&texts);
                let v = serde_json::to_value(&req).expect("serialize request");
                assert_eq!(v["input"], serde_json::json!(["法国的首都是哪里？"]));
                assert_eq!(v["input_type"], "query");
                assert_eq!(v["dimensions"], 2048);
            },
        );
    }

    // --- ResponseShape validation (fatal-error contract) ---

    /// Regression lock for the "silent degradation" trap: a document request
    /// answered with a flat (context-free) response MUST be rejected even
    /// when count and dims happen to match.
    #[test]
    fn document_shape_rejects_flat_response() {
        // Single-chunk document answered by a single flat item: count (1) and
        // dims (2) both match, so only the shape check can catch the
        // degradation.
        let body = br#"{"data": [{"embedding": [0.1, 0.2]}]}"#;
        let err = parse_and_validate(body, ResponseShape::NestedDocument, 1, 2)
            .expect_err("flat response must be rejected for document requests");
        assert!(
            err.contains("flat"),
            "error should name the flat-response degradation: {err}"
        );
    }

    #[test]
    fn document_shape_rejects_multi_document_response() {
        let body = br#"{"data": [
            {"data": [{"embedding": [0.1, 0.2]}]},
            {"data": [{"embedding": [0.3, 0.4]}]}
        ]}"#;
        let err = parse_and_validate(body, ResponseShape::NestedDocument, 2, 2)
            .expect_err("multi-document response must be rejected");
        assert!(err.contains("exactly 1 document"), "got: {err}");
    }

    #[test]
    fn count_mismatch_is_fatal() {
        let body = br#"{"data": [{"data": [{"embedding": [0.1, 0.2]}]}]}"#;
        let err = parse_and_validate(body, ResponseShape::NestedDocument, 2, 2)
            .expect_err("missing vector must be a count mismatch");
        assert!(err.contains("count mismatch"), "got: {err}");
        assert!(err.contains("expected 2, got 1"), "got: {err}");
    }

    #[test]
    fn dimension_mismatch_is_fatal() {
        let body = br#"{"data": [{"embedding": [0.1, 0.2, 0.3]}]}"#;
        let err = parse_and_validate(body, ResponseShape::Flat, 1, 2)
            .expect_err("wrong-dim vector must be a dimension mismatch");
        assert!(err.contains("dimension mismatch"), "got: {err}");
        assert!(err.contains("expected 2, got 3"), "got: {err}");
    }

    #[test]
    fn flat_shape_accepts_nested_query_response() {
        // Gateways answer contextualized *query* requests with a nested
        // single-vector response — the lenient Flat shape must flatten it.
        let body = br#"{"data": [{"data": [{"embedding": [0.1, 0.2]}]}]}"#;
        let vectors = parse_and_validate(body, ResponseShape::Flat, 1, 2).unwrap();
        assert_eq!(vectors, vec![vec![0.1, 0.2]]);
    }

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
        let vectors = flatten_response(resp);
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0], vec![0.1, 0.2, 0.3]);
        assert_eq!(vectors[1], vec![0.4, 0.5, 0.6]);
    }

    // Config-from-env gating is covered exhaustively in `config::tests`
    // (which serialises env-var mutations under a Mutex). Replicating that
    // here with bare `std::env::set_var` would race against the config
    // tests when cargo runs them in parallel and produced flaky failures
    // — the duplicate test was removed.
}
