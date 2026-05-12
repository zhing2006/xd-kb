// grafeo-backed KbStore: vectors + graph + file hashes share one DB.
//
// Schema:
//   :Chunk       { file_path, line_start, line_end, content, prefix?, embedding }
//   :FileHash    { path, vector_hash?, graph_hash?, llm_hash?, fulltext_hash?,
//                   summary_hash?, entity_hash?, vector_2_hash?, custom_hash?, custom_2_hash? }
//                  -- `path` is the unique key; each <dim>_hash is an optional string
//                     recording the file's blake3 at last successful processing for that
//                     pipeline dim. `index` only ever touches `vector_hash`; other dims
//                     are written exclusively by `check --type <dim> --mark`.
//   :IndexConfig { model, dim, max_chunk_chars }   -- DB-global singleton
//
// Each DB path is one KB (physical isolation).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use grafeo::{GrafeoDB, NodeId, Value};

const LABEL_CHUNK: &str = "Chunk";
const PROP_EMBEDDING: &str = "embedding";

// HNSW tuning. Defaults are m=16, ef_construction=128. We push both up for
// recall: m=32 doubles per-node neighbour fan-out (better high-dim coverage),
// ef_construction=400 widens the construction beam. Memory cost ~ +30%, build
// time ~ 1.5-2x; both are negligible at our < 10K-chunk-per-KB scale.
const HNSW_M: usize = 32;
const HNSW_EF_CONSTRUCTION: usize = 400;

pub struct KbStore {
    db: GrafeoDB,
}

pub struct ChunkInput<'a> {
    pub file_path: &'a str,
    pub line_start: u32,
    pub line_end: u32,
    pub content: &'a str,
    pub prefix: Option<&'a str>,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub content: String,
    /// cosine similarity in [-1, 1]; computed as 1 - distance
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexConfig {
    pub model: String,
    pub dim: usize,
    pub max_chunk_chars: u32,
}

/// Pipeline dim for `:FileHash` per-dim hash properties.
///
/// CLI surface uses kebab-case (clap default): `vector` / `vector-2` / `custom-2`.
/// grafeo property names are snake_case (`vector_hash` / `vector_2_hash` / ...) — GQL
/// identifiers do not allow `-`. The kebab → snake mapping lives entirely in
/// `property_name`; callers (CLI / storage / tests) never see the property literal.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum HashDim {
    Vector,
    Graph,
    Llm,
    Fulltext,
    Summary,
    Entity,
    // clap's default kebab-case rename does NOT insert `-` between a letter and a
    // digit (`Vector2` → `vector2`), but the spec mandates `vector-2`. Pin the CLI
    // name explicitly. Property side stays snake (`vector_2_hash`) — see property_name.
    #[value(name = "vector-2")]
    Vector2,
    Custom,
    #[value(name = "custom-2")]
    Custom2,
}

impl HashDim {
    /// grafeo node property name for this dim (`<name>_hash`).
    ///
    /// `match` over all variants — adding a new variant without updating this
    /// arm fails to compile. `format!("{}_hash", ...)` would silently accept
    /// new variants and write to a never-read property, so we avoid it.
    pub fn property_name(self) -> &'static str {
        match self {
            HashDim::Vector => "vector_hash",
            HashDim::Graph => "graph_hash",
            HashDim::Llm => "llm_hash",
            HashDim::Fulltext => "fulltext_hash",
            HashDim::Summary => "summary_hash",
            HashDim::Entity => "entity_hash",
            HashDim::Vector2 => "vector_2_hash",
            HashDim::Custom => "custom_hash",
            HashDim::Custom2 => "custom_2_hash",
        }
    }

    /// All 9 variants in canonical order — used for schema-wide reads
    /// (e.g. `remove_file_hash` 9-column SELECT) and for tests that walk every dim.
    pub const ALL: &'static [HashDim] = &[
        HashDim::Vector,
        HashDim::Graph,
        HashDim::Llm,
        HashDim::Fulltext,
        HashDim::Summary,
        HashDim::Entity,
        HashDim::Vector2,
        HashDim::Custom,
        HashDim::Custom2,
    ];
}

impl KbStore {
    pub fn open(db_path: &Path) -> Result<Self> {
        let db = GrafeoDB::open(db_path)
            .with_context(|| format!("failed to open grafeo DB at {}", db_path.display()))?;
        Ok(Self { db })
    }

    #[cfg(test)]
    fn open_in_memory() -> Self {
        Self {
            db: GrafeoDB::new_in_memory(),
        }
    }

    /// Returns the dim of any existing :Chunk node's embedding, or None if no chunks exist.
    /// Used for pre-flight dim validation in upsert.
    fn peek_existing_chunk_dim(&self) -> Result<Option<usize>> {
        let session = self.db.session();
        let res = session
            .execute("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN c.embedding LIMIT 1")
            .map_err(|e| anyhow!("peek existing chunk dim: {e}"))?;
        for row in res.rows() {
            if let Some(Value::Vector(v)) = row.first() {
                return Ok(Some(v.len()));
            }
        }
        Ok(None)
    }

    pub fn ensure_chunk_vector_index(&self, dim: usize) -> Result<()> {
        self.db
            .create_vector_index(
                LABEL_CHUNK,
                PROP_EMBEDDING,
                Some(dim),
                Some("cosine"),
                Some(HNSW_M),
                Some(HNSW_EF_CONSTRUCTION),
                None, // quantization disabled (precision over compression)
            )
            .with_context(|| format!("failed to create vector index :Chunk({PROP_EMBEDDING})"))
    }

    // ---------- IndexConfig ----------

    pub fn read_index_config(&self) -> Result<Option<IndexConfig>> {
        let session = self.db.session();
        let q = "MATCH (c:IndexConfig) RETURN c.model, c.dim, c.max_chunk_chars LIMIT 1";
        let res = session
            .execute(q)
            .map_err(|e| anyhow!("read IndexConfig failed: {e}"))?;
        let rows = res.rows();
        if rows.is_empty() {
            return Ok(None);
        }
        let row = &rows[0];
        let model = match &row[0] {
            Value::String(s) => s.to_string(),
            other => bail!("IndexConfig.model has unexpected value: {other:?}"),
        };
        let dim = match &row[1] {
            Value::Int64(n) => {
                usize::try_from(*n).map_err(|_| anyhow!("IndexConfig.dim out of range: {n}"))?
            }
            other => bail!("IndexConfig.dim has unexpected value: {other:?}"),
        };
        let max_chunk_chars = match &row[2] {
            Value::Int64(n) => u32::try_from(*n)
                .map_err(|_| anyhow!("IndexConfig.max_chunk_chars out of range: {n}"))?,
            other => bail!("IndexConfig.max_chunk_chars has unexpected value: {other:?}"),
        };
        Ok(Some(IndexConfig {
            model,
            dim,
            max_chunk_chars,
        }))
    }

    pub fn write_index_config(&self, model: &str, dim: usize, max_chunk_chars: u32) -> Result<()> {
        // MERGE without keys keeps :IndexConfig a DB-global singleton.
        let mut session = self.db.session();
        session
            .begin_transaction()
            .map_err(|e| anyhow!("begin txn for write_index_config: {e}"))?;
        let res = (|| -> Result<()> {
            // dim is user-controlled (XD_KB_DIM); guard against silent wrap on
            // 64-bit targets where usize > i64::MAX.
            let dim_i64 = i64::try_from(dim)
                .map_err(|_| anyhow!("IndexConfig.dim out of range for storage: {dim}"))?;
            let mut params = HashMap::new();
            params.insert("model".to_string(), Value::from(model));
            params.insert("dim".to_string(), Value::Int64(dim_i64));
            params.insert(
                "max_chunk_chars".to_string(),
                Value::Int64(i64::from(max_chunk_chars)),
            );
            let q = "MERGE (c:IndexConfig) SET c.model = $model, c.dim = $dim, c.max_chunk_chars = $max_chunk_chars";
            session
                .execute_with_params(q, params)
                .map_err(|e| anyhow!("write IndexConfig failed: {e}"))?;
            Ok(())
        })();
        match res {
            Ok(()) => session
                .commit()
                .map_err(|e| anyhow!("commit write_index_config: {e}")),
            Err(e) => {
                let _ = session.rollback();
                Err(e)
            }
        }
    }

    // ---------- Chunks ----------

    /// Replace all chunks for `file_path` and refresh its FileHash, atomically.
    ///
    /// Dim validation runs BEFORE the transaction opens — if any chunk has a vector dim
    /// that doesn't match the others, the call returns Err and no DB state changes.
    /// After commit, the vector index is rebuilt so subsequent `search` sees the new chunks.
    pub fn upsert_chunks_for_file(
        &self,
        file_path: &str,
        file_hash: &str,
        chunks: &[ChunkInput<'_>],
    ) -> Result<()> {
        // Pre-flight dim check. The session-level set_node_property does NOT validate
        // vector-index dim, so a wrong-dim insert would silently succeed and only
        // surface as a "Vector dimension mismatch" error later during the post-commit
        // index rebuild — at which point the bad chunks are already committed and
        // single-file rollback is broken. Catch dim mismatches BEFORE the txn opens.
        let chunk_dim = chunks.first().map(|c| c.embedding.len());
        if let Some(d) = chunk_dim {
            for (i, ck) in chunks.iter().enumerate() {
                if ck.embedding.len() != d {
                    bail!(
                        "chunk {i} of {file_path} has embedding dim {} but file expects dim {d}",
                        ck.embedding.len()
                    );
                }
            }
            if let Some(existing) = self.peek_existing_chunk_dim()?
                && existing != d
            {
                bail!("chunk dim {d} for {file_path} mismatches existing index dim {existing}");
            }
        }

        let mut session = self.db.session();
        session
            .begin_transaction()
            .map_err(|e| anyhow!("begin txn for upsert_chunks_for_file: {e}"))?;

        let res = (|| -> Result<()> {
            // 1. delete old chunks for this file
            let mut del_params = HashMap::new();
            del_params.insert("path".to_string(), Value::from(file_path));
            session
                .execute_with_params(
                    "MATCH (c:Chunk {file_path: $path}) DETACH DELETE c",
                    del_params,
                )
                .map_err(|e| anyhow!("delete old chunks for {file_path}: {e}"))?;

            // 2. insert each new chunk via GQL INSERT (NOT session.create_node).
            //
            // grafeo 0.5.41 quirk (verified empirically against on-disk DB): the
            // session-level create_node / set_node_property API is documented as
            // "low-level for testing" and does NOT write to the WAL even when called
            // inside a transaction. Nodes appear during the session and survive commit
            // in memory, but on the next process they are absent from the recovered
            // graph. GQL writes via execute / execute_with_params go through the WAL
            // path and persist correctly across process restarts.
            for ck in chunks {
                let mut params = HashMap::new();
                params.insert("file_path".to_string(), Value::from(ck.file_path));
                params.insert(
                    "line_start".to_string(),
                    Value::Int64(i64::from(ck.line_start)),
                );
                params.insert("line_end".to_string(), Value::Int64(i64::from(ck.line_end)));
                params.insert("content".to_string(), Value::from(ck.content));
                params.insert(
                    PROP_EMBEDDING.to_string(),
                    Value::Vector(Arc::from(ck.embedding.clone())),
                );
                let stmt = if let Some(p) = ck.prefix {
                    params.insert("prefix".to_string(), Value::from(p));
                    "INSERT (:Chunk {file_path: $file_path, line_start: $line_start, line_end: $line_end, content: $content, prefix: $prefix, embedding: $embedding})"
                } else {
                    "INSERT (:Chunk {file_path: $file_path, line_start: $line_start, line_end: $line_end, content: $content, embedding: $embedding})"
                };
                session.execute_with_params(stmt, params).map_err(|e| {
                    anyhow!(
                        "insert chunk {file_path}:{}-{}: {e}",
                        ck.line_start,
                        ck.line_end
                    )
                })?;
            }

            // 3. update FileHash node — `index` only ever writes to the vector dim;
            //    other dims are owned exclusively by `check --type <dim> --mark`.
            upsert_file_hash_in_session(&session, file_path, HashDim::Vector, file_hash)?;
            Ok(())
        })();

        match res {
            Ok(()) => session
                .commit()
                .map_err(|e| anyhow!("commit upsert_chunks_for_file: {e}"))?,
            Err(e) => {
                let _ = session.rollback();
                return Err(e);
            }
        }

        // Rebuild the vector index from current node state. The session-level
        // set_node_property does NOT auto-update the HNSW index (only
        // GrafeoDB::set_node_property does), so without this rebuild new chunks
        // are invisible to CALL grafeo.search.vector. Cheap because we're
        // already bounded by < 10K chunks per the design assumptions.
        if let Some(d) = chunk_dim {
            self.ensure_chunk_vector_index(d)?;
        }
        Ok(())
    }

    /// Per-dim cleanup for the **vector** pipeline: drop all `:Chunk` rows for
    /// `file_path`, REMOVE `:FileHash {path: $f}.vector_hash`, and DETACH DELETE
    /// the `:FileHash` node IFF removing `vector_hash` leaves no other dim
    /// hash on it. Other dims' hashes (graph_hash, llm_hash, ...) MUST stay
    /// intact when present — they belong to other pipelines and are owned
    /// exclusively by `check --type <dim> --mark`.
    ///
    /// This is the path `index --incremental` takes when a previously-indexed
    /// markdown file is missing from disk; using `delete_file_completely`
    /// here would unconditionally orphan other dims' hashes, which violates
    /// the per-dim FileHash contract documented in storage.rs:1-13.
    pub fn delete_vector_dim_for_file(&self, file_path: &str) -> Result<()> {
        let mut session = self.db.session();
        session
            .begin_transaction()
            .map_err(|e| anyhow!("begin txn for delete_vector_dim_for_file: {e}"))?;
        let res = (|| -> Result<()> {
            // 1. Drop all :Chunk rows for this file (vector pipeline owns them).
            let mut p_chunks = HashMap::new();
            p_chunks.insert("path".to_string(), Value::from(file_path));
            session
                .execute_with_params(
                    "MATCH (c:Chunk {file_path: $path}) DETACH DELETE c",
                    p_chunks,
                )
                .map_err(|e| anyhow!("delete chunks for {file_path}: {e}"))?;

            // 2. Read the FileHash node's 9 dim columns to decide whether to
            // DETACH DELETE the whole node (vector_hash was the last non-null
            // dim) or just REMOVE the vector_hash property. Same two-phase
            // shape as `remove_file_hash`; see that function's grafeo 0.5
            // probe notes for why a single-statement variant won't compile.
            let select_cols: Vec<String> = HashDim::ALL
                .iter()
                .map(|d| format!("h.{}", d.property_name()))
                .collect();
            let read_stmt = format!(
                "MATCH (h:FileHash {{path: $path}}) RETURN {}",
                select_cols.join(", ")
            );
            let mut read_params = HashMap::new();
            read_params.insert("path".to_string(), Value::from(file_path));
            let read_res = session
                .execute_with_params(&read_stmt, read_params)
                .map_err(|e| anyhow!("read FileHash dims for {file_path}: {e}"))?;

            let rows = read_res.rows();
            if rows.is_empty() {
                // No FileHash node — nothing to clean (e.g. file was indexed
                // then its FileHash was already wiped by force-rebuild).
                return Ok(());
            }

            let row = &rows[0];
            let mut non_null_count = 0usize;
            let mut vector_present = false;
            for (i, d) in HashDim::ALL.iter().enumerate() {
                let cell = row.get(i).unwrap_or(&Value::Null);
                if !matches!(cell, Value::Null) {
                    non_null_count += 1;
                    if *d == HashDim::Vector {
                        vector_present = true;
                    }
                }
            }

            // If vector_hash isn't on the node we have nothing to remove —
            // the chunks DELETE above already handled the vector-pipeline
            // side. Idempotent.
            if !vector_present {
                return Ok(());
            }

            let mut write_params = HashMap::new();
            write_params.insert("path".to_string(), Value::from(file_path));
            if non_null_count == 1 {
                // vector_hash was the only dim — drop the whole node.
                session
                    .execute_with_params(
                        "MATCH (h:FileHash {path: $path}) DETACH DELETE h",
                        write_params,
                    )
                    .map_err(|e| anyhow!("delete FileHash node for {file_path}: {e}"))?;
            } else {
                // Other dims still carry hashes — only strip vector_hash.
                session
                    .execute_with_params(
                        "MATCH (h:FileHash {path: $path}) REMOVE h.vector_hash",
                        write_params,
                    )
                    .map_err(|e| anyhow!("remove vector_hash for {file_path}: {e}"))?;
            }
            Ok(())
        })();
        match res {
            Ok(()) => session.commit().map_err(|e| anyhow!("commit: {e}")),
            Err(e) => {
                let _ = session.rollback();
                Err(e)
            }
        }
    }

    /// Force-rebuild helper: drop every node related to `file_path` — all
    /// `:Chunk` rows AND the whole `:FileHash` node, regardless of which
    /// `<dim>_hash` properties the node still carries.
    ///
    /// **Not used by `index --incremental`** — incremental cleanup goes
    /// through [`delete_vector_dim_for_file`] to honour the per-dim contract.
    /// Reserve this function for explicit force-rebuild paths where the
    /// caller knows it's safe to nuke other dims too.
    pub fn delete_file_completely(&self, file_path: &str) -> Result<()> {
        let mut session = self.db.session();
        session
            .begin_transaction()
            .map_err(|e| anyhow!("begin txn for delete_file_completely: {e}"))?;
        let res = (|| -> Result<()> {
            let mut p1 = HashMap::new();
            p1.insert("path".to_string(), Value::from(file_path));
            session
                .execute_with_params("MATCH (c:Chunk {file_path: $path}) DETACH DELETE c", p1)
                .map_err(|e| anyhow!("delete chunks: {e}"))?;

            let mut p2 = HashMap::new();
            p2.insert("path".to_string(), Value::from(file_path));
            session
                .execute_with_params("MATCH (h:FileHash {path: $path}) DETACH DELETE h", p2)
                .map_err(|e| anyhow!("delete FileHash: {e}"))?;
            Ok(())
        })();
        match res {
            Ok(()) => session.commit().map_err(|e| anyhow!("commit: {e}")),
            Err(e) => {
                let _ = session.rollback();
                Err(e)
            }
        }
    }

    // ---------- KNN ----------

    pub fn search(&self, query_vector: &[f32], top_n: usize) -> Result<Vec<SearchResult>> {
        if top_n == 0 {
            return Ok(Vec::new());
        }
        let rows = self
            .db
            .vector_search(LABEL_CHUNK, PROP_EMBEDDING, query_vector, top_n, None, None)
            .map_err(|e| anyhow!("vector_search failed: {e}"))?;
        let session = self.db.session();
        materialize_results(&session, &rows, top_n)
    }

    pub fn search_mmr(
        &self,
        query_vector: &[f32],
        top_n: usize,
        fetch_k: usize,
        lambda: f32,
    ) -> Result<Vec<SearchResult>> {
        if top_n == 0 {
            return Ok(Vec::new());
        }
        let rows = self
            .db
            .mmr_search(
                LABEL_CHUNK,
                PROP_EMBEDDING,
                query_vector,
                top_n,
                Some(fetch_k),
                Some(lambda),
                None,
                None,
            )
            .map_err(|e| anyhow!("mmr_search failed: {e}"))?;
        let session = self.db.session();
        materialize_results(&session, &rows, top_n)
    }

    // ---------- FileHash ----------

    pub fn get_file_hash(&self, path: &str, dim: HashDim) -> Result<Option<String>> {
        let session = self.db.session();
        let mut params = HashMap::new();
        params.insert("path".to_string(), Value::from(path));
        // dim.property_name() returns one of 9 fixed string literals, no injection risk.
        let stmt = format!(
            "MATCH (h:FileHash {{path: $path}}) RETURN h.{} LIMIT 1",
            dim.property_name()
        );
        let res = session
            .execute_with_params(&stmt, params)
            .map_err(|e| anyhow!("read FileHash.{}: {e}", dim.property_name()))?;
        let rows = res.rows();
        if rows.is_empty() {
            return Ok(None);
        }
        match &rows[0][0] {
            Value::String(s) => Ok(Some(s.to_string())),
            Value::Null => Ok(None),
            other => bail!(
                "FileHash.{} unexpected value: {other:?}",
                dim.property_name()
            ),
        }
    }

    pub fn set_file_hash(&self, path: &str, dim: HashDim, hash: &str) -> Result<()> {
        let mut session = self.db.session();
        session
            .begin_transaction()
            .map_err(|e| anyhow!("begin txn for set_file_hash: {e}"))?;
        let res = upsert_file_hash_in_session(&session, path, dim, hash);
        match res {
            Ok(()) => session.commit().map_err(|e| anyhow!("commit: {e}")),
            Err(e) => {
                let _ = session.rollback();
                Err(e)
            }
        }
    }

    /// Execute a raw GQL statement and return the underlying QueryResult.
    /// The statement is passed through verbatim — no predicate is auto-injected.
    pub fn execute_raw(&self, statement: &str) -> Result<grafeo::QueryResult> {
        let session = self.db.session();
        session
            .execute(statement)
            .map_err(|e| anyhow!("GQL execution failed: {e}"))
    }

    pub fn list_file_paths(&self) -> Result<Vec<String>> {
        let session = self.db.session();
        let res = session
            .execute("MATCH (h:FileHash) RETURN h.path")
            .map_err(|e| anyhow!("list FileHash paths: {e}"))?;
        let mut out = Vec::with_capacity(res.row_count());
        for row in res.rows() {
            if let Some(Value::String(s)) = row.first() {
                out.push(s.to_string());
            }
        }
        Ok(out)
    }

    /// Remove a single dim's hash from the `:FileHash` node for `path`.
    ///
    /// Two-phase same-session implementation (design decision 4 — empirical probe
    /// against grafeo 0.5.x ruled out single-statement variants: `FOREACH` does not
    /// parse, and `h[k]` with `k` as a variable silently yields NULL). When the
    /// removed dim is the LAST one with a value on the node, the entire node is
    /// `DETACH DELETE`d; otherwise only the target property is `REMOVE`d. Idempotent
    /// when the node or the target property does not exist.
    pub fn remove_file_hash(&self, path: &str, dim: HashDim) -> Result<()> {
        let mut session = self.db.session();
        session
            .begin_transaction()
            .map_err(|e| anyhow!("begin txn for remove_file_hash: {e}"))?;
        let res = (|| -> Result<()> {
            // Phase 1: read all 9 <dim>_hash properties in a single SELECT.
            // Property names are fixed literals (HashDim::ALL ordering), no injection risk.
            let select_cols: Vec<String> = HashDim::ALL
                .iter()
                .map(|d| format!("h.{}", d.property_name()))
                .collect();
            let read_stmt = format!(
                "MATCH (h:FileHash {{path: $path}}) RETURN {}",
                select_cols.join(", ")
            );
            let mut read_params = HashMap::new();
            read_params.insert("path".to_string(), Value::from(path));
            let read_res = session
                .execute_with_params(&read_stmt, read_params)
                .map_err(|e| anyhow!("read FileHash dims: {e}"))?;

            let rows = read_res.rows();
            if rows.is_empty() {
                // No node at all — nothing to remove (idempotent).
                return Ok(());
            }

            let row = &rows[0];
            let mut non_null_count = 0usize;
            let mut target_present = false;
            for (i, d) in HashDim::ALL.iter().enumerate() {
                let cell = row.get(i).unwrap_or(&Value::Null);
                if !matches!(cell, Value::Null) {
                    non_null_count += 1;
                    if *d == dim {
                        target_present = true;
                    }
                }
            }

            // Idempotent: nothing to do if the target dim was never set.
            // (Either no other dim either → node will linger with all-null until a future
            //  set_file_hash, which is harmless; or other dims still present → target was
            //  not on the node anyway.)
            if !target_present {
                return Ok(());
            }

            // Phase 2: choose write based on remaining dim count.
            let mut write_params = HashMap::new();
            write_params.insert("path".to_string(), Value::from(path));
            if non_null_count == 1 {
                // Target dim is the last one — drop the whole node.
                session
                    .execute_with_params(
                        "MATCH (h:FileHash {path: $path}) DETACH DELETE h",
                        write_params,
                    )
                    .map_err(|e| anyhow!("delete FileHash node: {e}"))?;
            } else {
                let prop = dim.property_name();
                let write_stmt = format!("MATCH (h:FileHash {{path: $path}}) REMOVE h.{prop}");
                session
                    .execute_with_params(&write_stmt, write_params)
                    .map_err(|e| anyhow!("remove FileHash.{prop}: {e}"))?;
            }
            Ok(())
        })();
        match res {
            Ok(()) => session.commit().map_err(|e| anyhow!("commit: {e}")),
            Err(e) => {
                let _ = session.rollback();
                Err(e)
            }
        }
    }
}

// ---------- helpers ----------

fn upsert_file_hash_in_session(
    session: &grafeo::Session,
    path: &str,
    dim: HashDim,
    hash: &str,
) -> Result<()> {
    let mut params = HashMap::new();
    params.insert("path".to_string(), Value::from(path));
    params.insert("hash".to_string(), Value::from(hash));
    let prop = dim.property_name();
    let stmt = format!("MERGE (h:FileHash {{path: $path}}) SET h.{prop} = $hash");
    session
        .execute_with_params(&stmt, params)
        .map_err(|e| anyhow!("upsert FileHash.{prop}: {e}"))?;
    Ok(())
}

/// Common field-readout for both `vector_search` and `mmr_search` outputs.
///
/// grafeo's `Vec<(NodeId, distance)>` carries no chunk fields; we re-read
/// `file_path` / `line_start` / `line_end` / `content` per node via the
/// session. `score` = `1.0 - distance` (cosine), matching the original
/// `KbStore::search` contract. Truncates to `top_n` so callers don't need to.
fn materialize_results(
    session: &grafeo::Session,
    rows: &[(NodeId, f32)],
    top_n: usize,
) -> Result<Vec<SearchResult>> {
    let mut out: Vec<SearchResult> = Vec::with_capacity(rows.len().min(top_n));
    for (nid, dist) in rows.iter().take(top_n) {
        let file_path = match session.get_node_property(*nid, "file_path") {
            Some(Value::String(s)) => s.to_string(),
            _ => String::new(),
        };
        let line_start = match session.get_node_property(*nid, "line_start") {
            Some(Value::Int64(n)) => u32::try_from(n).unwrap_or(0),
            _ => 0,
        };
        let line_end = match session.get_node_property(*nid, "line_end") {
            Some(Value::Int64(n)) => u32::try_from(n).unwrap_or(0),
            _ => 0,
        };
        let content = match session.get_node_property(*nid, "content") {
            Some(Value::String(s)) => s.to_string(),
            _ => String::new(),
        };
        out.push(SearchResult {
            file_path,
            line_start,
            line_end,
            content,
            score: 1.0 - *dist,
        });
    }
    Ok(out)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::tempdir;

    fn embed(values: &[f32]) -> Vec<f32> {
        // pad / trim to 4 dims for tests
        let mut v = values.to_vec();
        v.resize(4, 0.0);
        v
    }

    fn mk_chunk<'a>(
        file: &'a str,
        ls: u32,
        le: u32,
        body: &'a str,
        vec: Vec<f32>,
    ) -> ChunkInput<'a> {
        ChunkInput {
            file_path: file,
            line_start: ls,
            line_end: le,
            content: body,
            prefix: None,
            embedding: vec,
        }
    }

    #[test]
    fn open_creates_db_dir_without_ignore_files() {
        let tmp = tempdir().unwrap();
        let db_path = tmp.path().join("kb.db");
        {
            let store = KbStore::open(&db_path).expect("open db");
            store
                .write_index_config("model-x", 4, 8192)
                .expect("write IndexConfig");
        }

        for entry in std::fs::read_dir(tmp.path()).unwrap() {
            let e = entry.unwrap();
            let name = e.file_name();
            let s = name.to_string_lossy();
            assert!(
                s != ".p4ignore" && s != ".gitignore",
                "DB dir MUST NOT contain auto-written ignore file, found {s}"
            );
        }
        if db_path.is_dir() {
            for entry in std::fs::read_dir(&db_path).unwrap() {
                let e = entry.unwrap();
                let name = e.file_name();
                let s = name.to_string_lossy();
                assert!(
                    s != ".p4ignore" && s != ".gitignore",
                    "DB subdir MUST NOT contain auto-written ignore file, found {s}"
                );
            }
        }
    }

    #[test]
    fn index_config_round_trips() {
        let store = KbStore::open_in_memory();
        assert!(store.read_index_config().unwrap().is_none());
        store.write_index_config("voyage-4", 1024, 8192).unwrap();

        let cfg = store.read_index_config().unwrap().expect("present");
        assert_eq!(cfg.model, "voyage-4");
        assert_eq!(cfg.dim, 1024);
        assert_eq!(cfg.max_chunk_chars, 8192);

        // overwrite (singleton): MERGE should update in place, not duplicate
        store
            .write_index_config("text-3-small", 1536, 2048)
            .unwrap();
        let cfg2 = store.read_index_config().unwrap().expect("present");
        assert_eq!(cfg2.model, "text-3-small");
        assert_eq!(cfg2.dim, 1536);
        assert_eq!(cfg2.max_chunk_chars, 2048);
    }

    #[test]
    fn index_config_is_db_singleton() {
        // After multiple write_index_config calls the count of :IndexConfig
        // nodes MUST stay at exactly 1 (MERGE without key, not append).
        let store = KbStore::open_in_memory();
        store.write_index_config("m1", 4, 100).unwrap();
        store.write_index_config("m2", 8, 200).unwrap();
        store.write_index_config("m3", 16, 400).unwrap();

        let session = store.db.session();
        let res = session
            .execute("MATCH (c:IndexConfig) RETURN count(c)")
            .unwrap();
        let rows = res.rows();
        assert_eq!(rows.len(), 1, "count() row missing");
        let n = match &rows[0][0] {
            Value::Int64(n) => *n,
            other => panic!("unexpected count() value: {other:?}"),
        };
        assert_eq!(n, 1, ":IndexConfig MUST be a DB-global singleton, got {n}");
    }

    #[test]
    fn file_hash_per_dim_crud() {
        // Each of the 9 HashDim variants must support the full set/get/remove cycle
        // independently. Use a fresh store per dim so each dim's lifecycle is isolated.
        for dim in HashDim::ALL.iter().copied() {
            let store = KbStore::open_in_memory();
            assert!(
                store.get_file_hash("a.md", dim).unwrap().is_none(),
                "dim {dim:?}: initial get must be None"
            );
            store.set_file_hash("a.md", dim, "h1").unwrap();
            assert_eq!(
                store.get_file_hash("a.md", dim).unwrap().as_deref(),
                Some("h1"),
                "dim {dim:?}: after first set"
            );
            store.set_file_hash("a.md", dim, "h2").unwrap();
            assert_eq!(
                store.get_file_hash("a.md", dim).unwrap().as_deref(),
                Some("h2"),
                "dim {dim:?}: after overwrite"
            );
            store.remove_file_hash("a.md", dim).unwrap();
            assert!(
                store.get_file_hash("a.md", dim).unwrap().is_none(),
                "dim {dim:?}: after remove"
            );
        }
    }

    #[test]
    fn multi_dim_isolation_in_storage() {
        // Writing different dims on the same path must not overwrite each other,
        // and removing dims one by one must finally drop the node.
        let store = KbStore::open_in_memory();
        store.set_file_hash("a.md", HashDim::Vector, "vh").unwrap();
        store.set_file_hash("a.md", HashDim::Graph, "gh").unwrap();
        store.set_file_hash("a.md", HashDim::Llm, "lh").unwrap();

        assert_eq!(
            store
                .get_file_hash("a.md", HashDim::Vector)
                .unwrap()
                .as_deref(),
            Some("vh")
        );
        assert_eq!(
            store
                .get_file_hash("a.md", HashDim::Graph)
                .unwrap()
                .as_deref(),
            Some("gh")
        );
        assert_eq!(
            store
                .get_file_hash("a.md", HashDim::Llm)
                .unwrap()
                .as_deref(),
            Some("lh")
        );

        store.remove_file_hash("a.md", HashDim::Vector).unwrap();
        assert!(
            store
                .get_file_hash("a.md", HashDim::Vector)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            store
                .get_file_hash("a.md", HashDim::Graph)
                .unwrap()
                .as_deref(),
            Some("gh"),
            "graph_hash must survive vector_hash removal"
        );
        assert_eq!(
            store
                .get_file_hash("a.md", HashDim::Llm)
                .unwrap()
                .as_deref(),
            Some("lh"),
            "llm_hash must survive vector_hash removal"
        );

        store.remove_file_hash("a.md", HashDim::Graph).unwrap();
        store.remove_file_hash("a.md", HashDim::Llm).unwrap();

        // Last dim removed → whole node should be gone (count(h) == 0).
        let session = store.db.session();
        let mut params = HashMap::new();
        params.insert("path".to_string(), Value::from("a.md"));
        let res = session
            .execute_with_params("MATCH (h:FileHash {path: $path}) RETURN count(h)", params)
            .unwrap();
        let n = match &res.rows()[0][0] {
            Value::Int64(n) => *n,
            other => panic!("unexpected count: {other:?}"),
        };
        assert_eq!(
            n, 0,
            ":FileHash node should be DETACH DELETEd after last dim removal"
        );
    }

    #[test]
    fn remove_when_no_other_dim_deletes_node() {
        let store = KbStore::open_in_memory();
        store
            .set_file_hash("solo.md", HashDim::Vector, "h")
            .unwrap();
        store.remove_file_hash("solo.md", HashDim::Vector).unwrap();

        let session = store.db.session();
        let res = session
            .execute("MATCH (h:FileHash) RETURN count(h)")
            .unwrap();
        let n = match &res.rows()[0][0] {
            Value::Int64(n) => *n,
            other => panic!("unexpected count: {other:?}"),
        };
        assert_eq!(
            n, 0,
            "removing the only dim must DETACH DELETE the whole node"
        );
    }

    #[test]
    fn remove_when_other_dim_present_keeps_node() {
        let store = KbStore::open_in_memory();
        store
            .set_file_hash("two.md", HashDim::Vector, "vh")
            .unwrap();
        store.set_file_hash("two.md", HashDim::Graph, "gh").unwrap();

        store.remove_file_hash("two.md", HashDim::Vector).unwrap();

        // Node must remain: graph_hash still present, count(h) == 1
        let session = store.db.session();
        let res = session
            .execute("MATCH (h:FileHash) RETURN count(h)")
            .unwrap();
        let n = match &res.rows()[0][0] {
            Value::Int64(n) => *n,
            other => panic!("unexpected count: {other:?}"),
        };
        assert_eq!(
            n, 1,
            ":FileHash node must remain when other dim still has a hash"
        );
        assert_eq!(
            store
                .get_file_hash("two.md", HashDim::Graph)
                .unwrap()
                .as_deref(),
            Some("gh"),
            "graph_hash value must be untouched"
        );
        assert!(
            store
                .get_file_hash("two.md", HashDim::Vector)
                .unwrap()
                .is_none(),
            "vector_hash must be gone"
        );
    }

    #[test]
    fn upsert_chunks_replaces_old_and_writes_file_hash() {
        let store = KbStore::open_in_memory();
        store.ensure_chunk_vector_index(4).unwrap();

        let chunks_v1 = vec![
            mk_chunk("a.md", 1, 5, "first", embed(&[1.0, 0.0, 0.0])),
            mk_chunk("a.md", 6, 10, "second", embed(&[0.0, 1.0, 0.0])),
        ];
        store
            .upsert_chunks_for_file("a.md", "hash-v1", &chunks_v1)
            .unwrap();
        assert_eq!(
            store
                .get_file_hash("a.md", HashDim::Vector)
                .unwrap()
                .as_deref(),
            Some("hash-v1")
        );

        // re-upsert with one chunk: old chunks must be gone
        let chunks_v2 = vec![mk_chunk("a.md", 1, 9, "merged", embed(&[1.0, 1.0, 0.0]))];
        store
            .upsert_chunks_for_file("a.md", "hash-v2", &chunks_v2)
            .unwrap();
        assert_eq!(
            store
                .get_file_hash("a.md", HashDim::Vector)
                .unwrap()
                .as_deref(),
            Some("hash-v2")
        );

        let hits = store.search(&embed(&[1.0, 0.0, 0.0]), 10).unwrap();
        let bodies: Vec<_> = hits.iter().map(|h| h.content.clone()).collect();
        assert!(bodies.contains(&"merged".to_string()), "got {bodies:?}");
        assert!(
            !bodies.contains(&"first".to_string()),
            "old chunk leaked: {bodies:?}"
        );
        assert!(
            !bodies.contains(&"second".to_string()),
            "old chunk leaked: {bodies:?}"
        );
    }

    #[test]
    fn upsert_failure_rolls_back_file_hash() {
        // chunks list with an embedding of mismatched dimension forces grafeo to reject
        // the embedding write; the txn must roll back, leaving NO FileHash and NO chunks.
        let store = KbStore::open_in_memory();
        store.ensure_chunk_vector_index(4).unwrap();

        // baseline: insert a good file
        store
            .upsert_chunks_for_file(
                "good.md",
                "g-hash",
                &[mk_chunk(
                    "good.md",
                    1,
                    1,
                    "good body",
                    embed(&[1.0, 0.0, 0.0]),
                )],
            )
            .unwrap();
        assert_eq!(
            store
                .get_file_hash("good.md", HashDim::Vector)
                .unwrap()
                .as_deref(),
            Some("g-hash")
        );

        // bad ingest: wrong-dim vector
        let bad = vec![mk_chunk("bad.md", 1, 1, "bad body", vec![1.0, 0.0])]; // dim 2 vs index 4
        let result = store.upsert_chunks_for_file("bad.md", "b-hash", &bad);
        assert!(result.is_err(), "wrong-dim embedding should fail");

        // bad.md must have NO FileHash and NO chunks
        assert!(
            store
                .get_file_hash("bad.md", HashDim::Vector)
                .unwrap()
                .is_none(),
            "FileHash for failed file leaked"
        );
        let hits = store.search(&embed(&[1.0, 0.0, 0.0]), 10).unwrap();
        assert!(
            hits.iter().all(|h| h.file_path != "bad.md"),
            "bad chunk leaked: {hits:?}"
        );

        // good.md must still be there
        assert_eq!(
            store
                .get_file_hash("good.md", HashDim::Vector)
                .unwrap()
                .as_deref(),
            Some("g-hash")
        );
    }

    #[test]
    fn delete_file_completely_removes_chunks_and_hash() {
        let store = KbStore::open_in_memory();
        store.ensure_chunk_vector_index(4).unwrap();

        store
            .upsert_chunks_for_file(
                "x.md",
                "h",
                &[mk_chunk(
                    "x.md",
                    1,
                    1,
                    "to be removed",
                    embed(&[1.0, 0.0, 0.0]),
                )],
            )
            .unwrap();
        assert!(
            store
                .get_file_hash("x.md", HashDim::Vector)
                .unwrap()
                .is_some()
        );

        store.delete_file_completely("x.md").unwrap();
        assert!(
            store
                .get_file_hash("x.md", HashDim::Vector)
                .unwrap()
                .is_none()
        );

        let hits = store.search(&embed(&[1.0, 0.0, 0.0]), 10).unwrap();
        assert!(hits.iter().all(|h| h.file_path != "x.md"), "{hits:?}");
    }

    #[test]
    fn mmr_increases_file_path_diversity() {
        // Highly homogeneous corpus: doc-a.md has 5 near-identical chunks
        // (clustered around [1.0, 0.0, ...]) plus doc-b.md / doc-c.md each
        // contribute one slightly-farther chunk. A pure cosine top-3 will
        // bias toward doc-a's cluster; MMR with lambda=0.3 should pull at
        // least one doc-b / doc-c chunk in.
        let store = KbStore::open_in_memory();
        store.ensure_chunk_vector_index(4).unwrap();

        let a_chunks: Vec<ChunkInput<'_>> = (0..5)
            .map(|i| {
                ChunkInput {
                    file_path: "doc-a.md",
                    line_start: i + 1,
                    line_end: i + 1,
                    content: "a-cluster",
                    prefix: None,
                    // tiny jitter on the 2nd dim so the 5 vectors are not bit-identical
                    embedding: vec![1.0, 0.001 * (i as f32), 0.0, 0.0],
                }
            })
            .collect();
        store
            .upsert_chunks_for_file("doc-a.md", "ha", &a_chunks)
            .unwrap();
        store
            .upsert_chunks_for_file(
                "doc-b.md",
                "hb",
                &[mk_chunk(
                    "doc-b.md",
                    1,
                    1,
                    "b-topic",
                    vec![0.7, 0.7, 0.0, 0.0],
                )],
            )
            .unwrap();
        store
            .upsert_chunks_for_file(
                "doc-c.md",
                "hc",
                &[mk_chunk(
                    "doc-c.md",
                    1,
                    1,
                    "c-topic",
                    vec![0.6, 0.0, 0.8, 0.0],
                )],
            )
            .unwrap();

        let q = vec![1.0_f32, 0.0, 0.0, 0.0];
        let cosine = store.search(&q, 3).unwrap();
        let mmr = store.search_mmr(&q, 3, 12, 0.3).unwrap();

        let cosine_files: std::collections::HashSet<_> =
            cosine.iter().map(|h| h.file_path.clone()).collect();
        let mmr_files: std::collections::HashSet<_> =
            mmr.iter().map(|h| h.file_path.clone()).collect();
        assert!(
            mmr_files.len() > cosine_files.len(),
            "mmr file count ({}) must strictly exceed cosine file count ({}); cosine={cosine_files:?} mmr={mmr_files:?}",
            mmr_files.len(),
            cosine_files.len(),
        );
    }

    #[test]
    fn search_mmr_lambda_one_matches_search_within_tolerance() {
        // lambda=1.0 collapses MMR's diversity term — selection becomes pure
        // relevance, so the top-N node order should match `search` (allow
        // ties on equal distance to swap, but the *set* must match).
        let store = KbStore::open_in_memory();
        store.ensure_chunk_vector_index(4).unwrap();

        let chunks = vec![
            mk_chunk("a.md", 1, 1, "x1", embed(&[1.0, 0.0, 0.0])),
            mk_chunk("a.md", 2, 2, "x2", embed(&[0.9, 0.1, 0.0])),
            mk_chunk("a.md", 3, 3, "x3", embed(&[0.5, 0.5, 0.0])),
        ];
        store.upsert_chunks_for_file("a.md", "h", &chunks).unwrap();
        store
            .upsert_chunks_for_file(
                "b.md",
                "h2",
                &[mk_chunk("b.md", 1, 1, "y1", embed(&[-1.0, 0.0, 0.0]))],
            )
            .unwrap();

        let q = embed(&[1.0, 0.0, 0.0]);
        let cosine = store.search(&q, 3).unwrap();
        let mmr = store.search_mmr(&q, 3, 12, 1.0).unwrap();

        assert_eq!(cosine.len(), mmr.len(), "lengths differ");
        // All seed vectors carry distinct distances from the query, so the
        // expected order is deterministic — compare ordered Vec keys so an
        // ordering regression is caught (HashSet equality would mask it).
        let cosine_keys: Vec<_> = cosine
            .iter()
            .map(|h| (h.file_path.clone(), h.line_start, h.line_end))
            .collect();
        let mmr_keys: Vec<_> = mmr
            .iter()
            .map(|h| (h.file_path.clone(), h.line_start, h.line_end))
            .collect();
        assert_eq!(
            cosine_keys, mmr_keys,
            "lambda=1.0 mmr top-N order must equal search top-N order"
        );
    }

    // Suppress unused warning if HashMap is not exercised by other tests.
    #[allow(dead_code)]
    fn _hm_marker() -> HashMap<String, String> {
        HashMap::new()
    }
}
