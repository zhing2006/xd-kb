// Whole-DB migration: WAL-directory format -> single-file `.grafeo` format.
//
// grafeo's WAL-directory format never compacts its log (the KB directory
// only ever grows); the single-file `.grafeo` container checkpoints all
// data sections on close and removes its sidecar WAL. `migrate` copies
// every node, edge and property into a fresh single-file DB (embedding
// vectors verbatim; no re-embedding, no network), verifies counts, then
// swaps paths:
//
//   .migrate-tmp-<pid>-<name>.grafeo -> <name>.grafeo  (atomic no-clobber link)
//   <old-db-dir>                     -> <old-db-dir>.bak (kept; delete manually)
//
// A `<old-db-dir>.migrate-lock` file guards the whole swap: concurrent
// migrations exclude each other, and every other xd-kb command refuses to
// open the DB while the lock exists (see main.rs). Other commands check
// the lock once at startup rather than holding one for their lifetime —
// migration is a one-shot, manually-run operation, so the startup-window
// race has no real exposure and a full cross-command lock protocol is
// intentionally not implemented.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::storage::KbStore;

/// Lock file guarding a migration of `db_path`. Public so main.rs can refuse
/// to run other commands against a KB that is mid-migration. Lock identity
/// follows the canonical path, so symlink aliases of the same KB directory
/// map to the same lock file (falls back to the literal path while the DB
/// does not exist yet).
pub fn lock_path(db_path: &Path) -> PathBuf {
    let canonical = fs::canonicalize(db_path).unwrap_or_else(|_| db_path.to_path_buf());
    sidecar(&canonical, ".migrate-lock")
}

/// RAII migration lock: created atomically (`create_new`). Release
/// explicitly via [`Self::release`] to surface removal errors; `Drop` is a
/// best-effort fallback that can only warn.
struct MigrateLock {
    path: Option<PathBuf>,
}

impl MigrateLock {
    fn acquire(db_path: &Path) -> Result<Self> {
        let path = lock_path(db_path);
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(_) => Ok(Self { path: Some(path) }),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => bail!(
                "another migration of {} appears to be running (lock file {} exists) — \
                 wait for it to finish, or remove the stale lock if none is running",
                db_path.display(),
                path.display()
            ),
            Err(e) => Err(e).with_context(|| format!("create migration lock {}", path.display())),
        }
    }

    fn release(mut self) -> Result<()> {
        if let Some(path) = self.path.take() {
            fs::remove_file(&path)
                .with_context(|| format!("remove migration lock {}", path.display()))?;
        }
        Ok(())
    }
}

impl Drop for MigrateLock {
    fn drop(&mut self) {
        if let Some(path) = self.path.take()
            && fs::remove_file(&path).is_err()
            && path.exists()
        {
            eprintln!(
                "warning: migration lock {} could not be removed — remove it manually, \
                 other commands refuse to open the KB while it exists",
                path.display()
            );
        }
    }
}

/// Test-only failure injection: debug builds abort at the named point when
/// `XD_KB_MIGRATE_FAILPOINT` matches. Compiled out of release builds. Driven
/// from the CLI integration tests (separate process, so no env races with
/// parallel in-process tests).
fn failpoint(name: &str) -> Result<()> {
    #[cfg(debug_assertions)]
    if std::env::var("XD_KB_MIGRATE_FAILPOINT").as_deref() == Ok(name) {
        bail!("failpoint {name} triggered");
    }
    #[cfg(not(debug_assertions))]
    let _ = name;
    Ok(())
}

pub fn run(db_path: &Path, target: Option<PathBuf>) -> Result<()> {
    // Normalize away trailing separators: sidecar paths are built by
    // appending to the raw path, and `kb.db/` would otherwise put `.bak`
    // and the lock INSIDE the source directory.
    let db_path = &PathBuf::from(db_path.components().as_path());
    if !db_path.exists() {
        bail!("KB not found at {}", db_path.display());
    }
    if db_path.is_file() {
        bail!(
            "{} is already a single-file KB — nothing to migrate",
            db_path.display()
        );
    }
    ensure_legacy_kb_layout(db_path)?;

    let target = target.unwrap_or_else(|| db_path.with_extension("grafeo"));
    if target.extension().and_then(|e| e.to_str()) != Some("grafeo") {
        // grafeo's auto format detection for new paths is a case-sensitive
        // extension match; `out.GRAFEO` would silently create another WAL
        // directory, so require the exact lowercase form.
        bail!(
            "target must have a lowercase .grafeo extension, got {}",
            target.display()
        );
    }
    ensure_target_outside_source(db_path, &target)?;
    if target.exists() {
        bail!(
            "target {} already exists — remove it or pass --target",
            target.display()
        );
    }
    let bak = bak_path(db_path);
    if bak.exists() {
        bail!(
            "backup path {} already exists — remove it before migrating",
            bak.display()
        );
    }

    let lock = MigrateLock::acquire(db_path)?;

    let tmp = tmp_path(&target)?;
    // Leftovers from previous failed runs (any pid) are ours to discard.
    remove_stale_tmp_siblings(&target)?;

    let old_size = dir_size(db_path)?;
    let copied = copy_into(db_path, &tmp).and_then(|r| {
        failpoint("after-copy")?;
        // Belt and braces: if grafeo's format detection ever surprises us,
        // refuse to publish anything that is not a plain file.
        if !tmp.is_file() {
            bail!(
                "{} is not a plain file after the copy — refusing to publish it",
                tmp.display()
            );
        }
        Ok(r)
    });
    let report = match copied {
        Ok(r) => r,
        // Leave no half-built KB behind; if the cleanup itself fails, report
        // both instead of swallowing the cleanup error.
        Err(e) => return Err(combine_cleanup(e, remove_tmp_artifacts(&tmp))),
    };

    let published = failpoint("target-publish").and_then(|_| publish_no_clobber(&tmp, &target));
    if let Err(e) = published {
        return Err(combine_cleanup(e, remove_tmp_artifacts(&tmp)));
    }

    let swapped = failpoint("after-target-publish").and_then(|_| {
        fs::rename(db_path, &bak).with_context(|| {
            format!(
                "renaming the old directory {} to {} failed",
                db_path.display(),
                bak.display()
            )
        })
    });
    if let Err(e) = swapped {
        // Un-publish the new file so no half-switched state remains: the old
        // directory is still live at its original path, callers keep using it.
        let rollback = fs::remove_file(&target)
            .with_context(|| format!("un-publish {}", target.display()))
            .and_then(|_| remove_tmp_artifacts(&tmp));
        return Err(combine_cleanup(e, rollback));
    }

    // Post-publish tidy-up (best effort — the migration itself is complete):
    // grafeo leaves empty `.spill/` sidecar dirs behind for both stores.
    for leftover in [sidecar(&tmp, ".spill"), sidecar(db_path, ".spill")] {
        if leftover.is_dir() && fs::remove_dir_all(&leftover).is_err() {
            eprintln!("warning: could not remove leftover {}", leftover.display());
        }
    }

    let new_size = fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
    println!("migrated {} -> {}", db_path.display(), target.display());
    println!(
        "  nodes: {} ({} chunks, {} file hashes), edges: {}",
        report.nodes, report.chunks, report.file_hashes, report.edges
    );
    println!(
        "  vector index: not built here — query rebuilds it per process \
         (grafeo 0.5.x cannot reload a persisted index)"
    );
    println!(
        "  size: {} (directory) -> {} (single file)",
        human_size(old_size),
        human_size(new_size)
    );
    println!(
        "  old data kept at {} — delete it once the new KB checks out",
        bak.display()
    );
    println!("  point callers at --db {}", target.display());
    // Past the commit point (the old directory is renamed away) the
    // migration has succeeded; a failed lock cleanup must not turn that
    // into a reported failure — warn and leave removal to the user.
    if let Err(e) = failpoint("lock-release").and_then(|_| lock.release()) {
        eprintln!(
            "warning: {e:#} — remove the lock file manually, other commands \
             refuse to open the KB while it exists"
        );
    }
    Ok(())
}

/// Publish atomically WITHOUT clobbering: unlike rename, hard_link fails if
/// `target` sprang into existence during the copy, so a concurrent writer's
/// file survives and we fail instead of silently replacing it. The tmp link
/// must also go away — a leftover would alias the published file — and since
/// the commit point (source-directory rename) has not been passed yet, a
/// failed removal un-publishes the target and errors instead of half-succeeding.
fn publish_no_clobber(tmp: &Path, target: &Path) -> Result<()> {
    fs::hard_link(tmp, target).with_context(|| {
        format!(
            "verified KB written to {} but publishing it to {} failed \
             (did the target appear during the copy?)",
            tmp.display(),
            target.display()
        )
    })?;
    if let Err(e) =
        fs::remove_file(tmp).with_context(|| format!("remove tmp link {}", tmp.display()))
    {
        let rollback =
            fs::remove_file(target).with_context(|| format!("un-publish {}", target.display()));
        return Err(combine_cleanup(e, rollback));
    }
    Ok(())
}

struct CopyReport {
    nodes: u64,
    edges: u64,
    chunks: u64,
    file_hashes: u64,
}

/// A grafeo WAL-directory KB carries a `wal/` subdirectory holding
/// `wal_*.log` segments. Refuse anything else up front: `GrafeoDB::open`
/// would happily initialise a fresh WAL inside an arbitrary directory, and
/// the final swap would then rename that unrelated directory away as `.bak`.
fn ensure_legacy_kb_layout(db_path: &Path) -> Result<()> {
    let wal_dir = db_path.join("wal");
    let mut has_segment = false;
    if wal_dir.is_dir() {
        for entry in
            fs::read_dir(&wal_dir).with_context(|| format!("read {}", wal_dir.display()))?
        {
            let entry = entry?;
            // Match grafeo's own segment naming (`wal_<number>.log`, plain
            // file) so a stray `wal/wal_notes.log` cannot make an unrelated
            // directory pass as a KB.
            let is_file = entry.file_type().map(|t| t.is_file()).unwrap_or(false);
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let numbered = name
                .strip_prefix("wal_")
                .and_then(|s| s.strip_suffix(".log"))
                // grafeo's segment counter is a u64: reject digit strings
                // beyond that range (and signs, which parse alone would let
                // through) so an unrelated directory cannot pass as a KB.
                .is_some_and(|mid| {
                    mid.bytes().all(|b| b.is_ascii_digit()) && mid.parse::<u64>().is_ok()
                });
            if is_file && numbered {
                has_segment = true;
                break;
            }
        }
    }
    if !has_segment {
        bail!(
            "{} does not look like a grafeo WAL-directory KB (no wal/wal_*.log inside) — \
             refusing to migrate it",
            db_path.display()
        );
    }
    Ok(())
}

/// Reject a target (and thus its same-directory tmp sibling) inside the
/// source KB directory: the final `<old> -> <old>.bak` rename would drag the
/// freshly published file away with it. Canonicalization also catches
/// symlink aliases of the source directory.
fn ensure_target_outside_source(db_path: &Path, target: &Path) -> Result<()> {
    let src = fs::canonicalize(db_path)
        .with_context(|| format!("canonicalize source {}", db_path.display()))?;
    let parent = match target.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let parent = fs::canonicalize(parent)
        .with_context(|| format!("target directory {} must exist", parent.display()))?;
    if parent.starts_with(&src) {
        bail!(
            "target {} is inside the source KB directory {} — it would be moved away by the \
             .bak rename; pick a target outside it",
            target.display(),
            db_path.display()
        );
    }
    Ok(())
}

/// Attach a cleanup failure to the original error instead of swallowing it.
fn combine_cleanup(err: anyhow::Error, cleanup: Result<()>) -> anyhow::Error {
    match cleanup {
        Ok(()) => err,
        Err(ce) => err.context(format!("cleanup after the failure also failed: {ce:#}")),
    }
}

/// Dump the source KB and rebuild it at `tmp` (single-file format), verify
/// counts, and close both stores. On success `tmp` is a complete, durable
/// KB ready to be published (hard-linked) into place.
fn copy_into(db_path: &Path, tmp: &Path) -> Result<CopyReport> {
    let dst = KbStore::open(tmp)?;
    let report = copy_records(db_path, &dst)?;
    dst.close()?;
    Ok(report)
}

/// Copy every node and edge from the source KB into the already-open `dst`
/// and verify the four counts. Split from [`copy_into`] so tests can assert
/// on the destination store before it is closed. Deliberately does NOT build
/// the vector index: grafeo 0.5.x checkpoints HNSW topology into the
/// container but never reloads it on open (the load path drops the index
/// definitions before the VectorStore section is read), so anything built
/// here would be dead weight in the file. Query paths rebuild per process,
/// as on 0.3.0.
fn copy_records(db_path: &Path, dst: &KbStore) -> Result<CopyReport> {
    let src = KbStore::open(db_path)?;
    let nodes = src.dump_all_nodes()?;
    let edges = src.dump_all_edges()?;
    let src_nodes = src.count_nodes()?;
    let src_edges = src.count_edges()?;
    let src_chunks = src.count_label("Chunk")?;
    let src_hashes = src.count_label("FileHash")?;
    src.close()?;

    let mut id_map: HashMap<i64, grafeo::NodeId> = HashMap::with_capacity(nodes.len());
    for n in &nodes {
        let labels: Vec<&str> = n.labels.iter().map(String::as_str).collect();
        let new_id = dst.insert_node_raw(&labels, n.props.clone());
        id_map.insert(n.id, new_id);
    }
    for e in &edges {
        let (Some(&s), Some(&t)) = (id_map.get(&e.source), id_map.get(&e.target)) else {
            bail!(
                "edge {} -> {} references a node missing from the dump",
                e.source,
                e.target
            );
        };
        dst.insert_edge_raw(s, t, &e.edge_type, e.props.clone());
    }

    let checks = [
        ("nodes", src_nodes, dst.count_nodes()?),
        ("edges", src_edges, dst.count_edges()?),
        ("chunks", src_chunks, dst.count_label("Chunk")?),
        ("file hashes", src_hashes, dst.count_label("FileHash")?),
    ];
    for (what, expected, got) in checks {
        if expected != got {
            bail!("verification failed: {what} count {got} != source {expected}");
        }
    }
    failpoint("verify")?;

    Ok(CopyReport {
        nodes: src_nodes,
        edges: src_edges,
        chunks: src_chunks,
        file_hashes: src_hashes,
    })
}

fn bak_path(db_path: &Path) -> PathBuf {
    sidecar(db_path, ".bak")
}

/// `<path><suffix>` as a sibling path (e.g. `kb.db` -> `kb.db.bak`).
fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// Same-directory dot-prefixed sibling of `target`, keeping the `.grafeo`
/// extension so grafeo's auto detection creates it in single-file format.
/// The pid segment keeps two concurrent migrations pointed at the same
/// target from opening each other's half-written tmp file.
fn tmp_path(target: &Path) -> Result<PathBuf> {
    let Some(name) = target.file_name() else {
        bail!("target {} has no file name", target.display());
    };
    let mut tmp_name = std::ffi::OsString::from(format!(".migrate-tmp-{}-", std::process::id()));
    tmp_name.push(name);
    Ok(target.with_file_name(tmp_name))
}

/// True when directory entry `entry_name` is a tmp container (or one of its
/// sidecars) that a migration to `target_name` would have minted — with or
/// without a pid segment (pre-pid layouts included).
fn is_tmp_for(target_name: &str, entry_name: &str) -> bool {
    let Some(rest) = entry_name.strip_prefix(".migrate-tmp-") else {
        return false;
    };
    for suffix in ["", ".wal", ".spill"] {
        let want = format!("{target_name}{suffix}");
        if rest == want {
            return true;
        }
        if let Some(pid) = rest.strip_suffix(&want).and_then(|p| p.strip_suffix('-'))
            && !pid.is_empty()
            && pid.bytes().all(|b| b.is_ascii_digit())
        {
            return true;
        }
    }
    false
}

/// Remove tmp containers and sidecars left behind for `target` by previous
/// failed runs, whatever pid minted them (the migration lock guarantees no
/// live migration of this KB is running concurrently).
fn remove_stale_tmp_siblings(target: &Path) -> Result<()> {
    let Some(name) = target.file_name() else {
        return Ok(());
    };
    let name = name.to_string_lossy();
    let dir = match target.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let entry_name = entry.file_name();
        if !is_tmp_for(&name, &entry_name.to_string_lossy()) {
            continue;
        }
        let path = dir.join(&entry_name);
        let removed = if entry.file_type()?.is_dir() {
            fs::remove_dir_all(&path)
        } else {
            fs::remove_file(&path)
        };
        removed.with_context(|| format!("remove stale {}", path.display()))?;
    }
    Ok(())
}

/// Remove the tmp container file and grafeo's sidecars (`<tmp>.wal/` and
/// `<tmp>.spill/`). Errors are surfaced, not swallowed — a failed cleanup
/// leaves artifacts the caller must know about.
fn remove_tmp_artifacts(tmp: &Path) -> Result<()> {
    if tmp.exists() {
        fs::remove_file(tmp).with_context(|| format!("remove stale {}", tmp.display()))?;
    }
    for suffix in [".wal", ".spill"] {
        let dir = sidecar(tmp, suffix);
        if dir.exists() {
            fs::remove_dir_all(&dir).with_context(|| format!("remove stale {}", dir.display()))?;
        }
    }
    Ok(())
}

fn dir_size(path: &Path) -> Result<u64> {
    let mut total = 0u64;
    for entry in fs::read_dir(path).with_context(|| format!("read {}", path.display()))? {
        let entry = entry?;
        let meta = entry.metadata()?;
        total += if meta.is_dir() {
            dir_size(&entry.path())?
        } else {
            meta.len()
        };
    }
    Ok(total)
}

fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut v = bytes as f64;
    let mut unit = 0;
    while v >= 1024.0 && unit < UNITS.len() - 1 {
        v /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[0])
    } else {
        format!("{v:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{ChunkInput, DumpedEdge, DumpedNode};
    use grafeo::Value;
    use tempfile::tempdir;

    const DIM: usize = 4;
    /// Reopen-query-close cycles for the size-stability regression test.
    const QUERY_CYCLES: usize = 5;

    fn chunk<'a>(content: &'a str, embedding: &[f32]) -> ChunkInput<'a> {
        ChunkInput {
            file_path: "a.md",
            line_start: 1,
            line_end: 2,
            content,
            prefix: None,
            embedding: embedding.to_vec(),
        }
    }

    /// Old-format (WAL-directory) KB exercising the full production surface
    /// migrate must carry over: chunks (with prefix and line ranges), all
    /// nine FileHash dimensions, IndexConfig, a property-carrying edge, and
    /// underscore-prefixed user properties on both a node and an edge.
    fn build_source_kb(db_path: &Path) {
        let store = KbStore::open(db_path).expect("open source db");
        store
            .write_index_config("model-x", DIM, 20_000)
            .expect("write config");
        store
            .upsert_chunks_for_file(
                "a.md",
                "hash-a",
                &[
                    chunk("alpha", &[1.0, 0.0, 0.0, 0.0]),
                    ChunkInput {
                        file_path: "a.md",
                        line_start: 10,
                        line_end: 12,
                        content: "beta",
                        prefix: Some("# 标题"),
                        embedding: vec![0.0, 1.0, 0.0, 0.0],
                    },
                ],
            )
            .expect("upsert chunks");
        // Populate the remaining eight FileHash dimensions so the roundtrip
        // proves every dimension survives verbatim.
        store
            .execute_raw(
                "MATCH (h:FileHash {path:'a.md'}) SET h.graph_hash='g', h.llm_hash='l', \
                 h.fulltext_hash='f', h.summary_hash='s', h.entity_hash='e', \
                 h.vector_2_hash='v2', h.custom_hash='c', h.custom_2_hash='c2'",
            )
            .expect("set hash dims");
        store
            .execute_raw("INSERT (:Entity {name:'甲'})-[:KNOWS {w:3}]->(:Entity {name:'乙'})")
            .expect("insert graph");
        // Underscore-prefixed USER properties are data, not grafeo metadata —
        // regression guard for the dump filter dropping them.
        let a = store.insert_node_raw(
            &["Extra"],
            vec![
                ("_custom".to_string(), Value::from("keep-me")),
                ("normal".to_string(), Value::Int64(1)),
            ],
        );
        let b = store.insert_node_raw(&["Extra"], vec![("name".to_string(), Value::from("x"))]);
        store.insert_edge_raw(a, b, "REL", vec![("_weight".to_string(), Value::Int64(2))]);
        store.close().expect("close source");
    }

    /// No `.migrate-tmp-*` container/sidecar leftovers in `dir`.
    fn no_tmp_leftovers(dir: &Path) -> bool {
        std::fs::read_dir(dir).unwrap().all(|e| {
            !e.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".migrate-tmp-")
        })
    }

    /// Order-independent fingerprints for whole-KB content comparison
    /// (node ids differ across the copy, so compare labels + properties).
    fn node_keys(nodes: &[DumpedNode]) -> Vec<String> {
        let mut keys: Vec<String> = nodes
            .iter()
            .map(|n| {
                let mut labels = n.labels.clone();
                labels.sort();
                let mut props: Vec<String> =
                    n.props.iter().map(|(k, v)| format!("{k}={v:?}")).collect();
                props.sort();
                format!("{labels:?}|{props:?}")
            })
            .collect();
        keys.sort();
        keys
    }

    fn edge_keys(edges: &[DumpedEdge]) -> Vec<String> {
        let mut keys: Vec<String> = edges
            .iter()
            .map(|e| {
                let mut props: Vec<String> =
                    e.props.iter().map(|(k, v)| format!("{k}={v:?}")).collect();
                props.sort();
                format!("{}|{props:?}", e.edge_type)
            })
            .collect();
        keys.sort();
        keys
    }

    #[test]
    fn migrate_full_kb_roundtrip() {
        let tmp = tempdir().unwrap();
        let db_path = tmp.path().join("kb.db");
        build_source_kb(&db_path);

        // Pre-migration snapshot: full dump + top-N over a per-process index.
        let (pre_nodes, pre_edges, pre_hits) = {
            let store = KbStore::open(&db_path).expect("reopen source");
            let nodes = store.dump_all_nodes().unwrap();
            let edges = store.dump_all_edges().unwrap();
            store.ensure_chunk_vector_index(DIM).unwrap();
            let hits = store.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
            store.close().unwrap();
            (nodes, edges, hits)
        };

        run(&db_path, None).expect("migrate");

        let target = tmp.path().join("kb.grafeo");
        let bak = tmp.path().join("kb.db.bak");
        assert!(target.is_file(), "target must be a single file");
        assert!(bak.is_dir(), "old dir must be kept as .bak");
        assert!(!db_path.exists(), "old path must be renamed away");
        assert!(
            no_tmp_leftovers(tmp.path()),
            "tmp artifacts must not remain"
        );
        // grafeo sidecars and the migration lock must all be gone.
        assert!(!tmp.path().join("kb.grafeo.wal").exists());
        assert!(!tmp.path().join("kb.db.spill").exists());
        assert!(!lock_path(&db_path).exists());

        let store = KbStore::open(&target).expect("open migrated db");
        // 2 chunks + 1 FileHash + 1 IndexConfig + 2 entities + 2 extras
        // = 8 nodes; KNOWS + REL = 2 edges.
        assert_eq!(store.count_nodes().unwrap(), 8);
        assert_eq!(store.count_edges().unwrap(), 2);
        assert_eq!(store.count_label("Chunk").unwrap(), 2);
        assert_eq!(store.count_label("FileHash").unwrap(), 1);

        // Whole-KB content equality: every label and every property —
        // embeddings, all nine hash dims, underscore-prefixed user props —
        // must survive verbatim (ids are allowed to differ).
        let post_nodes = store.dump_all_nodes().unwrap();
        let post_edges = store.dump_all_edges().unwrap();
        assert_eq!(node_keys(&pre_nodes), node_keys(&post_nodes));
        assert_eq!(edge_keys(&pre_edges), edge_keys(&post_edges));

        let cfg = store.read_index_config().unwrap().expect("config present");
        assert_eq!(
            (cfg.model.as_str(), cfg.dim, cfg.max_chunk_chars),
            ("model-x", DIM, 20_000)
        );

        // Same query, same results as before migration: a per-process index
        // rebuild (the normal query path) over the migrated chunks. migrate
        // itself builds no index — grafeo 0.5.x could not reload it anyway.
        store.ensure_chunk_vector_index(DIM).expect("build index");
        let hits = store.search(&[1.0, 0.0, 0.0, 0.0], 2).expect("search");
        let contents: Vec<&str> = hits.iter().map(|h| h.content.as_str()).collect();
        let pre_contents: Vec<&str> = pre_hits.iter().map(|h| h.content.as_str()).collect();
        assert_eq!(contents, pre_contents, "top-N must match pre-migration");
        assert_eq!(contents[0], "alpha");

        let res = store
            .execute_raw("MATCH (:Entity {name:'甲'})-[r:KNOWS]->(b) RETURN r.w, b.name")
            .expect("graph query");
        let rows = res.rows();
        assert_eq!(rows.len(), 1, "graph edge must survive migration");
        assert_eq!(rows[0][0], Value::Int64(3), "edge property w");
        assert_eq!(rows[0][1], Value::from("乙"), "target node name");
    }

    #[test]
    fn migrate_copy_builds_no_vector_index() {
        let tmp = tempdir().unwrap();
        let db_path = tmp.path().join("kb.db");
        build_source_kb(&db_path);

        // Drive the copy layer directly so the destination can be inspected
        // BEFORE close — after a reopen the (buggy) grafeo load path would
        // hide a built index and the assertion would prove nothing.
        let tmp_file = tmp.path().join("out.grafeo");
        let dst = KbStore::open(&tmp_file).expect("open dst");
        copy_records(&db_path, &dst).expect("copy");
        let err = dst
            .search(&[1.0, 0.0, 0.0, 0.0], 1)
            .expect_err("migrate must not build a vector index");
        assert!(err.to_string().contains("No vector index"), "{err}");
        dst.close().unwrap();
    }

    #[test]
    fn migrate_refuses_existing_target() {
        let tmp = tempdir().unwrap();
        let db_path = tmp.path().join("kb.db");
        build_source_kb(&db_path);
        std::fs::write(tmp.path().join("kb.grafeo"), b"occupied").unwrap();

        let err = run(&db_path, None).unwrap_err();
        assert!(err.to_string().contains("already exists"), "{err}");
        assert!(db_path.is_dir(), "source must be untouched");
    }

    #[test]
    fn migrate_refuses_single_file_source() {
        let tmp = tempdir().unwrap();
        let db_path = tmp.path().join("kb.grafeo");
        std::fs::write(&db_path, b"x").unwrap();
        let err = run(&db_path, None).unwrap_err();
        assert!(err.to_string().contains("already a single-file"), "{err}");
    }

    #[test]
    fn migrate_requires_lowercase_grafeo_extension_on_target() {
        let tmp = tempdir().unwrap();
        let db_path = tmp.path().join("kb.db");
        build_source_kb(&db_path);
        for bad in ["kb.out", "kb.GRAFEO", "kb.Grafeo"] {
            let err = run(&db_path, Some(tmp.path().join(bad))).unwrap_err();
            assert!(
                err.to_string().contains(".grafeo extension"),
                "{bad}: {err}"
            );
        }
        assert!(db_path.is_dir(), "source must be untouched");
    }

    #[test]
    fn migrate_refuses_target_inside_source_dir() {
        let tmp = tempdir().unwrap();
        let db_path = tmp.path().join("kb.db");
        build_source_kb(&db_path);
        let err = run(&db_path, Some(db_path.join("out.grafeo"))).unwrap_err();
        assert!(err.to_string().contains("inside the source"), "{err}");
        assert!(db_path.is_dir(), "source must be untouched");
        assert!(!bak_path(&db_path).exists());
    }

    #[test]
    fn migrate_refuses_non_kb_directories() {
        let tmp = tempdir().unwrap();
        // A plain directory with unrelated content must never be renamed away.
        let plain = tmp.path().join("not-a-kb");
        std::fs::create_dir(&plain).unwrap();
        std::fs::write(plain.join("precious.txt"), b"data").unwrap();
        let err = run(&plain, None).unwrap_err();
        assert!(err.to_string().contains("WAL-directory"), "{err}");
        assert!(plain.join("precious.txt").exists(), "contents untouched");
        assert!(!sidecar(&plain, ".bak").exists());

        let empty = tmp.path().join("empty");
        std::fs::create_dir(&empty).unwrap();
        let err = run(&empty, None).unwrap_err();
        assert!(err.to_string().contains("WAL-directory"), "{err}");
        assert!(empty.is_dir());
    }

    #[test]
    fn migrate_refuses_when_lock_held() {
        let tmp = tempdir().unwrap();
        let db_path = tmp.path().join("kb.db");
        build_source_kb(&db_path);
        let lock = lock_path(&db_path);
        std::fs::write(&lock, b"").unwrap();

        let err = run(&db_path, None).unwrap_err();
        assert!(err.to_string().contains("lock"), "{err}");
        assert!(db_path.is_dir(), "source must be untouched");

        std::fs::remove_file(&lock).unwrap();
        run(&db_path, None).expect("migrate once the lock is released");
        assert!(!lock.exists(), "lock must be removed after success");
    }

    #[test]
    fn migrate_graph_only_kb_skips_index() {
        let tmp = tempdir().unwrap();
        let db_path = tmp.path().join("kb.db");
        {
            let store = KbStore::open(&db_path).expect("open source db");
            store
                .execute_raw("INSERT (:Entity {name:'甲'})-[:KNOWS]->(:Entity {name:'乙'})")
                .expect("insert graph");
            store.close().expect("close");
        }

        run(&db_path, None).expect("migrate graph-only KB");

        let store = KbStore::open(&tmp.path().join("kb.grafeo")).expect("open migrated");
        assert_eq!(store.count_nodes().unwrap(), 2);
        assert_eq!(store.count_edges().unwrap(), 1);
        assert_eq!(store.count_label("Chunk").unwrap(), 0);
    }

    /// Regression guard for the acceptance criterion "repeated queries must
    /// not keep growing the container": each query process rebuilds the HNSW
    /// (grafeo 0.5.x cannot reload it) and close() checkpoints the dirty
    /// VectorStore section back into the file. Empirically the checkpoint
    /// rewrites in place — after the first post-migration close the size is
    /// exactly constant, so assert strict equality rather than a tolerance
    /// (a tolerance would let steady 10–20% per-cycle growth pass).
    #[test]
    fn single_file_open_rebuild_close_cycle_size_stable() {
        let tmp = tempdir().unwrap();
        let db_path = tmp.path().join("kb.db");
        build_source_kb(&db_path);
        run(&db_path, None).expect("migrate");
        let target = tmp.path().join("kb.grafeo");
        let sidecar_wal = tmp.path().join("kb.grafeo.wal");

        let mut sizes = Vec::new();
        for _ in 0..QUERY_CYCLES {
            let store = KbStore::open(&target).expect("reopen");
            store.ensure_chunk_vector_index(DIM).expect("rebuild index");
            let hits = store.search(&[1.0, 0.0, 0.0, 0.0], 1).expect("search");
            assert_eq!(hits[0].content, "alpha");
            store.close().expect("close");
            assert!(
                !sidecar_wal.exists(),
                "sidecar WAL must be gone after every clean close"
            );
            sizes.push(std::fs::metadata(&target).unwrap().len());
        }
        // First close may append the VectorStore section; from then on the
        // size must be exactly stable.
        for (i, s) in sizes.iter().enumerate().skip(2) {
            assert_eq!(
                *s, sizes[1],
                "container must plateau after warm-up: sizes={sizes:?} (cycle {i})"
            );
        }
    }

    #[test]
    fn migrate_clears_stale_tmp_artifacts() {
        let tmp = tempdir().unwrap();
        let db_path = tmp.path().join("kb.db");
        build_source_kb(&db_path);
        // Pre-pid layout and another pid's leftovers must both be cleared;
        // a tmp file for a DIFFERENT target must be left alone.
        std::fs::write(tmp.path().join(".migrate-tmp-kb.grafeo"), b"stale").unwrap();
        std::fs::create_dir(tmp.path().join(".migrate-tmp-kb.grafeo.wal")).unwrap();
        std::fs::create_dir(tmp.path().join(".migrate-tmp-kb.grafeo.spill")).unwrap();
        std::fs::write(tmp.path().join(".migrate-tmp-424242-kb.grafeo"), b"stale").unwrap();
        std::fs::create_dir(tmp.path().join(".migrate-tmp-424242-kb.grafeo.spill")).unwrap();
        std::fs::write(tmp.path().join(".migrate-tmp-9-other.grafeo"), b"keep").unwrap();

        run(&db_path, None).expect("migrate despite stale tmp files");
        assert!(tmp.path().join("kb.grafeo").is_file());
        assert!(!tmp.path().join(".migrate-tmp-kb.grafeo").exists());
        assert!(!tmp.path().join(".migrate-tmp-kb.grafeo.wal").exists());
        assert!(!tmp.path().join(".migrate-tmp-kb.grafeo.spill").exists());
        assert!(!tmp.path().join(".migrate-tmp-424242-kb.grafeo").exists());
        assert!(
            !tmp.path()
                .join(".migrate-tmp-424242-kb.grafeo.spill")
                .exists()
        );
        assert!(
            tmp.path().join(".migrate-tmp-9-other.grafeo").exists(),
            "other targets' tmp files are not ours to delete"
        );
    }

    #[test]
    fn migrate_accepts_trailing_separator_on_source() {
        let tmp = tempdir().unwrap();
        let db_path = tmp.path().join("kb.db");
        build_source_kb(&db_path);

        let with_slash = PathBuf::from(format!("{}/", db_path.display()));
        run(&with_slash, None).expect("migrate with trailing separator");

        // Sidecars must land NEXT TO the source, not inside it.
        assert!(tmp.path().join("kb.grafeo").is_file());
        assert!(tmp.path().join("kb.db.bak").is_dir());
        assert!(!tmp.path().join("kb.db.bak").join(".bak").exists());
    }

    #[cfg(unix)]
    #[test]
    fn migrate_refuses_target_behind_symlink_alias_of_source() {
        let tmp = tempdir().unwrap();
        let db_path = tmp.path().join("kb.db");
        build_source_kb(&db_path);
        let alias = tmp.path().join("kb-alias");
        std::os::unix::fs::symlink(&db_path, &alias).unwrap();

        let err = run(&db_path, Some(alias.join("out.grafeo"))).unwrap_err();
        assert!(err.to_string().contains("inside the source"), "{err}");
        assert!(db_path.is_dir(), "source must be untouched");
        assert!(!bak_path(&db_path).exists());
    }

    #[cfg(unix)]
    #[test]
    fn lock_identity_follows_canonical_path() {
        let tmp = tempdir().unwrap();
        let db_path = tmp.path().join("kb.db");
        build_source_kb(&db_path);
        let alias = tmp.path().join("kb-alias");
        std::os::unix::fs::symlink(&db_path, &alias).unwrap();
        assert_eq!(
            lock_path(&db_path),
            lock_path(&alias),
            "symlink aliases of one KB must share one lock"
        );
    }

    #[test]
    fn migrate_refuses_fake_wal_segments() {
        let tmp = tempdir().unwrap();
        // Non-numeric segment names and directory-shaped segments must not
        // make an unrelated directory pass as a KB.
        let fake = tmp.path().join("fake.db");
        std::fs::create_dir_all(fake.join("wal")).unwrap();
        std::fs::write(fake.join("wal").join("wal_backup.log"), b"x").unwrap();
        std::fs::create_dir(fake.join("wal").join("wal_0.log")).unwrap();
        // One past u64::MAX: all digits, but no real grafeo counter.
        std::fs::write(fake.join("wal").join("wal_18446744073709551616.log"), b"x").unwrap();
        std::fs::write(fake.join("wal").join("wal_+7.log"), b"x").unwrap();
        std::fs::write(fake.join("precious.txt"), b"data").unwrap();

        let err = run(&fake, None).unwrap_err();
        assert!(err.to_string().contains("WAL-directory"), "{err}");
        assert!(fake.join("precious.txt").exists(), "contents untouched");
        assert!(!sidecar(&fake, ".bak").exists());
    }

    #[test]
    fn publish_refuses_preexisting_target_and_keeps_its_content() {
        let tmp = tempdir().unwrap();
        let staged = tmp.path().join(".migrate-tmp-1-kb.grafeo");
        let target = tmp.path().join("kb.grafeo");
        std::fs::write(&staged, b"new").unwrap();
        std::fs::write(&target, b"sentinel").unwrap();

        let err = publish_no_clobber(&staged, &target).unwrap_err();
        assert!(err.to_string().contains("publishing"), "{err}");
        assert_eq!(std::fs::read(&target).unwrap(), b"sentinel");
        assert!(staged.exists(), "tmp must survive a refused publish");
    }

    #[test]
    fn tmp_path_embeds_current_pid_and_keeps_target_name() {
        let tmp = tempdir().unwrap();
        let target = tmp.path().join("kb.grafeo");
        let staged = tmp_path(&target).unwrap();
        assert_eq!(staged.parent(), target.parent());
        assert_eq!(
            staged.file_name().unwrap().to_string_lossy(),
            format!(".migrate-tmp-{}-kb.grafeo", std::process::id())
        );
    }
}
