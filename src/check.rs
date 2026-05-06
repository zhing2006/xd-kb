use std::path::Path;

use anyhow::{Context, Result};

use crate::storage::{HashDim, KbStore};

/// Output line returned by [`compute_status`]. Tests assert against this directly
/// to avoid pulling in a stdout-capture dependency. `run` prints `text` verbatim
/// via `println!` and ignores `mutate`; `mutate` carries the storage write the
/// caller must perform after the line is decided.
struct StatusLine {
    text: String,
    mutate: Mutate,
}

enum Mutate {
    None,
    SetHash(String),
    RemoveHash,
}

/// Format a status line. Spec MUST: single-line `[<status>] <path>`, with an
/// optional ` -> [removed]` suffix when `--mark` actually cleaned up storage on
/// a deleted file. MUST NOT contain a `:dim` suffix — dim is selected by the
/// caller via `--type`, so echoing it back is redundant.
fn format_status_line(file: &str, status: &str, removed: bool) -> String {
    if removed {
        format!("[{status}] {file} -> [removed]")
    } else {
        format!("[{status}] {file}")
    }
}

/// Pure decision: given disk state (`exists`, `current_hash`), the prior stored
/// hash for this dim, and `--mark`, produce the line to print and the storage
/// mutation (if any). Extracted from `run` so tests can assert outputs without a
/// real DB / fs round trip on every variant.
fn compute_status(
    file: &str,
    exists: bool,
    current_hash: Option<&str>,
    stored: Option<&str>,
    mark: bool,
) -> StatusLine {
    if exists {
        let hash = current_hash.expect("compute_status: file exists requires a hash");
        match stored {
            None => StatusLine {
                text: format_status_line(file, "new", false),
                mutate: if mark {
                    Mutate::SetHash(hash.to_string())
                } else {
                    Mutate::None
                },
            },
            Some(prev) if prev == hash => StatusLine {
                text: format_status_line(file, "unchanged", false),
                mutate: Mutate::None,
            },
            Some(_) => StatusLine {
                text: format_status_line(file, "modified", false),
                mutate: if mark {
                    Mutate::SetHash(hash.to_string())
                } else {
                    Mutate::None
                },
            },
        }
    } else {
        match stored {
            Some(_) if mark => StatusLine {
                text: format_status_line(file, "deleted", true),
                mutate: Mutate::RemoveHash,
            },
            _ => StatusLine {
                text: format_status_line(file, "deleted", false),
                mutate: Mutate::None,
            },
        }
    }
}

pub fn run(db: &Path, file: &Path, dim: HashDim, mark: bool) -> Result<()> {
    let store = KbStore::open(db)?;
    let file_str = file.to_string_lossy().into_owned();
    let stored = store.get_file_hash(&file_str, dim)?;
    let exists = file.exists();

    let current_hash = if exists {
        let bytes = std::fs::read(file)
            .with_context(|| format!("read file for hashing: {}", file.display()))?;
        Some(blake3::hash(&bytes).to_hex().to_string())
    } else {
        None
    };

    let StatusLine { text, mutate } = compute_status(
        &file_str,
        exists,
        current_hash.as_deref(),
        stored.as_deref(),
        mark,
    );

    // Mutate first, print second. The status line claims a side effect (e.g.
    // ` -> [removed]` when --mark cleaned a deleted file's hash); printing it
    // before the storage call would make stdout lie if the storage write fails.
    match mutate {
        Mutate::None => {}
        Mutate::SetHash(h) => store.set_file_hash(&file_str, dim, &h)?,
        Mutate::RemoveHash => store.remove_file_hash(&file_str, dim)?,
    }
    println!("{text}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write_file(path: &Path, contents: &[u8]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    fn db_path(dir: &Path) -> std::path::PathBuf {
        dir.join("kb.db")
    }

    #[test]
    fn new_file_status_is_new() {
        let tmp = tempdir().unwrap();
        let db = db_path(tmp.path());
        let file = tmp.path().join("a.md");
        write_file(&file, b"hello");

        run(&db, &file, HashDim::Vector, false).expect("check new file");

        // mark=false should not write a hash
        let store = KbStore::open(&db).unwrap();
        assert!(
            store
                .get_file_hash(&file.to_string_lossy(), HashDim::Vector)
                .unwrap()
                .is_none(),
            "mark=false MUST NOT persist a hash"
        );
    }

    #[test]
    fn modified_file_status_is_modified() {
        let tmp = tempdir().unwrap();
        let db = db_path(tmp.path());
        let file = tmp.path().join("a.md");
        write_file(&file, b"v1");

        // seed: mark v1
        run(&db, &file, HashDim::Vector, true).expect("seed mark");

        // mutate
        write_file(&file, b"v2-different");
        run(&db, &file, HashDim::Vector, false).expect("check after modify");

        // hash in DB should still equal v1's hash because mark=false
        let store = KbStore::open(&db).unwrap();
        let stored = store
            .get_file_hash(&file.to_string_lossy(), HashDim::Vector)
            .unwrap()
            .expect("hash present");
        let v1_hash = blake3::hash(b"v1").to_hex().to_string();
        assert_eq!(
            stored, v1_hash,
            "modified check without mark MUST NOT update hash"
        );
    }

    #[test]
    fn unchanged_file_status_is_unchanged_and_preserves_node() {
        let tmp = tempdir().unwrap();
        let db = db_path(tmp.path());
        let file = tmp.path().join("a.md");
        write_file(&file, b"stable");

        // seed
        run(&db, &file, HashDim::Vector, true).expect("seed mark");
        let before = {
            let store = KbStore::open(&db).unwrap();
            store
                .get_file_hash(&file.to_string_lossy(), HashDim::Vector)
                .unwrap()
                .expect("hash present after seed")
        };

        // call check with mark=true on unchanged file: must not alter the node
        run(&db, &file, HashDim::Vector, true).expect("check unchanged with mark");
        let after = {
            let store = KbStore::open(&db).unwrap();
            store
                .get_file_hash(&file.to_string_lossy(), HashDim::Vector)
                .unwrap()
                .expect("hash still present")
        };
        assert_eq!(
            before, after,
            "unchanged file MUST NOT modify the FileHash node"
        );
    }

    #[test]
    fn deleted_file_without_mark_keeps_node() {
        let tmp = tempdir().unwrap();
        let db = db_path(tmp.path());
        let file = tmp.path().join("a.md");
        write_file(&file, b"to-be-deleted");

        run(&db, &file, HashDim::Vector, true).expect("seed mark");
        fs::remove_file(&file).unwrap();

        run(&db, &file, HashDim::Vector, false).expect("check deleted without mark");

        let store = KbStore::open(&db).unwrap();
        assert!(
            store
                .get_file_hash(&file.to_string_lossy(), HashDim::Vector)
                .unwrap()
                .is_some(),
            "FileHash node MUST remain when deleted is reported without --mark"
        );
    }

    #[test]
    fn mark_writes_to_grafeo_not_json_sidecar() {
        let tmp = tempdir().unwrap();
        let db = db_path(tmp.path());
        let file = tmp.path().join("a.md");
        write_file(&file, b"payload");

        run(&db, &file, HashDim::Vector, true).expect("mark new file");

        // sanity: hash actually written to grafeo
        {
            let store = KbStore::open(&db).unwrap();
            assert!(
                store
                    .get_file_hash(&file.to_string_lossy(), HashDim::Vector)
                    .unwrap()
                    .is_some(),
                "mark MUST persist hash via grafeo"
            );
        }

        // walk DB-related entries: no .json sidecar (e.g. check_hashes.json) anywhere
        fn walk_no_json(p: &Path) {
            if p.is_dir() {
                for entry in fs::read_dir(p).unwrap() {
                    let e = entry.unwrap();
                    walk_no_json(&e.path());
                }
            } else {
                let name = p.file_name().unwrap().to_string_lossy().to_lowercase();
                assert!(
                    !name.ends_with(".json"),
                    "check MUST NOT create JSON sidecar files; found {}",
                    p.display()
                );
            }
        }
        walk_no_json(tmp.path());
    }

    #[test]
    fn mark_on_deleted_file_removes_node() {
        let tmp = tempdir().unwrap();
        let db = db_path(tmp.path());
        let file = tmp.path().join("a.md");
        write_file(&file, b"will-vanish");

        run(&db, &file, HashDim::Vector, true).expect("seed mark");
        fs::remove_file(&file).unwrap();
        run(&db, &file, HashDim::Vector, true).expect("check deleted with mark");

        let store = KbStore::open(&db).unwrap();
        assert!(
            store
                .get_file_hash(&file.to_string_lossy(), HashDim::Vector)
                .unwrap()
                .is_none(),
            "--mark on a deleted file MUST remove the FileHash node"
        );
    }

    // ---- Multi-dim semantics (this change) ----

    #[test]
    fn mark_one_dim_does_not_touch_other_dims() {
        let tmp = tempdir().unwrap();
        let db = db_path(tmp.path());
        let file = tmp.path().join("multi.md");
        write_file(&file, b"shared body");
        let file_str = file.to_string_lossy().into_owned();

        // mark vector then graph for the same file
        run(&db, &file, HashDim::Vector, true).expect("mark vector");
        run(&db, &file, HashDim::Graph, true).expect("mark graph");

        let store = KbStore::open(&db).unwrap();
        let v = store
            .get_file_hash(&file_str, HashDim::Vector)
            .unwrap()
            .expect("vector_hash present");
        let g = store
            .get_file_hash(&file_str, HashDim::Graph)
            .unwrap()
            .expect("graph_hash present");
        let expected = blake3::hash(b"shared body").to_hex().to_string();
        assert_eq!(v, expected, "vector_hash equals current content hash");
        assert_eq!(g, expected, "graph_hash equals current content hash");
        assert_eq!(
            v, g,
            "both dims hold the same hash because the file content is identical"
        );

        // No collateral damage on other 7 dims.
        for other in [
            HashDim::Llm,
            HashDim::Fulltext,
            HashDim::Summary,
            HashDim::Entity,
            HashDim::Vector2,
            HashDim::Custom,
            HashDim::Custom2,
        ] {
            assert!(
                store.get_file_hash(&file_str, other).unwrap().is_none(),
                "dim {other:?} must not have been touched"
            );
        }
    }

    #[test]
    fn delete_with_other_dim_present_keeps_node_only_removes_property() {
        let tmp = tempdir().unwrap();
        let db = db_path(tmp.path());
        let file = tmp.path().join("two.md");
        write_file(&file, b"both dims");
        let file_str = file.to_string_lossy().into_owned();

        run(&db, &file, HashDim::Vector, true).expect("mark vector");
        run(&db, &file, HashDim::Graph, true).expect("mark graph");
        let graph_before = {
            let store = KbStore::open(&db).unwrap();
            store
                .get_file_hash(&file_str, HashDim::Graph)
                .unwrap()
                .expect("graph_hash seeded")
        };

        fs::remove_file(&file).unwrap();
        run(&db, &file, HashDim::Vector, true).expect("mark deleted vector");

        let store = KbStore::open(&db).unwrap();
        assert!(
            store
                .get_file_hash(&file_str, HashDim::Vector)
                .unwrap()
                .is_none(),
            "vector_hash MUST be cleaned"
        );
        assert_eq!(
            store
                .get_file_hash(&file_str, HashDim::Graph)
                .unwrap()
                .as_deref(),
            Some(graph_before.as_str()),
            "graph_hash MUST be untouched (node preserved, only property removed)"
        );

        // :FileHash node must still exist (count == 1).
        let qres = store
            .execute_raw(&format!(
                "MATCH (h:FileHash {{path: '{}'}}) RETURN count(h)",
                file_str.replace('\\', "\\\\").replace('\'', "\\'")
            ))
            .unwrap();
        let n = match &qres.rows()[0][0] {
            grafeo::Value::Int64(n) => *n,
            other => panic!("unexpected count value: {other:?}"),
        };
        assert_eq!(
            n, 1,
            ":FileHash node MUST remain when other dim hash is still present"
        );
    }

    #[test]
    fn delete_when_target_dim_is_last_removes_whole_node() {
        let tmp = tempdir().unwrap();
        let db = db_path(tmp.path());
        let file = tmp.path().join("solo.md");
        write_file(&file, b"only vector");
        let file_str = file.to_string_lossy().into_owned();

        run(&db, &file, HashDim::Vector, true).expect("mark vector");
        fs::remove_file(&file).unwrap();
        run(&db, &file, HashDim::Vector, true).expect("mark deleted vector");

        let store = KbStore::open(&db).unwrap();
        let qres = store
            .execute_raw(&format!(
                "MATCH (h:FileHash {{path: '{}'}}) RETURN count(h)",
                file_str.replace('\\', "\\\\").replace('\'', "\\'")
            ))
            .unwrap();
        let n = match &qres.rows()[0][0] {
            grafeo::Value::Int64(n) => *n,
            other => panic!("unexpected count value: {other:?}"),
        };
        assert_eq!(
            n, 0,
            "with no other dim hash, the whole :FileHash node MUST be DETACH DELETEd"
        );
    }

    #[test]
    fn mark_on_deleted_file_with_no_dim_hash_is_noop() {
        let tmp = tempdir().unwrap();
        let db = db_path(tmp.path());
        let file = tmp.path().join("never-marked.md");
        // Never write the file to disk — mimics a path the user never indexed before.
        let file_str = file.to_string_lossy().into_owned();

        // File doesn't exist on disk and FileHash node never written: --mark must
        // print [deleted] without ` -> [removed]` and must not create any node.
        run(&db, &file, HashDim::Vector, true).expect("noop on unknown deleted file");

        let store = KbStore::open(&db).unwrap();
        assert!(
            store
                .get_file_hash(&file_str, HashDim::Vector)
                .unwrap()
                .is_none(),
            "no node should have been created"
        );

        // Verify there are no :FileHash nodes at all.
        let qres = store
            .execute_raw("MATCH (h:FileHash) RETURN count(h)")
            .unwrap();
        let n = match &qres.rows()[0][0] {
            grafeo::Value::Int64(n) => *n,
            other => panic!("unexpected count value: {other:?}"),
        };
        assert_eq!(n, 0, "no :FileHash node may exist after noop --mark");

        // Independently verify the line shape via the helper.
        let line = compute_status(&file_str, false, None, None, true);
        assert_eq!(
            line.text,
            format!("[deleted] {file_str}"),
            "noop --mark on never-seen file must NOT append ` -> [removed]`"
        );
    }

    #[test]
    fn deleted_mark_with_existing_hash_outputs_removed_suffix() {
        // Direct unit test of the helper: when --mark is set on a deleted file
        // whose dim hash exists in storage, the status line MUST append
        // ` -> [removed]` AND the request to remove storage MUST be emitted.
        let line = compute_status("deleted.md", false, None, Some("old-hash"), true);

        assert_eq!(line.text, "[deleted] deleted.md -> [removed]");
        assert!(
            matches!(line.mutate, Mutate::RemoveHash),
            "deleted mark with existing dim hash must request storage cleanup"
        );
    }

    #[test]
    fn output_format_has_no_dim_suffix() {
        let tmp = tempdir().unwrap();
        let file = tmp.path().join("fmt.md");
        let file_str = file.to_string_lossy().into_owned();
        let some_hash = "abc123".to_string();

        // Exercise the helper for each (dim, status) combo that could leak a suffix.
        // The helper output is dim-agnostic by construction, so this also serves as
        // a regression lock against any future "let's add `:dim` for clarity" PR.
        let cases = [
            // (status, exists, stored, mark) -> compute_status uses some_hash as the
            // current content hash whenever exists=true.
            ("new", true, None::<&str>, false),
            ("unchanged", true, Some(some_hash.as_str()), false),
            ("modified", true, Some("different-hash"), false),
            ("deleted", false, Some(some_hash.as_str()), false),
        ];
        for (status, exists, stored, mark) in cases {
            let line = compute_status(
                &file_str,
                exists,
                if exists {
                    Some(some_hash.as_str())
                } else {
                    None
                },
                stored,
                mark,
            );
            let expected = format!("[{status}] {file_str}");
            assert_eq!(line.text, expected, "({status}) line must equal {expected}");

            // MUST NOT contain a dim-name suffix — cover both kebab CLI form and
            // snake property form so a future change leaking either gets caught.
            // We deliberately do NOT forbid the bare `:` character, because Windows
            // file paths contain drive letters like `H:`.
            for forbidden in [":graph", ":vector-2", ":vector_2", ":custom-2"] {
                assert!(
                    !line.text.contains(forbidden),
                    "({status}) line `{}` MUST NOT contain dim suffix `{forbidden}`",
                    line.text
                );
            }
        }
    }
}
