use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::config::find_xd_kb_yaml_root;
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

/// Resolution of a `--root` decision. Carries the explicit/implicit distinction
/// so `validate_and_normalize` can use the correct coordinate system for `<file>`:
///
/// - `Explicit`: user passed `--root <dir>`; `<file>` is **root-relative**.
/// - `Implicit`: discovered via upward `.xd-kb.yaml` walk from `cwd`; `<file>`
///   is **cwd-relative** (per spec scenario at xd-kb-cli/spec.md:225-226).
#[derive(Debug)]
enum RootResolution {
    Explicit { root: PathBuf },
    Implicit { root: PathBuf, cwd: PathBuf },
}

impl RootResolution {
    fn root(&self) -> &Path {
        match self {
            Self::Explicit { root } | Self::Implicit { root, .. } => root,
        }
    }
}

/// Resolve the effective KB root.
///
/// - explicit `--root` always wins (even if the dir doesn't carry `.xd-kb.yaml`);
/// - otherwise walk up from `cwd` looking for `.xd-kb.yaml`;
/// - otherwise return `None` — the caller falls back to the legacy verbatim
///   path behaviour for backward compatibility with existing CI scripts.
///
/// `cwd` is taken as an argument (rather than read from `std::env::current_dir`
/// internally) so unit tests can exercise the implicit-detect branch under a
/// controlled directory tree without leaking into the test process's actual
/// cwd. `run` passes `std::env::current_dir()?` from production.
fn resolve_root(explicit: Option<&Path>, cwd: &Path) -> Option<RootResolution> {
    if let Some(p) = explicit {
        return Some(RootResolution::Explicit {
            root: p.to_path_buf(),
        });
    }
    find_xd_kb_yaml_root(cwd).map(|root| RootResolution::Implicit {
        root,
        cwd: cwd.to_path_buf(),
    })
}

/// Validate `<file>` against the resolved root and produce the
/// `(disk_path, store_path)` pair.
///
/// Coordinate system depends on the resolution mode:
/// - `Explicit { root }`: `<file>` is interpreted **relative to root**.
///   `disk = root.join(file)`, store key = `file`.
/// - `Implicit { root, cwd }`: `<file>` is interpreted **relative to cwd**
///   (per spec scenario for `.xd-kb.yaml` discovered from a sub-directory).
///   `disk = cwd.join(file)`, store key = `cwd.strip_prefix(root).join(file)`.
///
/// Both modes share the same safety checks:
/// - reject absolute `<file>`;
/// - reject `..` escape (string-level fast check on the relative store path,
///   plus symlink-aware canonicalize check when the file exists);
/// - normalize `\` → `/` in the store key so DB keys are identical across
///   Windows and POSIX KBs.
fn validate_and_normalize(
    resolution: &RootResolution,
    file: &Path,
) -> Result<(PathBuf, String)> {
    if file.is_absolute() {
        bail!(
            "<file> {} must be a relative path when --root or implicit .xd-kb.yaml is in effect",
            file.display()
        );
    }

    // Up-front separator normalisation: on POSIX, `PathBuf::from("故事\\X.md")`
    // is one literal segment ("故事\X.md") and disk lookup would fail. We
    // canonicalise `\` → `/` so the same input produces the same multi-segment
    // path on every OS.
    let file_rel = PathBuf::from(file.to_string_lossy().replace('\\', "/"));

    let (disk_path, store_rel_raw) = match resolution {
        RootResolution::Explicit { root } => (root.join(&file_rel), file_rel),
        RootResolution::Implicit { root, cwd } => {
            let cwd_rel = cwd.strip_prefix(root).with_context(|| {
                format!(
                    "cwd {} is not under implicit root {}",
                    cwd.display(),
                    root.display()
                )
            })?;
            (cwd.join(&file_rel), cwd_rel.join(&file_rel))
        }
    };

    // Pure-string `.`-drop / `..`-pop normaliser. Returns `None` on escape
    // attempts (used as the safety check for both existing and deleted files);
    // the *normalised* result becomes the store key so `index` and `check`
    // produce identical `:Chunk.file_path` / `:FileHash.path` for the same file.
    let Some(store_rel) = string_normalise_under_root(&store_rel_raw) else {
        bail!(
            "<file> {} canonicalised escapes --root via .. segments",
            file.display()
        );
    };

    // Symlink-aware escape detection when the file actually exists.
    let root = resolution.root();
    let canon_root = root
        .canonicalize()
        .with_context(|| format!("canonicalize --root {}", root.display()))?;
    if disk_path.exists() {
        let canon_full = disk_path
            .canonicalize()
            .with_context(|| format!("canonicalize <file> {}", disk_path.display()))?;
        if !canon_full.starts_with(&canon_root) {
            bail!(
                "<file> {} canonicalised to {} which escapes --root {}",
                file.display(),
                canon_full.display(),
                canon_root.display()
            );
        }
    }

    // Forward-slash normalise the (already `.`-stripped, `..`-popped) store key
    // so DB keys are identical across Windows and POSIX KBs.
    let store_path = store_rel.to_string_lossy().replace('\\', "/");
    Ok((disk_path, store_path))
}

/// Pure-string `..`-pop normaliser used in the deletion scenario where
/// `canonicalize` would fail. Returns `None` if any prefix of the path pops
/// above its starting directory (i.e. the relative path escapes the root).
fn string_normalise_under_root(rel: &Path) -> Option<PathBuf> {
    use std::path::Component;
    let mut stack: Vec<std::ffi::OsString> = Vec::new();
    for comp in rel.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                if stack.pop().is_none() {
                    return None;
                }
            }
            Component::Normal(seg) => stack.push(seg.to_os_string()),
            Component::RootDir | Component::Prefix(_) => {
                // Already filtered out by `is_absolute()`, but be defensive.
                return None;
            }
        }
    }
    let mut out = PathBuf::new();
    for seg in stack {
        out.push(seg);
    }
    Some(out)
}

pub fn run(
    db: &Path,
    file: &Path,
    dim: HashDim,
    mark: bool,
    explicit_root: Option<&Path>,
) -> Result<()> {
    let store = KbStore::open(db)?;

    // Resolve disk_path / store_path based on whether a root is in effect.
    // `display_path` is what we print in the status line (always echoes the
    // user-facing form, so output stays predictable for shell pipelines).
    let cwd = std::env::current_dir().context("read cwd to resolve implicit --root")?;
    let (disk_path, store_path) = match resolve_root(explicit_root, &cwd) {
        Some(resolution) => validate_and_normalize(&resolution, file)?,
        None => {
            // Legacy verbatim behaviour: read file as given, store its
            // to_string_lossy() form. Required by the spec's
            // backward-compatibility scenario for users without `.xd-kb.yaml`.
            let raw = file.to_string_lossy().into_owned();
            (file.to_path_buf(), raw)
        }
    };
    let display_path = file.to_string_lossy().into_owned();

    let stored = store.get_file_hash(&store_path, dim)?;
    let exists = disk_path.exists();

    let current_hash = if exists {
        let bytes = std::fs::read(&disk_path)
            .with_context(|| format!("read file for hashing: {}", disk_path.display()))?;
        Some(blake3::hash(&bytes).to_hex().to_string())
    } else {
        None
    };

    let StatusLine { text, mutate } = compute_status(
        &display_path,
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
        Mutate::SetHash(h) => store.set_file_hash(&store_path, dim, &h)?,
        Mutate::RemoveHash => store.remove_file_hash(&store_path, dim)?,
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

        run(&db, &file, HashDim::Vector, false, None).expect("check new file");

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
        run(&db, &file, HashDim::Vector, true, None).expect("seed mark");

        // mutate
        write_file(&file, b"v2-different");
        run(&db, &file, HashDim::Vector, false, None).expect("check after modify");

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
        run(&db, &file, HashDim::Vector, true, None).expect("seed mark");
        let before = {
            let store = KbStore::open(&db).unwrap();
            store
                .get_file_hash(&file.to_string_lossy(), HashDim::Vector)
                .unwrap()
                .expect("hash present after seed")
        };

        // call check with mark=true on unchanged file: must not alter the node
        run(&db, &file, HashDim::Vector, true, None).expect("check unchanged with mark");
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

        run(&db, &file, HashDim::Vector, true, None).expect("seed mark");
        fs::remove_file(&file).unwrap();

        run(&db, &file, HashDim::Vector, false, None).expect("check deleted without mark");

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

        run(&db, &file, HashDim::Vector, true, None).expect("mark new file");

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

        run(&db, &file, HashDim::Vector, true, None).expect("seed mark");
        fs::remove_file(&file).unwrap();
        run(&db, &file, HashDim::Vector, true, None).expect("check deleted with mark");

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
        run(&db, &file, HashDim::Vector, true, None).expect("mark vector");
        run(&db, &file, HashDim::Graph, true, None).expect("mark graph");

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

        run(&db, &file, HashDim::Vector, true, None).expect("mark vector");
        run(&db, &file, HashDim::Graph, true, None).expect("mark graph");
        let graph_before = {
            let store = KbStore::open(&db).unwrap();
            store
                .get_file_hash(&file_str, HashDim::Graph)
                .unwrap()
                .expect("graph_hash seeded")
        };

        fs::remove_file(&file).unwrap();
        run(&db, &file, HashDim::Vector, true, None).expect("mark deleted vector");

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

        run(&db, &file, HashDim::Vector, true, None).expect("mark vector");
        fs::remove_file(&file).unwrap();
        run(&db, &file, HashDim::Vector, true, None).expect("mark deleted vector");

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
        run(&db, &file, HashDim::Vector, true, None).expect("noop on unknown deleted file");

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

    // ---- Section 3.5: --root + implicit detection + path normalization ----
    //
    // BDD-style coverage of the spec scenarios for the `--root` flag.

    /// Spec scenario "--root 显式指定时拼盘读、规范化路径写入":
    /// disk read joins `<root>/<file>`; storage writes `<file>` (relative,
    /// `/` separators); the same path key reaches the FileHash node that
    /// `index <root>` would write for the same file.
    #[test]
    fn root_explicit_disk_read_and_normalised_store_write() {
        let tmp = tempdir().unwrap();
        let db = db_path(tmp.path());
        let root = tmp.path().join("Wiki/example-kb");
        write_file(&root.join("故事/X.md"), b"hello-from-explicit-root");

        // <file> argument is relative to <root>.
        let file_arg = std::path::PathBuf::from("故事/X.md");
        run(&db, &file_arg, HashDim::Graph, true, Some(&root))
            .expect("explicit root mark");

        // FileHash MUST be reachable via the relative key with `/` separators.
        let store = KbStore::open(&db).unwrap();
        assert!(
            store
                .get_file_hash("故事/X.md", HashDim::Graph)
                .unwrap()
                .is_some(),
            "store key must be the forward-slash relative form"
        );
        // And MUST NOT be reachable via the joined absolute form.
        let joined = root.join("故事/X.md").to_string_lossy().into_owned();
        assert!(
            store
                .get_file_hash(&joined, HashDim::Graph)
                .unwrap()
                .is_none(),
            "absolute disk path must NOT have leaked into the store key"
        );
    }

    /// Spec scenario "--root 隐式从 .xd-kb.yaml 探测":
    /// no `--root` argument; `resolve_root` walks up from cwd and finds
    /// `.xd-kb.yaml` in an ancestor directory, returning that ancestor as
    /// the implicit root (and remembering cwd for cwd-relative `<file>`).
    #[test]
    fn root_implicit_detected_from_xd_kb_yaml_in_ancestor() {
        let tmp = tempdir().unwrap();
        let kb_root = tmp.path().join("Wiki/example-kb");
        let cwd = kb_root.join("故事");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::write(kb_root.join(".xd-kb.yaml"), "include:\n  - \"**/*.md\"\n").unwrap();

        let resolution = resolve_root(None, &cwd).expect("ancestor with .xd-kb.yaml");
        match &resolution {
            RootResolution::Implicit { root, cwd: r_cwd } => {
                assert_eq!(root, &kb_root, "root MUST be the dir containing .xd-kb.yaml");
                assert_eq!(r_cwd, &cwd, "implicit MUST remember cwd for cwd-relative <file>");
            }
            other => panic!("expected Implicit, got {other:?}"),
        }
    }

    /// Spec scenario (xd-kb-cli/spec.md:225-226): cwd is inside an
    /// ancestor `.xd-kb.yaml`'d KB; `check X.md` MUST read `cwd/X.md` and
    /// store `<cwd-rel-to-root>/X.md` — NOT `root/X.md` and `X.md`.
    #[test]
    fn root_implicit_resolves_file_relative_to_cwd_not_root() {
        let tmp = tempdir().unwrap();
        let kb_root = tmp.path().join("Wiki/example-kb");
        let cwd = kb_root.join("故事");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::write(kb_root.join(".xd-kb.yaml"), "include:\n  - \"**/*.md\"\n").unwrap();
        write_file(&cwd.join("X.md"), b"implicit-body");

        // Drive `validate_and_normalize` directly with a synthetic Implicit
        // resolution — avoids mutating the test process's actual cwd while
        // still exercising the production code path used by `run`.
        let resolution = RootResolution::Implicit {
            root: kb_root.clone(),
            cwd: cwd.clone(),
        };
        let (disk_path, store_path) =
            validate_and_normalize(&resolution, Path::new("X.md")).expect("validate implicit");

        assert_eq!(
            disk_path,
            cwd.join("X.md"),
            "disk read MUST resolve <file> relative to cwd, not root"
        );
        assert_eq!(
            store_path, "故事/X.md",
            "store key MUST be cwd.strip_prefix(root).join(file), forward-slash form"
        );
    }

    /// Implicit root with cwd == root: `cwd.strip_prefix(root) == ""` so the
    /// store key collapses to just `<file>`. Covers the "user runs from KB
    /// root with .xd-kb.yaml right there" path.
    #[test]
    fn root_implicit_cwd_equals_root_yields_bare_file_store_key() {
        let tmp = tempdir().unwrap();
        let kb_root = tmp.path().join("Wiki/example-kb");
        std::fs::create_dir_all(&kb_root).unwrap();
        std::fs::write(kb_root.join(".xd-kb.yaml"), "include:\n  - \"**/*.md\"\n").unwrap();
        write_file(&kb_root.join("README.md"), b"root-body");

        let resolution = RootResolution::Implicit {
            root: kb_root.clone(),
            cwd: kb_root.clone(),
        };
        let (disk_path, store_path) =
            validate_and_normalize(&resolution, Path::new("README.md")).expect("validate at root");

        assert_eq!(disk_path, kb_root.join("README.md"));
        assert_eq!(
            store_path, "README.md",
            "cwd == root → store key is just the file name"
        );
    }

    /// Spec scenario "未找到 .xd-kb.yaml 时维持向后兼容":
    /// no `--root`, no ancestor `.xd-kb.yaml` → legacy verbatim path
    /// behaviour: read file as given, store the user's original string form.
    #[test]
    fn root_not_found_falls_back_to_verbatim_path() {
        let tmp = tempdir().unwrap();
        let db = db_path(tmp.path());
        // The file lives at an absolute path; no `.xd-kb.yaml` anywhere relevant.
        let file = tmp.path().join("docs/x.md");
        write_file(&file, b"backward-compat");

        // Direct call of `run` with explicit_root=None.
        // The test process cwd is somewhere under the workspace; the workspace
        // tree carries no `.xd-kb.yaml`, so resolve_root returns None and run
        // takes the legacy branch.
        run(&db, &file, HashDim::Vector, true, None)
            .expect("fallback verbatim mark");

        // Store key is the file's verbatim to_string_lossy() form (absolute).
        let store = KbStore::open(&db).unwrap();
        let verbatim = file.to_string_lossy().into_owned();
        assert!(
            store
                .get_file_hash(&verbatim, HashDim::Vector)
                .unwrap()
                .is_some(),
            "legacy verbatim key MUST be the stored path"
        );
    }

    /// Spec scenario "--root 模式下绝对路径被拒绝":
    /// passing an absolute `<file>` together with `--root` is a hard error.
    #[test]
    fn root_explicit_rejects_absolute_file() {
        let tmp = tempdir().unwrap();
        let db = db_path(tmp.path());
        let root = tmp.path().join("kb");
        std::fs::create_dir_all(&root).unwrap();

        // Use a clearly absolute path; on Windows tmp.path() is already absolute.
        let absolute = tmp.path().join("anywhere.md");
        let res = run(&db, &absolute, HashDim::Vector, false, Some(&root));
        let err = res.expect_err("absolute file under --root MUST error");
        let msg = format!("{err}");
        assert!(
            msg.to_lowercase().contains("relative") || msg.to_lowercase().contains("absolute"),
            "error must mention path-relativity, got: {msg}"
        );
    }

    /// Spec scenario "--root 模式下 .. 逃逸被拒绝":
    /// `<file>` containing `..` segments that would escape `<root>` is rejected
    /// (verified via the helper for a deleted-file path; the helper covers both
    /// canonicalize and string-fallback branches).
    #[test]
    fn root_explicit_rejects_dotdot_escape() {
        let tmp = tempdir().unwrap();
        let root = tmp.path().join("kb");
        std::fs::create_dir_all(&root).unwrap();

        // Deleted-file scenario: file does not exist on disk, so the helper
        // takes the string-normalise fallback. `故事/../../etc/passwd` pops
        // above the root.
        let escaping = std::path::PathBuf::from("故事/../../etc/passwd");
        let resolution = RootResolution::Explicit { root: root.clone() };
        let res = validate_and_normalize(&resolution, &escaping);
        let err = res.expect_err(".. escape MUST error");
        let msg = format!("{err}");
        assert!(
            msg.to_lowercase().contains("escape"),
            "error must mention escape, got: {msg}"
        );
    }

    /// Round 2 follow-up: `./` segments in the user-supplied `<file>` MUST be
    /// dropped before being written to `:FileHash.path`, otherwise the key
    /// `./故事/X.md` would not match `index` writing `故事/X.md` for the same
    /// file. Verified directly via the helper since the bug is in store-key
    /// derivation, not disk read.
    #[test]
    fn root_explicit_drops_dot_segments_in_store_key() {
        let tmp = tempdir().unwrap();
        let root = tmp.path().join("Wiki/example-kb");
        std::fs::create_dir_all(&root).unwrap();
        write_file(&root.join("故事/X.md"), b"dot-test");

        let resolution = RootResolution::Explicit { root: root.clone() };
        let (disk_path, store_path) =
            validate_and_normalize(&resolution, Path::new("./故事/X.md"))
                .expect("validate ./");

        // disk lookup must hit the same file as without the `./`.
        assert_eq!(disk_path, root.join("./故事/X.md"));
        // BUT the store key must NOT carry the `./` (otherwise check & index
        // produce divergent `:FileHash.path` for the same file).
        assert_eq!(store_path, "故事/X.md");
    }

    /// Round 2 follow-up: a non-escaping `..` segment in `<file>` MUST be
    /// popped before being written to the store key, e.g. `故事/../X.md`
    /// resolves to `X.md` (the file in the root) rather than literally
    /// `故事/../X.md`.
    #[test]
    fn root_explicit_pops_non_escaping_dotdot_in_store_key() {
        let tmp = tempdir().unwrap();
        let root = tmp.path().join("Wiki/example-kb");
        std::fs::create_dir_all(&root).unwrap();
        write_file(&root.join("X.md"), b"dotdot-test");

        let resolution = RootResolution::Explicit { root: root.clone() };
        let (_disk, store_path) =
            validate_and_normalize(&resolution, Path::new("故事/../X.md"))
                .expect("validate non-escaping ..");

        assert_eq!(
            store_path, "X.md",
            "non-escaping `..` MUST be popped before becoming the store key"
        );
    }

    /// Round 2 follow-up: implicit-root `<file> = ./X.md` MUST normalise to
    /// `故事/X.md` (cwd-relative-to-root, with `./` dropped). Mirrors the
    /// `root_implicit_resolves_file_relative_to_cwd_not_root` test but with
    /// a leading `./` that the normaliser must collapse.
    #[test]
    fn root_implicit_drops_dot_segments_in_store_key() {
        let tmp = tempdir().unwrap();
        let kb_root = tmp.path().join("Wiki/example-kb");
        let cwd = kb_root.join("故事");
        std::fs::create_dir_all(&cwd).unwrap();
        write_file(&cwd.join("X.md"), b"implicit-dot-test");

        let resolution = RootResolution::Implicit {
            root: kb_root.clone(),
            cwd: cwd.clone(),
        };
        let (_disk, store_path) =
            validate_and_normalize(&resolution, Path::new("./X.md")).expect("validate implicit ./");

        assert_eq!(store_path, "故事/X.md");
    }

    /// Spec scenario "--root 模式下混合路径分隔符规范化":
    /// `<file>` with `\` or mixed `\` / `/` separators is normalised to `/`
    /// before being written to `:FileHash.path`.
    #[test]
    fn root_explicit_normalises_path_separators_to_forward_slash() {
        let tmp = tempdir().unwrap();
        let db = db_path(tmp.path());
        let root = tmp.path().join("Wiki/example-kb");
        write_file(&root.join("故事/X.md"), b"separator-test");

        // Backslash separator on input — common when scripts copy in Windows
        // paths verbatim.
        let file_arg = std::path::PathBuf::from(r"故事\X.md");
        run(&db, &file_arg, HashDim::Vector, true, Some(&root))
            .expect("backslash file mark");

        let store = KbStore::open(&db).unwrap();
        assert!(
            store
                .get_file_hash("故事/X.md", HashDim::Vector)
                .unwrap()
                .is_some(),
            "store key MUST be the forward-slash form"
        );
        assert!(
            store
                .get_file_hash(r"故事\X.md", HashDim::Vector)
                .unwrap()
                .is_none(),
            "store key MUST NOT preserve the backslash form"
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
