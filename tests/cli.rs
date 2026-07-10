//! CLI-level integration tests: drive the real binary in a separate process,
//! so env-based migration failpoints (debug builds only) and the stderr
//! migration-hint contract can be asserted without cross-test env races.

use std::path::Path;
use std::process::{Command, Output};

use tempfile::tempdir;

fn xd_kb(db: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_xd-kb"));
    cmd.arg("--db").arg(db);
    cmd
}

fn run_gql(db: &Path, statement: &str) -> Output {
    xd_kb(db)
        .args(["gql", statement])
        .output()
        .expect("spawn xd-kb")
}

/// Create a legacy WAL-directory KB via the CLI itself (a directory path
/// without a `.grafeo` extension makes grafeo pick the WAL-directory format).
fn build_legacy_kb(db: &Path) {
    let out = run_gql(db, "INSERT (:Entity {name:'a'})");
    assert!(
        out.status.success(),
        "seeding legacy KB failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(db.is_dir(), "legacy KB must be a directory");
}

fn stderr_hint_lines(out: &Output) -> usize {
    String::from_utf8_lossy(&out.stderr)
        .lines()
        .filter(|l| l.contains("migrate"))
        .count()
}

fn assert_source_intact_no_leftovers(dir: &Path, db: &Path) {
    assert!(db.is_dir(), "source KB must remain a directory");
    assert!(db.join("wal").is_dir(), "source KB wal/ must remain");
    let name = db.file_name().unwrap().to_string_lossy().to_string();
    for leftover in [
        format!("{name}.bak"),
        "kb.grafeo".to_string(),
        format!("{name}.migrate-lock"),
    ] {
        assert!(
            !dir.join(&leftover).exists(),
            "leftover {leftover} must not exist after a failed migration"
        );
    }
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry_name = entry.unwrap().file_name();
        assert!(
            !entry_name.to_string_lossy().starts_with(".migrate-tmp-"),
            "tmp leftover {entry_name:?} must not exist after a failed migration"
        );
    }
}

#[test]
fn migration_hint_printed_once_for_legacy_dir_kb() {
    let tmp = tempdir().unwrap();
    let db = tmp.path().join("kb.db");
    build_legacy_kb(&db);

    let out = run_gql(&db, "MATCH (n) RETURN count(n)");
    assert!(out.status.success());
    assert_eq!(
        stderr_hint_lines(&out),
        1,
        "exactly one migration hint line expected on stderr, got: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn no_hint_for_single_file_kb_and_migrate_succeeds_via_cli() {
    let tmp = tempdir().unwrap();
    let db = tmp.path().join("kb.db");
    build_legacy_kb(&db);

    let out = xd_kb(&db).arg("migrate").output().expect("spawn migrate");
    assert!(
        out.status.success(),
        "migrate failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    // The migrate command itself must not print the hint.
    assert_eq!(stderr_hint_lines(&out), 0);

    let target = tmp.path().join("kb.grafeo");
    assert!(target.is_file());
    let out = run_gql(&target, "MATCH (n) RETURN count(n)");
    assert!(out.status.success());
    assert_eq!(
        stderr_hint_lines(&out),
        0,
        "single-file KB must not trigger the hint: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn migration_lock_blocks_other_commands() {
    let tmp = tempdir().unwrap();
    let db = tmp.path().join("kb.db");
    build_legacy_kb(&db);
    let lock = tmp.path().join("kb.db.migrate-lock");
    std::fs::write(&lock, b"").unwrap();

    let out = run_gql(&db, "MATCH (n) RETURN count(n)");
    assert!(!out.status.success(), "commands must refuse a locked KB");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("lock"),
        "stderr should mention the lock: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Failure safety: whichever stage fails — count verification, just after
/// the copy, at the publish step itself, or between publish and the final
/// directory swap — the source KB must be untouched and no target / tmp /
/// sidecar / lock artifacts may remain.
#[test]
fn migrate_failpoints_leave_source_intact_and_clean() {
    for failpoint in [
        "verify",
        "after-copy",
        "target-publish",
        "after-target-publish",
    ] {
        let tmp = tempdir().unwrap();
        let db = tmp.path().join("kb.db");
        build_legacy_kb(&db);

        let out = xd_kb(&db)
            .arg("migrate")
            .env("XD_KB_MIGRATE_FAILPOINT", failpoint)
            .output()
            .expect("spawn migrate");
        assert!(
            !out.status.success(),
            "failpoint {failpoint} should make migrate fail"
        );
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("failpoint"),
            "stderr should carry the failpoint error ({failpoint}): {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_source_intact_no_leftovers(tmp.path(), &db);

        // The KB must still be fully usable after the failed attempt.
        let out = run_gql(&db, "MATCH (n) RETURN count(n)");
        assert!(
            out.status.success(),
            "source KB unusable after failpoint {failpoint}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// Past the commit point (old directory renamed to `.bak`) a failed lock
/// cleanup must not fail the migration: exit 0, warn on stderr, migrated
/// pair left in place.
#[test]
fn lock_release_failure_after_commit_still_succeeds() {
    let tmp = tempdir().unwrap();
    let db = tmp.path().join("kb.db");
    build_legacy_kb(&db);

    let out = xd_kb(&db)
        .arg("migrate")
        .env("XD_KB_MIGRATE_FAILPOINT", "lock-release")
        .output()
        .expect("spawn migrate");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "lock cleanup failure must not fail a committed migration: {stderr}"
    );
    assert!(
        stderr.contains("warning") && stderr.contains("lock"),
        "stderr should warn about the lock: {stderr}"
    );
    assert!(tmp.path().join("kb.grafeo").is_file());
    assert!(tmp.path().join("kb.db.bak").is_dir());
}
