use std::path::Path;

use anyhow::Result;
use grafeo::Value;

use crate::storage::KbStore;

fn display_width(s: &str) -> usize {
    s.chars()
        .map(|c| {
            if ('\u{1100}'..='\u{115F}').contains(&c)      // Hangul Jamo
                || ('\u{2E80}'..='\u{9FFF}').contains(&c)   // CJK
                || ('\u{AC00}'..='\u{D7AF}').contains(&c)   // Hangul Syllables
                || ('\u{F900}'..='\u{FAFF}').contains(&c)   // CJK Compatibility
                || ('\u{FE30}'..='\u{FE6F}').contains(&c)   // CJK Compatibility Forms
                || ('\u{FF01}'..='\u{FF60}').contains(&c)   // Fullwidth Forms
                || ('\u{FFE0}'..='\u{FFE6}').contains(&c)   // Fullwidth Signs
                || ('\u{20000}'..='\u{2FA1F}').contains(&c)
            // CJK Ext B–F
            {
                2
            } else {
                1
            }
        })
        .sum()
}

fn pad_to_width(s: &str, target: usize) -> String {
    let w = display_width(s);
    if w >= target {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(target - w))
    }
}

fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int64(i) => serde_json::Value::Number((*i).into()),
        Value::Float64(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Value::String(s) => serde_json::Value::String(s.to_string()),
        Value::List(items) => serde_json::Value::Array(items.iter().map(value_to_json).collect()),
        Value::Map(map) => {
            let obj: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.to_string(), value_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        Value::Path { nodes, edges } => {
            serde_json::json!({
                "nodes": nodes.iter().map(value_to_json).collect::<Vec<_>>(),
                "edges": edges.iter().map(value_to_json).collect::<Vec<_>>(),
            })
        }
        other => serde_json::Value::String(format!("{other}")),
    }
}

fn format_value(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Int64(i) => i.to_string(),
        Value::Float64(f) => f.to_string(),
        Value::String(s) => s.replace('\n', "\\n"),
        Value::List(_) | Value::Map(_) | Value::Path { .. } => {
            serde_json::to_string(&value_to_json(v)).unwrap_or_else(|_| format!("{v}"))
        }
        other => format!("{other}"),
    }
}

fn render_table(columns: &[String], rows: &[Vec<Value>]) -> String {
    let col_count = columns.len();
    let mut out = String::new();

    let mut widths: Vec<usize> = columns.iter().map(|c| display_width(c)).collect();
    for row in rows {
        for (i, val) in row.iter().enumerate() {
            if i < col_count {
                let w = display_width(&format_value(val));
                if w > widths[i] {
                    widths[i] = w;
                }
            }
        }
    }

    let top: String = widths
        .iter()
        .map(|w| "─".repeat(w + 2))
        .collect::<Vec<_>>()
        .join("┬");
    out.push_str(&format!("┌{top}┐\n"));

    let header: String = columns
        .iter()
        .enumerate()
        .map(|(i, c)| format!(" {} ", pad_to_width(c, widths[i])))
        .collect::<Vec<_>>()
        .join("│");
    out.push_str(&format!("│{header}│\n"));

    let sep: String = widths
        .iter()
        .map(|w| "─".repeat(w + 2))
        .collect::<Vec<_>>()
        .join("┼");
    out.push_str(&format!("├{sep}┤\n"));

    for row in rows {
        let cells: String = (0..col_count)
            .map(|i| {
                let val = row.get(i).unwrap_or(&Value::Null);
                format!(" {} ", pad_to_width(&format_value(val), widths[i]))
            })
            .collect::<Vec<_>>()
            .join("│");
        out.push_str(&format!("│{cells}│\n"));
    }

    let bottom: String = widths
        .iter()
        .map(|w| "─".repeat(w + 2))
        .collect::<Vec<_>>()
        .join("┴");
    out.push_str(&format!("└{bottom}┘\n"));

    out
}

/// Scan for a keyword token in a GQL statement, skipping string literals
/// and treating all whitespace as token separators.
fn contains_keyword(statement: &str, keyword: &str) -> bool {
    let upper = statement.to_uppercase();
    let kw = keyword.to_uppercase();
    let kw_len = kw.len();
    let bytes = upper.as_bytes();
    let kw_bytes = kw.as_bytes();
    let mut in_single_quote = false;
    let mut in_double_quote = false;

    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\'' && !in_double_quote {
            in_single_quote = !in_single_quote;
        } else if b == b'"' && !in_single_quote {
            in_double_quote = !in_double_quote;
        } else if !in_single_quote
            && !in_double_quote
            && i + kw_len <= bytes.len()
            && &bytes[i..i + kw_len] == kw_bytes
        {
            let before_ok =
                i == 0 || (!bytes[i - 1].is_ascii_alphanumeric() && bytes[i - 1] != b'_');
            let after_ok = i + kw_len >= bytes.len()
                || (!bytes[i + kw_len].is_ascii_alphanumeric() && bytes[i + kw_len] != b'_');
            if before_ok && after_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Check if a GQL statement starts with a known write keyword.
fn is_write_statement(statement: &str) -> bool {
    let trimmed = statement.trim_start().to_uppercase();
    const PURE_WRITE_PREFIXES: &[&str] = &["INSERT", "MERGE", "DELETE", "SET", "CREATE", "DROP"];
    for prefix in PURE_WRITE_PREFIXES {
        if trimmed.starts_with(prefix) {
            return true;
        }
    }
    if trimmed.starts_with("MATCH") {
        return !contains_keyword(statement, "RETURN");
    }
    false
}

fn format_result(result: &grafeo::QueryResult, is_write: bool) -> String {
    if is_write && !result.columns.iter().any(|c| !c.starts_with("_anon_")) {
        let affected = result.row_count();
        match &result.status_message {
            Some(msg) => format!("OK: {msg}"),
            None if affected > 0 => format!("OK: {affected} row(s) affected"),
            None => "OK".to_string(),
        }
    } else if result.columns.is_empty() || result.is_empty() {
        if result.columns.is_empty() {
            "OK".to_string()
        } else {
            "(empty result)".to_string()
        }
    } else {
        render_table(&result.columns, result.rows())
            .trim_end()
            .to_string()
    }
}

pub fn run(db_path: &Path, statement: &str) -> Result<()> {
    let output = execute_and_format(db_path, statement)?;
    println!("{output}");
    Ok(())
}

fn execute_and_format(db_path: &Path, statement: &str) -> Result<String> {
    let store = KbStore::open(db_path)?;
    let result = store.execute_raw(statement)?;
    Ok(format_result(&result, is_write_statement(statement)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup_db() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    // === Spec scenario 1: read statement returns table with column header ===
    #[test]
    fn read_statement_returns_table_with_header() {
        let dir = setup_db();
        let db_path = dir.path().join("kb.db");

        // seed: insert a Chunk node
        execute_and_format(&db_path, "INSERT (:Chunk {file_path: 'a.md'})").unwrap();

        // spec example: MATCH (c:Chunk) RETURN c.file_path LIMIT 5
        let output =
            execute_and_format(&db_path, "MATCH (c:Chunk) RETURN c.file_path LIMIT 5").unwrap();

        assert!(output.contains('┌'), "table top border missing: {output}");
        assert!(
            output.contains('└'),
            "table bottom border missing: {output}"
        );
        assert!(
            output.contains("c.file_path"),
            "column header missing: {output}"
        );
        assert!(output.contains("a.md"), "data row missing: {output}");
    }

    // === Spec scenario 2: write statement returns OK ===
    #[test]
    fn write_statement_returns_ok() {
        let dir = setup_db();
        let db_path = dir.path().join("kb.db");

        let insert_out = execute_and_format(&db_path, "INSERT (:Note {name: 'x'})").unwrap();
        assert!(
            insert_out.starts_with("OK"),
            "INSERT should return OK: {insert_out}"
        );

        let merge_out = execute_and_format(&db_path, "MERGE (:Note {name: 'x'})").unwrap();
        assert!(
            merge_out.starts_with("OK"),
            "MERGE should return OK: {merge_out}"
        );

        let set_out =
            execute_and_format(&db_path, "MATCH (n:Note {name: 'x'}) SET n.tag = 'a'").unwrap();
        assert!(
            set_out.starts_with("OK"),
            "MATCH...SET should return OK: {set_out}"
        );

        let delete_out =
            execute_and_format(&db_path, "MATCH (n:Note {name: 'x'}) DELETE n").unwrap();
        assert!(
            delete_out.starts_with("OK"),
            "MATCH...DELETE should return OK: {delete_out}"
        );
    }

    // === Spec scenario 3: GQL syntax error => non-zero exit (Err in Rust) ===
    #[test]
    fn syntax_error_returns_err() {
        let dir = setup_db();
        let db_path = dir.path().join("kb.db");
        let result = execute_and_format(&db_path, "INVALID GQL SYNTAX HERE");
        assert!(
            result.is_err(),
            "syntax error must propagate as Err, got Ok({:?})",
            result.ok()
        );
    }

    // gql does NOT auto-inject any predicate. The statement is passed through verbatim.
    #[test]
    fn statement_is_passed_through_verbatim() {
        let dir = setup_db();
        let db_path = dir.path().join("kb.db");

        execute_and_format(&db_path, "INSERT (:Chunk {file_path: 'l1.md'})").unwrap();
        execute_and_format(&db_path, "INSERT (:Chunk {file_path: 'o1.md'})").unwrap();

        // statement matches all chunks — must see both rows because the CLI
        // doesn't auto-filter.
        let out = execute_and_format(&db_path, "MATCH (c:Chunk) RETURN c.file_path").unwrap();

        assert!(out.contains("l1.md"), "first row missing: {out}");
        assert!(out.contains("o1.md"), "second row missing: {out}");
    }

    // === pure unit tests for helpers ===

    #[test]
    fn is_write_recognizes_writes() {
        assert!(is_write_statement("INSERT (:N {})"));
        assert!(is_write_statement("MERGE (:N {})"));
        assert!(is_write_statement("DELETE n"));
        assert!(is_write_statement("SET n.x = 1"));
        assert!(is_write_statement("MATCH (n) DELETE n"));
        assert!(is_write_statement("MATCH (n)\nSET n.x = 1"));
    }

    #[test]
    fn is_write_recognizes_reads() {
        assert!(!is_write_statement("MATCH (n) RETURN n"));
        assert!(!is_write_statement("MATCH (n)\nRETURN n"));
    }

    #[test]
    fn is_write_ignores_return_in_string_literal() {
        assert!(is_write_statement("INSERT (:T {desc: 'RETURN value'})"));
    }

    #[test]
    fn format_value_basics() {
        assert_eq!(format_value(&Value::Null), "null");
        assert_eq!(format_value(&Value::Bool(true)), "true");
        assert_eq!(format_value(&Value::Int64(42)), "42");
        assert_eq!(format_value(&Value::String("a\nb".into())), "a\\nb");
    }

    #[test]
    fn display_width_handles_cjk() {
        assert_eq!(display_width("hello"), 5);
        assert_eq!(display_width("科尔"), 4);
        assert_eq!(display_width("hello科尔"), 9);
    }
}
