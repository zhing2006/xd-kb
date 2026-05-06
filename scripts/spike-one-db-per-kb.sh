#!/usr/bin/env bash
# spike-one-db-per-kb.sh
#
# End-to-end spike for xd-kb's one-DB-per-KB model.
# Verifies the CLI shape (--db <path> only), physical DB-level isolation,
# and cross-process WAL persistence.
#
# Exit codes:
#   0  = all 10 spike steps passed
#   77 = SKIPPED (missing required env vars or release binary)
#   1  = a step failed (assertion mismatch or non-zero exit from xd-kb)

set -euo pipefail

readonly SKIP_EXIT=77

# ---------------------------------------------------------------------------
# Locate script + repo root, then locate the xd-kb release binary.
# Works under git bash / WSL / real Linux. Prefer Unix-style binary, fall
# back to .exe (running git bash on Windows), else SKIP.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XD_KB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

XD_KB_BIN=""
if [[ -x "${XD_KB_DIR}/target/release/xd-kb" ]]; then
  XD_KB_BIN="${XD_KB_DIR}/target/release/xd-kb"
elif [[ -x "${XD_KB_DIR}/target/release/xd-kb.exe" ]]; then
  XD_KB_BIN="${XD_KB_DIR}/target/release/xd-kb.exe"
elif [[ -f "${XD_KB_DIR}/target/release/xd-kb.exe" ]]; then
  # On Windows / git bash the executable bit may not be set; -f is enough.
  XD_KB_BIN="${XD_KB_DIR}/target/release/xd-kb.exe"
elif [[ -f "${XD_KB_DIR}/target/release/xd-kb" ]]; then
  XD_KB_BIN="${XD_KB_DIR}/target/release/xd-kb"
fi

if [[ -z "${XD_KB_BIN}" ]]; then
  echo "SKIPPED: xd-kb release binary not found under ${XD_KB_DIR}/target/release/ (run: cargo build --release)" >&2
  exit "${SKIP_EXIT}"
fi

# ---------------------------------------------------------------------------
# Required env vars (the CLI itself needs these to call the embedding API
# and the chat model). Missing any -> SKIP with code 77 (NOT 0).
# ---------------------------------------------------------------------------
missing=()
[[ -z "${XD_KB_API_BASE:-}" ]] && missing+=("XD_KB_API_BASE")
[[ -z "${XD_KB_API_KEY:-}"  ]] && missing+=("XD_KB_API_KEY")
[[ -z "${XD_KB_MODEL:-}"    ]] && missing+=("XD_KB_MODEL")
if (( ${#missing[@]} > 0 )); then
  echo "SKIPPED: missing required env vars: ${missing[*]}" >&2
  exit "${SKIP_EXIT}"
fi

# Optional vars are left to the CLI's own defaults (XD_KB_DIM=1024,
# XD_KB_MAX_CHUNK_CHARS=8192). We don't override them here.

# ---------------------------------------------------------------------------
# Temp workspace + cleanup trap.
# ---------------------------------------------------------------------------
TMP="$(mktemp -d -t xd-kb-spike-XXXXXX)"
cleanup() {
  local rc=$?
  if [[ -n "${TMP:-}" && -d "${TMP}" ]]; then
    rm -rf "${TMP}"
  fi
  exit "${rc}"
}
trap cleanup EXIT

KB_A="${TMP}/kb_a.db"
KB_B="${TMP}/kb_b.db"
FX_A="${TMP}/fixtures_a"
FX_B="${TMP}/fixtures_b"
# All fixtures must sit under a recognized category subtree (e.g. 世界观/),
# otherwise xd-kb's category detector rejects them and `index` produces no
# chunks. We also use the relative path (世界观/<file>.md) as the key when
# calling `check`, because that's what `index` stores in :FileHash.path.
FX_A_WORLD="${FX_A}/世界观"
FX_B_WORLD="${FX_B}/世界观"
A_REL="世界观/a.md"
B_REL="世界观/b.md"
C_REL="世界观/c.md"
mkdir -p "${FX_A_WORLD}" "${FX_B_WORLD}"

# ---------------------------------------------------------------------------
# Assertion helper: fail loudly with diagnostic output on mismatch.
# ---------------------------------------------------------------------------
assert_contains() {
  local haystack="$1"
  local needle="$2"
  local label="$3"
  if ! printf '%s' "${haystack}" | grep -F -q -- "${needle}"; then
    {
      echo "FAIL: ${label}"
      echo "  expected substring: ${needle}"
      echo "  actual output:"
      printf '%s\n' "${haystack}" | sed 's/^/    /'
    } >&2
    exit 1
  fi
}

assert_matches() {
  local haystack="$1"
  local pattern="$2"
  local label="$3"
  if ! printf '%s' "${haystack}" | grep -E -q -- "${pattern}"; then
    {
      echo "FAIL: ${label}"
      echo "  expected regex: ${pattern}"
      echo "  actual output:"
      printf '%s\n' "${haystack}" | sed 's/^/    /'
    } >&2
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Step 2: prepare fixtures (generated at runtime via heredoc).
#   a.md: long mixed zh/en doc (>4096 chars) to trigger auto-chunking
#   b.md: short doc with the spike-marker phrase used by the query step
#   c.md: multi H2 sections to verify prefix preservation
# ---------------------------------------------------------------------------

# a.md: build a long mixed-language body deterministically.
A_MD="${FX_A}/${A_REL}"
{
  echo "# 文档 A: 长文档自动切段验证 / Long Document Auto-Chunking"
  echo
  echo "本文档用于验证 xd-kb index 在文档长度超过 XD_KB_MAX_CHUNK_CHARS 时的自动切段行为。"
  echo "This document verifies the auto-chunking behavior when the document length exceeds the configured max chunk size."
  echo
  para_zh="一段中文段落：知识库系统需要把长文档切成可被 embedding 模型处理的小块，每一块在向量库里独立存储。"
  para_en="An English paragraph: a knowledge base needs to split long documents into smaller chunks so that each chunk can be embedded and stored independently in the vector store."
  for i in $(seq 1 60); do
    echo
    echo "## Section ${i} / 第 ${i} 节"
    echo
    echo "${para_zh} (segment ${i})"
    echo "${para_en} (segment ${i})"
    echo
    echo "Mixed sentence ${i}: 这是第 ${i} 段，混合了 Chinese 和 English 来制造足够的字符数 trigger auto chunking."
  done
} > "${A_MD}"

# Sanity check: ensure a.md is genuinely long (>4096 bytes) to trigger chunking.
a_size=$(wc -c < "${A_MD}")
if (( a_size < 4096 )); then
  echo "FAIL: setup: a.md is only ${a_size} bytes, need >4096 to trigger auto-chunking" >&2
  exit 1
fi

# b.md: short doc carrying the marker phrase used by the query step.
SPIKE_PHRASE="spike-marker-phrase-bbb 是一个用来在 query 测试中精确命中的句子"
B_MD="${FX_A}/${B_REL}"
cat > "${B_MD}" <<EOF
# 文档 B: 标志短语命中验证

## 标志短语

这是一篇短文档，仅用于验证 query 子命令能精确命中标志性短语。

${SPIKE_PHRASE}。

The marker above is intentionally distinctive so that a top-3 query will surface this file.
EOF

# c.md: multi H2 sections, verifies prefix preservation in chunk metadata.
C_MD="${FX_A}/${C_REL}"
cat > "${C_MD}" <<'EOF'
# 文档 C: 多段 H2 验证 prefix 保留

## 第一节 引言

第一节的正文内容。介绍 xd-kb 的 chunk prefix 保留机制。
This section introduces prefix preservation.

## 第二节 设计

第二节的正文内容。讨论一 DB 一 KB 的物理隔离方案。
Section two discusses the design.

## 第三节 验证

第三节的正文内容。说明 spike 脚本的端到端验证流程。
Section three covers the spike verification flow.
EOF

# fixtures_b: completely different filenames AND content. Must NOT contain
# the spike-marker phrase, otherwise the isolation step would be ambiguous.
cat > "${FX_B_WORLD}/x.md" <<'EOF'
# Document X: isolation fixture

## Isolation section

This file lives in fixtures_b and must never be returned by a query against kb_a.
其内容与 fixtures_a 完全不同。
EOF

cat > "${FX_B_WORLD}/y.md" <<'EOF'
# Document Y: another isolation fixture

## Isolation section

完全不同的内容，不包含任何 marker phrase。
Y stands alone, used only by kb_b.
EOF

cat > "${FX_B_WORLD}/z.md" <<'EOF'
# Document Z: third isolation fixture

## Isolation section

Content for kb_b only. 物理隔离断言依赖 fixtures_b 与 fixtures_a 完全不重叠。
EOF

echo "spike: tmp=${TMP}"
echo "spike: xd-kb=${XD_KB_BIN}"

# ---------------------------------------------------------------------------
# Step 1 (implicit): KB_A path is established (file is created by the index step).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 3: index fixtures_a into kb_a.db.
#   Expect: "Index complete:" line.
# ---------------------------------------------------------------------------
echo "--- step 3: index fixtures_a -> kb_a.db ---"
out3="$("${XD_KB_BIN}" --db "${KB_A}" index "${FX_A}" 2>&1)"
assert_contains "${out3}" "Index complete" "step 3: index missing 'Index complete' marker"
assert_matches  "${out3}" "indexed=[1-9][0-9]*" "step 3: index must index at least one file"

# ---------------------------------------------------------------------------
# Step 4: check unchanged file -> [unchanged].
# ---------------------------------------------------------------------------
echo "--- step 4: check a.md (unchanged) ---"
# Use the same relative path key index stored in :FileHash.path; cd into the
# fixtures root so the file argument resolves to "世界观/a.md" (matching index's
# walkdir + strip_prefix output).
out4="$(cd "${FX_A}" && "${XD_KB_BIN}" --db "${KB_A}" check "${A_REL}" 2>&1)"
assert_contains "${out4}" "[unchanged]" "step 4: expected [unchanged] for freshly-indexed a.md"

# ---------------------------------------------------------------------------
# Step 5: modify a.md, then check -> [modified].
# ---------------------------------------------------------------------------
echo "--- step 5: modify a.md then check ---"
echo >> "${A_MD}"
echo "## 追加段落用于触发 modified" >> "${A_MD}"
echo "This appended paragraph mutates the file hash so that 'check' must report [modified]." >> "${A_MD}"
out5="$(cd "${FX_A}" && "${XD_KB_BIN}" --db "${KB_A}" check "${A_REL}" 2>&1)"
assert_contains "${out5}" "[modified]" "step 5: expected [modified] after appending to a.md"

# ---------------------------------------------------------------------------
# Step 6: delete a.md, then check --mark -> [deleted] ... -> [removed].
# ---------------------------------------------------------------------------
echo "--- step 6: delete a.md then check --mark ---"
rm -f "${A_MD}"
out6="$(cd "${FX_A}" && "${XD_KB_BIN}" --db "${KB_A}" check "${A_REL}" --mark 2>&1)"
assert_contains "${out6}" "[deleted]"  "step 6: expected [deleted] after rm a.md"
assert_contains "${out6}" "[removed]"  "step 6: expected [removed] from --mark cleanup"

# ---------------------------------------------------------------------------
# Step 7: query the spike marker phrase. Top hit path must live under fixtures_a.
# ---------------------------------------------------------------------------
echo "--- step 7: query for spike marker phrase ---"
out7="$("${XD_KB_BIN}" --db "${KB_A}" query "${SPIKE_PHRASE}" --top 3 2>&1)"
if [[ -z "${out7//[[:space:]]/}" ]]; then
  echo "FAIL: step 7: query returned empty output" >&2
  exit 1
fi
# The top hit should reference b.md inside fixtures_a. We accept either the
# full fixtures_a path or the basename, since path normalization may differ.
if ! printf '%s' "${out7}" | grep -F -q "b.md"; then
  {
    echo "FAIL: step 7: top hits do not reference b.md"
    printf '%s\n' "${out7}" | sed 's/^/    /'
  } >&2
  exit 1
fi
# And whatever fixtures path appears, it must be from fixtures_a (not fixtures_b).
if printf '%s' "${out7}" | grep -F -q "fixtures_b"; then
  {
    echo "FAIL: step 7: query against kb_a returned a fixtures_b path"
    printf '%s\n' "${out7}" | sed 's/^/    /'
  } >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 8: gql read - MATCH (c:Chunk) RETURN count(c).
#   Expect tabular output with header 'count(c)' and a non-zero numeric row.
# ---------------------------------------------------------------------------
echo "--- step 8: gql read count(c) ---"
# grafeo renames bare aggregation functions in the result column header
# (RETURN count(c) → "countnonnull(...)"), so we use an explicit AS alias
# to pin the column name. Spec intent ("含列头 + 非零数字行") is preserved:
# we still assert a deterministic header AND a non-zero numeric row.
out8="$("${XD_KB_BIN}" --db "${KB_A}" gql "MATCH (c:Chunk) RETURN count(c) AS chunk_count" 2>&1)"
assert_contains "${out8}" "chunk_count" "step 8: gql output missing 'chunk_count' column header"
# Find a row that contains a non-zero integer (e.g. '12', '| 12 |', etc.).
if ! printf '%s' "${out8}" | grep -E -q '(^|[^0-9])[1-9][0-9]*([^0-9]|$)'; then
  {
    echo "FAIL: step 8: gql output has no non-zero numeric row"
    printf '%s\n' "${out8}" | sed 's/^/    /'
  } >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 8a: gql write - MERGE (:Smoke {name: 'one-db-per-kb'}).
#   Expect output starting with 'OK'.
# ---------------------------------------------------------------------------
echo "--- step 8a: gql write MERGE ---"
out8a="$("${XD_KB_BIN}" --db "${KB_A}" gql "MERGE (:Smoke {name: 'one-db-per-kb'})" 2>&1)"
# Allow leading whitespace, then OK at the start of the first non-empty line.
first_line="$(printf '%s\n' "${out8a}" | sed -n '/[^[:space:]]/{p;q;}')"
if [[ "${first_line}" != OK* ]]; then
  {
    echo "FAIL: step 8a: MERGE output does not start with 'OK'"
    echo "  first non-empty line: ${first_line}"
    printf '%s\n' "${out8a}" | sed 's/^/    /'
  } >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 9: physical isolation. Index fixtures_b into kb_b.db, then run the
# SAME query as step 7. Hits must come from fixtures_b only; fixtures_a paths
# (and the spike marker phrase itself) must not appear.
# ---------------------------------------------------------------------------
echo "--- step 9: physical isolation via kb_b.db ---"
out9_index="$("${XD_KB_BIN}" --db "${KB_B}" index "${FX_B}" 2>&1)"
assert_contains "${out9_index}" "Index complete" "step 9: kb_b index missing 'Index complete'"
assert_matches  "${out9_index}" "indexed=[1-9][0-9]*" "step 9: kb_b index must index at least one file"

out9_query="$("${XD_KB_BIN}" --db "${KB_B}" query "${SPIKE_PHRASE}" --top 3 2>&1)"
if printf '%s' "${out9_query}" | grep -F -q "fixtures_a"; then
  {
    echo "FAIL: step 9: query against kb_b returned a fixtures_a path (isolation broken)"
    printf '%s\n' "${out9_query}" | sed 's/^/    /'
  } >&2
  exit 1
fi
# At least one fixtures_b file (x/y/z.md) should appear.
if ! printf '%s' "${out9_query}" | grep -E -q '(x|y|z)\.md'; then
  {
    echo "FAIL: step 9: query against kb_b returned no fixtures_b files"
    printf '%s\n' "${out9_query}" | sed 's/^/    /'
  } >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 10: process isolation / WAL persistence across processes.
# Spawn a fresh xd-kb process against kb_a.db; query must still work
# (data was written by earlier processes and persisted via WAL).
# ---------------------------------------------------------------------------
echo "--- step 10: cross-process WAL persistence on kb_a.db ---"
out10="$("${XD_KB_BIN}" --db "${KB_A}" query "${SPIKE_PHRASE}" --top 3 2>&1)"
if [[ -z "${out10//[[:space:]]/}" ]]; then
  echo "FAIL: step 10: query in a fresh process returned empty output (WAL persistence broken?)" >&2
  exit 1
fi
if ! printf '%s' "${out10}" | grep -F -q "b.md"; then
  {
    echo "FAIL: step 10: cross-process query lost reference to b.md"
    printf '%s\n' "${out10}" | sed 's/^/    /'
  } >&2
  exit 1
fi

echo "OK: spike-one-db-per-kb passed (10/10)"
exit 0
