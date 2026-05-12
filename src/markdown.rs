use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};

pub struct Chunk {
    pub content: String,
    pub line_start: usize,
    pub line_end: usize,
    pub prefix: String,
}

pub struct ChunkLimits {
    pub max_chars: u32,
}

struct RawChunk {
    pub content: String,
    pub line_start: usize,
    pub line_end: usize,
    pub heading_path: Vec<String>,
    // Byte range of `content` within the original source `content` string.
    // Invariant: source[byte_start..byte_end] == self.content.
    // When the producer cannot guarantee this invariant (e.g. content was
    // synthesized by joining non-contiguous regions), `byte_start == byte_end`
    // signals "no precise byte mapping available" and split_oversized falls
    // back to inheriting the parent line range.
    pub byte_start: usize,
    pub byte_end: usize,
}

// `frontmatter_end_offset` only needs to skip a leading YAML frontmatter block
// so chunkers don't include it as document body. Field-level YAML parsing is
// no longer performed: chunk prefixes carry only generic heading_path metadata,
// not KB-specific frontmatter projections.
fn frontmatter_end_offset(content: &str) -> usize {
    if !content.starts_with("---") {
        return 0;
    }
    let rest = &content[3..];
    if let Some(pos) = rest.find("\n---") {
        let end = 3 + pos + 4;
        if content.len() > end && content.as_bytes().get(end) == Some(&b'\n') {
            end + 1
        } else {
            end
        }
    } else {
        0
    }
}

// --- Line number mapping ---

fn build_line_starts(text: &str) -> Vec<usize> {
    let mut starts = vec![0usize];
    for (i, byte) in text.as_bytes().iter().enumerate() {
        if *byte == b'\n' {
            starts.push(i + 1);
        }
    }
    starts
}

fn offset_to_line(line_starts: &[usize], offset: usize) -> usize {
    match line_starts.binary_search(&offset) {
        Ok(i) => i + 1,
        Err(i) => i,
    }
}

fn heading_level_num(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

// Trim leading and trailing ASCII / unicode whitespace from a byte slice and
// return the (start, end) offsets in the original `content` such that
// `&content[start..end]` is the trimmed substring.
fn trim_byte_range(content: &str, start: usize, end: usize) -> (usize, usize) {
    let slice = &content[start..end];
    let trimmed = slice.trim();
    if trimmed.is_empty() {
        return (start, start);
    }
    let leading = slice.len() - slice.trim_start().len();
    let trailing = slice.len() - slice.trim_end().len();
    (start + leading, end - trailing)
}

// --- Chunking ---

struct HeadingMark {
    offset: usize,
    level: HeadingLevel,
    text: String,
}

fn chunk_by_heading(content: &str, target_levels: &[HeadingLevel]) -> Vec<RawChunk> {
    let line_starts = build_line_starts(content);
    let content_start = frontmatter_end_offset(content);

    let all_headings: Vec<HeadingMark> = parse_all_headings(content)
        .into_iter()
        .filter(|h| h.offset >= content_start)
        .collect();

    let split_indices: Vec<usize> = all_headings
        .iter()
        .enumerate()
        .filter(|(_, h)| target_levels.contains(&h.level))
        .map(|(i, _)| i)
        .collect();

    let mut chunks = Vec::new();

    // Determine whether there is non-empty preamble text before the first
    // split-point heading. If so it gets merged into the first chunk by
    // EXTENDING the chunk's byte range backwards, so the resulting
    // `content` substring still corresponds 1:1 to a contiguous source range.
    let preamble_start = split_indices
        .first()
        .map(|&i| all_headings[i].offset)
        .and_then(|first_offset| {
            if first_offset > content_start {
                let pre = content[content_start..first_offset].trim();
                if !pre.is_empty() {
                    return Some(content_start);
                }
            }
            None
        });

    for (si, &idx) in split_indices.iter().enumerate() {
        let raw_start = if si == 0 {
            preamble_start.unwrap_or(all_headings[idx].offset)
        } else {
            all_headings[idx].offset
        };
        let raw_end = split_indices
            .get(si + 1)
            .map(|&next_idx| all_headings[next_idx].offset)
            .unwrap_or(content.len());

        let (byte_start, byte_end) = trim_byte_range(content, raw_start, raw_end);
        if byte_start == byte_end {
            continue;
        }
        let chunk_content = &content[byte_start..byte_end];

        let path = build_heading_path(&all_headings, idx);

        chunks.push(RawChunk {
            content: chunk_content.to_string(),
            line_start: offset_to_line(&line_starts, byte_start),
            line_end: offset_to_line(&line_starts, byte_end.saturating_sub(1).max(byte_start)),
            heading_path: path,
            byte_start,
            byte_end,
        });
    }

    chunks
}

fn parse_all_headings(content: &str) -> Vec<HeadingMark> {
    let parser = Parser::new_ext(content, Options::empty());
    let mut headings = Vec::new();
    let mut in_heading = false;
    let mut heading_text = String::new();
    let mut heading_offset = 0;
    let mut heading_level = HeadingLevel::H1;

    for (event, range) in parser.into_offset_iter() {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                in_heading = true;
                heading_offset = range.start;
                heading_level = level;
                heading_text.clear();
            }
            Event::Text(ref text) if in_heading => {
                heading_text.push_str(text);
            }
            Event::Code(ref code) if in_heading => {
                heading_text.push_str(code);
            }
            Event::End(TagEnd::Heading(_)) if in_heading => {
                headings.push(HeadingMark {
                    offset: heading_offset,
                    level: heading_level,
                    text: heading_text.clone(),
                });
                in_heading = false;
            }
            _ => {}
        }
    }

    headings
}

fn build_heading_path(all_headings: &[HeadingMark], target_idx: usize) -> Vec<String> {
    let mut path = vec![all_headings[target_idx].text.clone()];
    let mut current_level = heading_level_num(all_headings[target_idx].level);

    for i in (0..target_idx).rev() {
        let level = heading_level_num(all_headings[i].level);
        if level < current_level {
            path.push(all_headings[i].text.clone());
            current_level = level;
        }
    }

    path.reverse();
    path
}

fn chunk_whole_file(content: &str) -> Vec<RawChunk> {
    // Skip leading YAML frontmatter so heading-less docs with frontmatter
    // don't leak the YAML block into the indexed chunk body.
    let content_start = frontmatter_end_offset(content);
    let (byte_start, byte_end) = trim_byte_range(content, content_start, content.len());
    if byte_start == byte_end {
        return Vec::new();
    }
    let line_starts = build_line_starts(content);
    vec![RawChunk {
        content: content[byte_start..byte_end].to_string(),
        line_start: offset_to_line(&line_starts, byte_start),
        line_end: offset_to_line(&line_starts, byte_end.saturating_sub(1).max(byte_start)),
        heading_path: Vec::new(),
        byte_start,
        byte_end,
    }]
}

// Generic chunker for any markdown file: split on H2 headings; if the file has
// no H2, fall back to H3; if neither, treat the whole file as one chunk.
// Splitting policy is intentionally simple — KB-specific tuning belongs in the
// KB's own `.xd-kb.yaml`-driven preprocessing, not in the engine.
fn chunk_file(content: &str) -> Vec<RawChunk> {
    let chunks = chunk_by_heading(content, &[HeadingLevel::H2]);
    if !chunks.is_empty() {
        return chunks;
    }
    let chunks = chunk_by_heading(content, &[HeadingLevel::H3]);
    if !chunks.is_empty() {
        return chunks;
    }
    chunk_whole_file(content)
}

// --- Oversized chunk splitting ---

// Given a substring boundary list for a single parent chunk, produce sub-chunk
// byte ranges (relative to the parent's `content`) such that each sub-chunk's
// char count is <= max_chars whenever possible.
//
// `boundaries` is a sorted list of candidate split offsets (relative to the
// parent slice, exclusive of 0 and slice.len()) where splitting is preferred.
// The function greedily extends each sub-chunk to the largest boundary that
// keeps the resulting char count within `max_chars`. If no such boundary
// exists for a piece (i.e. even the smallest extension exceeds the limit),
// returns None to signal the caller to fall back to a finer granularity.
fn try_split_at_boundaries(
    slice: &str,
    boundaries: &[usize],
    max_chars: usize,
) -> Option<Vec<(usize, usize)>> {
    if slice.chars().count() <= max_chars {
        return Some(vec![(0, slice.len())]);
    }
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    let mut cursor = 0usize;
    let total_len = slice.len();

    let mut all_points: Vec<usize> = boundaries.to_vec();
    all_points.push(total_len);

    while cursor < total_len {
        // Find the largest boundary > cursor whose chars().count() from cursor is <= max_chars.
        let mut best: Option<usize> = None;
        for &b in &all_points {
            if b <= cursor {
                continue;
            }
            let piece = &slice[cursor..b];
            if piece.chars().count() <= max_chars {
                best = Some(b);
            } else {
                break;
            }
        }
        match best {
            Some(end) => {
                pieces.push((cursor, end));
                cursor = end;
            }
            None => return None,
        }
    }

    Some(pieces)
}

// Force-split `slice` purely on char boundaries into sub-pieces of <= max_chars chars.
fn force_char_split(slice: &str, max_chars: usize) -> Vec<(usize, usize)> {
    let mut pieces = Vec::new();
    let mut piece_start = 0usize;
    let mut chars_in_piece = 0usize;
    let mut last_idx = 0usize;

    for (i, _ch) in slice.char_indices() {
        if chars_in_piece == max_chars {
            pieces.push((piece_start, i));
            piece_start = i;
            chars_in_piece = 0;
        }
        chars_in_piece += 1;
        last_idx = i;
    }
    // Push trailing piece.
    let _ = last_idx;
    if piece_start < slice.len() {
        pieces.push((piece_start, slice.len()));
    }
    pieces
}

// Find paragraph break offsets (positions just after a blank line) within `slice`.
// Recognises both LF (`\n\n`) and CRLF (`\r\n\r\n`) blank-line forms — Windows
// markdown files routinely use CRLF and would otherwise skip the paragraph tier
// of split_oversized and fall through to sentence/forced-char splitting.
fn paragraph_boundaries(slice: &str) -> Vec<usize> {
    let bytes = slice.as_bytes();
    let mut result = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        // Match a blank-line separator at position i. A blank-line is two
        // consecutive newline runs where each newline is `\n` or `\r\n`.
        let first = newline_len_at(bytes, i);
        if first > 0 {
            let after_first = i + first;
            let second = newline_len_at(bytes, after_first);
            if second > 0 {
                // We have a blank line; consume any further trailing newlines
                // so the next sub-chunk starts on non-empty content.
                let mut j = after_first + second;
                loop {
                    let n = newline_len_at(bytes, j);
                    if n == 0 {
                        break;
                    }
                    j += n;
                }
                if j < bytes.len() {
                    result.push(j);
                }
                i = j;
                continue;
            }
        }
        i += 1;
    }
    result
}

// If `bytes[i..]` begins with a newline (`\r\n` or `\n`), return its byte
// length; otherwise return 0. Handles bare `\r` as not-a-newline (markdown
// convention; pulldown_cmark also ignores bare CR).
fn newline_len_at(bytes: &[u8], i: usize) -> usize {
    if i >= bytes.len() {
        return 0;
    }
    if bytes[i] == b'\r' && i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
        2
    } else if bytes[i] == b'\n' {
        1
    } else {
        0
    }
}

// Find sentence-end boundaries within `slice`. Boundary offset is positioned
// immediately after the punctuation char so the punctuation stays with the
// preceding sentence.
fn sentence_boundaries(slice: &str) -> Vec<usize> {
    let mut result = Vec::new();
    let len = slice.len();
    for (i, ch) in slice.char_indices() {
        let is_terminator = matches!(ch, '。' | '！' | '？' | '.' | '!' | '?');
        if is_terminator {
            let after = i + ch.len_utf8();
            if after < len {
                result.push(after);
            }
        }
    }
    result
}

fn split_oversized(
    chunks: Vec<RawChunk>,
    _content: &str,
    line_starts: &[usize],
    max_chars: usize,
) -> Vec<RawChunk> {
    if max_chars == 0 {
        return chunks;
    }
    let mut out: Vec<RawChunk> = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        if chunk.content.chars().count() <= max_chars {
            out.push(chunk);
            continue;
        }

        // Try paragraph boundaries first.
        let slice = chunk.content.as_str();
        let para_pts = paragraph_boundaries(slice);
        let pieces = if let Some(p) = try_split_at_boundaries(slice, &para_pts, max_chars) {
            p
        } else {
            // Per-paragraph fallback to sentence, then to forced char split.
            let mut pts = vec![0usize];
            pts.extend(para_pts.iter().copied());
            pts.push(slice.len());
            let mut acc: Vec<(usize, usize)> = Vec::new();
            for w in pts.windows(2) {
                let (ps, pe) = (w[0], w[1]);
                let para = &slice[ps..pe];
                if para.chars().count() <= max_chars {
                    acc.push((ps, pe));
                    continue;
                }
                // Try sentence boundaries within this paragraph.
                let sent_pts_local = sentence_boundaries(para);
                if let Some(sp) = try_split_at_boundaries(para, &sent_pts_local, max_chars) {
                    for (a, b) in sp {
                        acc.push((ps + a, ps + b));
                    }
                    continue;
                }
                // Per-sentence fallback to force char split.
                let mut spts = vec![0usize];
                spts.extend(sent_pts_local.iter().copied());
                spts.push(para.len());
                for sw in spts.windows(2) {
                    let (ss, se) = (sw[0], sw[1]);
                    let sent = &para[ss..se];
                    if sent.chars().count() <= max_chars {
                        acc.push((ps + ss, ps + se));
                        continue;
                    }
                    for (fa, fb) in force_char_split(sent, max_chars) {
                        acc.push((ps + ss + fa, ps + ss + fb));
                    }
                }
            }
            acc
        };

        let has_byte_mapping = chunk.byte_end > chunk.byte_start
            && (chunk.byte_end - chunk.byte_start) == chunk.content.len();
        let parent_line_start = chunk.line_start;
        let parent_line_end = chunk.line_end;

        for (a, b) in pieces {
            let sub_text = &chunk.content[a..b];
            let trimmed = sub_text.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Locate the trimmed sub-slice byte offsets within sub_text.
            let leading = sub_text.len() - sub_text.trim_start().len();
            let trailing = sub_text.len() - sub_text.trim_end().len();
            let sub_a = a + leading;
            let sub_b = b - trailing;

            let (line_start, line_end) = if has_byte_mapping {
                let abs_a = chunk.byte_start + sub_a;
                let abs_b = chunk.byte_start + sub_b;
                let ls =
                    offset_to_line(line_starts, abs_a).clamp(parent_line_start, parent_line_end);
                let le = offset_to_line(line_starts, abs_b.saturating_sub(1).max(abs_a))
                    .clamp(parent_line_start, parent_line_end);
                (ls, le)
            } else {
                (parent_line_start, parent_line_end)
            };

            out.push(RawChunk {
                content: trimmed.to_string(),
                line_start,
                line_end,
                heading_path: chunk.heading_path.clone(),
                byte_start: 0,
                byte_end: 0,
            });
        }
    }

    out
}

// --- Heading-path prefix ---

// Heading-path prefix: the chunk's heading ancestors joined by " > ", used
// only as :Chunk.prefix metadata. Empty when the chunk has no heading_path
// (e.g. whole-file fallback for a heading-less doc).
fn generate_heading_prefix(heading_path: &[String]) -> String {
    heading_path.join(" > ")
}

pub fn process_file(content: &str, _rel_path: &str, limits: &ChunkLimits) -> Vec<Chunk> {
    let raw_chunks = chunk_file(content);
    let line_starts = build_line_starts(content);
    let raw_chunks = split_oversized(raw_chunks, content, &line_starts, limits.max_chars as usize);

    raw_chunks
        .into_iter()
        .map(|raw| {
            let prefix = generate_heading_prefix(&raw.heading_path);
            Chunk {
                content: raw.content,
                line_start: raw.line_start,
                line_end: raw.line_end,
                prefix,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unlimited() -> ChunkLimits {
        ChunkLimits {
            max_chars: u32::MAX,
        }
    }

    fn make_raw(content: &str, line_start: usize, line_end: usize, byte_start: usize) -> RawChunk {
        RawChunk {
            content: content.to_string(),
            line_start,
            line_end,
            heading_path: vec!["H".to_string()],
            byte_start,
            byte_end: byte_start + content.len(),
        }
    }

    #[test]
    fn frontmatter_end_offset_skips_yaml_block() {
        let content = "---\nname: 科尔\n---\n# Body\n";
        let off = frontmatter_end_offset(content);
        assert!(off > 0);
        // Body starts after the closing ---\n
        assert!(content[off..].starts_with("# Body"));
    }

    #[test]
    fn frontmatter_end_offset_returns_zero_when_absent() {
        assert_eq!(frontmatter_end_offset("# Body\n"), 0);
        assert_eq!(frontmatter_end_offset(""), 0);
    }

    #[test]
    fn test_chunk_by_h2_line_ranges() {
        let content = "---\nname: test\n---\n## Section A\nContent A\n## Section B\nContent B\n";
        let chunks = chunk_by_heading(content, &[HeadingLevel::H2]);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].content.contains("Section A"));
        assert!(chunks[1].content.contains("Section B"));
        assert_eq!(chunks[0].heading_path, vec!["Section A"]);
        assert_eq!(chunks[1].heading_path, vec!["Section B"]);
        // Line ranges should be sane: chunk 0 starts at line 4 ("## Section A"),
        // chunk 1 starts at line 6 ("## Section B")
        assert_eq!(chunks[0].line_start, 4);
        assert_eq!(chunks[1].line_start, 6);
    }

    #[test]
    fn test_chunk_h3_with_h2_parent() {
        let content = "## Geography\n### Country A\nContent A\n### Country B\nContent B\n## History\n### Event 1\nContent 1\n";
        let chunks = chunk_by_heading(content, &[HeadingLevel::H3]);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].heading_path, vec!["Geography", "Country A"]);
        assert_eq!(chunks[1].heading_path, vec!["Geography", "Country B"]);
        assert_eq!(chunks[2].heading_path, vec!["History", "Event 1"]);
    }

    #[test]
    fn test_preamble_merged_into_first_chunk() {
        let content =
            "Intro text before any heading\n\n## Section A\nContent A\n## Section B\nContent B\n";
        let chunks = chunk_by_heading(content, &[HeadingLevel::H2]);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].content.contains("Intro text before any heading"));
        assert!(chunks[0].content.contains("Section A"));
        assert_eq!(chunks[0].line_start, 1);
    }

    #[test]
    fn test_chunk_whole_file_line_count() {
        let content = "Some content\nLine 2\nLine 3";
        let chunks = chunk_whole_file(content);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].line_start, 1);
        assert_eq!(chunks[0].line_end, 3);
    }

    #[test]
    fn process_file_uses_h2_split_with_heading_path_prefix() {
        let content = "## 背景\n背景描述\n## 能力\n能力描述\n";
        let chunks = process_file(content, "any/path.md", &unlimited());
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].prefix, "背景");
        assert_eq!(chunks[1].prefix, "能力");
        // Generic prefix MUST NOT contain any KB-specific category label
        for ck in &chunks {
            assert!(!ck.prefix.contains("角色档案"));
            assert!(!ck.prefix.contains("台本"));
            assert!(!ck.prefix.contains('['));
        }
    }

    #[test]
    fn process_file_falls_back_to_h3_when_no_h2() {
        let content = "### Sub A\ncontent A\n### Sub B\ncontent B\n";
        let chunks = process_file(content, "any.md", &unlimited());
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].prefix, "Sub A");
        assert_eq!(chunks[1].prefix, "Sub B");
    }

    #[test]
    fn process_file_falls_back_to_whole_file_when_no_headings() {
        let content = "just some prose, no headings at all.\n";
        let chunks = process_file(content, "any.md", &unlimited());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].prefix, "");
        assert!(chunks[0].content.contains("just some prose"));
    }

    #[test]
    fn process_file_whole_file_fallback_skips_frontmatter() {
        // Heading-less doc with YAML frontmatter: the YAML block MUST NOT be
        // included in the chunk content. Regression for the whole-file path
        // that previously trimmed from byte 0 and indexed the YAML.
        // Layout: line 1 `---`, line 2 `tags: [a, b]`, line 3 `---`,
        //         line 4 `just some prose, no headings.`, line 5 (empty).
        let content = "---\ntags: [a, b]\n---\njust some prose, no headings.\n";
        let chunks = process_file(content, "any.md", &unlimited());
        assert_eq!(chunks.len(), 1);
        assert!(
            !chunks[0].content.contains("tags:"),
            "frontmatter MUST NOT leak into whole-file chunk: {:?}",
            chunks[0].content
        );
        assert!(
            !chunks[0].content.contains("---"),
            "frontmatter delimiter MUST NOT leak into chunk: {:?}",
            chunks[0].content
        );
        assert!(chunks[0].content.contains("just some prose"));

        // Line range MUST land on the body, not on the YAML block. A
        // regression that trims from byte 0 would report line_start = 1.
        assert_eq!(
            chunks[0].line_start, 4,
            "whole-file chunk after frontmatter MUST start at body line 4, got {}",
            chunks[0].line_start
        );
        assert_eq!(
            chunks[0].line_end, 4,
            "whole-file chunk's body ends on line 4, got {}",
            chunks[0].line_end
        );
    }

    // --- split_oversized tests ---

    #[test]
    fn test_split_oversized_paragraph_boundary() {
        // Three paragraphs, each ~10 chars; max_chars=12 forces splitting at \n\n only.
        let content = "AAAAAAAAAA\n\nBBBBBBBBBB\n\nCCCCCCCCCC";
        let line_starts = build_line_starts(content);
        let parent = make_raw(content, 1, 5, 0);
        let out = split_oversized(vec![parent], content, &line_starts, 12);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].content, "AAAAAAAAAA");
        assert_eq!(out[1].content, "BBBBBBBBBB");
        assert_eq!(out[2].content, "CCCCCCCCCC");
        for sub in &out {
            assert!(sub.content.chars().count() <= 12);
            assert_eq!(sub.heading_path, vec!["H".to_string()]);
        }
    }

    #[test]
    fn test_split_oversized_paragraph_boundary_crlf() {
        // Same as the LF test but with Windows CRLF line endings — the splitter
        // MUST recognise `\r\n\r\n` as a paragraph boundary or it would skip
        // the paragraph tier on Windows-authored markdown and degrade to
        // sentence/forced-char splitting unnecessarily.
        let content = "AAAAAAAAAA\r\n\r\nBBBBBBBBBB\r\n\r\nCCCCCCCCCC";
        let line_starts = build_line_starts(content);
        let parent = make_raw(content, 1, 5, 0);
        let out = split_oversized(vec![parent], content, &line_starts, 12);
        assert_eq!(
            out.len(),
            3,
            "CRLF paragraph splitter must produce 3 sub-chunks, got {} ({:?})",
            out.len(),
            out.iter().map(|c| c.content.as_str()).collect::<Vec<_>>()
        );
        // Each sub-chunk is exactly one paragraph; trailing CRLFs are trimmed.
        assert_eq!(out[0].content, "AAAAAAAAAA");
        assert_eq!(out[1].content, "BBBBBBBBBB");
        assert_eq!(out[2].content, "CCCCCCCCCC");
    }

    #[test]
    fn test_split_oversized_sentence_fallback() {
        // Single paragraph (no \n\n) with multiple sentences. Total chars 30 > max_chars 14
        // but each sentence individually <= 14, so output should split at sentence boundaries.
        // Mix Chinese and English terminators.
        let content = "这是第一句。这是第二句。end.";
        // chars: 6 + 6 + 4 = 16 > 12 (max). Each piece <= 12.
        let line_starts = build_line_starts(content);
        let parent = make_raw(content, 1, 1, 0);
        let out = split_oversized(vec![parent], content, &line_starts, 12);
        assert!(
            out.len() >= 2,
            "expected >= 2 sub-chunks, got {}",
            out.len()
        );
        for sub in &out {
            assert!(
                sub.content.chars().count() <= 12,
                "sub-chunk too large: {:?} ({} chars)",
                sub.content,
                sub.content.chars().count()
            );
        }
        // No sub-chunk should split mid-sentence: every sub-chunk should end with a terminator
        // OR be the remainder that still fits.
        let joined: String = out.iter().map(|c| c.content.as_str()).collect();
        assert!(joined.contains("这是第一句"));
        assert!(joined.contains("这是第二句"));
        assert!(joined.contains("end"));
    }

    #[test]
    fn test_split_oversized_force_char_boundary() {
        // No paragraph or sentence breaks; mixed Chinese + emoji multibyte chars.
        // 12 chars total, each multibyte. max_chars=4 forces 3 forced splits.
        let content = "中文😀文字😎更多内容🎉end";
        // chars: 中(1)文(2)😀(3)文(4)字(5)😎(6)更(7)多(8)内(9)容(10)🎉(11)e(12)n(13)d(14) = 14
        let line_starts = build_line_starts(content);
        let parent = make_raw(content, 1, 1, 0);
        let out = split_oversized(vec![parent], content, &line_starts, 4);
        assert!(out.len() >= 3);
        // Reassembly verifies UTF-8 was preserved (no replacement chars, no panic).
        let joined: String = out.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(joined, content);
        for sub in &out {
            assert!(sub.content.chars().count() <= 4);
            assert!(!sub.content.is_empty());
        }
    }

    #[test]
    fn test_split_oversized_line_numbers_accurate() {
        // (a) Multi-line paragraphs separated by \n\n: row ranges should not overlap and
        // should sit within the parent's line range.
        // Content lines:
        // 1: AAA
        // 2: AAA
        // 3: (blank)
        // 4: BBB
        // 5: BBB
        // 6: (blank)
        // 7: CCC
        // 8: CCC
        let content = "AAA\nAAA\n\nBBB\nBBB\n\nCCC\nCCC";
        let line_starts = build_line_starts(content);
        let parent = RawChunk {
            content: content.to_string(),
            line_start: 1,
            line_end: 8,
            heading_path: vec!["H".into()],
            byte_start: 0,
            byte_end: content.len(),
        };
        let out = split_oversized(vec![parent], content, &line_starts, 8);
        assert_eq!(out.len(), 3);
        // Each sub-chunk's row range must be within parent and not overlap with neighbours.
        assert_eq!(out[0].line_start, 1);
        assert_eq!(out[0].line_end, 2);
        assert_eq!(out[1].line_start, 4);
        assert_eq!(out[1].line_end, 5);
        assert_eq!(out[2].line_start, 7);
        assert_eq!(out[2].line_end, 8);

        // (b) Single source line cut by sentence boundary into multiple sub-chunks
        // should report overlapping line ranges (all on the shared line) but stay
        // within the parent's range.
        let content_b = "句子一。句子二。句子三。";
        // All on line 1 since no '\n'. chars: 4*3=12. max=4 forces split per sentence.
        let line_starts_b = build_line_starts(content_b);
        let parent_b = RawChunk {
            content: content_b.to_string(),
            line_start: 1,
            line_end: 1,
            heading_path: vec!["H".into()],
            byte_start: 0,
            byte_end: content_b.len(),
        };
        let out_b = split_oversized(vec![parent_b], content_b, &line_starts_b, 4);
        assert!(out_b.len() >= 3);
        for sub in &out_b {
            assert_eq!(sub.line_start, 1);
            assert_eq!(sub.line_end, 1);
        }
    }

    #[test]
    fn test_split_oversized_preserves_prefix() {
        // RawChunk-level "prefix" is the heading_path. Verify each sub-chunk inherits it.
        let content = "111\n\n222\n\n333";
        let line_starts = build_line_starts(content);
        let parent = RawChunk {
            content: content.to_string(),
            line_start: 1,
            line_end: 5,
            heading_path: vec!["顶级".into(), "二级".into()],
            byte_start: 0,
            byte_end: content.len(),
        };
        let out = split_oversized(vec![parent], content, &line_starts, 4);
        assert!(out.len() >= 2);
        for sub in &out {
            assert_eq!(
                sub.heading_path,
                vec!["顶级".to_string(), "二级".to_string()]
            );
        }
    }

    #[test]
    fn test_no_split_when_under_limit() {
        let content = "short content";
        let line_starts = build_line_starts(content);
        let parent = RawChunk {
            content: content.to_string(),
            line_start: 7,
            line_end: 9,
            heading_path: vec!["only".into()],
            byte_start: 0,
            byte_end: content.len(),
        };
        let out = split_oversized(vec![parent], content, &line_starts, 999);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].content, content);
        assert_eq!(out[0].line_start, 7);
        assert_eq!(out[0].line_end, 9);
        assert_eq!(out[0].heading_path, vec!["only".to_string()]);
    }
}
