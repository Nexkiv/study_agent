"""
Section detection for structured documents.

Detects section headers in raw text to preserve document structure
during chunking. Conservative heuristics — false negatives (missing a header)
are preferred over false positives (treating content as a header).
"""
import re


# Minimum content length for a section to be kept (avoids empty/trivial sections)
MIN_SECTION_LENGTH = 20


def _strip_parenthetical(text: str) -> str:
    """Strip parenthetical content from a line for header detection.

    E.g., "Spanish Renaissance (Also called mannerism in Spain. I will accept either)"
    → "Spanish Renaissance"
    """
    return re.sub(r'\s*\(.*?\)\s*', ' ', text).strip()


def _is_markdown_header(line: str) -> bool:
    """Check if a line is a markdown-style header (# Header)."""
    return bool(re.match(r'^#{1,4}\s+\S', line))


def _extract_markdown_title(line: str) -> str:
    """Extract title text from a markdown header line."""
    return re.sub(r'^#{1,4}\s+', '', line).strip()


def _is_title_case_header(line: str) -> bool:
    """
    Check if a line looks like a title-case section header.

    Strips parenthetical content before checking, so instructor notes
    like "(Also called mannerism in Spain)" don't affect detection.

    Criteria:
    - Under 120 characters (after stripping parentheticals)
    - At least 2 words
    - Most words are capitalized (allowing for small words like "of", "the", "and")
    - Not ending with sentence punctuation (., ?, !)
    """
    # Strip parenthetical content for the check
    stripped = _strip_parenthetical(line.strip())
    if not stripped or len(stripped) > 120:
        return False

    # Must not end with sentence punctuation
    if stripped[-1] in '.?!:':
        return False

    words = stripped.split()
    if len(words) < 2:
        return False

    # Small words that don't need to be capitalized
    small_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for',
                   'and', 'but', 'or', 'nor', 'with', 'by', 'as', 'also',
                   'ca', 'ca.', 'c.', 'vs'}

    # Count capitalized words (excluding small words)
    significant_words = [w for w in words if w.lower() not in small_words]
    if not significant_words:
        return False

    capitalized = sum(1 for w in significant_words if w[0].isupper())
    ratio = capitalized / len(significant_words)

    # At least 75% of significant words should be capitalized
    return ratio >= 0.75


def _is_all_caps_header(line: str) -> bool:
    """
    Check if a line is an ALL-CAPS header.

    Criteria:
    - Under 120 characters
    - At least 2 alphabetic characters
    - All alphabetic characters are uppercase
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False

    alpha_chars = [c for c in stripped if c.isalpha()]
    if len(alpha_chars) < 2:
        return False

    return all(c.isupper() for c in alpha_chars)


def detect_sections(text: str) -> list[tuple[str, str]]:
    """
    Detect document sections from raw text.

    Identifies section headers using conservative heuristics:
    1. Markdown headers (# Title, ## Title)
    2. ALL-CAPS lines (< 120 chars)
    3. Title-case lines (< 120 chars, preceded by blank line)

    Subsection headers (titles that repeat, e.g., "Terms, People, and Places to Know")
    are emitted with compound names: "Parent > Subsection". This preserves both the
    parent context and subsection identity for embedding.

    Args:
        text: Raw document text

    Returns:
        List of (section_title, section_text) tuples.
        If no sections detected, returns [("", text)].
        Section titles are cleaned (no markdown #, no extra whitespace).
    """
    if not text or not text.strip():
        return [("", text)]

    lines = text.split('\n')
    # Collect (line_index, title) for detected headers
    headers: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Check markdown headers first (most reliable)
        if _is_markdown_header(stripped):
            title = _extract_markdown_title(stripped)
            headers.append((i, title))
            continue

        # Check ALL-CAPS headers
        if _is_all_caps_header(stripped):
            # Must be preceded by a blank line or be at the start
            if i == 0 or not lines[i - 1].strip():
                headers.append((i, stripped))
                continue

        # For title-case check, strip parentheticals first
        check_line = _strip_parenthetical(stripped)
        if _is_title_case_header(check_line):
            if i == 0 or not lines[i - 1].strip():
                # Store the cleaned title (without parenthetical notes)
                headers.append((i, check_line))
                continue

    # No headers found — return entire text as one section
    if not headers:
        return [("", text)]

    # Identify subsection headers — headers that repeat across the document
    # (e.g., "Terms, People, and Places to Know" appears multiple times).
    title_counts: dict[str, int] = {}
    for _, title in headers:
        title_counts[title] = title_counts.get(title, 0) + 1

    subsection_titles = {title for title, count in title_counts.items() if count > 1}

    # Build sections with compound names for subsections
    sections: list[tuple[str, str]] = []

    # Content before the first header (preamble)
    if headers[0][0] > 0:
        preamble = '\n'.join(lines[:headers[0][0]]).strip()
        if len(preamble) >= MIN_SECTION_LENGTH:
            sections.append(("", preamble))

    current_parent_title = ""
    # Track where current parent's own content starts (after the header line)
    parent_content_start = headers[0][0] + 1

    for idx, (line_idx, title) in enumerate(headers):
        is_subsection = title in subsection_titles

        # Find where this header's content ends (at the next header)
        if idx + 1 < len(headers):
            next_header_line = headers[idx + 1][0]
        else:
            next_header_line = len(lines)

        if is_subsection:
            # Emit parent content accumulated before this subsection
            parent_text = '\n'.join(lines[parent_content_start:line_idx]).strip()
            if parent_text and len(parent_text) >= MIN_SECTION_LENGTH:
                sections.append((current_parent_title, parent_text))

            # Emit the subsection with a compound name
            subsection_text = '\n'.join(lines[line_idx + 1:next_header_line]).strip()
            if subsection_text:
                compound_title = f"{current_parent_title} > {title}" if current_parent_title else title
                sections.append((compound_title, subsection_text))

            # Parent content resumes after this subsection
            parent_content_start = next_header_line
        else:
            # New parent section — emit previous parent's remaining content
            if idx > 0:
                parent_text = '\n'.join(lines[parent_content_start:line_idx]).strip()
                if parent_text and len(parent_text) >= MIN_SECTION_LENGTH:
                    sections.append((current_parent_title, parent_text))

            current_parent_title = title
            parent_content_start = line_idx + 1

    # Emit the last parent section's remaining content
    last_text = '\n'.join(lines[parent_content_start:]).strip()
    if last_text:
        sections.append((current_parent_title, last_text))

    # If we detected headers but all sections were empty, fall back
    if not sections:
        return [("", text)]

    return sections
