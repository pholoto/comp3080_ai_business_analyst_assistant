"""Helpers for inferring section headings from plain text."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

HeadingMetadata = Dict[str, Any]

_NUMERIC_PATTERN = re.compile(r"^(?P<identifier>\d+(?:\.\d+)*)(?:[\.)-])?\s+(?P<title>.+)")
_ROMAN_PATTERN = re.compile(
    r"^(?P<identifier>[IVXLCDM]+)(?:[\.)-])?\s+(?P<title>.+)", re.IGNORECASE
)
_UPPER_ALPHA_PATTERN = re.compile(r"^(?P<identifier>[A-Z])(?:[\.)-])?\s+(?P<title>.+)")
_LOWER_ALPHA_PATTERN = re.compile(r"^(?P<identifier>[a-z])(?:[\.)-])?\s+(?P<title>.+)")
_UPPERCASE_WORDS_PATTERN = re.compile(r"^(?P<title>[A-Z][A-Z\s]{3,})$")
_BULLET_PATTERN = re.compile(r"^(?P<identifier>[\-\*•])\s+(?P<title>.+)")

_HEADING_PATTERNS = (
    _NUMERIC_PATTERN,
    _ROMAN_PATTERN,
    _UPPER_ALPHA_PATTERN,
    _LOWER_ALPHA_PATTERN,
    _UPPERCASE_WORDS_PATTERN,
    _BULLET_PATTERN,
)

__all__ = ["detect_section_heading", "match_heading_line"]


def detect_section_heading(text: str, *, max_lines: int = 4) -> Optional[HeadingMetadata]:
    """Return heading metadata inferred from the first *max_lines* of *text*."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:max_lines]:
        heading = match_heading_line(line)
        if heading:
            return heading
    return None


def match_heading_line(line: str) -> Optional[HeadingMetadata]:
    """Match a single line against known heading patterns."""
    stripped = line.strip()
    if not stripped:
        return None

    for pattern in _HEADING_PATTERNS:
        match = pattern.match(stripped)
        if not match:
            continue

        groups = match.groupdict()
        identifier = groups.get("identifier")
        if pattern is _BULLET_PATTERN:
            identifier = None

        title = groups.get("title") or stripped
        title = _normalise_title(title)

        heading: HeadingMetadata = {
            "heading": stripped,
            "identifier": _normalise_identifier(identifier),
            "title": title,
        }
        heading["path"] = _derive_path(heading.get("identifier"), title)
        return heading

    return None


def _normalise_title(title: str) -> str:
    cleaned = " ".join(title.split())
    return cleaned.strip()


def _normalise_identifier(identifier: Optional[str]) -> Optional[str]:
    if not identifier:
        return None

    stripped = identifier.strip()
    if not stripped:
        return None

    if stripped.isalpha() and len(stripped) == 1:
        return stripped.upper()
    if re.fullmatch(r"[IVXLCDM]+", stripped, re.IGNORECASE):
        return stripped.upper()

    return stripped


def _derive_path(identifier: Optional[str], title: str) -> List[str]:
    cleaned = _normalise_title(title)
    tokens: List[str] = []

    if identifier:
        collapsed = identifier.replace(" ", "")
        if collapsed.replace(".", "").isdigit():
            tokens = [part for part in collapsed.split(".") if part]
        elif re.fullmatch(r"[IVXLCDM]+", collapsed, re.IGNORECASE):
            tokens = [collapsed.upper()]
        elif len(collapsed) == 1 and collapsed.isalpha():
            tokens = [collapsed.upper()]
        else:
            tokens = [collapsed]

    if cleaned and cleaned not in tokens:
        tokens.append(cleaned)

    return tokens or ([cleaned] if cleaned else [])
