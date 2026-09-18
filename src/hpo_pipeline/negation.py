from __future__ import annotations

import re

_PRE_NEGATION = re.compile(
    r"(?:\bno\b|\bnot\b|\bwithout\b|\bden(?:y|ies|ied)\b|"
    r"\bnegative (?:for|with respect to)\b|\bfree of\b|\babsence of\b|"
    r"\babsent\b|\bnever\b)",
    re.IGNORECASE,
)
_POST_NEGATION = re.compile(
    r"^\s*(?:[A-Z](?:\s*,\s*|\s+and\s+))*"
    r"(?:(?:was|were|is|are)\s+(?:negative|absent|within (?:the )?normal range)|"
    r"(?:has|have|had)\s+not\s+(?:occurred|developed|been present)|"
    r"(?:was|were)?\s*ruled out|not present)\b",
    re.IGNORECASE,
)
_SCOPE_BREAK = re.compile(r"[.;:]|\b(?:but|however|although|except|yet)\b", re.IGNORECASE)


def is_negated(text: str, start: int, end: int, window: int = 100) -> bool:
    left = text[max(0, start - window):start]
    right = text[end:min(len(text), end + 100)]
    right = re.split(r"[.;:]", right, maxsplit=1)[0]
    breaks = list(_SCOPE_BREAK.finditer(left))
    if breaks:
        left = left[breaks[-1].end():]
    matches = list(_PRE_NEGATION.finditer(left))
    if matches and len(left) - matches[-1].end() <= 70:
        return True
    return bool(_POST_NEGATION.search(right))
