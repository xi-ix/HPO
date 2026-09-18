from __future__ import annotations

import re

_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)
_SPACE_RE = re.compile(r"\s+")


def normalize_equal_length(text: str) -> str:
    """Normalize case and punctuation without changing character offsets."""
    return text.translate(_TRANSLATION).lower()


def normalize_key(text: str) -> str:
    return _SPACE_RE.sub(" ", normalize_equal_length(text).strip())


def is_word_character(character: str) -> bool:
    return character.isalnum() or character == "_"


def inflection_variants(surface: str) -> set[str]:
    """Return conservative singular/plural variants of the final alphabetic word."""
    match = re.search(r"([a-zA-Z]+)$", surface)
    if not match or len(match.group(1)) < 4:
        return set()
    word = match.group(1)
    stem = surface[: match.start(1)]
    variants = set()
    lower = word.lower()
    if lower.endswith("ies") and len(word) > 4:
        variants.add(stem + word[:-3] + "y")
    elif lower.endswith(("ches", "shes", "ses", "xes", "zes")):
        variants.add(stem + word[:-2])
    elif lower.endswith("s") and not lower.endswith(("ss", "us", "is")):
        variants.add(stem + word[:-1])
    elif lower.endswith("y") and len(word) > 1 and lower[-2] not in "aeiou":
        variants.add(stem + word[:-1] + "ies")
    elif lower.endswith(("ch", "sh", "x", "z")):
        variants.add(stem + word + "es")
    elif not lower.endswith(("s", "us", "is")):
        variants.add(stem + word + "s")
    return variants
