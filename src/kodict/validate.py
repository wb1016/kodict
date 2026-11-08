"""Validation and frequency mapping for extracted Korean words."""

from __future__ import annotations

import math
import re
import unicodedata

# Hangul syllable block only (precomposed U+AC00..U+D7A3)
_HANGUL = re.compile(r"^[가-힣]+$")
# pure-Latin loanwords (konglish), e.g. "coffee", "Netflix"
_LATIN = re.compile(r"^[A-Za-z]+$")

MAX_WORD_LEN = 15
RANK_CAP = 300_000  # ranks beyond this map to the minimum frequency


def normalize(text: str) -> str:
    """NFC-normalize and strip surrounding whitespace."""
    return unicodedata.normalize("NFC", text).strip()


def is_valid_word(word: str) -> bool:
    """True for a plausible dictionary entry: Hangul or pure-Latin token."""
    if not word or len(word) > MAX_WORD_LEN:
        return False
    if _HANGUL.match(word):
        return True
    if _LATIN.match(word) and len(word) >= 2:
        return True
    return False


def rank_to_freq(rank: int, cap: int = RANK_CAP) -> int:
    """Map a global 1-based rank to AOSP's log frequency scale (0..255).

    rank 1 -> 255, rank == cap -> 1, everything beyond cap -> 1.
    """
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if rank >= cap:
        return 1
    value = round(255 * (1 - math.log10(rank) / math.log10(cap)))
    return min(255, max(1, value))


def clean_words(words: "list[str]") -> "list[str]":
    """NFC-normalize, dedupe (preserving order) and drop invalid entries."""
    seen: set[str] = set()
    result: list[str] = []
    for raw in words:
        word = normalize(raw)
        if word in seen or not is_valid_word(word):
            continue
        seen.add(word)
        result.append(word)
    return result
