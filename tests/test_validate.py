"""Tests for validation and rank → frequency mapping."""

import pytest

from kodict.validate import clean_words, is_valid_word, normalize, rank_to_freq


def test_normalize_nfc():
    # decomposed 한글 (U+1112 U+1161 U+11AB ...) -> precomposed
    assert normalize("  한글  ") == "한글"
    assert normalize("한글") == "한글"


@pytest.mark.parametrize("word", ["사과", "학교", "대한민국", "아이스크림", "커피"])
def test_valid_hangul(word):
    assert is_valid_word(word)


@pytest.mark.parametrize("word", ["coffee", "Netflix", "AI"])
def test_valid_loanwords(word):
    assert is_valid_word(word)


@pytest.mark.parametrize("word", [
    "", " ", "사과!", "한글123", "hello world", "안녕-하세요",
    "a", "a" * 16, "가" * 16, "😀", "abc123",
])
def test_invalid_words(word):
    assert not is_valid_word(word)


def test_rank_to_freq_bounds():
    assert rank_to_freq(1) == 255
    assert rank_to_freq(300_000) == 1
    assert rank_to_freq(10_000_000) == 1


def test_rank_to_freq_monotonic():
    ranks = [1, 10, 100, 1_000, 10_000, 100_000, 300_000]
    freqs = [rank_to_freq(r) for r in ranks]
    assert freqs == sorted(freqs, reverse=True)
    assert all(1 <= f <= 255 for f in freqs)


def test_rank_to_freq_rejects_zero():
    with pytest.raises(ValueError):
        rank_to_freq(0)


def test_clean_words():
    words = ["사과", "사과", " coffee ", "bad!", "학교"]
    assert clean_words(words) == ["사과", "coffee", "학교"]
