"""Golden-format tests for the .combined writer."""

import warnings

import pytest

from kodict.wordlist import Header, Wordlist, WordlistError


def make_wordlist() -> Wordlist:
    wl = Wordlist(Header(locale="ko", dict_type="main", description="Korean",
                         version=20, date=1000000000))
    wl.add_word("사과", 187)
    wl.add_word("학교", 150)
    wl.add_word("커피", 90)
    wl.add_bigram("사과", "먹다", 1)
    wl.add_bigram("사과", "주다", 2)
    wl.add_bigram("사과", "열다", 3)
    return wl


def test_header_format():
    header = Header(locale="ko", dict_type="main", description="Korean",
                    version=20, date=1000000000)
    assert header.write() == (
        "dictionary=main:ko,locale=ko,description=Korean,"
        "date=1000000000,version=20"
    )


def test_golden_output():
    wl = make_wordlist()
    assert wl.to_combined() == (
        "dictionary=main:ko,locale=ko,description=Korean,date=1000000000,version=20\n"
        " word=사과,f=187\n"
        "  bigram=먹다,f=210\n"
        "  bigram=주다,f=200\n"
        "  bigram=열다,f=190\n"
        " word=학교,f=150\n"
        " word=커피,f=90\n"
    )


def test_words_sorted_by_descending_freq():
    wl = make_wordlist()
    wl.add_word("아주", 200)
    words = [line.split("=")[1].split(",")[0]
             for line in wl.to_combined().splitlines() if line.startswith(" word=")]
    assert words == ["아주", "사과", "학교", "커피"]


def test_missing_bigram_parent_raises():
    wl = Wordlist(Header())
    with pytest.raises(WordlistError):
        wl.add_bigram("없는단어", "다음", 1)


def test_dangling_bigram_target_dropped():
    wl = make_wordlist()
    wl.add_bigram("사과", "고양이", 1)  # 고양이 not in wordlist
    wl.normalize()
    assert ("고양이", 1) not in wl._words["사과"].bigrams


def test_bigrams_limited_to_three_and_renumbered():
    wl = make_wordlist()
    for next_word in ["하나", "둘", "셋", "넷"]:
        wl.add_word(next_word, 50)
    for i, next_word in enumerate(["하나", "둘", "셋", "넷"], start=1):
        wl.add_bigram("학교", next_word, i)
    wl.normalize()
    bigrams = wl._words["학교"].bigrams
    assert bigrams == [("하나", 1), ("둘", 2), ("셋", 3)]


def test_duplicate_bigram_keeps_best_rank():
    wl = make_wordlist()
    wl.add_word("공부", 50)
    wl.add_bigram("학교", "공부", 3)
    wl.add_bigram("학교", "공부", 1)
    wl.normalize()
    assert wl._words["학교"].bigrams == [("공부", 1)]


def test_add_word_keeps_max_freq():
    wl = Wordlist(Header())
    wl.add_word("사과", 10)
    wl.add_word("사과", 50)
    wl.add_word("사과", 20)
    assert wl._words["사과"].f == 50


def test_f_clamped_to_255():
    wl = Wordlist(Header())
    wl.add_word("사과", 999)
    wl.normalize()
    assert wl._words["사과"].f == 255


def test_merge():
    a, b = make_wordlist(), Wordlist(Header())
    b.add_word("아주", 200)
    b.add_word("사과", 100)
    a.merge(b)
    assert len(a) == 4
    assert a._words["사과"].f == 187  # max kept


def test_write_requires_header():
    wl = Wordlist()
    wl.add_word("사과", 10)
    with pytest.raises(WordlistError):
        wl.to_combined()


def test_empty_description_rejected():
    with pytest.raises(WordlistError):
        Header(description="")


def test_comma_in_description_rejected():
    with pytest.raises(WordlistError):
        Header(description="Korean, words")


def test_low_version_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Header(version=5)
    assert any("version 5" in str(w.message) for w in caught)


def test_write_file(tmp_path):
    wl = make_wordlist()
    path = tmp_path / "main_ko.combined"
    wl.write(str(path))
    content = path.read_text(encoding="utf-8")
    assert content.startswith("dictionary=main:ko,")
    assert content.endswith(" word=커피,f=90\n")


def test_decompose_writes_nfd():
    import unicodedata
    wl = make_wordlist()
    wl.decompose = True
    out = wl.to_combined()
    line = next(l for l in out.splitlines() if l.startswith(" word=사과".replace("사과", unicodedata.normalize("NFD", "사과"))))
    written = line.split("=")[1].split(",")[0]
    assert written == unicodedata.normalize("NFD", "사과")
    assert unicodedata.normalize("NFC", written) == "사과"
    # bigram targets are decomposed identically (they must match word entries)
    bigram_line = next(l for l in out.splitlines() if l.startswith("  bigram="))
    b = bigram_line.split("=")[1].split(",")[0]
    assert b == unicodedata.normalize("NFD", "먹다")
    # words already in NFD are left untouched (idempotent)
    assert unicodedata.normalize("NFD", written) == written


def test_no_decompose_by_default():
    import unicodedata
    out = make_wordlist().to_combined()
    assert " word=사과,f=187" in out  # NFC as authored


def test_decompose_emits_nfc_mirror_for_bigram_words():
    import unicodedata
    wl = make_wordlist()
    wl.decompose = True
    lines = wl.to_combined().splitlines()
    nfd_word = unicodedata.normalize("NFD", "사과")
    nfd_idx = next(i for i, l in enumerate(lines) if l.startswith(f" word={nfd_word},"))
    # NFC twin exists with the low mirror frequency and the same bigrams
    nfc_idx = next(i for i, l in enumerate(lines) if l == " word=사과,f=1")
    nfd_bigrams = [l for l in lines[nfd_idx + 1:nfd_idx + 4] if l.startswith("  bigram=")]
    nfc_bigrams = [l for l in lines[nfc_idx + 1:nfc_idx + 4] if l.startswith("  bigram=")]
    assert nfd_bigrams == nfc_bigrams and len(nfc_bigrams) == 3
    # 학교 has no bigrams -> no mirror
    assert not any(l == " word=학교,f=1" for l in lines)


def test_no_mirrors_without_decompose():
    lines = make_wordlist().to_combined().splitlines()
    assert not any(l.endswith(",f=1") and l.startswith(" word=") for l in lines)
