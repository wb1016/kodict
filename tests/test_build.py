"""Tests for the .combined assembly and .dict compilation."""

import shutil
from pathlib import Path

import pytest

from kodict.build import BuildError, build, build_wordlist, load_shards, verify_dict
from kodict.config import Config, OutputConfig

JAR = Path("deps/aosp-dictionary-tools/dicttool_aosp.jar")
needs_java = pytest.mark.skipif(
    shutil.which("java") is None or not JAR.is_file(),
    reason="java or dicttool_aosp.jar missing (run scripts/fetch_deps.sh)",
)


def write_shard(raw_dir: Path, name: str, lines: list[str]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_shards(raw_dir: Path) -> None:
    write_shard(raw_dir, "freq-000000.jsonl", [
        '{"w": "사과", "r": 1}',
        '{"w": "학교", "r": 2}',
        '{"w": "먹다", "r": 6}',
        '{"w": "주다", "r": 7}',
        '{"w": "열다", "r": 8}',
        '{"w": "coffee", "r": 50}',
        '{"w": "사과!", "r": 3}',      # invalid: punctuation
        '{"w": "wrong", "r": "x"}',    # invalid: rank not an int
        '{"w": "한글", "r": 4}',       # decomposed NFC form of 한글
    ])
    write_shard(raw_dir, "cat-001.jsonl", [
        '{"w": "한글", "r": 100001}',  # duplicate of decomposed, worse rank
        '{"w": "나무", "r": 5}',
    ])
    write_shard(raw_dir, "bigram-0000.jsonl", [
        '{"w": "사과", "b": ["먹다", "주다"]}',
        '{"w": "사과", "b": ["열다", "고양이"]}',   # merged: 4 candidates
        '{"w": "없는단어", "b": ["사과"]}',          # parent not extracted
        '{"w": "학교", "b": "not-a-list"}',
    ])


def test_load_shards_merges_ranks_and_candidates(tmp_path):
    make_shards(tmp_path / "raw")
    ranks, followers = load_shards(tmp_path / "raw")
    assert ranks["사과"] == 1
    assert ranks["한글"] == 4        # NFC-normalized, best rank kept
    assert ranks["나무"] == 5
    assert followers["사과"] == ["먹다", "주다", "열다", "고양이"]


def test_build_wordlist(tmp_path):
    make_shards(tmp_path / "raw")
    wl = build_wordlist(Config(), tmp_path / "raw")
    # invalid entries dropped, NFC duplicate merged
    assert "사과!" not in wl and "wrong" not in wl
    assert "사과" in wl and "한글" in wl and len(wl) == 8
    # rank -> log frequency: rank 1 beats rank 50
    assert wl._words["사과"].f > wl._words["coffee"].f
    # bigrams: parent must exist, dangling targets dropped, top-3 kept
    assert wl._words["사과"].bigrams == [("먹다", 1), ("주다", 2), ("열다", 3)]
    assert "고양이" not in wl
    assert wl._words["학교"].bigrams == []
    combined = wl.to_combined()
    assert combined.startswith("dictionary=main:ko,locale=ko,description=Korean,")
    # decompose defaults on: words are written NFD (canonical jamo), which
    # NFC-normalizes back to the original for assertions
    import unicodedata
    written = {unicodedata.normalize("NFC", line.split("=")[1].split(",")[0])
               for line in combined.splitlines() if line.startswith(" word=")}
    assert "사과" in written
    bigrams = {unicodedata.normalize("NFC", line.split("=")[1].split(",")[0])
               for line in combined.splitlines() if line.startswith("  bigram=")}
    assert "먹다" in bigrams
    # ...and the raw output really is decomposed (jamo, not syllables)
    raw_word_line = next(l for l in combined.splitlines()
                         if l.startswith(" word=") and l.split("=")[1].split(",")[0] != "coffee")
    raw_word = raw_word_line.split("=")[1].split(",")[0]
    assert unicodedata.normalize("NFC", raw_word) != raw_word  # NFD != NFC for Hangul


def test_build_wordlist_empty_shards_raises(tmp_path):
    (tmp_path / "raw").mkdir()
    with pytest.raises(BuildError, match="run `kodict extract`"):
        build_wordlist(Config(), tmp_path / "raw")


def test_build_wordlist_all_invalid_raises(tmp_path):
    write_shard(tmp_path / "raw", "x.jsonl", ['{"w": "oops!", "r": 1}'])
    with pytest.raises(BuildError, match="invalid"):
        build_wordlist(Config(), tmp_path / "raw")


def test_missing_jar_message(tmp_path):
    from kodict.build import compile_dict
    cfg = Config(output=OutputConfig(dicttool="does/not/exist.jar"))
    combined = tmp_path / "x.combined"
    combined.write_text("x", encoding="utf-8")
    with pytest.raises(BuildError, match="fetch_deps"):
        compile_dict(cfg, combined, tmp_path / "out")


@needs_java
def test_full_build_end_to_end(tmp_path):
    make_shards(tmp_path / "raw")
    cfg = Config(output=OutputConfig(dicttool=str(JAR), dir=str(tmp_path / "out")))
    combined, dict_path, words, size = build(cfg, tmp_path / "raw",
                                             tmp_path / "out")
    assert words == 8
    assert size == dict_path.stat().st_size > 0
    assert combined.is_file() and combined.suffix == ".combined"
    assert dict_path.is_file() and dict_path.name == "main_ko.dict"


@needs_java
def test_verify_dict_checks_header(tmp_path):
    make_shards(tmp_path / "raw")
    cfg = Config(output=OutputConfig(dicttool=str(JAR)))
    from kodict.build import compile_dict
    wl = build_wordlist(cfg, tmp_path / "raw")
    combined = tmp_path / "small.combined"
    wl.write(str(combined))
    dict_path = compile_dict(cfg, combined, tmp_path)
    assert verify_dict(dict_path) == dict_path.stat().st_size


def test_verify_dict_rejects_bad_magic(tmp_path):
    bad = tmp_path / "bad.dict"
    bad.write_bytes(b"NOPE" + b"\x00" * 10)
    with pytest.raises(BuildError, match="magic"):
        verify_dict(bad)
