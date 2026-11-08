"""Tests for the Wikipedia path: cleaning, tokenization, auth, mining,
and a parquet → shards → wordlist end-to-end run."""

import unicodedata
from collections import Counter
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kodict import wiki
from kodict.build import build_wordlist, load_shards
from kodict.config import Config, MineConfig, OutputConfig
from kodict.mine import break_short_cycles, mine, short_cycles

# ---------------------------------------------------------------- clean


def test_strips_templates_nested():
    text = "앞 {{cite web|title={{1990}}|url=x}} 뒤"
    (line,) = wiki.clean_wikitext(text)
    assert line.split() == ["앞", "뒤"]


def test_unbalanced_template_survives():
    text = "수식 {{ unclosed"
    assert "unclosed" in wiki.clean_wikitext(text)[0]


def test_strips_refs_comments_tags():
    text = "본문<ref name=\"x\">각주</ref>끝<br/>다음 <!-- 주석 -->글"
    (line,) = wiki.clean_wikitext(text)
    assert "각주" not in line
    assert "주석" not in line
    assert "본문" in line and "글" in line


def test_namespace_links_removed_regular_links_kept():
    text = "나라 [[분류:국가]] 도시 [[서울]] 메인 [[대한민국|한국]]"
    (line,) = wiki.clean_wikitext(text)
    assert "분류" not in line
    assert "서울" in line and "한국" in line


def test_link_replacement_invents_no_spaces():
    # rendered text has no space where markup abutted — a fake space
    # here would create a fake bigram
    (line,) = wiki.clean_wikitext("'''[[대한민국|한국]]이라고도 부른다.")
    assert line == "한국이라고도 부른다."


def test_heading_keeps_words_drops_marks():
    assert wiki.clean_wikitext("== 개요 ==") == ["개요"]


def test_tables_dropped_entirely():
    text = "앞 문장\n{| class=\"x\"\n| 1948 || 건국일\n|}\n뒤 문장"
    lines = wiki.clean_wikitext(text)
    assert lines == ["앞 문장", "뒤 문장"]
    assert not any("건국일" in line for line in lines)


def test_list_markers_stripped():
    assert wiki.clean_wikitext("* 첫 항목\n# 둘째 항목") == ["첫 항목", "둘째 항목"]


def test_external_links_keep_display_text_only():
    (line,) = wiki.clean_wikitext("참고 [https://example.com 문서] 여기")
    assert line == "참고 문서 여기"


# ---------------------------------------------------------------- tokens


def test_strip_edges():
    assert wiki.strip_edges("「사과」") == ("사과", "」")
    assert wiki.strip_edges("학교에...") == ("학교에", "...")
    assert wiki.strip_edges("그냥") == ("그냥", "")


def test_segments_break_on_sentence_and_invalid_tokens():
    segs = wiki.tokenize_line("그 사람이었다. 2024년 나는 갔다!")
    assert segs == [["그", "사람이었다"], ["나는", "갔다"]]


def test_mixed_script_tokens_break_the_chain():
    segs = wiki.tokenize_line("한국 KBO리그 좋다")
    assert segs == [["한국"], ["좋다"]]


# ---------------------------------------------------------------- auth


def test_token_prefers_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", " hf_env ")
    assert wiki.token() == "hf_env"


def test_token_falls_back_to_workspace_file(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    marker = tmp_path.parent / ".hftok"
    marker.write_text("hf_file\n", encoding="utf-8")
    try:
        assert wiki.token() == "hf_file"
    finally:
        marker.unlink()


def test_list_shards_falls_back_offline(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise httpx.HTTPError("offline")

    monkeypatch.setattr(wiki.httpx, "get", boom)
    shards = wiki.list_shards(make_cfg(tmp_path))
    assert [s.name for s in shards] == [n for n, _, _ in wiki.FALLBACK_SHARDS]
    assert all(len(s.sha256) == 64 for s in shards)


def test_parquet_paths_returns_cached_without_network(tmp_path):
    cfg = make_cfg(tmp_path)
    path = write_parquet(tmp_path, [("제목", "본문")])
    assert wiki.parquet_paths(cfg) == [path]


# ---------------------------------------------------------------- mine

ARTICLES = [
    ("대한민국", """대한민국은 동아시아의 국가이다. 수많은 사람들이 모인다.
'''대한민국'''은 [[대한민국|한국]]이라고도 부른다. {{출처 필요}} [[분류:국가]]
<ref name="x">리퍼런스</ref>
{| class="infobox"
| 1948 || 건국일
|}
== 개요 ==
* 첫 번째 항목
"""),
    ("사람", """그 학교는 커다란 건물이 있다. 사람 도 함께 갔다.
그 사람이 먼저 도착했다. 미국의 수도는 워싱턴이다.
사람과 자연이 중요하다.
"""),
]


def make_cfg(tmp_path, **mine_over) -> Config:
    mine_kw = dict(
        cache_dir=str(tmp_path / "cache"),
        out_dir=str(tmp_path / "shards"),
        min_count=1,
        max_words=1000,
        bigram_parents=100,
        bigram_children=100,
        min_bigram_count=1,
        shard_words=3,
    )
    mine_kw.update(mine_over)
    return Config(
        output=OutputConfig(dir=str(tmp_path / "out")),
        mine=MineConfig(**mine_kw),
    )


def write_parquet(tmp_path, articles) -> "Path":
    table = pa.table({
        "id": [str(i) for i in range(len(articles))],
        "url": [f"https://ko.wikipedia.org/wiki/{t}" for t, _ in articles],
        "title": [t for t, _ in articles],
        "text": [text for _, text in articles],
    })
    path = tmp_path / "cache" / "train-00000-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def test_mine_end_to_end(tmp_path):
    cfg = make_cfg(tmp_path)
    stats = mine(cfg, [write_parquet(tmp_path, ARTICLES)])

    assert stats["words"] > 0
    assert stats["shards"] == -(-stats["words"] // cfg.mine.shard_words)
    ranks, followers = load_shards(Path(cfg.mine.out_dir))
    assert ranks and min(ranks.values()) == 1
    assert "사람" in followers

    wl = build_wordlist(cfg, Path(cfg.mine.out_dir))

    # unigrams from cleaned prose, markup content excluded
    for word in ("대한민국은", "사람이", "사람들이", "개요", "항목", "한국이라고도"):
        assert word in wl, word
    for junk in ("출처", "리퍼런스", "건국일", "분류"):
        assert junk not in wl, junk

    # THE regression: 사람 must not suggest 이 (particles attach without
    # a space in Korean, so that pair cannot occur across a real space)
    people = dict(wl._words["사람"].bigrams)
    assert "이" not in people
    assert "도" in people  # "사람 도 함께 갔다" — a real space-separated pair

    # sentence/line boundaries never form bigrams
    assert wl._words["모인다"].bigrams == []
    assert "수많은" not in dict(wl._words["국가이다"].bigrams)

    # no slot-position graph may contain a short loop
    for pos in range(3):
        graph = {w: cs[pos] for w, cs in followers.items() if len(cs) > pos}
        assert short_cycles(graph) == []


def test_build_writes_nfd_and_nfc_mirrors(tmp_path):
    cfg = make_cfg(tmp_path)
    mine(cfg, [write_parquet(tmp_path, ARTICLES)])
    wl = build_wordlist(cfg, Path(cfg.mine.out_dir))
    out = tmp_path / "out" / "main_ko.combined"
    out.parent.mkdir(parents=True, exist_ok=True)
    wl.write(str(out))
    text = out.read_text(encoding="utf-8")

    nfc = "사람이"
    nfd = unicodedata.normalize("NFD", nfc)
    assert f"word={nfd},f=" in text          # primary entry is decomposed
    assert f"word={nfc},f=1" in text          # NFC mirror carries bigrams


# ---------------------------------------------------------------- cycles


def _counts(links: dict[str, list[str]]) -> tuple[list[Counter], list[str],
                                                   list[str], dict[str, int]]:
    """Fixture plumbing for break_short_cycles: evidence falls with the
    child's rank, ties broken by parent frequency (rarest loses)."""
    parents = list(links)
    children = sorted({c for cs in links.values() for c in cs})
    pidx = {w: i for i, w in enumerate(parents)}
    cidx = {w: i for i, w in enumerate(children)}
    counts: list[Counter] = [Counter() for _ in parents]
    for parent, cs in links.items():
        for i, child in enumerate(cs, 1):
            counts[pidx[parent]][cidx[child]] = 10 - i
    ranks = {w: i for i, w in enumerate(parents, 1)}
    return counts, parents, children, ranks


def test_short_cycles_finds_small_loops_only():
    assert short_cycles({"a": "b", "b": "c", "c": "a"}) == [["a", "b", "c"]]
    assert short_cycles({"x": "x"}) == [["x"]]
    long_cycle = {f"w{i}": f"w{(i + 1) % 5}" for i in range(5)}
    assert short_cycles(long_cycle) == []


def test_break_short_cycles_kills_the_observed_device_loop():
    # the on-device loop: strip slot 1 renders our rank 2, so tapping it
    # walked 사람 → 중 → 한 → 작품 → 중
    links = {
        "사람": ["등을", "중", "모두"],
        "중": ["하나이다", "한", "가장"],
        "한": ["영화", "작품", "번"],
        "작품": ["목록", "중", "번호", "많다"],   # 4th = backfill candidate
    }
    counts, parents, children, ranks = _counts(links)
    assert break_short_cycles(links, counts, parents, children, ranks) == 1
    # the loop-closing edge went, and the slot refilled from the extra candidate
    assert "중" not in links["작품"]
    assert links["작품"] == ["목록", "번호", "많다"]
    assert links["사람"] == ["등을", "중", "모두"]   # untouched outside the cycle
    for pos in range(3):
        graph = {w: cs[pos] for w, cs in links.items() if len(cs) > pos}
        assert short_cycles(graph) == []


def test_break_short_cycles_removes_self_loop():
    links = {"No": ["No", "the", "and"], "the": ["end", "of", "world"]}
    counts, parents, children, ranks = _counts(links)
    assert break_short_cycles(links, counts, parents, children, ranks) == 1
    assert links["No"] == ["the", "and"]


def test_break_short_cycles_leaves_long_cycles_alone():
    words = [f"w{i}" for i in range(5)]
    links = {w: [f"a{i}", words[(i + 1) % 5], f"b{i}"]
             for i, w in enumerate(words)}   # rank-2 graph is a 5-cycle
    before = {w: list(cs) for w, cs in links.items()}
    counts, parents, children, ranks = _counts(links)
    assert break_short_cycles(links, counts, parents, children, ranks) == 0
    assert links == before
