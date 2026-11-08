"""Two-pass frequency mining over the Wikipedia corpus.

Pass 1 counts eojeol unigrams (per shard, with a memory-bounding hapax
prune) and derives the bigram keep-sets: the ``bigram_parents`` most
frequent eojeols may carry next-word data, the ``bigram_children`` most
frequent ones may be suggested as next words. Pass 2 re-reads the corpus
and counts parent→child pairs, but only *within* token segments — so a
pair exists only if the two words were actually adjacent across a single
space in real text.

Output: JSONL shards in the exact ``{"w": word, "r": rank,
"b": [next, ...]}`` shape ``build.load_shards`` already consumes; ranks
are corpus counts sorted, bigrams are count-ranked children (build maps
them to BIGRAM_F 210/200/190), after ``break_short_cycles`` has removed
the short loops a tap-the-same-slot user could walk into.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from kodict import wiki
from kodict.config import Config


# shard-local types above this → the shard is "big" and its hapax are
# pruned when merged (memory guard for the real corpus; small shards
# merge in full so tiny test corpora lose nothing)
PRUNE_TYPES = 500_000

# candidates kept per parent, one more than the strip shows, so a cycle
# broken below can be backfilled instead of shrinking the suggestion set
MAX_LINKS = 4

# chains that tap the same strip slot walk our rank-N graph (the device
# renders slots as [rank2, rank1, rank3]), and any deterministic bigram
# graph has cycles — tapping slot 1 forever through a function-word
# cycle looped on device. Drop the weakest edge of every cycle up to
# this length; longer cycles are harmless and kept.
SHORT_CYCLE = 3


def _log(msg: str) -> None:
    print(msg, flush=True)


def count_words(cfg: Config, paths: list[Path]) -> Counter:
    """Pass 1: unigram counts.

    Counts are accumulated per shard. Merging a *large* shard
    (``> PRUNE_TYPES`` types) drops its hapax: wiki typos/one-off names
    seen once in the shard and never before, which bounds how much the
    global counter can grow. Small shards merge in full. With
    ``min_count >= 3`` the only loss on the real corpus is a word
    spread as 1+1+1 over all three shards — worth the memory.
    """
    total: Counter = Counter()
    for path in paths:
        shard: Counter = Counter()
        articles = 0
        for title, text in wiki.iter_articles(path):
            for segment in wiki.iter_segments(title, text):
                shard.update(segment)
            articles += 1
            if articles % 50_000 == 0:
                _log(f"[pass 1] {path.name}: {articles} articles, "
                     f"{len(total) + len(shard)} types seen")
        if len(shard) > PRUNE_TYPES:
            merge = {w: c for w, c in shard.items() if c > 1 or w in total}
        else:
            merge = dict(shard)
        total.update(merge)
        _log(f"[pass 1] {path.name}: {articles} articles done, "
             f"{len(shard)} types ({len(merge)} kept, hapax pruned)")
    _log(f"[pass 1] total: {len(total)} distinct words")
    return total


def rank_words(cfg: Config, words: Counter) -> list[tuple[str, int]]:
    """``min_count``-filtered, count-descending, capped at ``max_words``."""
    kept = [(w, c) for w, c in words.items() if c >= cfg.mine.min_count]
    kept.sort(key=lambda kv: (-kv[1], kv[0]))
    return kept[: cfg.mine.max_words]


def count_bigrams(cfg: Config, paths: list[Path],
                  parents: list[str], children: list[str]) -> list[Counter]:
    """Pass 2: ``counts[pid][cid]`` adjacency counts, segments only.

    Tokens outside the keep-sets break adjacency (a real intervening
    word must never be skipped over) — matching how the keyboard sees
    context: only the immediately previous committed word qualifies.
    """
    roles: dict[str, tuple[int, int]] = {}
    for i, w in enumerate(parents):
        roles[w] = (i, -1)
    for i, w in enumerate(children):
        pid, _ = roles.get(w, (-1, -1))
        roles[w] = (pid, i)
    counts: list[Counter] = [Counter() for _ in parents]

    for path in paths:
        articles = 0
        pairs = 0
        for title, text in wiki.iter_articles(path):
            for segment in wiki.iter_segments(title, text):
                mapped = [roles.get(tok) for tok in segment]
                for a, b in zip(mapped, mapped[1:]):
                    if a is None or b is None:
                        continue
                    pid, _ = a
                    _, cid = b
                    if cid < 0 or pid < 0:
                        continue
                    counts[pid][cid] += 1
                    pairs += 1
            articles += 1
            if articles % 50_000 == 0:
                _log(f"[pass 2] {path.name}: {articles} articles, "
                     f"{pairs} candidate pairs counted")
        _log(f"[pass 2] {path.name}: {articles} articles done, "
             f"{pairs} candidate pairs")
    return counts


def collect_links(cfg: Config, words: list[tuple[str, int]],
                  counts: list[Counter], parents: list[str],
                  children: list[str]) -> dict[str, list[str]]:
    """Per parent: top-``MAX_LINKS`` count-ranked children passing the
    evidence floor; both ends must have made the wordlist. The caller
    trims to the three the strip can show, after cycle-breaking has had
    its chance to backfill a dropped slot."""
    word_set = {w for w, _ in words}
    links: dict[str, list[str]] = {}
    for pid, parent in enumerate(parents):
        if parent not in word_set:
            continue
        candidates = [
            (children[cid], n) for cid, n in counts[pid].items()
            if n >= cfg.mine.min_bigram_count and children[cid] in word_set
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda kv: (-kv[1], kv[0]))
        links[parent] = [w for w, _ in candidates[:MAX_LINKS]]
    return links


def short_cycles(graph: dict[str, str], max_len: int = SHORT_CYCLE
                 ) -> list[list[str]]:
    """Cycles of length ≤ ``max_len`` in a functional graph (one
    successor per node). Longer cycles are left in place."""
    seen: set[str] = set()
    found: list[list[str]] = []
    for start in graph:
        if start in seen:
            continue
        path: list[str] = []
        at: dict[str, int] = {}
        node = start
        while node in graph and node not in seen:
            seen.add(node)
            at[node] = len(path)
            path.append(node)
            node = graph[node]
        if node in at:
            cycle = path[at[node]:]
            if len(cycle) <= max_len:
                found.append(cycle)
    return found


def break_short_cycles(links: dict[str, list[str]], counts: list[Counter],
                       parents: list[str], children: list[str],
                       ranks: dict[str, int]) -> int:
    """Remove every short cycle from each slot-position graph.

    Slot *p* of the strip shows our rank *p*+1 link, so a user tapping
    the same slot repeatedly walks ``links[word][p]``; that graph is
    broken by dropping the least-evidenced edge of each cycle (rarer
    parent loses a tie) — the dropped child's slot then backfills from
    the extra candidate ``collect_links`` kept. Returns links dropped.
    """
    pidx = {w: i for i, w in enumerate(parents)}
    cidx = {w: i for i, w in enumerate(children)}
    drops = 0
    while True:
        victim: tuple[str, str] | None = None
        for pos in range(3):
            graph = {w: cs[pos] for w, cs in links.items() if len(cs) > pos}
            for cycle in short_cycles(graph):
                best: tuple[tuple[int, int], str, str] | None = None
                for i, parent in enumerate(cycle):
                    child = cycle[(i + 1) % len(cycle)]
                    key = (counts[pidx[parent]][cidx[child]],
                           -ranks.get(parent, 0))
                    if best is None or key < best[0]:
                        best = (key, parent, child)
                assert best is not None
                victim = best[1], best[2]
                break
            if victim:
                break
        if victim is None:
            return drops
        parent, child = victim
        links[parent].remove(child)
        drops += 1


def write_shards(cfg: Config, words: list[tuple[str, int]],
                 links: dict[str, list[str]]) -> tuple[Path, int]:
    """Emit JSONL shards for ``build``; returns (out_dir, shard count)."""
    out_dir = Path(cfg.mine.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jsonl"):
        stale.unlink()
    shard_id, in_shard = 0, 0
    handle = None
    shards = 0
    try:
        for rank, (word, _) in enumerate(words, start=1):
            if handle is None or in_shard >= cfg.mine.shard_words:
                if handle is not None:
                    handle.close()
                shard_id += 1
                shards += 1
                path = out_dir / f"shard-{shard_id:05d}.jsonl"
                handle = path.open("w", encoding="utf-8")
                in_shard = 0
            entry = {"w": word, "r": rank}
            if word in links:
                entry["b"] = links[word]
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            in_shard += 1
    finally:
        if handle is not None:
            handle.close()
    return out_dir, shards


def mine(cfg: Config, paths: list[Path]) -> dict:
    """Full mining run; returns stats for the CLI/status."""
    words_counter = count_words(cfg, paths)
    ranked = rank_words(cfg, words_counter)
    parents = [w for w, _ in ranked[: cfg.mine.bigram_parents]]
    children = [w for w, _ in ranked[: cfg.mine.bigram_children]]
    _log(f"[keep sets] {len(parents)} parents, {len(children)} children")
    counts = count_bigrams(cfg, paths, parents, children)
    links = collect_links(cfg, ranked, counts, parents, children)
    ranks = {w: i for i, (w, _) in enumerate(ranked, start=1)}
    dropped = break_short_cycles(links, counts, parents, children, ranks)
    if dropped:
        _log(f"[cycles] dropped {dropped} links closing short "
             "prediction loops")
    links = {w: cs[:3] for w, cs in links.items() if cs}
    out_dir, shards = write_shards(cfg, ranked, links)
    stats = {
        "articles_types": len(words_counter),
        "words": len(ranked),
        "parents": len(parents),
        "children": len(children),
        "parents_with_bigrams": len(links),
        "bigram_links": sum(len(v) for v in links.values()),
        "links_dropped": dropped,
        "shards": shards,
        "out_dir": str(out_dir),
    }
    _log("[mine] {words} words, {parents_with_bigrams} parents with bigrams, "
         "{bigram_links} links -> {shards} shards in {out_dir}".format(**stats))
    return stats