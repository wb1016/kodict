# kodict

Wikipedia-mined Korean dictionary for AOSP keyboard forks.

Words and next-word (bigram) data
are mined from the Korean Wikipedia(`wikimedia/wikipedia`, pinned subdir `20231101.ko`: 647,897 articles, 783 MB of parquet)
by two counting passes, assembled into `main_ko.combined` and compiled into `main_ko.dict` with AOSP's own `dicttool`.
Current result: 100,000 words + 138,837 next-word links over 47,055 parents -> 2.5 MB dict.

There is currently no `main_ko` dictionary in [aosp-dictionaries](https://codeberg.org/Helium314/aosp-dictionaries),
and this repo is made to fill that absence.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
bash scripts/fetch_deps.sh          # ~400 KB sparse checkout of dicttool_aosp.jar
# optional, avoids HuggingFace download rate limits:
#   echo hf_... > .hftok  (or export HF_TOKEN=hf_...)
```

## Usage

```bash
uv run python -m kodict fetch    # download the corpus (sha256-verified, resumable)
uv run python -m kodict mine     # two passes -> state/wiki/shards/*.jsonl
uv run python -m kodict build    # out/main_ko.combined + out/main_ko.dict
uv run python -m kodict status   # corpus + shard overview
uv run pytest                    # test suite (offline, 76 tests)
```

`mine` is idempotent and takes a few minutes; `build` reads whatever is
in `state/wiki/shards` (override with `--raw-dir`).

## How it works

```
- kodict fetch
- - state/wiki/*.parquet (3 shards, 783 MB, sha256 verified)

- kodict mine

- - pass 1
- - - clean wikitext
- - - eojeol tokens
- - - unigram counts (2.17M types)
- - - keep sets: 50k parents / 50k children may carry bigrams

- - pass 2
- - - segment-local adjacency counts (only both-in-keep-set pairs)
- - - evidence floor >= 3, top 3 per parent (a 4th candidate is held back as backfill for cycle-breaking)

- state/wiki/shards/*.jsonl      {"w": word, "r": rank, "b": [next...]}

- build
- - NFC validate
- - rank
- - log frequency (0..255)
- - NFD write + NFC mirrors

- out/main_ko.combined
- java -jar dicttool_aosp.jar makedict
- out/main_ko.dict      (AOSP magic 0x9BC13AFE)
```

Cleaning (`wiki.py`): templates, references, tables, headings, links, HTML and emphasis markers are stripped before counting.
and replacement does not invents a space: rendered text has no space where markup abutted.
`[[대한민국|한국]]이라고도` stays one token. A fake space there would become a fake bigram.

Segments are the unit of mining: a bigram is only formed between two tokens of the same segment,
so predictions never cross a line, a sentence end (`학교에 갔다.`),
or a dropped token (numbers, mixed script, bare punctuation).

Keep sets bound memory: only the most frequent eojeols may carry or receive next-word data (`[mine] bigram_parents` / `bigram_children`).
At 50k/50k roughly half the wordlist carries next-word data; the original 10k setting left every mid-frequency context falling back to history unigrams.

Cycles: tapping one strip slot repeatedly walks a single rank graph,
and every deterministic bigram graph has cycles (the device renders slots as `[rank2, rank1, rank3]`, so the leftmost is our rank 2).
After counting, `mine` drops the least-evidenced edge of every cycle of length <= 3; self-loops like `No -> No` included.
from each slot graph and backfills from the held-back candidate, so a loop such as `중 한 작품 중 ...` can't survive.

Frequency: corpus count -> global rank -> log frequency 0..255
(`rank_to_freq`), the `f` field of the AOSP format.

## Repository layout

```
config.toml               [output] dict header/NFD, [mine] corpus + thresholds
scripts/fetch_deps.sh     blobless sparse clone of dicttool_aosp.jar @ pinned SHA
src/kodict/
  wiki.py                 HF auth + download (sha256), wikitext cleaning,
                          eojeol tokenization into segments
  mine.py                 two-pass counting, keep sets, shard emission
  validate.py             NFC, word validation, rank -> frequency mapping
  wordlist.py             .combined writer (written from the documented format)
  build.py                assembly + dicttool compilation + header verification
  cli.py                  fetch / mine / build / status
tests/                    76 tests, all offline
state/wiki/               corpus cache + mined shards (gitignored)
```

## Format notes

Words are written NFD (decomposed jamo), not NFC; this is required.
LeanType (and HeliBoard-lineage keyboards) wrap Korean dictionaries in `KoreanDictionary`,
which NFD-decomposes the typed word into canonical jamo (`사` -> `ᄉ`+`ᅡ`, U+1109 U+1161) before the native trie lookup.
the native code passes code points through with no normalization, and converts candidates back to NFC for display.
An NFC dictionary therefore never matches at the trie root and Korean prediction silently returns nothing.
Validation and dedup still run in NFC (normalization is idempotent); only the final `.combined` output is decomposed (`[output] decompose = true`).

Two lookups hit the same native trie, so bigram-carrying words additionally get an NFC mirror entry (frequency 1).
the typed word is transformed to NFD, but the ngram context (the previous committed word, used for next-word prediction) is passed through untransformed.
without the mirror, `getWordId` misses and next-word prediction silently degrades to context-free history unigrams.
Bigram ranks are also written as real probabilities (`BIGRAM_F`: 210/200/190) instead of 1/2/3,
because LeanType boosts history unigrams to ~182; an AOSP-sample-style `f=1..3` can never win there,
and the app's history-damping only activates above 180.

`.combined` (from aosp-dictionaries' README):

```
dictionary=main:ko,locale=ko,description=Korean,date=...,version=20
 word=사과,f=187
  bigram=먹다,f=1
```

`f` is the log frequency, 0-255 (higher = more likely to be suggested);
`version` must be >= 18 for all AOSP keyboards. The `.combined` writer here is an independent implementation of that documented format.

## Licenses

kodict itself is [CC0-1.0](LICENSE).

The corpus is [Korean Wikipedia](https://ko.wikipedia.org) content (`wikimedia/wikipedia` on HuggingFace, CC BY-SA 4.0 / GFDL, attribution: Wikimedia Foundation).
The mined dictionary stores word frequencies and adjacency counts; facts, not creative expression. but the corpus origin is credited here regardless.

`scripts/fetch_deps.sh` fetches [remi0s/aosp-dictionary-tools](https://github.com/remi0s/aosp-dictionary-tools)
(`dicttool_aosp.jar`, AOSP LatinIME, Apache-2.0) into the gitignored `deps/` directory.
it is not distributed with this repository, and the fetch is a blobless sparse checkout (~400 KB, pinned commit).

## extra notes about LLM
this repo contains LLM-generated codes. but i tried doing by my own where i can; the original idea, repo structure, planning, some logics, and the README of course.