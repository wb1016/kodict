# kodict

LLM-extracted Korean dictionary for AOSP keyboard forks.

Words and next-word (bigram) data are extracted from an LLM by **16 parallel
streaming jobs — each with its own terminal blinker that toggles on every
received token** — then assembled into `main_ko.combined` and compiled into
`main_ko.dict` with AOSP's own `dicttool`.

There is currently no `main_ko` dictionary in
[aosp-dictionaries](https://codeberg.org/Helium314/aosp-dictionaries), so this
fills that gap.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
bash scripts/fetch_deps.sh          # ~400 KB sparse checkout of dicttool_aosp.jar
export OPENROUTER_API_KEY=...       # https://openrouter.ai/settings/keys
```

## Usage

```bash
uv run python -m kodict extract --dry-run --limit 4   # demo the dashboard, no API calls
uv run python -m kodict extract                      # full run: words, then bigrams
uv run python -m kodict extract --limit 10           # small real smoke test
uv run python -m kodict status                       # progress + tokens + cost
uv run python -m kodict build                        # out/main_ko.combined + out/main_ko.dict
uv run pytest                                        # test suite (offline)
```

Interrupted runs are resumable: completed tasks are checkpointed in
`state/checkpoint.json` and their output stored per-task in `state/raw/*.jsonl`.

## How it works

```
tasks.py ──▶ asyncio.Queue ──▶ 16 workers streaming from OpenRouter
                                   │  every token: blinker ●↔○ + incremental JSONL parse
                                   ▼
                        state/raw/<task>.jsonl  (checkpointed, resumable)
                                   ▼
     validate: NFC, Hangul/loanword filter, dedupe, rank → log frequency (0-255)
                                   ▼
                     out/main_ko.combined  (AOSP .combined format)
                                   ▼
     java -jar dicttool_aosp.jar makedict  →  out/main_ko.dict (AOSP magic verified)
```

**Model & routing** (`config.toml`): `qwen/qwen3.6-35b-a3b` with
`provider: { quantizations: ["fp8"] }` — only fp8 providers are used.

**Phase 1 – words**: 250 frequency-band tasks (top 100k ranks, 400 words each)
+ 25 category shards (nouns, verbs, particles, loanwords, slang, …). The model
numbers each list from 1; the runner adds the task's global `rank_base`.

**Phase 2 – bigrams**: generated from phase-1 output — the top 3 000 words in
chunks of 100, each asking for the 3 most likely following words (written as
`bigram=` lines under the word, ranks renumbered 1–3).

**Blinkers**: plain ANSI, no dependencies — 16 status lines redrawn as a single
cursor-up block. Each job's glyph toggles ● ↔ ○ per token, dims to · after 1 s
of silence, and becomes ✓ / ✗ when finished. Automatically disabled when
stdout is not a TTY.

## Repository layout

```
config.toml               model, fp8 routing, job count, output paths
scripts/fetch_deps.sh     blobless sparse clone of dicttool_aosp.jar @ pinned SHA
src/kodict/
  tasks.py                task/prompt generation (words, categories, bigrams)
  llm.py                  OpenRouter SSE streaming client, retries/backoff
  parse.py                incremental JSONL parsing of the token stream
  blinker.py              16-job ANSI dashboard with per-job blinkers
  runner.py               asyncio workers, checkpoint/resume, stats
  validate.py             NFC, word validation, rank → frequency mapping
  wordlist.py             .combined writer (written from the documented format)
  build.py                assembly + dicttool compilation + header verification
  cli.py                  extract / status / build
tests/                    86 tests, all offline
```

## Format notes

**Words are written NFD (decomposed jamo), not NFC** — this is required.
LeanType (and HeliBoard-lineage keyboards) wrap Korean dictionaries in
`KoreanDictionary`, which NFD-decomposes the typed word into canonical jamo
(`사` → `ᄉ`+`ᅡ`, U+1109 U+1161) before the native trie lookup — the native
code passes code points through with **no normalization** — and converts
candidates back to NFC for display. An NFC dictionary therefore never matches
at the trie root and Korean prediction silently returns nothing. Validation
and dedup still run in NFC (normalization is idempotent); only the final
`.combined` output is decomposed (`[output] decompose = true`).

Two lookups hit the same native trie, so bigram-carrying words additionally get
an **NFC mirror entry** (frequency 1): the *typed word* is transformed to NFD,
but the *ngram context* (the previous committed word, used for next-word
prediction) is passed through **untransformed** — without the mirror,
`getWordId` misses and next-word prediction silently degrades to context-free
history unigrams. Bigram ranks are also written as real probabilities
(`BIGRAM_F`: 210/200/190) instead of 1/2/3, because LeanType boosts history
unigrams to ~182 — an AOSP-sample-style `f=1..3` can never win there, and the
app's history-damping only activates above 180.

`.combined` (from aosp-dictionaries' README):

```
dictionary=main:ko,locale=ko,description=Korean,date=...,version=20
 word=사과,f=187
  bigram=먹다,f=1
```

`f` is the log frequency, 0–255 (higher = more likely to be suggested);
`version` must be ≥ 18 for all AOSP keyboards. The `.combined` writer here is
an independent implementation of that documented format.

## Licenses

kodict itself is [CC0-1.0](LICENSE).

`scripts/fetch_deps.sh` fetches
[remi0s/aosp-dictionary-tools](https://github.com/remi0s/aosp-dictionary-tools)
(`dicttool_aosp.jar`, AOSP LatinIME, **Apache-2.0**) into the gitignored
`deps/` directory — it is not distributed with this repository, and the fetch
is a blobless sparse checkout (~400 KB, pinned commit).
