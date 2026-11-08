"""Task generation: what the 16 jobs ask the model for.

Two phases:

* **words** — frequency bands ("the 400 most common Korean words, numbered
  1-400") plus category shards (nouns, verbs, particles, loanwords, ...).
  The model numbers each list from 1; the runner adds the task's
  ``rank_base`` so all ranks merge into one global ordering later.
* **bigrams** — generated *after* phase 1 from the extracted words: the top
  words split into chunks, each asking for the 3 most likely next words.

Output contract (JSONL, one object per line):

* words:   ``{"w": "사과", "r": 41}``
* bigrams: ``{"w": "사과", "b": ["먹다", "주다", "열다"]}``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

BAND_SIZE = 400        # words per frequency-band task (fits one 8192-token reply)
MAX_RANK = 100_000     # covers the top 100k ranks via bands
CATEGORY_RANK_BASE = 100_000  # category words rank after the bands
CATEGORY_SIZE = 300
BIGRAM_WORDS_PER_TASK = 100
BIGRAM_MAX_WORDS = 3_000  # only the top-N words get next-word data

SYSTEM_WORDS = (
    "You are building data for a Korean input-method dictionary used by "
    "AOSP-based keyboards. Your entire reply must be raw JSON Lines: one "
    "standalone JSON object per line, nothing else. Never wrap the objects in "
    'a JSON array "[...]", never wrap them in one outer object, never use '
    "markdown code fences, never add explanations or headings; the reply must "
    "start with the first object. Each object must be compact and fit on a "
    "single line — no indentation, no line breaks inside an object. "
    "Every word must be in Korean Hangul (Hangeul), written in NFC, without "
    "spaces, punctuation or digits; common loanwords written in Latin letters "
    "are allowed. No proper nouns, no abbreviations unless very common, no "
    "duplicates, no word forms a user would never type. "
    "Always produce the full requested list in one reply; do not stop early, "
    "do not summarize, do not sample or abbreviate the list."
)

SYSTEM_BIGRAM = (
    "You are building next-word data for a Korean input-method dictionary "
    "used by AOSP-based keyboards. Your entire reply must be raw JSON Lines: "
    "one standalone JSON object per line, starting with the first object. "
    "Never wrap the objects in a JSON array or an outer object, never use "
    "code fences, no explanations. Each line is "
    '{"w": "<word>", "b": ["<next 1>", "<next 2>", "<next 3>"]} where the '
    "three entries are the words most likely to follow <word>, most likely "
    "first. Only Hangul words are allowed in b."
)

CATEGORIES: tuple[str, ...] = (
    "everyday nouns",
    "food and cooking nouns",
    "school and education nouns",
    "work and office nouns",
    "technology nouns",
    "nature and weather nouns",
    "body and health nouns",
    "clothing nouns",
    "travel and transportation nouns",
    "family and relationship nouns",
    "emotion and personality nouns",
    "verbs of movement and action",
    "verbs of communication",
    "verbs of thinking and feeling",
    "adjectives describing things",
    "adjectives describing people and feelings",
    "adverbs",
    "particles and connective endings",
    "counters and measure words",
    "Sino-Korean nouns",
    "loanwords used in daily life",
    "onomatopoeia and mimesis",
    "abbreviations and short forms",
    "internet and messaging slang",
    # NOTE: "common two-word expressions" was removed — spaced entries are
    # invalid by design (is_valid_word rejects spaces), so that category
    # could never produce a valid entry and burned retries forever
)


@dataclass(frozen=True)
class Task:
    id: str
    kind: str            # "words" | "bigrams"
    prompt: str
    rank_base: int = 0   # words: global rank = rank_base + model's "r"
    expected: int = 0    # entries wanted; 0 = unknown (no completeness gate)
    # fraction of `expected` that still counts as usable output; category
    # shards are supplementary, models reliably return fewer valid singles
    min_fraction: float = 0.5


def generate_word_tasks() -> list[Task]:
    tasks: list[Task] = []
    for start in range(0, MAX_RANK, BAND_SIZE):
        base = start
        end = start + BAND_SIZE
        prompt = (
            f"List the {BAND_SIZE} most common Korean words, numbered 1 to "
            f"{BAND_SIZE} in descending order of frequency (1 = most common); "
            f"they are ranks {base + 1} to {end} globally. "
            f"You must output all {BAND_SIZE} entries — never stop early."
        )
        tasks.append(Task(id=f"freq-{start:06d}", kind="words",
                          prompt=prompt, rank_base=base, expected=BAND_SIZE))
    for i, category in enumerate(CATEGORIES):
        base = CATEGORY_RANK_BASE + i * CATEGORY_SIZE
        prompt = (
            f"List {CATEGORY_SIZE} common Korean {category}, numbered 1 to "
            f"{CATEGORY_SIZE} in descending order of frequency. "
            f"You must output all {CATEGORY_SIZE} entries — never stop early."
        )
        tasks.append(Task(id=f"cat-{i:03d}", kind="words",
                          prompt=prompt, rank_base=base, expected=CATEGORY_SIZE,
                          min_fraction=0.25))
    return tasks


def load_extracted_words(raw_dir: str | Path) -> dict[str, int]:
    """word -> best (lowest) rank, from all completed word-task shards."""
    best: dict[str, int] = {}
    raw_dir = Path(raw_dir)
    for path in sorted(raw_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                word, rank = entry.get("w"), entry.get("r")
                if not isinstance(word, str) or not isinstance(rank, int):
                    continue
                if word not in best or rank < best[word]:
                    best[word] = rank
    return best


def generate_bigram_tasks(raw_dir: str | Path) -> list[Task]:
    """Phase 2: top words (by rank) → next-word tasks."""
    words = load_extracted_words(raw_dir)
    top = [w for w, _ in sorted(words.items(), key=lambda item: item[1])[:BIGRAM_MAX_WORDS]]
    tasks: list[Task] = []
    for i in range(0, len(top), BIGRAM_WORDS_PER_TASK):
        chunk = top[i:i + BIGRAM_WORDS_PER_TASK]
        listing = " ".join(chunk)
        prompt = (
            "For each of these Korean words, output the 3 most likely "
            f"following words as JSONL lines — one line per word, "
            f"{len(chunk)} lines total:\n{listing}"
        )
        tasks.append(Task(id=f"bigram-{i // BIGRAM_WORDS_PER_TASK:04d}",
                          kind="bigrams", prompt=prompt, expected=len(chunk)))
    return tasks
