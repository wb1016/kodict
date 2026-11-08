"""AOSP ``.combined`` wordlist writer.

Implements the format documented in aosp-dictionaries' README
("Wordlist information"):

    dictionary=main:ko,locale=ko,description=Korean,date=1234567890,version=20
     word=사과,f=187
      bigram=먹다,f=1
      bigram=주다,f=2
      bigram=열다,f=3

* header is one line, all fields required; ``version`` must be > 18 for the
  dictionary to work with all AOSP keyboards
* ``f`` is the logarithm of the word frequency, integer 0..255 (higher = more
  likely to be suggested / corrected)
* up to 3 ``bigram`` lines per word, ranked 1 = most likely next word;
  the target word must exist in the wordlist
* words are written sorted by descending ``f``

This is an independent implementation written from the documented format;
no code from the (GPL-3.0) aosp-dictionaries repository is used here.
"""

from __future__ import annotations

import time
import unicodedata
import warnings
from dataclasses import dataclass, field

MAX_F = 255
MAX_BIGRAMS = 3
MIN_VERSION = 18  # lower versions are ignored by some AOSP-based keyboards
# frequency for NFC mirror entries (see Wordlist docstring): low enough that
# they never surface as suggestions themselves, structure is all that matters
NFC_MIRROR_F = 1
# bigram rank -> probability written as the bigram line's `f`.
# The native side uses this value RAW as the prediction score
# (PatriciaTriePolicy.getWordAttributesInContext -> getWordAttributes(bigramProbability)),
# while LeanType boosts context-free history unigrams by ~+32 on top of
# their ~150 score. AOSP's sample dictionaries use f=1..3 which cannot win
# there (nor cross the >180 threshold that dampens history), so use real
# probabilities: rank 1 must stay above 180 to activate the damping.
BIGRAM_F = {1: 210, 2: 200, 3: 190}


class WordlistError(ValueError):
    """The wordlist cannot be written in a valid format."""


@dataclass
class Header:
    locale: str = "ko"
    dict_type: str = "main"
    description: str = "Korean"
    version: int = 20
    date: int = field(default_factory=lambda: int(time.time()))

    def __post_init__(self) -> None:
        if not self.locale:
            raise WordlistError("header requires a locale")
        if not self.description:
            raise WordlistError("header requires a non-empty description")
        for name, value in (("description", self.description), ("locale", self.locale)):
            if any(c in value for c in ",=\n"):
                raise WordlistError(f"header {name} may not contain ',', '=' or newlines")
        if self.version < MIN_VERSION:
            warnings.warn(
                f"version {self.version} < {MIN_VERSION}: the dictionary may be "
                "ignored by some AOSP-based keyboard apps",
                stacklevel=2,
            )

    def write(self) -> str:
        return (
            f"dictionary={self.dict_type}:{self.locale.lower()},locale={self.locale},"
            f"description={self.description},date={self.date},version={self.version}"
        )


@dataclass
class Word:
    word: str
    f: int
    # (next word, rank), rank 1 = most likely next word
    bigrams: list[tuple[str, int]] = field(default_factory=list)
    possibly_offensive: bool = False
    not_a_word: bool = False


class Wordlist:
    """Mutable wordlist; call :meth:`normalize` before writing.

    ``decompose`` writes words NFD-normalized (canonical jamo sequences).
    LeanType/HeliBoard-style Korean handling NFD-decomposes the typed word
    before the native trie lookup and converts suggestions back to NFC for
    display — so a Korean dictionary must store decomposed words or lookups
    never match.

    Two different lookups hit the same native trie:

    * **typed word** → transformed to NFD by ``KoreanDictionary.processInput``
      → needs NFD entries (the primary ones),
    * **ngram context** (previous committed word, for next-word prediction) →
      passed through **untransformed** as NFC → needs NFC entries.

    Since ``getPredictions`` resolves the previous word with ``getWordId``
    and silently returns nothing when it misses, every word that carries
    bigram data gets an additional NFC *mirror* entry (low frequency) with
    the same bigram lines — otherwise next-word prediction falls back to
    context-free history unigrams. The mirror is only written in decompose
    mode; NFC and NFD forms are distinct trie nodes, so typed-word matching
    is unaffected.
    """

    def __init__(self, header: Header | None = None, *, decompose: bool = False):
        self.header = header
        self.decompose = decompose
        self._words: dict[str, Word] = {}

    def __len__(self) -> int:
        return len(self._words)

    def __contains__(self, word: str) -> bool:
        return word in self._words

    def add_word(self, word: str, f: int) -> None:
        """Add or update a word, keeping the highest frequency seen."""
        if not word:
            raise WordlistError("empty word")
        existing = self._words.get(word)
        if existing is None:
            self._words[word] = Word(word=word, f=f)
        else:
            existing.f = max(existing.f, f)

    def add_bigram(self, word: str, next_word: str, rank: int) -> None:
        """Attach a next-word suggestion to ``word``, keeping the best rank."""
        if word not in self._words:
            raise WordlistError(f"bigram parent {word!r} is not in the wordlist")
        if rank < 1:
            raise WordlistError(f"bigram rank must be >= 1, got {rank}")
        bigrams = self._words[word].bigrams
        for i, (existing_word, existing_rank) in enumerate(bigrams):
            if existing_word == next_word:
                if rank < existing_rank:
                    bigrams[i] = (next_word, rank)
                return
        bigrams.append((next_word, rank))

    def merge(self, other: "Wordlist") -> None:
        """Merge another wordlist into this one (max f, best bigram ranks)."""
        for word in other._words.values():
            self.add_word(word.word, word.f)
            for next_word, rank in word.bigrams:
                if next_word in self._words:
                    self.add_bigram(word.word, next_word, rank)

    def normalize(self) -> None:
        """Clamp frequencies, drop dangling bigrams, keep top 3, renumber 1..3."""
        for word in self._words.values():
            word.f = min(MAX_F, max(0, word.f))
            bigrams = [(n, r) for (n, r) in word.bigrams if n in self._words]
            bigrams.sort(key=lambda item: (item[1], item[0]))
            word.bigrams = [(n, i + 1) for i, (n, _) in enumerate(bigrams[:MAX_BIGRAMS])]

    def _out(self, text: str) -> str:
        return unicodedata.normalize("NFD", text) if self.decompose else text

    def to_combined(self) -> str:
        if self.header is None:
            raise WordlistError("cannot write a wordlist without a header")
        lines = [self.header.write()]
        for word in sorted(self._words.values(), key=lambda w: (-w.f, w.word)):
            flags = ""
            if word.possibly_offensive:
                flags += ",possibly_offensive=true"
            if word.not_a_word:
                flags += ",not_a_word=true"
            lines.append(f" word={self._out(word.word)},f={word.f}{flags}")
            lines.extend(self._bigram_lines(word.bigrams))
        if self.decompose:
            lines.extend(self._nfc_mirrors())
        return "\n".join(lines) + "\n"

    def _bigram_lines(self, bigrams: "list[tuple[str, int]]") -> list[str]:
        return [f"  bigram={self._out(next_word)},f={BIGRAM_F.get(rank, rank)}"
                for next_word, rank in sorted(bigrams, key=lambda b: b[1])]

    def _nfc_mirrors(self) -> list[str]:
        """NFC twins of bigram-carrying words (see class docstring)."""
        out: list[str] = []
        for word in sorted(self._words.values(), key=lambda w: w.word):
            if not word.bigrams:
                continue
            nfc = unicodedata.normalize("NFC", word.word)
            if nfc == self._out(word.word):
                continue  # nothing to mirror (already NFC, e.g. Latin)
            out.append(f" word={nfc},f={NFC_MIRROR_F}")
            out.extend(self._bigram_lines(word.bigrams))
        return out

    def write(self, path: str) -> None:
        self.normalize()
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_combined())
