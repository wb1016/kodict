"""Parsing of streamed model output.

The prompts ask for JSON Lines, but real models wrap output in all sorts of
ways, so parsing is layered:

1. **Line parsing** — complete lines are parsed as they arrive (fast path
   for well-behaved JSONL), tolerating fences, bullets, trailing prose and
   numbered/bare word lines.
2. **Whole-text fallback** — at :meth:`JsonlParser.flush`, every flat
   ``{...}`` object in the accumulated raw text is extracted with a regex.
   This catches output wrapped in a JSON array, a single outer object,
   pretty-printed JSON, or truncated output (all *complete* objects survive).

The raw text is always kept on the parser so callers can persist it for
debugging.
"""

from __future__ import annotations

import json
import re

# flat objects: our entries never nest, so braces inside strings are not a concern
_OBJECT_RE = re.compile(r"\{[^{}]*\}")
# bare entry lines: "1. 사과", "사과", "coffee"
_WORD_LINE_RE = re.compile(r"^(?:(\d+)[.)]\s*)?([가-힣]{1,15}|[A-Za-z]{2,15})$")
_RANK_KEYS = ("r", "rank", "f", "no", "index", "n")


def extract_objects(text: str) -> list[dict]:
    """Every JSON object that can be found anywhere in ``text``."""
    found: list[dict] = []
    for match in _OBJECT_RE.finditer(text):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            found.append(obj)
    return found


def _loads(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class JsonlParser:
    """Feed arbitrary text chunks, collect parsed entry objects."""

    def __init__(self) -> None:
        self._buf = ""
        self.raw = ""            # full accumulated stream text
        self.entries: list[dict] = []
        self.bad_lines = 0

    def feed(self, text: str) -> list[dict]:
        """Consume a chunk, return the entries completed by it."""
        self.raw += text
        self._buf += text
        *lines, self._buf = self._buf.split("\n")
        completed: list[dict] = []
        for line in lines:
            completed.extend(self._consume(line))
        return completed

    def flush(self) -> list[dict]:
        """Finish the stream: last line + whole-text fallback extraction."""
        completed: list[dict] = []
        if self._buf:
            completed.extend(self._consume(self._buf))
            self._buf = ""
        # Fallback: if line parsing missed entries (array / wrapper object /
        # pretty-printed or truncated JSON), pull every complete object out
        # of the raw text.
        for obj in extract_objects(self.raw):
            if obj not in self.entries:
                self.entries.append(obj)
                completed.append(obj)
        return completed

    def _consume(self, line: str) -> list[dict]:
        s = line.strip()
        if not s or s.startswith("```"):
            return []
        if s in ("{", "["):
            # opening line of a pretty-printed object: complete in later
            # lines, recovered by the flush-time fallback — not an error
            return []
        if s[:2] in ("- ", "* "):
            s = s[2:].strip()

        if s.startswith("{") or s.startswith("["):
            return self._consume_json(s)

        # bare / numbered word line: "1. 사과"
        match = _WORD_LINE_RE.match(s.rstrip(","))
        if match:
            entry: dict = {"w": match.group(2)}
            if match.group(1):
                entry["r"] = int(match.group(1))
            self.entries.append(entry)
            return [entry]
        return []  # prose: ignored, not counted as bad

    def _consume_json(self, s: str) -> list[dict]:
        obj = _loads(s)
        if obj is None and s.startswith("{"):
            end = s.rfind("}")
            if end > 0:
                obj = _loads(s[:end + 1])  # trailing prose / comma after object
        if isinstance(obj, dict):
            self.entries.append(obj)
            return [obj]
        if isinstance(obj, list):
            return [self._add(e) for e in obj if isinstance(e, dict)]
        # looks like JSON but did not parse: count it. If the object is
        # actually split across lines, the flush-time fallback still recovers
        # it from the raw text.
        self.bad_lines += 1
        return []

    def _add(self, entry: dict) -> dict:
        self.entries.append(entry)
        return entry