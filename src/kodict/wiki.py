"""Wikipedia corpus: auth/download, wikitext cleaning, eojeol tokenization.

The dataset is `wikimedia/wikipedia`, dated subdir pinned in config
(``20231101.ko``): three parquet shards with columns ``id, url, title,
text``.

Cleaning reduces an article to plain text lines: templates, references,
tables, link targets and other markup would otherwise poison both the
wordlist (URLs, template parameter names) and the bigram table. Token
*segments* are the unit of bigram mining — a bigram is only ever formed
between two tokens of the same segment, so predictions never cross a
line, a dropped token (numbers, mixed script, bare punctuation) or a
sentence-ending mark. This is what makes mined bigrams grammatical:
Korean particles attach without a space, so `사람 이` simply never
occurs as a space-separated pair in real text.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import httpx
import pyarrow.parquet as pq

from kodict.config import Config
from kodict.validate import is_valid_word

DATASET_API = "https://huggingface.co/api/datasets/{dataset}/tree/main/{subdir}"
DATASET_RESOLVE = "https://huggingface.co/datasets/{dataset}/resolve/main/{subdir}/{name}"

# Offline fallback: names/sizes/sha256 of the pinned shards, recorded
# when the API was queried on 2026-11-08. Used if the tree listing is
# unreachable (rate limit, offline).
FALLBACK_SHARDS = (
    ("train-00000-of-00003.parquet", 400000634,
     "28ff69a402878eba3be42dfd187f37876b917b72eb9b05896ed7469fa681bb03"),
    ("train-00001-of-00003.parquet", 205287522,
     "60795ca350aec5c6813651e181c8e506a1c1b1461d56575b54c25632ed28dd14"),
    ("train-00002-of-00003.parquet", 177388905,
     "5e16f59e258717c95f2bb1ca5840085e02a4a88177df95447aefe9420e2be94a"),
)

# ---------------------------------------------------------------- auth


def token() -> str | None:
    """HF token: ``HF_TOKEN`` env first, else ``.hftok``"""
    env = os.environ.get("HF_TOKEN", "").strip()
    if env:
        return env
    for path in (Path(".hftok"), Path(".hftok")):
        if path.is_file():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
    return None


def _headers(tok: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {tok}"} if tok else {}


# ---------------------------------------------------------------- fetch


@dataclass(frozen=True)
class Shard:
    name: str
    size: int
    sha256: str


def list_shards(cfg: Config) -> list[Shard]:
    """Shards from the HF tree API, falling back to the pinned list."""
    url = DATASET_API.format(dataset=cfg.mine.dataset, subdir=cfg.mine.subdir)
    tok = token()
    for headers in (_headers(tok), {}):
        try:
            resp = httpx.get(url, headers=headers, timeout=30.0,
                             follow_redirects=True)
        except httpx.HTTPError:
            break
        if resp.status_code in (401, 403) and headers:
            print("kodict: HF token rejected — falling back to anonymous",
                  file=sys.stderr)
            continue
        if resp.status_code != 200:
            break
        shards = []
        for entry in resp.json():
            path = entry.get("path", "")
            if not path.endswith(".parquet"):
                continue
            lfs = entry.get("lfs") or {}
            shards.append(Shard(path.rsplit("/", 1)[-1], entry["size"],
                                lfs.get("oid", "")))
        if shards:
            return sorted(shards, key=lambda s: s.name)
    print("kodict: tree listing unavailable — using pinned shard list",
          file=sys.stderr)
    return [Shard(name, size, sha) for name, size, sha in FALLBACK_SHARDS]


def fetch(cfg: Config) -> list[Path]:
    """Download the parquet shards into the cache dir; verify size +
    sha256; skip files already present. Returns the local paths."""
    cache = Path(cfg.mine.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    shards = list_shards(cfg)
    base = DATASET_RESOLVE.format(dataset=cfg.mine.dataset,
                                  subdir=cfg.mine.subdir, name="")
    out: list[Path] = []
    for shard in shards:
        dest = cache / shard.name
        if dest.is_file() and dest.stat().st_size == shard.size:
            print(f"  {shard.name}: cached ({shard.size} bytes)")
            out.append(dest)
            continue
        _download(base + shard.name, dest, shard)
        out.append(dest)
    return out


def _download(url: str, dest: Path, shard: Shard) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    tok = token()
    for headers in (_headers(tok), {}):
        digest = hashlib.sha256()
        done = 0
        try:
            with httpx.Client(timeout=httpx.Timeout(60.0, read=180.0),
                              follow_redirects=True,
                              headers=headers) as client:
                with client.stream("GET", url) as resp:
                    if resp.status_code in (401, 403) and headers:
                        print(f"  {shard.name}: token rejected — retrying "
                              "anonymously", file=sys.stderr)
                        continue
                    resp.raise_for_status()
                    with open(part, "wb") as f:
                        for chunk in resp.iter_bytes(1 << 20):
                            f.write(chunk)
                            digest.update(chunk)
                            done += len(chunk)
                            if done % (64 << 20) < (1 << 20):
                                pct = 100 * done / shard.size
                                print(f"  {shard.name}: {pct:5.1f}%")
        except httpx.HTTPError as e:
            raise RuntimeError(f"download failed for {shard.name}: {e}") from e
        if done != shard.size:
            raise RuntimeError(
                f"{shard.name}: got {done} bytes, expected {shard.size}")
        if shard.sha256 and digest.hexdigest() != shard.sha256:
            raise RuntimeError(f"{shard.name}: sha256 mismatch")
        part.replace(dest)
        print(f"  {shard.name}: {shard.size} bytes, sha256 ok")
        return
    raise RuntimeError(f"{shard.name}: authentication failed")


def parquet_paths(cfg: Config) -> list[Path]:
    """Cached shards, fetched if necessary."""
    cache = Path(cfg.mine.cache_dir)
    paths = sorted(cache.glob("*.parquet"))
    if not paths:
        paths = fetch(cfg)
    return paths


def iter_articles(path: Path):
    """Yield ``(title, text)`` rows, streaming row-group batches so peak
    memory stays small."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=256, columns=["title", "text"]):
        titles = batch.column("title").to_pylist()
        texts = batch.column("text").to_pylist()
        yield from zip(titles, texts)


# ---------------------------------------------------------------- clean

_REF_RE = re.compile(r"<ref\b[^>]*>.*?</ref>|<ref\b[^>]*/>", re.S)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_TAG_RE = re.compile(r"<[^<>\s][^>]*>")
_MAGIC_RE = re.compile(r"__[A-Za-z]+__")
_HEADING_RE = re.compile(r"^\s*(={1,6})\s*(.*?)\s*\1\s*$")
_LIST_RE = re.compile(r"^\s*[*#:;]+[ \t]*")
_NS_LINK_RE = re.compile(r"\[\[\s*[^\]\[|:]{1,30}:[^\]\[]*(?:\|[^\]]*)?\]\]")
_PIPE_LINK_RE = re.compile(r"\[\[\s*([^\]|]+?)\s*\|\s*([^\]]+?)\s*\]\]")
_PLAIN_LINK_RE = re.compile(r"\[\[\s*([^\]]+?)\s*\]\]")
_EXT_TEXT_RE = re.compile(r"\[https?://[^\]\s]+\s+([^\]]+)\]")
_EXT_BARE_RE = re.compile(r"\[https?://[^\]]*\]")
_QUOTE_RE = re.compile(r"''+")


def strip_templates(text: str) -> str:
    """Remove ``{{...}}`` blocks, innermost-outermost in one pass.

    Templates are citations, hatnotes, infoboxes and parser functions —
    almost all noise for word frequencies, so the whole block goes,
    nested content included. Unbalanced braces are left untouched.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("{{", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("{{", j):
                    depth += 1
                    j += 2
                elif text.startswith("}}", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            if depth == 0:
                out.append("")
                i = j
                continue
            out.append("{{")
            i += 2
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def clean_wikitext(text: str) -> list[str]:
    """Wikitext → plain text lines (markup stripped, tables dropped)."""
    text = _COMMENT_RE.sub("", text)
    text = _REF_RE.sub("", text)
    text = strip_templates(text)
    lines: list[str] = []
    in_table = False
    for raw in text.split("\n"):
        stripped = raw.strip()
        if in_table:
            if stripped.startswith("|}"):
                in_table = False
            continue
        if stripped.startswith("{|"):
            in_table = True
            continue
        if stripped.startswith("|"):  # stray table row/cell line
            continue
        line = _MAGIC_RE.sub("", raw)
        heading = _HEADING_RE.match(line)
        if heading:
            line = heading.group(2)  # keep heading words, drop the '='s
        line = _LIST_RE.sub(" ", line)
        line = _NS_LINK_RE.sub("", line)
        line = _EXT_TEXT_RE.sub(r"\1", line)
        line = _EXT_BARE_RE.sub("", line)
        line = _PIPE_LINK_RE.sub(r"\2", line)
        line = _PLAIN_LINK_RE.sub(r"\1", line)
        line = _TAG_RE.sub(" ", line)
        line = _QUOTE_RE.sub("", line)  # ''italics''/'''bold'''
        line = line.strip()
        if line:
            lines.append(line)
    return lines


# ---------------------------------------------------------------- tokens

# characters that terminate a sentence when trailing a token
_SENTENCE_END = frozenset(".!?…")


def strip_edges(token: str) -> tuple[str, str]:
    """``(core, tail)``: leading/trailing punctuation stripped off; the
    trailing characters come back so sentence ends stay detectable."""
    start, end = 0, len(token)
    while start < end and unicodedata.category(token[start])[0] in "PS":
        start += 1
    while end > start and unicodedata.category(token[end - 1])[0] in "PS":
        end -= 1
    return token[start:end], token[end:]


def tokenize_line(line: str) -> list[list[str]]:
    """One cleaned line → segments of valid eojeol tokens.

    A segment ends before any token that fails ``is_valid_word``
    (numbers, mixed script, punctuation-only) and after any token that
    ends in sentence punctuation — bigrams never span segments.
    """
    segments: list[list[str]] = []
    current: list[str] = []
    for raw in unicodedata.normalize("NFC", line).split():
        core, tail = strip_edges(raw)
        if not core or not is_valid_word(core):
            if current:
                segments.append(current)
                current = []
            continue
        current.append(core)
        if any(ch in _SENTENCE_END for ch in tail):
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def iter_segments(title: str, text: str):
    """All token segments of one article (title counts as a line)."""
    yield from tokenize_line(title)
    for line in clean_wikitext(text):
        yield from tokenize_line(line)