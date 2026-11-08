"""Command line interface: kodict {fetch,mine,build,status}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kodict.build import BuildError, build
from kodict.config import Config, load
from kodict.mine import mine
from kodict.wiki import fetch, parquet_paths


def _fetch(args: argparse.Namespace) -> int:
    cfg = load(args.config)
    try:
        paths = fetch(cfg)
    except (RuntimeError, OSError) as e:
        print(f"kodict: {e}", file=sys.stderr)
        return 1
    total = sum(p.stat().st_size for p in paths)
    print(f"{len(paths)} parquet shards ({total} bytes) in {cfg.mine.cache_dir}")
    return 0


def _mine(args: argparse.Namespace) -> int:
    cfg = load(args.config)
    try:
        paths = parquet_paths(cfg)
        stats = mine(cfg, paths)
    except (RuntimeError, OSError) as e:
        print(f"kodict: {e}", file=sys.stderr)
        return 1
    print(f"mined {stats['words']} words "
          f"({stats['bigram_links']} bigrams over "
          f"{stats['parents_with_bigrams']} parents"
          + (f", {stats['links_dropped']} loop-links dropped"
             if stats.get("links_dropped") else "")
          + f") -> {stats['out_dir']}")
    return 0


def _status(args: argparse.Namespace) -> int:
    cfg = load(args.config)
    cache = Path(cfg.mine.cache_dir)
    parquet = sorted(cache.glob("*.parquet"))
    if parquet:
        size = sum(p.stat().st_size for p in parquet)
        print(f"corpus: {len(parquet)} parquet shards, {size} bytes "
              f"in {cache}")
    else:
        print(f"corpus: not fetched yet — run `kodict fetch` (cache: {cache})")

    shard_dir = Path(cfg.mine.out_dir)
    shards = sorted(shard_dir.glob("*.jsonl"))
    if not shards:
        print(f"mined shards: none in {shard_dir} — run `kodict mine`")
        return 0
    entries = with_bigrams = links = 0
    for path in shards:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entries += 1
                if entry.get("b"):
                    with_bigrams += 1
                    links += len(entry["b"])
    print(f"mined: {entries} words in {len(shards)} shards ({shard_dir})")
    print(f"  bigrams: {links} links over {with_bigrams} parents "
          f"(min_count={cfg.mine.min_count}, "
          f"min_bigram_count={cfg.mine.min_bigram_count})")
    print(f"  build with: kodict build --raw-dir {cfg.mine.out_dir}")
    return 0


def _build(args: argparse.Namespace) -> int:
    cfg = load(args.config)
    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(cfg.mine.out_dir)
    if not raw_dir.is_dir() or not list(raw_dir.glob("*.jsonl")):
        print(f"kodict: no mined shards in {raw_dir} — run `kodict mine` first",
              file=sys.stderr)
        return 1
    try:
        combined, dict_path, words, size = build(cfg, raw_dir, Path(cfg.output.dir))
    except BuildError as e:
        print(f"kodict: {e}", file=sys.stderr)
        return 1
    print(f"{words} words -> {combined}")
    print(f"compiled  -> {dict_path} ({size} bytes, AOSP magic verified)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kodict",
        description="Wikipedia-mined Korean dictionary for AOSP keyboard forks",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download the Wikipedia parquet shards")
    p_fetch.add_argument("--config", default="config.toml", help="path to config.toml")

    p_mine = sub.add_parser("mine", help="count words/bigrams and write JSONL shards")
    p_mine.add_argument("--config", default="config.toml", help="path to config.toml")

    p_build = sub.add_parser("build", help="write main_ko.combined and compile main_ko.dict")
    p_build.add_argument("--config", default="config.toml", help="path to config.toml")
    p_build.add_argument("--raw-dir", default=None,
                         help="directory of mined .jsonl shards "
                              "(default: [mine].out_dir)")

    p_status = sub.add_parser("status", help="show corpus cache and mined shards")
    p_status.add_argument("--config", default="config.toml", help="path to config.toml")

    args = parser.parse_args(argv)
    handler = {"fetch": _fetch, "mine": _mine, "build": _build,
               "status": _status}[args.command]
    try:
        return handler(args)
    except BuildError as e:
        print(f"kodict: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())