"""Command line interface: kodict {extract,build,status}."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from kodict import tasks
from kodict.blinker import Dashboard
from kodict.build import BuildError, build
from kodict.config import Config, load
from kodict.runner import (CHECKPOINT_PATH, RAW_DIR, Checkpoint, RunStats,
                           run_tasks, shard_is_degenerate, shard_is_incomplete)


def _extract_async(cfg: Config, args: argparse.Namespace) -> RunStats:
    async def go() -> RunStats:
        dash = Dashboard(cfg.jobs.count)

        # phase 1: words (frequency bands + categories)
        word_tasks = tasks.generate_word_tasks()
        stats = await run_tasks(cfg, word_tasks, dry_run=args.dry_run,
                                limit=args.limit, dashboard=dash)
        if stats.fatal:
            return stats

        # phase 2: bigrams, derived from what phase 1 extracted
        # (skipped for --limit smoke tests and dry runs)
        if not args.dry_run and args.limit is None:
            bigram_tasks = tasks.generate_bigram_tasks(RAW_DIR)
            if bigram_tasks:
                dash.log(f"phase 2: {len(bigram_tasks)} bigram tasks")
                more = await run_tasks(cfg, bigram_tasks, dry_run=args.dry_run,
                                       dashboard=dash)
                stats = _combine(stats, more)
        return stats

    return asyncio.run(go())


def _combine(a: RunStats, b: RunStats) -> RunStats:
    return RunStats(
        total=a.total + b.total,
        completed=a.completed + b.completed,
        failed=a.failed + b.failed,
        retried=a.retried + b.retried,
        entries=a.entries + b.entries,
        tokens=a.tokens + b.tokens,
        cost=a.cost + b.cost,
        fatal=b.fatal or a.fatal,
        skipped=a.skipped + b.skipped,
    )


def _print_summary(stats: RunStats) -> None:
    state = "ABORTED" if stats.fatal else ("OK" if stats.ok else "DONE WITH ERRORS")
    print(f"extract [{state}]: {stats.completed}/{stats.total} tasks "
          f"({stats.skipped} skipped), {stats.entries} entries, "
          f"{stats.tokens} tokens, ${stats.cost:.4f}")
    if stats.failed:
        print(f"  {stats.failed} failed tasks (see state/checkpoint.json)")
    if stats.fatal:
        print(f"  fatal: {stats.fatal}", file=sys.stderr)


def _extract(args: argparse.Namespace) -> int:
    cfg = load(args.config)
    try:
        stats = _extract_async(cfg, args)
    except KeyboardInterrupt:
        print("\nkodict: interrupted — progress is checkpointed; "
              "rerun `kodict extract` to resume", file=sys.stderr)
        return 130
    except RuntimeError as e:  # missing OPENROUTER_API_KEY
        print(f"kodict: {e}", file=sys.stderr)
        return 1
    _print_summary(stats)
    return 0 if stats.ok else 1


def _status(args: argparse.Namespace) -> int:
    cfg = load(args.config)
    if not CHECKPOINT_PATH.is_file():
        print("no checkpoint yet — run `kodict extract` first")
        return 0
    cp = Checkpoint(CHECKPOINT_PATH, enabled=False)
    raw_completed = cp.data.get("completed", {})
    word_tasks = tasks.generate_word_tasks()
    bigram_tasks = tasks.generate_bigram_tasks(RAW_DIR)
    task_by_id = {t.id: t for t in word_tasks + bigram_tasks}
    empty = [k for k, v in raw_completed.items() if v.get("entries", 0) == 0]
    looped = [k for k, v in raw_completed.items()
              if v.get("entries", 0) > 0 and shard_is_degenerate(RAW_DIR, k)]
    short = [k for k, v in raw_completed.items()
             if v.get("entries", 0) > 0 and k not in looped
             and task_by_id.get(k) is not None
             and shard_is_incomplete(RAW_DIR, task_by_id[k])]
    completed = {k: v for k, v in raw_completed.items()
                 if k not in empty and k not in looped and k not in short}
    failed = cp.data.get("failed", {})
    word_total = len(word_tasks)
    word_done = sum(1 for t in completed if not t.startswith("bigram-"))
    bigram_done = sum(1 for t in completed if t.startswith("bigram-"))

    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"  words:   {word_done}/{word_total} tasks done")
    print(f"  bigrams: {bigram_done}/{len(bigram_tasks)} tasks done")
    if empty:
        print(f"  note:    {len(empty)} tasks produced 0 entries and will be retried")
    if looped:
        print(f"  note:    {len(looped)} tasks looped (repetition) and will be "
              f"retried: {', '.join(looped[:5])}")
    if short:
        print(f"  note:    {len(short)} tasks stopped early (<50% of requested) "
              f"and will be retried: {', '.join(short[:5])}")
    print(f"  entries: {sum(v.get('entries', 0) for v in completed.values())}")
    print(f"  tokens:  {cp.data.get('tokens', 0)}   cost: ${cp.data.get('cost', 0.0):.4f}")
    if failed:
        print(f"  failed:  {len(failed)} ({', '.join(sorted(failed)[:10])})")
    print(f"  dicttool: {cfg.output.dicttool}")
    return 0


def _build(args: argparse.Namespace) -> int:
    cfg = load(args.config)
    try:
        combined, dict_path, words, size = build(
            cfg, RAW_DIR, Path(cfg.output.dir))
    except BuildError as e:
        print(f"kodict: {e}", file=sys.stderr)
        return 1
    print(f"{words} words -> {combined}")
    print(f"compiled  -> {dict_path} ({size} bytes, AOSP magic verified)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kodict",
        description="LLM-extracted Korean dictionary for AOSP keyboard forks",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="extract words/bigrams from the LLM")
    p_extract.add_argument("--config", default="config.toml", help="path to config.toml")
    p_extract.add_argument("--limit", type=int, default=None,
                           help="run at most N word tasks, skip bigrams (smoke test)")
    p_extract.add_argument("--dry-run", action="store_true",
                           help="render the dashboard with fake tokens, no API calls")

    p_build = sub.add_parser("build", help="write main_ko.combined and compile main_ko.dict")
    p_build.add_argument("--config", default="config.toml", help="path to config.toml")

    p_status = sub.add_parser("status", help="show checkpoint progress and cost so far")
    p_status.add_argument("--config", default="config.toml", help="path to config.toml")

    args = parser.parse_args(argv)
    handler = {"extract": _extract, "build": _build, "status": _status}[args.command]
    try:
        return handler(args)
    except BuildError as e:
        print(f"kodict: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
