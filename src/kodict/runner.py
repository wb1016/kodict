"""Parallel extraction runner: N jobs, one blinker each, checkpointed.

Workers pull tasks from a queue and stream them from OpenRouter (fp8-only
provider routing). Every token toggles the owning job's blinker. Completed
tasks are appended to ``state/raw/<task>.jsonl`` and recorded in
``state/checkpoint.json`` so interrupted runs resume where they stopped.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from kodict.blinker import Dashboard
from kodict.config import Config, api_key
from kodict.llm import LLMError, StreamResult, stream_chat
from kodict.parse import JsonlParser
from kodict.tasks import SYSTEM_BIGRAM, SYSTEM_WORDS, Task  # noqa: F401 (Task used in annotations)

STATE_DIR = Path("state")
RAW_DIR = STATE_DIR / "raw"
CHECKPOINT_PATH = STATE_DIR / "checkpoint.json"

# a result this unique (out of a long list) means the model was looping
DEGENERATE_MIN_UNIQUE = 20


def is_degenerate(total: int, unique: int) -> bool:
    """True for repetition-loop output: heavy duplication AND few unique words."""
    if total <= 0:
        return False
    return unique < total * 0.5 and unique < DEGENERATE_MIN_UNIQUE


def shard_counts(raw_dir: Path, task_id: str) -> tuple[int, int]:
    """(total entries, unique words) of an already-written shard."""
    path = raw_dir / f"{task_id}.jsonl"
    if not path.is_file():
        return 0, 0
    total, words = 0, set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                words.add(json.loads(line).get("w"))
            except json.JSONDecodeError:
                continue
    return total, len(words)


def shard_is_degenerate(raw_dir: Path, task_id: str) -> bool:
    """True when an existing shard looks like a repetition loop."""
    total, unique = shard_counts(raw_dir, task_id)
    return is_degenerate(total, unique)


def shard_is_incomplete(raw_dir: Path, task: Task) -> bool:
    """True when an existing shard holds less than the task's usable minimum."""
    if task.expected <= 0:
        return False
    total, _ = shard_counts(raw_dir, task.id)
    return 0 < total < task.expected * task.min_fraction


@dataclass
class RunStats:
    total: int = 0
    completed: int = 0
    failed: int = 0
    retried: int = 0
    entries: int = 0
    tokens: int = 0
    cost: float = 0.0
    fatal: str | None = None
    skipped: int = 0

    @property
    def ok(self) -> bool:
        return self.fatal is None and self.failed == 0


class Checkpoint:
    """JSON record of completed/failed tasks; atomic writes."""

    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.data: dict = {"completed": {}, "failed": {}, "tokens": 0, "cost": 0.0}
        if path.is_file():
            try:
                self.data.update(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass  # corrupt checkpoint: start fresh, shards still on disk

    def done(self, task_id: str) -> bool:
        """Only tasks with usable output count as done (0-entry records —
        e.g. from a format mismatch — are retried on the next run)."""
        record = self.data["completed"].get(task_id)
        return bool(record and record.get("entries", 0) > 0)

    def mark_done(self, task_id: str, entries: int, cost: float, tokens: int) -> None:
        self.data["completed"][task_id] = {"entries": entries, "cost": cost}
        self.data["failed"].pop(task_id, None)
        self.data["tokens"] = self.data.get("tokens", 0) + tokens
        self.data["cost"] = self.data.get("cost", 0.0) + cost
        self.save()

    def mark_failed(self, task_id: str, error: str) -> None:
        self.data["failed"][task_id] = error
        self.data["completed"].pop(task_id, None)  # invalidate any stale record
        self.save()

    def save(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)


@dataclass
class _Ctx:
    cfg: Config
    dash: Dashboard
    stats: RunStats
    checkpoint: Checkpoint
    raw_dir: Path
    dry_run: bool
    backoff_base: float
    api_key: str | None = None
    stream_fn: Callable = stream_chat
    http: httpx.AsyncClient | None = None
    fatal: str | None = None
    total: int = 0
    log_path: Path | None = None  # persistent run log (None = dashboard only)


def _log(ctx: _Ctx, message: str) -> None:
    """Dashboard line + persistent timestamped log (the dashboard keeps only
    the latest line, so retries/errors would otherwise be lost)."""
    ctx.dash.log(message)
    if ctx.log_path is not None:
        ctx.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ctx.log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


async def run_tasks(
    cfg: Config,
    tasks: list[Task],
    *,
    dry_run: bool = False,
    limit: int | None = None,
    dashboard: Dashboard | None = None,
    stream_fn: Callable | None = None,
    backoff_base: float = 1.0,
    checkpoint_path: Path = CHECKPOINT_PATH,
    raw_dir: Path = RAW_DIR,
    persist: bool = True,
) -> RunStats:
    """Run tasks with ``cfg.jobs.count`` parallel workers; returns stats."""
    if limit is not None:
        tasks = tasks[:limit]
    persist = persist and not dry_run
    checkpoint = Checkpoint(checkpoint_path, enabled=persist)
    if dry_run:
        pending = list(tasks)  # demo mode: always show work to do
    else:
        # resume skips completed tasks — unless their shard looks like a
        # repetition loop or is missing more than half the expected entries
        pending = [
            t for t in tasks
            if not checkpoint.done(t.id)
            or shard_is_degenerate(raw_dir, t.id)
            or shard_is_incomplete(raw_dir, t)
        ]

    stats = RunStats(total=len(tasks), completed=len(tasks) - len(pending),
                     skipped=len(tasks) - len(pending))
    dash = dashboard if dashboard is not None else Dashboard(cfg.jobs.count)

    if not pending:
        dash.log(f"all {len(tasks)} tasks already completed")
        return stats

    ctx = _Ctx(cfg=cfg, dash=dash, stats=stats, checkpoint=checkpoint,
               raw_dir=raw_dir, dry_run=dry_run, backoff_base=backoff_base,
               stream_fn=stream_fn or stream_chat, total=len(tasks),
               log_path=(checkpoint_path.parent / "run.log") if persist else None)
    if not dry_run:
        ctx.api_key = api_key()
        ctx.http = httpx.AsyncClient()

    dash.log(f"{len(pending)}/{len(tasks)} tasks pending, "
             f"{cfg.jobs.count} jobs, model={cfg.model.name}")
    dash.start()
    queue: asyncio.Queue[Task] = asyncio.Queue()
    for task in pending:
        queue.put_nowait(task)
    workers = [asyncio.create_task(_worker(i, queue, ctx))
               for i in range(cfg.jobs.count)]
    try:
        while not ctx.fatal:
            try:
                await asyncio.wait_for(queue.join(), timeout=0.2)
                break
            except asyncio.TimeoutError:
                dash.tick()
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        dash.stop()
        if ctx.http is not None:
            await ctx.http.aclose()

    stats.fatal = ctx.fatal
    if ctx.fatal:
        dash.log(f"aborted: {ctx.fatal}")
    return stats


async def _worker(index: int, queue: asyncio.Queue, ctx: _Ctx) -> None:
    blinker = ctx.dash.blinkers[index]
    while not ctx.fatal:
        task = await queue.get()
        try:
            await _run_task(index, task, ctx)
        except Exception as e:  # noqa: BLE001 — one bad task must not kill a job
            blinker.status = "failed"
            ctx.stats.failed += 1
            ctx.checkpoint.mark_failed(task.id, repr(e))
            _log(ctx, f"job-{index:02d} {task.id} crashed: {e!r}")
        finally:
            queue.task_done()


async def _run_task(index: int, task: Task, ctx: _Ctx) -> None:
    blinker = ctx.dash.blinkers[index]
    blinker.status = "running"
    blinker.task = task.id
    _log(ctx, f"job-{index:02d} → {task.id}")

    system = SYSTEM_WORDS if task.kind == "words" else SYSTEM_BIGRAM
    result = StreamResult()

    for attempt in range(ctx.cfg.jobs.max_retries + 1):
        if ctx.fatal:
            return
        if attempt > 0:
            ctx.stats.retried += 1
            blinker.status = "retry"
            delay = ctx.backoff_base * (2 ** (attempt - 1)) * (0.5 + random.random())
            _log(ctx, f"job-{index:02d} {task.id}: retry {attempt} in {delay:.1f}s")
            await asyncio.sleep(delay)
            blinker.status = "running"

        parser = JsonlParser()  # fresh per attempt: text from a failed or
        #                         short stream must not leak into the next one

        def on_token(text: str) -> None:
            blinker.token(text)   # blink on every received content token
            parser.feed(text)     # and parse it incrementally

        user_prompt = task.prompt
        if attempt > 0 and task.expected:
            # the previous reply stopped early or came back malformed
            user_prompt += (
                f"\n\nYour previous reply did not contain all {task.expected} "
                f"entries. Output the complete list: all {task.expected} "
                "entries, each as ONE compact JSON object on its own line, "
                "from the first entry to the last — no pretty-printing, no "
                "early stop."
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        try:
            if ctx.dry_run:
                result = await _fake_stream(task, blinker)
            else:
                result = await ctx.stream_fn(ctx.http, ctx.cfg.model, messages,
                                             ctx.api_key, on_token=on_token,
                                             on_activity=blinker.token)
        except LLMError as e:
            if e.fatal:
                ctx.fatal = str(e)
                blinker.status = "failed"
                _log(ctx, f"job-{index:02d} FATAL: {e} (task {task.id} aborted)")
                _save_meta(ctx, task, result, "fatal", str(e), detail=e.detail)
                return
            if e.retryable and attempt < ctx.cfg.jobs.max_retries:
                _log(ctx, f"job-{index:02d} {task.id}: attempt {attempt + 1} failed: {e}")
                # write meta on every attempt so failures are inspectable
                # while the retries are still running
                _save_meta(ctx, task, result, "retry", str(e), detail=e.detail)
                continue
            ctx.stats.failed += 1
            blinker.status = "failed"
            ctx.checkpoint.mark_failed(task.id, str(e))
            _log(ctx, f"job-{index:02d} {task.id} failed after {attempt + 1} "
                      f"attempt(s): {e}")
            _save_meta(ctx, task, result, "error", str(e), detail=e.detail)
            return

        # ---- quality gate: run on every attempt, so bad-but-successful
        # ---- streams retry *within this run* (with the early-stop nudge)
        parser.flush()          # line parsing + whole-text fallback both land here
        entries = list(parser.entries)  # (flush() must NOT be added again:
        valid = _shape_filter(task, entries)  # it returns what it just appended)
        unique = len({e["w"] for e in valid})
        if not ctx.dry_run:
            _save_raw(ctx.raw_dir, task, parser.raw)

        notes = []
        if result.finish_reason == "length":
            notes.append("truncated at max_tokens")
        if result.salvaged_from_reasoning:
            notes.append("salvaged from reasoning")
        if result.events == 0:
            notes.append("no content streamed")
        if parser.bad_lines:
            notes.append(f"{parser.bad_lines} bad lines")

        looping = is_degenerate(len(valid), unique)
        short = 0 < task.expected and len(valid) < task.expected * task.min_fraction
        if not ctx.dry_run and (not valid or looping or short):
            if not valid:
                reason = "0 valid entries"
            elif looping:
                reason = f"repetition loop: only {unique} unique of {len(valid)}"
            else:
                reason = f"incomplete: only {len(valid)}/{task.expected} entries"
            if attempt < ctx.cfg.jobs.max_retries:
                _log(ctx, f"job-{index:02d} {task.id}: {reason} — retrying in run")
                _save_meta(ctx, task, result, "retry", reason)
                continue
            ctx.stats.failed += 1
            blinker.status = "failed"
            ctx.checkpoint.mark_failed(task.id, f"{reason} (raw saved to {task.id}.raw.txt)")
            _log(ctx, f"job-{index:02d} ✗ {task.id} — {reason} after "
                      f"{attempt + 1} attempt(s); raw kept"
                  + (f" [{'; '.join(notes)}]" if notes else ""))
            _save_meta(ctx, task, result, "unusable", reason)
            return

        # ---- usable output ----
        if not ctx.dry_run:
            _write_shard(ctx.raw_dir, task, valid)
            ctx.checkpoint.mark_done(task.id, len(valid), result.cost, result.events)
        ctx.stats.completed += 1
        ctx.stats.entries += len(valid)
        ctx.stats.tokens += result.events
        ctx.stats.cost += result.cost
        blinker.status = "done"
        suffix = f" [{' '.join(notes)}]" if notes else ""
        _log(ctx, f"job-{index:02d} ✓ {task.id} — {len(valid)} entries "
                  f"({unique} unique), {result.events} tokens{suffix}")
        _save_meta(ctx, task, result, "done",
                   f"{len(valid)} entries, {unique} unique")
        return


def _save_meta(ctx: _Ctx, task: Task, result: StreamResult, status: str,
               note: str = "", detail: dict | None = None) -> None:
    """Sidecar diagnostics: what the provider actually returned.

    ``detail`` comes from the raised LLMError and carries the *real* stream
    stats (the local ``result`` may be stale when a stream failed)."""
    if ctx.dry_run:
        return
    ctx.raw_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "task": task.id,
        "kind": task.kind,
        "status": status,
        "note": note,
    }
    meta.update(result.detail())
    if detail:
        meta.update(detail)
    meta["salvaged_from_reasoning"] = result.salvaged_from_reasoning
    meta["errors"] = list(result.errors)
    (ctx.raw_dir / f"{task.id}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


async def _fake_stream(task: Task, blinker) -> StreamResult:
    """Dry-run: emit canned tokens so the dashboard can be demoed offline."""
    result = StreamResult()
    lines = 12
    for i in range(1, lines + 1):
        await asyncio.sleep(0.04)
        chunk = (f'{{"w": "단어{i}", "r": {i}}}\n' if task.kind == "words"
                 else f'{{"w": "단어{i}", "b": ["다음1", "다음2", "다음3"]}}\n')
        result.text += chunk
        result.events += 1
        blinker.token(chunk)
    result.finish_reason = "stop"
    return result


_WORD_KEYS = ("w", "word", "kr", "kor")
_RANK_KEYS = ("r", "rank", "f", "no", "index", "n", "order", "id")
_BIGRAM_KEYS = ("b", "next", "bigrams", "follows")


def _first(entry: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in entry:
            return entry[key]
    return None


def _as_rank(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _shape_filter(task: Task, entries: list[dict]) -> list[dict]:
    """Keep well-formed entries; words get their global rank (rank_base + r).

    Deliberately lenient about key names: models sometimes emit
    ``{"word": ..., "rank": ...}`` instead of ``{"w": ..., "r": ...}``,
    quote strings, or even a plain ``word -> rank`` mapping.
    """
    valid: list[dict] = []
    for entry in entries:
        word = _first(entry, _WORD_KEYS)
        if not isinstance(word, str):
            # plain mapping: {"사과": 1, "학교": 2}
            for key, value in entry.items():
                if key in (*_WORD_KEYS, *_RANK_KEYS, *_BIGRAM_KEYS):
                    continue
                rank = _as_rank(value)
                if isinstance(key, str) and key and rank is not None:
                    word, fallback_rank = key, rank
                    break
            else:
                continue
        else:
            fallback_rank = None
        word = str(word).strip().strip("\"'“”‘’").strip()
        if not word:
            continue

        if task.kind == "words":
            rank = _as_rank(_first(entry, _RANK_KEYS))
            if rank is None:
                rank = fallback_rank
            if rank is None:
                # no rank in the entry: output order is frequency order
                rank = len(valid) + 1
            if rank < 1:
                continue
            if task.rank_base and task.expected >= 1 and rank > task.expected:
                # the model numbered with global ranks (e.g. 1600 for a band
                # of 1201-1600) instead of local 1..N — use them as-is
                global_rank = rank
            else:
                global_rank = task.rank_base + rank
            valid.append({"w": word, "r": global_rank})
        elif task.kind == "bigrams":
            bigrams = _first(entry, _BIGRAM_KEYS)
            if not isinstance(bigrams, list):
                continue
            b = [w.strip() for w in bigrams[:3] if isinstance(w, str) and w.strip()]
            if b:
                valid.append({"w": word, "b": b})
    return valid


def _save_raw(raw_dir: Path, task: Task, text: str) -> None:
    """Keep the raw model output for debugging/re-parsing."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{task.id}.raw.txt").write_text(text, encoding="utf-8")


def _write_shard(raw_dir: Path, task: Task, entries: list[dict]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{task.id}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
