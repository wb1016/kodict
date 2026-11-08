"""Tests for the parallel extraction runner (mocked streaming)."""

import asyncio
import io
import json

import pytest

from kodict.blinker import Dashboard
from kodict.config import Config, JobsConfig
from kodict.llm import LLMError, StreamResult
from kodict.runner import run_tasks
from kodict.tasks import Task


def make_cfg(jobs: int = 2, retries: int = 2) -> Config:
    return Config(jobs=JobsConfig(count=jobs, max_retries=retries))


def make_tasks(n: int = 4, kind: str = "words") -> list[Task]:
    return [Task(id=f"t{i}", kind=kind, prompt=f"prompt-{i}", rank_base=100 * i)
            for i in range(n)]


def make_dashboard(jobs: int) -> Dashboard:
    return Dashboard(jobs, out=io.StringIO(), enabled=True)


def fake_ok(entries: int = 2, calls: list | None = None):
    async def stream_fn(http, model, messages, api_key, on_token=None, **kwargs):
        if calls is not None:
            calls.append(messages[1]["content"])
        await asyncio.sleep(0)  # yield so workers interleave like real streams
        result = StreamResult(cost=0.001)
        for i in range(entries):
            line = json.dumps({"w": f"단어{i}", "r": i + 1}, ensure_ascii=False) + "\n"
            result.text += line
            result.events += 1
            if on_token:
                on_token(line)
        result.finish_reason = "stop"
        return result
    return stream_fn


def run(cfg, tasks, dash, tmp_path, **kwargs) -> object:
    kwargs.setdefault("checkpoint_path", tmp_path / "checkpoint.json")
    kwargs.setdefault("raw_dir", tmp_path / "raw")
    kwargs.setdefault("dashboard", dash)
    kwargs.setdefault("backoff_base", 0.001)
    kwargs.setdefault("stream_fn", fake_ok())  # never hit the real API in tests
    return asyncio.run(run_tasks(cfg, tasks, **kwargs))


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")


def test_success_writes_shards_and_checkpoint(tmp_path):
    cfg, tasks = make_cfg(jobs=2), make_tasks(4)
    dash = make_dashboard(2)
    stats = run(cfg, tasks, dash, tmp_path)

    assert stats.completed == 4 and stats.failed == 0 and stats.ok
    assert stats.entries == 8
    assert stats.cost == pytest.approx(0.004)
    # shards contain global ranks (rank_base + local r); t1 has rank_base=100
    first = json.loads((tmp_path / "raw" / "t1.jsonl").read_text().splitlines()[0])
    assert first == {"w": "단어0", "r": 101}
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert sorted(checkpoint["completed"]) == ["t0", "t1", "t2", "t3"]
    # blinkers that ran a task ended in the done state, none failed
    assert all(b.status in ("done", "idle") for b in dash.blinkers)
    assert any(b.status == "done" for b in dash.blinkers)
    assert dash.blinkers[0].tokens > 0  # it actually blinked


def test_resume_skips_completed_tasks(tmp_path):
    cfg, tasks = make_cfg(), make_tasks(4)
    calls: list[str] = []
    run(cfg, tasks, make_dashboard(2), tmp_path, stream_fn=fake_ok(calls=calls))
    assert len(calls) == 4

    calls2: list[str] = []
    dash = make_dashboard(2)
    stats = run(cfg, tasks, dash, tmp_path, stream_fn=fake_ok(calls=calls2))
    assert stats.skipped == 4
    assert stats.completed == 4
    assert calls2 == []  # nothing re-requested


def test_retry_then_success(tmp_path):
    cfg, tasks = make_cfg(jobs=1, retries=3), make_tasks(1)
    attempts = {"n": 0}

    async def flaky(http, model, messages, api_key, on_token=None, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise LLMError("HTTP 429", status=429, retryable=True)
        return await fake_ok()(http, model, messages, api_key, on_token)

    stats = run(cfg, tasks, make_dashboard(1), tmp_path, stream_fn=flaky)
    assert stats.completed == 1
    assert stats.retried == 1
    assert stats.failed == 0
    assert attempts["n"] == 2


def test_non_retryable_failure_is_recorded(tmp_path):
    cfg, tasks = make_cfg(jobs=1, retries=2), make_tasks(1)

    async def broken(http, model, messages, api_key, on_token=None, **kwargs):
        raise LLMError("HTTP 400: bad request", status=400)

    stats = run(cfg, tasks, make_dashboard(1), tmp_path, stream_fn=broken)
    assert stats.failed == 1 and not stats.ok
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert "t0" in checkpoint["failed"]
    assert not (tmp_path / "raw" / "t0.jsonl").exists()


def test_fatal_error_aborts_run(tmp_path):
    cfg, tasks = make_cfg(jobs=1, retries=3), make_tasks(3)

    async def unauthorized(http, model, messages, api_key, on_token=None, **kwargs):
        raise LLMError("HTTP 401: invalid key", status=401, fatal=True)

    stats = run(cfg, tasks, make_dashboard(1), tmp_path, stream_fn=unauthorized)
    assert stats.fatal is not None and "401" in stats.fatal
    assert stats.completed == 0


def test_limit_caps_tasks(tmp_path):
    cfg, tasks = make_cfg(), make_tasks(6)
    stats = run(cfg, tasks, make_dashboard(2), tmp_path, limit=2)
    assert stats.total == 2
    assert stats.completed == 2


def test_dry_run_needs_no_key_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    cfg, tasks = make_cfg(jobs=3), make_tasks(3)
    dash = make_dashboard(3)
    stats = run(cfg, tasks, dash, tmp_path, dry_run=True)
    assert stats.completed == 3
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "checkpoint.json").exists()
    assert all(b.status == "done" for b in dash.blinkers)
    assert dash.blinkers[0].tokens > 0


def test_bigram_entries_filtered_and_written(tmp_path):
    cfg, tasks = make_cfg(jobs=1), make_tasks(1, kind="bigrams")

    async def bigrams(http, model, messages, api_key, on_token=None, **kwargs):
        result = StreamResult()
        for line in ['{"w": "사과", "b": ["먹다", 5, "주다"]}',
                     '{"w": "학교", "b": "not-a-list"}',
                     '{"w": "커피", "b": ["마시다", "얼음", "한잔", "더"]}\n']:
            text = line if line.endswith("\n") else line + "\n"
            result.text += text
            result.events += 1
            if on_token:
                on_token(text)
        return result

    stats = run(cfg, tasks, make_dashboard(1), tmp_path, stream_fn=bigrams)
    assert stats.entries == 2  # malformed shapes dropped, good ones kept
    lines = (tmp_path / "raw" / "t0.jsonl").read_text().splitlines()
    assert json.loads(lines[0]) == {"w": "사과", "b": ["먹다", "주다"]}
    assert json.loads(lines[1]) == {"w": "커피", "b": ["마시다", "얼음", "한잔"]}


def test_zero_entries_marks_failed_and_keeps_raw(tmp_path):
    cfg, tasks = make_cfg(jobs=1), make_tasks(1)

    async def prose_only(http, model, messages, api_key, on_token=None, **kwargs):
        result = StreamResult()
        for text in ["Here are the words you asked for:\n", "사과와 학교 등등\n"]:
            result.text += text
            result.events += 1
            if on_token:
                on_token(text)
        result.finish_reason = "stop"
        return result

    stats = run(cfg, tasks, make_dashboard(1), tmp_path, stream_fn=prose_only)
    assert stats.failed == 1 and stats.completed == 0 and not stats.ok
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert "t0" in checkpoint["failed"] and "t0" not in checkpoint["completed"]
    # raw output kept for inspection, so the format can be diagnosed
    raw = (tmp_path / "raw" / "t0.raw.txt").read_text()
    assert "사과와 학교" in raw
    # next run will retry it (only "completed" tasks are skipped)


def test_shape_filter_accepts_aliases(tmp_path):
    from kodict.runner import _shape_filter
    from kodict.tasks import Task

    task = Task(id="x", kind="words", prompt="p", rank_base=1000)
    entries = [
        {"word": "사과", "rank": "12"},     # alias keys, string rank
        {"kr": "학교", "f": 7},              # more aliases
        {"사과나무": 3},                      # plain word -> rank mapping
        {"w": "coffee", "quoted": True},     # no rank at all -> sequential
        {"w": "", "r": 1},                   # empty word -> dropped
        {"w": "길", "r": -5},                # negative rank -> dropped
    ]
    out = _shape_filter(task, entries)
    words = {e["w"]: e["r"] for e in out}
    assert words["사과"] == 1012
    assert words["학교"] == 1007
    assert words["사과나무"] == 1003
    assert words["coffee"] == 1004        # sequential fallback (len(valid)+1)
    assert "길" not in words and "" not in words


def test_raw_saved_on_success(tmp_path):
    cfg, tasks = make_cfg(jobs=1), make_tasks(1)
    stats = run(cfg, tasks, make_dashboard(1), tmp_path)
    assert stats.completed == 1
    raw = tmp_path / "raw" / "t0.raw.txt"
    assert raw.is_file() and '"w"' in raw.read_text()


def test_completed_tasks_with_no_entries_are_retried(tmp_path):
    """A task recorded with 0 entries must not count as done."""
    cfg, tasks = make_cfg(jobs=1), make_tasks(1)
    from kodict.runner import Checkpoint
    cp = Checkpoint(tmp_path / "checkpoint.json", enabled=True)
    cp.mark_done("t0", 0, 0.0, 0)   # simulate the buggy first run
    assert not cp.done("t0")
    cp.mark_done("t0", 5, 0.1, 10)
    assert cp.done("t0")
    cp.mark_failed("t0", "boom")
    assert not cp.done("t0")
    cp.mark_done("t0", 5, 0.1, 10)  # success clears the failed record
    assert "t0" not in cp.data["failed"]


def test_is_degenerate():
    from kodict.runner import is_degenerate
    assert is_degenerate(401, 5)        # the real loop we captured
    assert not is_degenerate(401, 300)  # healthy band
    assert not is_degenerate(5, 5)      # small but all unique
    assert not is_degenerate(1, 1)
    assert not is_degenerate(0, 0)


def test_repetition_loop_marked_failed_and_raw_kept(tmp_path):
    cfg, tasks = make_cfg(jobs=1), make_tasks(1)

    async def looping(http, model, messages, api_key, on_token=None, **kwargs):
        result = StreamResult()
        cycle = ["저", "나", "너"]
        for i in range(100):
            line = json.dumps({"rank": i + 1, "word": cycle[i % 3]},
                              ensure_ascii=False) + "\n"
            result.text += line
            result.events += 1
            if on_token:
                on_token(line)
        result.finish_reason = "stop"
        return result

    stats = run(cfg, tasks, make_dashboard(1), tmp_path, stream_fn=looping)
    assert stats.failed == 1 and stats.completed == 0 and not stats.ok
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert "repetition loop" in checkpoint["failed"]["t0"]
    assert (tmp_path / "raw" / "t0.raw.txt").is_file()


def test_degenerate_shard_is_retried_on_resume(tmp_path):
    """A checkpointed task whose shard loops must be re-run, not skipped."""
    cfg, tasks = make_cfg(jobs=1), make_tasks(1)
    from kodict.runner import Checkpoint
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # simulate the broken first run: checkpoint says done, shard loops
    lines = [json.dumps({"w": w}, ensure_ascii=False)
             for i in range(100) for w in [["저", "나", "너"][i % 3]]]
    (raw_dir / "t0.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cp = Checkpoint(tmp_path / "checkpoint.json", enabled=True)
    cp.mark_done("t0", 100, 0.5, 1000)

    calls: list[str] = []
    stats = run(cfg, tasks, make_dashboard(1), tmp_path,
                stream_fn=fake_ok(calls=calls))
    assert len(calls) == 1          # re-requested instead of skipped
    assert stats.skipped == 0 and stats.completed == 1


def test_healthy_shard_is_skipped_on_resume(tmp_path):
    cfg, tasks = make_cfg(jobs=1), make_tasks(1)
    from kodict.runner import Checkpoint
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    lines = [json.dumps({"w": f"단어{i}", "r": i + 1}, ensure_ascii=False)
             for i in range(30)]
    (raw_dir / "t0.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cp = Checkpoint(tmp_path / "checkpoint.json", enabled=True)
    cp.mark_done("t0", 30, 0.5, 1000)

    calls: list[str] = []
    stats = run(cfg, tasks, make_dashboard(1), tmp_path,
                stream_fn=fake_ok(calls=calls))
    assert calls == []
    assert stats.skipped == 1 and stats.completed == 1


def test_meta_sidecar_and_run_log(tmp_path):
    cfg, tasks = make_cfg(jobs=1), make_tasks(2)

    async def prose_only(http, model, messages, api_key, on_token=None, **kwargs):
        result = StreamResult(finish_reason="stop")
        result.text = "some prose in korean\n"
        result.events = 1
        if on_token:
            on_token(result.text)
        return result

    # success first (t0 via default fake), then unusable output for t1
    run(cfg, tasks[:1], make_dashboard(1), tmp_path)
    meta = json.loads((tmp_path / "raw" / "t0.meta.json").read_text())
    assert meta["status"] == "done"
    assert meta["finish_reason"] == "stop"
    assert meta["note"].startswith("2 entries")

    run(cfg, tasks[1:], make_dashboard(1), tmp_path, stream_fn=prose_only)
    meta = json.loads((tmp_path / "raw" / "t1.meta.json").read_text())
    assert meta["status"] == "unusable"
    assert meta["note"] == "0 valid entries"

    log = (tmp_path / "run.log").read_text()
    assert "✓ t0" in log and "✗ t1" in log
    assert "20" in log[:20]  # timestamped


def test_error_meta_on_llm_failure(tmp_path):
    cfg, tasks = make_cfg(jobs=1, retries=0), make_tasks(1)

    async def broken(http, model, messages, api_key, on_token=None, **kwargs):
        raise LLMError("HTTP 400: bad request", status=400)

    run(cfg, tasks, make_dashboard(1), tmp_path, stream_fn=broken)
    meta = json.loads((tmp_path / "raw" / "t0.meta.json").read_text())
    assert meta["status"] == "error"
    assert "HTTP 400" in meta["note"]
    log = (tmp_path / "run.log").read_text()
    assert "failed after 1 attempt" in log


def test_error_meta_carries_stream_detail(tmp_path):
    cfg, tasks = make_cfg(jobs=1, retries=0), make_tasks(1)

    async def thinking_provider(http, model, messages, api_key, on_token=None, **kw):
        raise LLMError("empty stream (finish=stop, provider=CoreWeave)",
                       retryable=True,
                       detail={"finish_reason": "stop", "provider": "CoreWeave",
                               "reasoning_chars": 11933})

    run(cfg, tasks, make_dashboard(1), tmp_path, stream_fn=thinking_provider)
    meta = json.loads((tmp_path / "raw" / "t0.meta.json").read_text())
    assert meta["status"] == "error"
    assert meta["provider"] == "CoreWeave"
    assert meta["reasoning_chars"] == 11933
    assert "empty stream" in meta["note"]


def test_incomplete_output_marks_failed(tmp_path):
    """4 of 400 requested entries must never count as success."""
    from kodict.tasks import Task
    tasks = [Task(id="freq-x", kind="words", prompt="p",
                  rank_base=0, expected=400)]

    async def stops_early(http, model, messages, api_key, on_token=None, **kw):
        result = StreamResult(finish_reason="stop")
        for i in range(1, 5):
            line = json.dumps({"rank": i, "word": f"단어{i}"}, ensure_ascii=False) + "\n"
            result.text += line
            result.events += 1
            if on_token:
                on_token(line)
        return result

    cfg = make_cfg(jobs=1, retries=1)
    stats = run(cfg, tasks, make_dashboard(1), tmp_path, stream_fn=stops_early)
    assert stats.failed == 1 and stats.completed == 0
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert "incomplete: only 4/400" in checkpoint["failed"]["freq-x"]
    # the retry nudge was included in the second attempt's prompt
    assert (tmp_path / "run.log").read_text()  # attempts logged


def test_incomplete_shard_is_retried_on_resume(tmp_path):
    from kodict.tasks import Task
    from kodict.runner import Checkpoint
    tasks = [Task(id="freq-x", kind="words", prompt="p", expected=10)]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    lines = [json.dumps({"w": f"단어{i}", "r": i + 1}) for i in range(4)]
    (raw_dir / "freq-x.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cp = Checkpoint(tmp_path / "checkpoint.json", enabled=True)
    cp.mark_done("freq-x", 4, 0.1, 100)

    calls: list[str] = []
    stats = run(make_cfg(jobs=1), tasks, make_dashboard(1), tmp_path,
                stream_fn=fake_ok(entries=8, calls=calls))
    assert len(calls) == 1  # re-run, not skipped
    assert stats.completed == 1


def test_retry_prompt_nudges_for_completion(tmp_path):
    from kodict.tasks import Task
    tasks = [Task(id="freq-x", kind="words", prompt="list them", expected=400)]
    prompts: list[str] = []
    attempts = {"n": 0}

    async def flaky(http, model, messages, api_key, on_token=None, **kw):
        prompts.append(messages[1]["content"])
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise LLMError("empty stream", retryable=True)
        return await fake_ok()(http, model, messages, api_key, on_token, **kw)

    run(make_cfg(jobs=1, retries=2), tasks, make_dashboard(1), tmp_path,
        stream_fn=flaky, backoff_base=0.001)
    assert "list them" == prompts[0]
    assert "did not contain all 400" in prompts[1]


def test_quality_retry_completes_within_one_run(tmp_path):
    """A short stream retries in-run with the early-stop nudge."""
    from kodict.tasks import Task
    tasks = [Task(id="freq-x", kind="words", prompt="list them",
                  rank_base=0, expected=20)]
    prompts: list[str] = []
    calls = {"n": 0}

    async def short_then_full(http, model, messages, api_key, on_token=None, **kw):
        prompts.append(messages[1]["content"])
        calls["n"] += 1
        count = 4 if calls["n"] == 1 else 25
        result = StreamResult(finish_reason="stop")
        for i in range(1, count + 1):
            line = json.dumps({"rank": i, "word": f"단어{i}"}, ensure_ascii=False) + "\n"
            result.text += line
            result.events += 1
            if on_token:
                on_token(line)
        return result

    stats = run(make_cfg(jobs=1, retries=2), tasks, make_dashboard(1), tmp_path,
                stream_fn=short_then_full, backoff_base=0.001)
    assert stats.completed == 1 and stats.failed == 0
    assert stats.retried == 1
    assert calls["n"] == 2
    assert "list them" == prompts[0]
    assert "did not contain all 20" in prompts[1]  # nudge reached attempt 2


def test_failed_attempt_text_does_not_leak_into_next_attempt(tmp_path):
    cfg, tasks = make_cfg(jobs=1, retries=1), make_tasks(1)
    calls = {"n": 0}

    async def partial_then_good(http, model, messages, api_key, on_token=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            if on_token:
                on_token('{"w": "고양이", "r": 1}\n{"w": ')  # dies mid-line
            raise LLMError("provider died mid-stream", retryable=True)
        return await fake_ok()(http, model, messages, api_key, on_token, **kw)

    stats = run(cfg, tasks, make_dashboard(1), tmp_path,
                stream_fn=partial_then_good, backoff_base=0.001)
    assert stats.completed == 1
    shard = (tmp_path / "raw" / "t0.jsonl").read_text()
    assert "고양이" not in shard       # attempt 1's partial text is gone
    assert "단어0" in shard            # attempt 2's output only


def test_shape_filter_distinguishes_global_ranks():
    from kodict.runner import _shape_filter
    from kodict.tasks import Task

    task = Task(id="x", kind="words", prompt="p", rank_base=1200, expected=400)
    entries = [{"w": "기억", "r": 1600},    # model used global ranks
               {"w": "결정", "r": 1599},    # ...descending, still global
               {"w": "단어", "r": 7}]       # local rank
    ranks = {e["w"]: e["r"] for e in _shape_filter(task, entries)}
    assert ranks == {"기억": 1600, "결정": 1599, "단어": 1207}


def test_category_tasks_use_relaxed_fraction():
    from kodict import tasks as T
    cats = [t for t in T.generate_word_tasks() if t.id.startswith("cat-")]
    freq = [t for t in T.generate_word_tasks() if t.id.startswith("freq-")]
    assert cats and all(t.min_fraction == 0.25 for t in cats)
    assert freq and all(t.min_fraction == 0.5 for t in freq)
    assert all(t.id != "cat-024" for t in cats)  # doomed category removed
    assert any(t.id == "cat-023" for t in cats)  # but 0..023 are intact


def test_relaxed_fraction_allows_partial_category(tmp_path):
    from kodict.tasks import Task
    from kodict.runner import Checkpoint
    task = Task(id="cat-x", kind="words", prompt="p", expected=300,
                min_fraction=0.25)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    lines = [json.dumps({"w": f"단어{i}", "r": i + 1}) for i in range(130)]
    (raw_dir / "cat-x.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cp = Checkpoint(tmp_path / "checkpoint.json", enabled=True)
    cp.mark_done("cat-x", 130, 0.1, 100)
    # 130/300 = 43%: below the default 50% but above the category's 25%
    assert not __import__("kodict.runner", fromlist=["x"]).shard_is_incomplete(raw_dir, task)

    async def partial(http, model, messages, api_key, on_token=None, **kw):
        result = StreamResult(finish_reason="stop")
        for i in range(1, 131):
            line = json.dumps({"id": i, "word": f"단어{i}"}, ensure_ascii=False) + "\n"
            result.text += line
            result.events += 1
            if on_token:
                on_token(line)
        return result

    stats = run(make_cfg(jobs=1), [task], make_dashboard(1), tmp_path,
                stream_fn=partial)
    assert stats.skipped == 1 and stats.completed == 1  # shard accepted as-is
