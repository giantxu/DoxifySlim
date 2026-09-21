"""作业注册表：发布/订阅、cursor 重放、淘汰、终态。

运行: cd /Users/xuzheng/CodingProjects/Doxify && conda run -n mineru python -m pytest tests/test_job_registry.py -v

背景见 docs/superpowers/specs/2026-08-07-job-registry-reconnect-design.md：
任务此前绑在 SSE 请求生命周期上，客户端一断事件就没人接收，任务变成孤儿。
"""
import asyncio

import pytest

import app


def _mk(kind="parse"):
    return app.Job(id="j1", kind=kind, status="running", created_at=100.0)


def test_publish_assigns_increasing_index():
    job = _mk()
    app._publish(job, {"type": "a"})
    app._publish(job, {"type": "b"})
    assert [e["i"] for e in job.events] == [0, 1]
    assert job.events[1]["type"] == "b"


def test_subscribe_returns_backlog_from_cursor():
    job = _mk()
    for n in range(5):
        app._publish(job, {"type": "e", "n": n})
    q, backlog = app._subscribe(job, 2)
    assert [e["n"] for e in backlog] == [2, 3, 4]
    assert q in job.subscribers


def test_subscribe_clamps_out_of_range_cursor():
    job = _mk()
    app._publish(job, {"type": "e"})
    assert app._subscribe(job, 99)[1] == []
    assert len(app._subscribe(job, -5)[1]) == 1


def test_no_gap_no_duplicate_across_subscribe_boundary():
    """订阅与快照之间不得插入 await，否则会漏事件。

    这是整个重连机制的正确性基石：如果 _subscribe 先取快照再加订阅，
    两步之间发布的事件就会永久丢失，且丢失只在高并发下偶现、极难排查。
    """
    async def run():
        job = _mk()
        for n in range(3):
            app._publish(job, {"type": "e", "n": n})
        q, backlog = app._subscribe(job, 0)
        for n in range(3, 6):
            app._publish(job, {"type": "e", "n": n})
        live = []
        while not q.empty():
            live.append(q.get_nowait())
        return [e["n"] for e in backlog] + [e["n"] for e in live]

    assert asyncio.run(run()) == [0, 1, 2, 3, 4, 5]


def test_slow_subscriber_is_evicted_without_affecting_others():
    async def run():
        job = _mk()
        slow = asyncio.Queue(maxsize=1)
        fast = asyncio.Queue(maxsize=100)
        job.subscribers.update({slow, fast})
        for n in range(5):
            app._publish(job, {"type": "e", "n": n})
        return slow in job.subscribers, fast in job.subscribers, fast.qsize()

    slow_alive, fast_alive, fast_n = asyncio.run(run())
    assert not slow_alive, "队列溢出的订阅者必须被踢掉，否则死标签页会吃光内存"
    assert fast_alive and fast_n == 5


def test_eviction_keeps_running_jobs_and_drops_oldest_finished(monkeypatch):
    monkeypatch.setattr(app, "JOB_HISTORY_MAX", 2)
    monkeypatch.setattr(app, "_jobs", {})
    for n in range(4):
        j = app.Job(id=f"done{n}", kind="parse", status="done", created_at=float(n))
        app._jobs[j.id] = j
    running = app.Job(id="run", kind="parse", status="running", created_at=-1.0)
    app._jobs["run"] = running

    app._evict_old_jobs()

    assert "run" in app._jobs, "运行中的作业永不淘汰"
    assert sorted(k for k in app._jobs if k != "run") == ["done2", "done3"]


@pytest.mark.parametrize("outcome,expected", [
    ("ok", "done"),
    ("boom", "error"),
])
def test_terminal_paths_publish_job_end(outcome, expected, monkeypatch):
    monkeypatch.setattr(app, "_jobs", {})

    async def runner(job):
        if outcome == "boom":
            raise ValueError("炸了")

    async def run():
        job = app._start_job("parse", runner)
        await asyncio.gather(job.task, return_exceptions=True)
        return job

    job = asyncio.run(run())
    assert job.status == expected
    end = job.events[-1]
    assert end["type"] == "job_end" and end["status"] == expected
    if expected == "error":
        assert "炸了" in end["error"]


def test_cancel_publishes_job_end_with_cancelled(monkeypatch):
    """取消也必须发终止事件，否则前端会无限重连、看起来像程序挂死。"""
    monkeypatch.setattr(app, "_jobs", {})

    async def runner(job):
        await asyncio.sleep(30)

    async def run():
        job = app._start_job("parse", runner)
        await asyncio.sleep(0.05)
        job.task.cancel()
        await asyncio.gather(job.task, return_exceptions=True)
        return job

    job = asyncio.run(run())
    assert job.status == "cancelled"
    assert job.events[-1] == {"type": "job_end", "status": "cancelled",
                              "error": None, "i": 0}
