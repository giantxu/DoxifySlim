"""作业端点：重放、cursor 续接、取消、404。

运行: cd /Users/xuzheng/CodingProjects/Doxify && conda run -n mineru python -m pytest tests/test_job_endpoints.py -v

本项目未装 pytest-asyncio，测试一律用 asyncio.run() 驱动 httpx.AsyncClient。
"""
import asyncio
import json

import httpx

import app


def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app),
                             base_url="http://test")


async def _read_events(resp, limit=None):
    """从 SSE 响应里读事件，读满 limit 条就停（用于模拟中途断线）。"""
    out, buf = [], ""
    async for chunk in resp.aiter_text():
        buf += chunk
        lines = buf.split("\n")
        buf = lines.pop()
        for line in lines:
            if not line.startswith("data: "):
                continue
            out.append(json.loads(line[6:]))
            if limit is not None and len(out) >= limit:
                return out
    return out


def test_events_replays_all_then_ends():
    async def runner(job):
        for n in range(3):
            app._publish(job, {"type": "tick", "n": n})

    async def run():
        app._jobs.clear()
        job = app._start_job("parse", runner)
        await job.task
        async with _client() as c:
            async with c.stream("GET", f"/jobs/{job.id}/events?cursor=0") as r:
                return await _read_events(r)

    evts = asyncio.run(run())
    assert evts[0]["type"] == "_replay" and evts[0]["count"] == 4
    assert [e["n"] for e in evts[1:4]] == [0, 1, 2]
    assert evts[-1]["type"] == "job_end" and evts[-1]["status"] == "done"


def test_reconnect_with_cursor_loses_and_duplicates_nothing():
    """断线重连的核心测试：断开后按 cursor 重连，事件序号必须连续无缺口。

    httpx 0.28.1 的 ASGITransport 不支持真正的增量流式传输——
    `handle_async_request` 内部 `await self.app(...)` 要等整个 ASGI 调用
    跑完（生成器 return）才会把 Response 交回给调用方（源码见
    httpx/_transports/asgi.py）。若把"读 3 条就断线"和"gate.set() 放行
    下半段"顺序写在同一条协程里，协程会卡在 `async with c.stream(...)`
    上——它必须等 job_end 事件出现生成器才会返回，而 job_end 又要等
    这条协程自己执行到 gate.set() 那一行才会发生，自己等自己，死锁
    （已用一个不依赖 app.py 的最小复现脚本单独验证）。用
    asyncio.create_task 把第一段读流放进独立任务，主协程才能在它
    卡住之后继续跑到 gate.set()。
    """
    async def run():
        # Event 必须在运行中的循环内创建：3.10 虽容忍循环外构造，但那是不该依赖的行为
        gate = asyncio.Event()

        async def runner(job):
            for n in range(3):
                app._publish(job, {"type": "tick", "n": n})
            await gate.wait()
            for n in range(3, 6):
                app._publish(job, {"type": "tick", "n": n})

        app._jobs.clear()
        job = app._start_job("parse", runner)
        await asyncio.sleep(0.05)
        async with _client() as c:
            async def read_first():
                # 第一段：只读 3 条就"断线"
                async with c.stream("GET", f"/jobs/{job.id}/events?cursor=0") as r:
                    return await _read_events(r, limit=3)

            first_task = asyncio.create_task(read_first())
            await asyncio.sleep(0.05)
            gate.set()
            first = await first_task
            cursor = first[-1]["i"] + 1
            await asyncio.gather(job.task, return_exceptions=True)
            # 第二段：带 cursor 重连
            async with c.stream("GET", f"/jobs/{job.id}/events?cursor={cursor}") as r:
                second = await _read_events(r)
        return first, second

    first, second = asyncio.run(run())
    got = [e["i"] for e in first if "i" in e] + [e["i"] for e in second if "i" in e]
    assert got == list(range(len(got))), f"事件序号必须连续无缺口，实得 {got}"
    assert second[-1]["type"] == "job_end"


def test_live_subscriber_receives_job_end_queued_while_sending():
    """作业在流发送途中结束时，已入队的 job_end 不能被丢掉。

    订阅发生在作业仍 running 时、backlog 为空（连上一个正在跑的作业）。作业在
    generator 发送 _replay 那一刻结束，job_end 被 put_nowait 进已注册的队列——
    此时若只看 status 就 return，客户端拿不到终止事件，会误判为断线并重连。
    httpx 的 ASGITransport 会整体缓冲响应，复现不了这个交错，故直接驱动生成器。
    """
    async def run():
        app._jobs.clear()
        gate = asyncio.Event()

        async def runner(job):
            await gate.wait()

        job = app._start_job("parse", runner)
        await asyncio.sleep(0.05)

        resp = await app.job_events(job.id, cursor=0)
        it = resp.body_iterator
        first = await it.__anext__()          # _replay 元信息
        gate.set()                            # 让作业在"发送途中"结束
        await asyncio.gather(job.task, return_exceptions=True)
        rest = [chunk async for chunk in it]
        return first, rest

    # wait_for 不是防御性装饰：这个测试驱动的正是 job_events 生成器，任何让它停在
    # await q.get() 的回归都会把整个测试套件挂死，而不是给出一条清晰的失败。
    first, rest = asyncio.run(asyncio.wait_for(run(), timeout=5))
    assert '"_replay"' in first
    assert any('"job_end"' in c for c in rest), f"终止事件被丢弃了: {rest}"


def test_overflowed_subscriber_stream_ends_instead_of_hanging(monkeypatch):
    """慢订阅者被踢掉后，SSE 生成器必须收流，绝不能永久停在 await q.get()。

    _publish 在队列满时把队列从 job.subscribers 里 discard 掉，此后再无人向它投递。
    生成器若继续 await 这条无人持有的队列，就是**永久静默**：不出字节、不关闭、不报错，
    客户端的 for await 永不 resolve，job-client.js 的整套重连/退避/放弃机制结构上
    无法启动。收流后客户端才会落进既有路径：流结束但无 job_end → 带 cursor 重连补齐。
    """
    monkeypatch.setattr(app, "SUBSCRIBER_QUEUE_MAX", 3)

    async def run():
        app._jobs.clear()
        gate = asyncio.Event()

        async def runner(job):
            await gate.wait()

        job = app._start_job("parse", runner)
        await asyncio.sleep(0.05)

        resp = await app.job_events(job.id, cursor=0)
        it = resp.body_iterator
        await it.__anext__()                  # _replay，此时 backlog 为空
        # 生成器停在 await q.get()，同步连发 4 条把 maxsize=3 的队列撑爆：
        # 第 4 条触发 QueueFull，订阅者被踢
        for n in range(4):
            app._publish(job, {"type": "tick", "n": n})
        assert not job.subscribers, "前提不成立：队列没被撑爆"

        out = []
        try:
            while True:
                # 每条都设超时：修复前这里会在排空已入队事件后永久挂起
                out.append(await asyncio.wait_for(it.__anext__(), timeout=2))
        except StopAsyncIteration:
            pass
        finally:
            gate.set()
            await asyncio.gather(job.task, return_exceptions=True)
        return out

    out = asyncio.run(run())
    # 收到的事件必须是从 0 开始连续的前缀——客户端据此推进 cursor，重连补齐其余
    got = [json.loads(c[6:])["n"] for c in out]
    assert got == list(range(len(got))), f"事件必须连续无缺口，实得 {got}"


def test_events_on_finished_job_with_stale_cursor_closes_immediately():
    """作业已结束、cursor 又越过终止事件时必须立即关闭，不能等实时事件。

    这种 cursor 来自已消费过 job_end 的客户端、过期的 localStorage 或 /jobs 恢复
    路径。此时 backlog 为空，若继续 await q.get() 会永久挂住连接。
    """
    async def runner(job):
        app._publish(job, {"type": "tick"})

    async def run():
        app._jobs.clear()
        job = app._start_job("parse", runner)
        await asyncio.gather(job.task, return_exceptions=True)
        stale = len(job.events)          # 越过 job_end
        async with _client() as c:
            async def read():
                async with c.stream("GET", f"/jobs/{job.id}/events?cursor={stale}") as r:
                    return await _read_events(r)
            return await asyncio.wait_for(read(), timeout=5)

    evts = asyncio.run(run())
    assert [e["type"] for e in evts] == ["_replay"]
    assert evts[0]["count"] == 0


def test_unknown_job_returns_404():
    async def run():
        async with _client() as c:
            return (await c.get("/jobs/deadbeef/events")).status_code

    assert asyncio.run(run()) == 404


def test_cancel_endpoint_moves_job_to_cancelled():
    async def runner(job):
        await asyncio.sleep(30)

    async def run():
        app._jobs.clear()
        job = app._start_job("parse", runner)
        await asyncio.sleep(0.05)
        async with _client() as c:
            r = await c.post(f"/jobs/{job.id}/cancel")
        await asyncio.gather(job.task, return_exceptions=True)
        return r.status_code, job.status, job.events[-1]

    code, status, last = asyncio.run(run())
    assert code == 200 and status == "cancelled"
    assert last["type"] == "job_end" and last["status"] == "cancelled"


def test_cancel_unknown_job_returns_404():
    async def run():
        async with _client() as c:
            return (await c.post("/jobs/nope/cancel")).status_code

    assert asyncio.run(run()) == 404


def test_jobs_list_reports_status():
    async def runner(job):
        app._publish(job, {"type": "tick"})

    async def run():
        app._jobs.clear()
        job = app._start_job("translate", runner)
        await job.task
        async with _client() as c:
            return (await c.get("/jobs")).json()

    data = asyncio.run(run())
    assert len(data["jobs"]) == 1
    row = data["jobs"][0]
    assert row["kind"] == "translate" and row["status"] == "done"
    assert "created_at" in row and "id" in row


def test_parse_job_returns_job_id_immediately(monkeypatch, tmp_path):
    """端点必须立即返回 job_id，不能等解析跑完。"""
    started = asyncio.Event()

    async def fake_parse(data, filename, file_id, queue):
        started.set()
        await queue.put({"type": "file_done", "file_id": file_id,
                         "filename": filename, "markdown": "x", "has_images": False})

    monkeypatch.setattr(app, "parse_pdf_streaming",
                        lambda d, fn, fid, q, *a, **k: fake_parse(d, fn, fid, q))

    async def run():
        app._jobs.clear()
        async with _client() as c:
            r = await c.post("/jobs/parse",
                             files={"files": ("a.pdf", b"%PDF-1.4", "application/pdf")},
                             data={"mode": "vlm"})
        body = r.json()
        job = app._jobs[body["job_id"]]
        await asyncio.gather(job.task, return_exceptions=True)
        return r.status_code, body, [e["type"] for e in job.events]

    code, body, types = asyncio.run(run())
    assert code == 200 and "job_id" in body
    assert types[0] == "init" and types[-1] == "job_end"


def test_parse_job_rejects_non_pdf():
    async def run():
        async with _client() as c:
            r = await c.post("/jobs/parse",
                             files={"files": ("a.txt", b"hi", "text/plain")},
                             data={"mode": "vlm"})
        return r.json()

    assert asyncio.run(run())["success"] is False


def test_old_parse_endpoint_is_gone():
    """旧端点必须删除：单用户工具保留两条路径只会导致逻辑漂移。"""
    async def run():
        async with _client() as c:
            return (await c.post("/parse_pdf_stream")).status_code

    assert asyncio.run(run()) == 404


def test_translate_job_returns_job_id_and_ends(monkeypatch):
    async def fake_chunk_stream(client, chunk, idx, total, lang, max_tokens=6000):
        yield "译文"

    monkeypatch.setattr(app, "translate_chunk_stream", fake_chunk_stream)
    monkeypatch.setattr(app, "_detect_residual_english", lambda buf: [])

    async def run():
        app._jobs.clear()
        async with _client() as c:
            r = await c.post("/jobs/translate",
                             data={"text": "hello world", "target_lang": "中文"})
        body = r.json()
        job = app._jobs[body["job_id"]]
        await asyncio.gather(job.task, return_exceptions=True)
        return body, [e["type"] for e in job.events]

    body, types = asyncio.run(run())
    assert "job_id" in body
    assert types[0] == "init" and types[-1] == "job_end"
    assert "all_done" in types


def test_translate_child_task_crash_ends_job_with_file_error(monkeypatch):
    """翻译子任务抛未捕获异常时，作业必须落到终态并发出 job_end。

    子任务一死就再也不投递 file_done，而 runner 的 `while done_count < len(tasks)`
    会永久阻塞：作业永远停在 running（_evict_old_jobs 又永不淘汰运行中作业），
    SSE 生成器停在 await q.get()，客户端看到一条永不结束的静默流——不重连、
    不放弃、无任何提示。注入点选 _detect_src_lang_code：它在 _process_chunk 的
    try 之外，只有文件级兜底能接住它。
    """
    async def fake_chunk_stream(client, chunk, idx, total, lang, max_tokens=6000):
        yield "译文"

    monkeypatch.setattr(app, "translate_chunk_stream", fake_chunk_stream)
    monkeypatch.setattr(app, "_detect_residual_english", lambda buf: [])

    def boom(text):
        raise RuntimeError("语种识别炸了")

    monkeypatch.setattr(app, "_detect_src_lang_code", boom)

    async def run():
        app._jobs.clear()
        async with _client() as c:
            r = await c.post("/jobs/translate",
                             data={"text": "hello world", "target_lang": "中文"})
        job = app._jobs[r.json()["job_id"]]
        await asyncio.wait_for(
            asyncio.gather(job.task, return_exceptions=True), timeout=5)
        return job.status, [e["type"] for e in job.events]

    status, types = asyncio.run(run())
    assert status != "running", "作业卡在 running：永久静默"
    assert types[-1] == "job_end"
    assert "file_error" in types, f"未发出 file_error，卡片会无解释地一直转: {types}"


def test_translate_residual_detection_crash_still_finishes_file(monkeypatch):
    """纵深防御：残留英文检测抛异常不该让整块任务死掉，file_done 照发。"""
    async def fake_chunk_stream(client, chunk, idx, total, lang, max_tokens=6000):
        yield "译文"

    monkeypatch.setattr(app, "translate_chunk_stream", fake_chunk_stream)

    def boom(buf):
        raise RuntimeError("残留检测炸了")

    monkeypatch.setattr(app, "_detect_residual_english", boom)

    async def run():
        app._jobs.clear()
        async with _client() as c:
            r = await c.post("/jobs/translate",
                             data={"text": "hello world", "target_lang": "中文"})
        job = app._jobs[r.json()["job_id"]]
        await asyncio.wait_for(
            asyncio.gather(job.task, return_exceptions=True), timeout=5)
        return job.status, [e["type"] for e in job.events]

    status, types = asyncio.run(run())
    assert status != "running"
    assert types[-1] == "job_end"
    assert "file_done" in types and "all_done" in types


def test_translate_job_rejects_empty_input():
    async def run():
        async with _client() as c:
            r = await c.post("/jobs/translate", data={"text": "  "})
        return r.json()

    assert asyncio.run(run())["success"] is False


def test_old_translate_endpoint_is_gone():
    async def run():
        async with _client() as c:
            return (await c.post("/translate_stream")).status_code

    assert asyncio.run(run()) == 404
