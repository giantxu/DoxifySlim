"""解析任务的错误兜底：单个文件解析抛异常不得让 SSE 连接挂起。

bug 复现：VLM 路径 parse_pdf_streaming 在 pdf_to_images 阶段对损坏 PDF 会抛异常，
若不兜底则 task 直接死亡、不发 file_done/file_error，
导致 parse_pdf_stream._generate 的 `while done_count + error_count < total_files`
永久挂起。修复：用 _run_parse_task 包装 task_fn，任何异常都转成一条 file_error 事件。
（MinerU / PaddleOCR 路径自身已发 file_error 并正常返回，包装器不会重复发。）
"""
import asyncio

import app


def _drain(queue: asyncio.Queue) -> list[dict]:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def test_raising_task_emits_file_error_not_exception():
    async def boom(data, filename, file_id, queue):
        raise ValueError("corrupt pdf")

    async def run():
        queue: asyncio.Queue = asyncio.Queue()
        await app._run_parse_task(boom, b"x", "broken.pdf", "fid-err", queue)
        return _drain(queue)

    events = asyncio.run(run())  # 不抛异常即第一关
    errors = [e for e in events if e.get("type") == "file_error"]
    assert errors, f"expected a file_error event, got: {events}"
    err = errors[0]
    assert err["file_id"] == "fid-err"
    assert err["filename"] == "broken.pdf"
    assert "corrupt pdf" in err.get("error", "")


def test_successful_task_does_not_inject_extra_file_error():
    """后端自身已发终止事件时，包装器不得再补一条 file_error（防止计数错乱）。"""
    async def ok(data, filename, file_id, queue):
        await queue.put({"type": "file_done", "file_id": file_id})

    async def run():
        queue: asyncio.Queue = asyncio.Queue()
        await app._run_parse_task(ok, b"x", "ok.pdf", "fid-ok", queue)
        return _drain(queue)

    events = asyncio.run(run())
    assert [e["type"] for e in events] == ["file_done"], events
