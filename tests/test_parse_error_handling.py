"""解析路径的错误兜底：损坏 PDF 不得让 SSE 连接挂起。

bug 复现：parse_pdf_streaming 在 pdf_to_images 阶段对损坏 PDF 会抛异常，
若不兜底则 task 直接死亡、不发 file_done/file_error，
导致 parse_pdf_stream._generate 的 `while done_count + error_count < total_files`
永久挂起。修复目标：任何异常都转成一条 file_error 事件。
"""
import asyncio

import app


def _drain(queue: asyncio.Queue) -> list[dict]:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def test_corrupt_pdf_emits_file_error_not_exception():
    async def run():
        queue: asyncio.Queue = asyncio.Queue()
        # 不是合法 PDF —— pdf_to_images 会在 fitz.open 阶段抛错
        await app._parse_pdf_safe(
            b"this is definitely not a pdf",
            "broken.pdf",
            "fid-test-1",
            queue,
            True,
            True,
        )
        return _drain(queue)

    events = asyncio.run(run())  # 不抛异常即第一关
    errors = [e for e in events if e.get("type") == "file_error"]
    assert errors, f"expected a file_error event, got: {events}"
    err = errors[0]
    assert err["file_id"] == "fid-test-1"
    assert err["filename"] == "broken.pdf"
    assert err.get("error"), "file_error should carry a non-empty error message"
