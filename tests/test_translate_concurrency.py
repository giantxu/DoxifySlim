"""翻译并发上限：同时在途的流不得超过 TRANSLATE_CONCURRENCY。

运行: cd /Users/xuzheng/CodingProjects/Doxify && conda run -n mineru python -m pytest tests/test_translate_concurrency.py -v

背景（2026-08-12）：此前 translate_chunk_stream 里的信号量只包住 build_request +
send(stream=True)。stream=True 时 send() 拿到响应头就返回，正文的流式读取全在
信号量之外——限的是「发起请求」的瞬时并发，不是真正在途的流。所有块拿到响应头后
就一起流，实际并发几乎不受控。要真正限流，信号量必须覆盖整个流的生命周期。
"""
import asyncio

import app


class _FakeResponse:
    """假的流式响应：进入正文迭代时记一次「在途」，迭代结束时销账。"""

    def __init__(self, tracker, n_events=4):
        self.status_code = 200
        self._tracker = tracker
        self._n = n_events

    async def aiter_bytes(self):
        self._tracker.enter()
        try:
            for i in range(self._n):
                await asyncio.sleep(0.01)          # 让出控制权，制造真实交错
                yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n'
            yield b"data: [DONE]\n"
        finally:
            self._tracker.exit()

    async def aread(self):
        return b""

    async def aclose(self):
        pass


class _Tracker:
    def __init__(self):
        self.cur = 0
        self.peak = 0

    def enter(self):
        self.cur += 1
        self.peak = max(self.peak, self.cur)

    def exit(self):
        self.cur -= 1


class _FakeClient:
    def __init__(self, tracker):
        self._tracker = tracker

    def build_request(self, *a, **k):
        return object()

    async def send(self, req, stream=False):
        await asyncio.sleep(0)                      # 模拟拿到响应头的往返
        return _FakeResponse(self._tracker)


def _drain(sem_limit, n_chunks, monkeypatch):
    async def run():
        app._api_semaphore = asyncio.Semaphore(64)          # 全局放宽，只考察翻译侧
        app._translate_semaphore = asyncio.Semaphore(sem_limit)
        tracker = _Tracker()
        client = _FakeClient(tracker)

        async def one(i):
            async for _ in app.translate_chunk_stream(client, f"chunk {i}", i, n_chunks, "中文"):
                pass

        await asyncio.gather(*(one(i) for i in range(1, n_chunks + 1)))
        return tracker.peak

    return asyncio.run(run())


def test_concurrent_streams_never_exceed_limit(monkeypatch):
    """8 个块、上限 3：同时在途的流峰值不得超过 3。"""
    peak = _drain(3, 8, monkeypatch)
    assert peak <= 3, f"同时在途的流达到 {peak} 个，超过上限 3"


def test_limit_is_actually_reached(monkeypatch):
    """反向确认测试有效：上限 3 时峰值应真的到 3，否则说明测试没造出并发。"""
    peak = _drain(3, 8, monkeypatch)
    assert peak == 3, f"峰值只有 {peak}，测试没有制造出足够并发，结论不可信"


def test_limit_one_serialises_completely(monkeypatch):
    peak = _drain(1, 5, monkeypatch)
    assert peak == 1


def test_default_limit_is_three():
    assert app.TRANSLATE_CONCURRENCY == 3
