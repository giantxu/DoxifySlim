"""流式翻译的首段缓冲与占位符 UX（2026-09-05）。

背景：网关背后的 kimi-2.6 从 09-04 起实际是 GLM，思考关不掉。流式路径原本只认
`<think>` 标签，无标签的中文推理（「1. **拆解用户请求：**…」）会被逐 token 当译文
直接流进用户的输出框——比空白更糟，用户看到一段像模像样的分析，还以为那是译文。

对策：前 240 个真实字符先攒着不下发，攒满或流结束时用 strip_thinking 判一次；
判为推理就整段丢弃并告警，判为正文就整段下发，之后逐 token 直通。代价是首字延迟，
用 `_process_chunk` 的占位符 + chunk_replace 补上（chunk_replace 前端就是整块覆盖）。

假流写法参考 tests/test_translate_concurrency.py 的 _FakeClient。

运行: cd /Users/xuzheng/CodingProjects/Doxify && conda run -n mineru python -m pytest tests/test_stream_think_buffer.py -q
"""
import asyncio
import contextlib
import json
import logging

import httpx

import app


@contextlib.contextmanager
def _capture_warnings():
    """app 的 logger 设了 propagate=False（隔离 PaddlePaddle），caplog 抓不到，
    只能自己挂一个 handler。"""
    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                records.append(record.getMessage())

    h = _H()
    app.log.addHandler(h)
    try:
        yield records
    finally:
        app.log.removeHandler(h)


# ---------------------------------------------------------------- 假 SSE 流

class _FakeStreamResp:
    def __init__(self, deltas, status_code=200):
        self.status_code = status_code
        self._deltas = deltas

    async def aiter_bytes(self):
        for d in self._deltas:
            payload = json.dumps({"choices": [{"delta": d}]}, ensure_ascii=False)
            yield f"data: {payload}\n".encode()
        yield b"data: [DONE]\n"

    async def aread(self):
        return b""

    async def aclose(self):
        pass


class _FakeClient:
    """按脚本回放若干条流；记录每次请求体（用于断言 max_tokens 翻倍）。"""

    def __init__(self, streams):
        self._streams = list(streams)
        self.bodies = []

    def build_request(self, method, url, json=None, headers=None):
        self.bodies.append(json)
        return object()

    async def send(self, req, stream=False):
        await asyncio.sleep(0)
        idx = min(len(self.bodies) - 1, len(self._streams) - 1)
        return self._streams[idx]


def _tokens(text, size=8):
    return [{"content": text[i:i + size]} for i in range(0, len(text), size)]


def _drain(client, max_tokens=None):
    async def run():
        app._api_semaphore = asyncio.Semaphore(8)
        app._translate_semaphore = asyncio.Semaphore(3)
        kwargs = {} if max_tokens is None else {"max_tokens": max_tokens}
        out = []
        async for tok in app.translate_chunk_stream(
                client, "source chunk", 1, 1, "中文", **kwargs):
            out.append(tok)
        return out

    return asyncio.run(run())


# ---------------------------------------------------------------- 首段缓冲

REASONING = ("1.  **拆解用户请求：**\n    *   **核心任务：**把这段 Markdown 译成中文。"
             "\n    *   **关键约束：**保留格式。\n2. **头脑风暴：**先看有没有代码块。"
             "\n    *   表格要保持结构。\n    *   链接只译文字。\n")


def test_untagged_reasoning_is_never_yielded():
    """无标签中文推理：一个字都不许流给用户，并且要留下告警。"""
    with _capture_warnings() as warns:
        out = _drain(_FakeClient([_FakeStreamResp(_tokens(REASONING))]))
    assert "".join(out) == "", f"推理被当译文下发了: {''.join(out)[:80]}"
    assert any("思考泄漏" in m for m in warns), \
        "丢弃了推理却没有任何告警，线上没人会发现档案配错了"


def test_normal_translation_is_delivered_in_full():
    body = "这是一段完整的中文译文。" * 40          # 远超 240 字符
    out = _drain(_FakeClient([_FakeStreamResp(_tokens(body))]))
    assert "".join(out) == body


def test_short_translation_shorter_than_buffer_still_delivered():
    """流结束时缓冲没攒满也必须冲出去，否则短块整块丢失。"""
    body = "很短的一句译文。"
    out = _drain(_FakeClient([_FakeStreamResp(_tokens(body))]))
    assert "".join(out) == body


def test_first_yield_is_the_whole_buffered_segment():
    """第一次 yield 应当是攒满的整段，而不是单个 token——前端要用它整块替换占位符。"""
    body = "译文片段。" * 100
    out = _drain(_FakeClient([_FakeStreamResp(_tokens(body))]))
    assert len(out[0]) >= 240, f"首段只有 {len(out[0])} 字符，缓冲没生效"


def test_reasoning_prefix_before_real_body_is_stripped():
    """推理在前、正文在后：剥掉前缀，正文照常下发。"""
    body = REASONING + "1949 年，中华人民共和国成立。"
    out = _drain(_FakeClient([_FakeStreamResp(_tokens(body))]))
    joined = "".join(out)
    assert "拆解用户请求" not in joined
    assert "1949 年，中华人民共和国成立。" in joined


def test_think_tags_still_filtered_and_body_survives():
    body = "<think>先想想怎么译</think>" + "这是正文。" * 60
    out = _drain(_FakeClient([_FakeStreamResp(_tokens(body))]))
    joined = "".join(out)
    assert "<think>" not in joined and "先想想怎么译" not in joined
    assert joined == "这是正文。" * 60


def test_reasoning_content_deltas_are_ignored():
    """思考走 reasoning_content 是正常形态：不下发，但要计入泄漏告警。"""
    deltas = [{"reasoning_content": "先分析一下用户的请求"} for _ in range(5)]
    deltas += _tokens("正常的中文译文。" * 40)
    with _capture_warnings() as warns:
        out = _drain(_FakeClient([_FakeStreamResp(deltas)]))
    joined = "".join(out)
    assert "先分析一下用户的请求" not in joined
    assert joined == "正常的中文译文。" * 40
    assert any("reasoning_content" in m for m in warns)


def test_max_tokens_parameter_reaches_the_payload():
    c = _FakeClient([_FakeStreamResp(_tokens("译文"))])
    _drain(c, max_tokens=12000)
    assert c.bodies[0]["max_tokens"] == 12000


def test_default_max_tokens_is_6000():
    c = _FakeClient([_FakeStreamResp(_tokens("译文"))])
    _drain(c)
    assert c.bodies[0]["max_tokens"] == 6000


def test_stream_payload_carries_profile():
    c = _FakeClient([_FakeStreamResp(_tokens("译文"))])
    _drain(c)
    assert c.bodies[0]["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert "enable_thinking" not in c.bodies[0]


# ---------------------------------------------------------------- 占位符 UX

def _run_translate_job(monkeypatch, fake_stream, text="hello world"):
    """跑一次 /jobs/translate，返回事件列表（单块输入 → 单块事件序列）。"""
    monkeypatch.setattr(app, "translate_chunk_stream", fake_stream)
    monkeypatch.setattr(app, "_detect_residual_english", lambda buf: [])

    async def run():
        app._jobs.clear()
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app.app),
                base_url="http://test") as c:
            r = await c.post("/jobs/translate",
                             data={"text": text, "target_lang": "中文"})
        job = app._jobs[r.json()["job_id"]]
        await asyncio.wait_for(asyncio.gather(job.task, return_exceptions=True),
                               timeout=10)
        return job.events

    return asyncio.run(run())


def test_chunk_event_order_is_start_placeholder_replace_then_tokens(monkeypatch):
    async def fake(client, chunk, idx, total, lang, max_tokens=6000):
        yield "第一段译文"
        yield "后续 token"

    events = _run_translate_job(monkeypatch, fake)
    seq = [e["type"] for e in events if e["type"].startswith("chunk_")]
    assert seq == ["chunk_start", "chunk_token", "chunk_replace",
                   "chunk_token", "chunk_done"], seq

    placeholder = next(e for e in events if e["type"] == "chunk_token")
    assert placeholder["token"] == app._CHUNK_PLACEHOLDER
    replace = next(e for e in events if e["type"] == "chunk_replace")
    assert replace["text"] == "第一段译文", "第一段必须整块替换掉占位符"


def test_placeholder_is_a_reassuring_chinese_notice():
    assert app._CHUNK_PLACEHOLDER == "翻译进行中，请稍后……"


def test_empty_chunk_is_retried_once_with_doubled_max_tokens(monkeypatch):
    seen = []

    async def fake(client, chunk, idx, total, lang, max_tokens=6000):
        seen.append(max_tokens)
        if len(seen) == 1:
            return                      # 第一次一个 token 都没有
        yield "重发后拿到的译文"

    events = _run_translate_job(monkeypatch, fake)
    assert seen == [6000, 12000], f"空块未按翻倍额度重发: {seen}"
    replace = [e for e in events if e["type"] == "chunk_replace"]
    assert replace and replace[-1]["text"] == "重发后拿到的译文"


def test_twice_empty_chunk_replaces_placeholder_with_notice(monkeypatch):
    async def fake(client, chunk, idx, total, lang, max_tokens=6000):
        return
        yield                            # pragma: no cover — 让它成为生成器

    events = _run_translate_job(monkeypatch, fake)
    replace = [e for e in events if e["type"] == "chunk_replace"]
    assert replace, "两次都空时必须把占位符换掉，不能把「翻译进行中」留在译文里"
    assert "译文为空" in replace[-1]["text"]
    assert app._CHUNK_PLACEHOLDER not in replace[-1]["text"]


def test_no_new_sse_event_types_were_invented(monkeypatch):
    async def fake(client, chunk, idx, total, lang, max_tokens=6000):
        yield "译文"

    events = _run_translate_job(monkeypatch, fake)
    known = {"init", "file_start", "chunk_start", "chunk_token", "chunk_replace",
             "chunk_done", "file_done", "file_error", "all_done", "job_end"}
    assert {e["type"] for e in events} <= known


def test_stream_exception_replaces_placeholder_instead_of_leaving_it(monkeypatch):
    """块任务抛异常时占位符必须被换掉——否则「翻译进行中，请稍后……」会原样留在
    前端的块缓冲里，混进最终译文和下载的文件。"""
    async def fake(client, chunk, idx, total, lang, max_tokens=6000):
        raise RuntimeError("流炸了")
        yield                            # pragma: no cover — 让它成为生成器

    events = _run_translate_job(monkeypatch, fake)
    replace = [e for e in events if e["type"] == "chunk_replace"]
    assert replace, "异常路径没有替换占位符"
    assert app._CHUNK_PLACEHOLDER not in replace[-1]["text"]
    assert "翻译异常" in replace[-1]["text"]
    assert [e["type"] for e in events][-1] == "job_end"


def test_source_starting_with_numbered_bold_keeps_its_first_line():
    """源块以「1. **标题**」开头时译文也会——那仍可能撞上推理前缀的形状。清洗器
    v2 收紧后「1. **公司概况**」已经安全，但加粗内容里带「任务/需求/用户/分析」等
    词的合法标题（「1. **任务分工**」）照旧命中，首行会被静默吃掉。这种块只剥
    <think>，不做前缀猜测。"""
    body = "1. **任务分工**\n甲方负责设计，乙方负责施工。" * 8

    async def run():
        app._api_semaphore = asyncio.Semaphore(8)
        app._translate_semaphore = asyncio.Semaphore(3)
        out = []
        async for tok in app.translate_chunk_stream(
                _FakeClient([_FakeStreamResp(_tokens(body))]),
                "1. **Task Allocation**\nParty A designs.", 1, 1, "中文"):
            out.append(tok)
        return "".join(out)

    assert asyncio.run(run()) == body


def test_normal_source_still_strips_reasoning_prefix():
    """反向确认上一条没有把启发式整个关掉：普通源块仍然剥推理前缀。"""
    out = _drain(_FakeClient([_FakeStreamResp(_tokens(REASONING))]))
    assert "".join(out) == ""


class _TimeoutAfterReasoning:
    """先吐一段无标签推理，然后连接超时——首段缓冲还没结算就被中断。"""

    def __init__(self):
        self.status_code = 200

    async def aiter_bytes(self):
        for d in _tokens(REASONING):
            payload = json.dumps({"choices": [{"delta": d}]}, ensure_ascii=False)
            yield f"data: {payload}\n".encode()
        raise httpx.TimeoutException("read timeout")

    async def aread(self):
        return b""

    async def aclose(self):
        pass


def test_timeout_after_reasoning_still_logs_the_leak():
    """超时路径也会丢掉一整段被判为推理的首段缓冲——不告警等于彻底静默。"""
    with _capture_warnings() as warns:
        out = _drain(_FakeClient([_TimeoutAfterReasoning()]))
    assert "拆解用户请求" not in "".join(out)
    assert any("思考泄漏" in m for m in warns), \
        f"超时路径丢掉推理却没有告警: {warns}"


def test_short_reasoning_head_warns_regardless_of_length():
    """20 字符阈值是给 <think> 标签统计用的；首段缓冲被判为推理，多短都要报。"""
    with _capture_warnings() as warns:
        out = _drain(_FakeClient([_FakeStreamResp(_tokens("头脑风暴：先看格式。"))]))
    assert "".join(out) == ""
    assert any("丢弃推理首段" in m for m in warns), warns


def test_whitespace_only_head_is_not_reported_as_a_leak():
    """纯空白不是推理，不许谎报泄漏（源块带缩进时首段就可能全是空白）。"""
    with _capture_warnings() as warns:
        _drain(_FakeClient([_FakeStreamResp(_tokens("   \n  "))]))
    assert not any("思考泄漏" in m for m in warns), warns


def test_partially_stripped_head_is_also_reported():
    """首段被剥掉前缀但仍有正文——此前完全静默。剥掉的可能是真思考（档案错），
    也可能是被误伤的正文，两种都必须看得见。"""
    body = REASONING + "1949 年，中华人民共和国成立。"
    with _capture_warnings() as warns:
        out = _drain(_FakeClient([_FakeStreamResp(_tokens(body))]))
    assert "1949 年，中华人民共和国成立。" in "".join(out)
    assert any("首段剥前缀" in m for m in warns), f"部分剥离没有告警: {warns}"


def test_long_reasoning_fills_the_buffer_and_is_judged_there():
    """推理长过 STREAM_HEAD_CHARS：判定发生在「攒满 240 就结算」那条分支上，而不是
    流结束时那条。两条分支写法不同（一条在 _accept 里，一条在 _finalize 里），
    此前只有后者有测试。

    钉法是算术：token 8 字符一片、阈值 240 整除 8，所以缓冲恰好在 240 处结算并被
    整段丢弃，此后逐 token 直通——下发的内容必须精确等于 full[240:]。若判定只在流
    末尾做，整条流会被判为推理、一个字都不下发（joined == ""）。

    顺带钉住 common-brief 承认的边界：推理超过 240 时，尾巴会漏出来。缓冲只是兜底，
    档案配对才是第一防线。
    """
    long_reasoning = REASONING * 4                     # 520 字符，远超 240
    tail = "其后的正文照常下发。"
    full = long_reasoning + tail
    assert len(long_reasoning) > app.STREAM_HEAD_CHARS
    assert app.STREAM_HEAD_CHARS % 8 == 0              # 让下面的等式成立

    with _capture_warnings() as warns:
        out = _drain(_FakeClient([_FakeStreamResp(_tokens(full))]))
    joined = "".join(out)

    assert joined != "", "整条流都被吞了：判定没发生在「攒满」分支上"
    assert joined == full[app.STREAM_HEAD_CHARS:], \
        f"下发的不是 full[240:]，实际前 30 字符: {joined[:30]!r}"
    # 不再断言 full[:240] not in joined——REASONING 是重复文本，后面的重复段里
    # 本来就含有相同的字符串。上面的等式已经把该证的都证了。
    assert tail in joined
    assert any("丢弃推理首段" in m for m in warns), warns


# ── F1：流式截断只告警、不重跑 ────────────────────────────────────────

class _TruncatedStreamResp(_FakeStreamResp):
    """末条事件带 finish_reason=length、delta 为空——实测网关就是这么发的。"""

    async def aiter_bytes(self):
        for d in self._deltas:
            payload = json.dumps({"choices": [{"delta": d}]}, ensure_ascii=False)
            yield f"data: {payload}\n".encode()
        yield (b'data: ' + json.dumps(
            {"choices": [{"index": 0, "finish_reason": "length", "delta": {}}]},
            ensure_ascii=False).encode() + b"\n")
        yield b"data: [DONE]\n"


def test_truncated_stream_warns_but_still_delivers_everything():
    """已经流出去的 token 收不回来，所以不重跑；但必须告诉人这块是被掐断的。"""
    body = "这是一段被掐断的译文，" * 30
    with _capture_warnings() as warns:
        out = _drain(_FakeClient([_TruncatedStreamResp(_tokens(body))]))
    assert "".join(out) == body, "告警归告警，内容一个字都不能少"
    assert any("译文可能不完整" in m and "finish_reason=length" in m for m in warns), warns


def test_normal_stream_does_not_warn_about_truncation():
    body = "完整收尾的译文。" * 30
    with _capture_warnings() as warns:
        _drain(_FakeClient([_FakeStreamResp(_tokens(body))]))
    assert not any("译文可能不完整" in m for m in warns), warns


def test_truncated_stream_is_not_re_run(monkeypatch):
    """非空截断不重发——重发会让同一块在前端出现两遍。"""
    seen = []

    async def fake(client, chunk, idx, total, lang, max_tokens=6000):
        seen.append(max_tokens)
        yield "被掐断的译文"

    events = _run_translate_job(monkeypatch, fake)
    assert seen == [6000], f"非空截断被重发了: {seen}"
    assert [e["type"] for e in events][-1] == "job_end"


class _TruncatedThenErrorThenBadClose(_TruncatedStreamResp):
    """截断收尾 → 流内错误事件 → aclose() 也炸。

    这是 _finalize() 真的会被走两次的路径：错误事件分支先调一次、随后 aclose()
    抛出的异常被外层 `except Exception` 接住又调一次。没有闩锁，同一块会报两次
    「译文可能不完整」，看日志的人会以为出了两个问题。
    """

    async def aiter_bytes(self):
        async for chunk in super().aiter_bytes():
            if chunk.strip() == b"data: [DONE]":
                break
            yield chunk
        yield (b'data: ' + json.dumps({"error": {"message": "upstream reset"}},
                                      ensure_ascii=False).encode() + b"\n")

    async def aclose(self):
        raise RuntimeError("连接关闭时也炸了")


def test_length_warning_is_emitted_at_most_once_per_chunk():
    with _capture_warnings() as warns:
        _drain(_FakeClient([_TruncatedThenErrorThenBadClose(_tokens("被掐断的译文。" * 20))]))
    hits = [m for m in warns if "译文可能不完整" in m]
    assert len(hits) == 1, f"截断告警发了 {len(hits)} 次: {hits}"
