"""VLM 单页识别的响应清洗与「思考吃光额度」防护（2026-09-05）。

`vlm_recognize_page` 是四种解析模式里唯一的远程调用点，此前没有任何测试：
手写的 `<think>` 剥离、思考把 max_tokens 吃光后返回空正文，都是静默失败——
一页 Markdown 变成一段中文推理，或者干脆是空字符串，只有翻到那页才发现。

OCR 转写与其余调用点的关键差别：转写结果本身可能以「1. **公司概况**」这类
编号加粗标题开头，正好命中推理前缀启发式，所以这一处必须 prefix_heuristic=False。

运行: cd /Users/xuzheng/CodingProjects/Doxify && conda run -n mineru python -m pytest tests/test_vlm_recognize.py -q
"""
import asyncio
import copy

import app


class _FakeResp:
    def __init__(self, content, finish_reason="stop", status_code=200):
        self.status_code = status_code
        self.text = ""
        self._payload = {"choices": [{"finish_reason": finish_reason,
                                      "message": {"role": "assistant",
                                                  "content": content}}]}

    def json(self):
        return self._payload


class _FakeClient:
    """按脚本逐次返回响应，并记录每次请求体（用于断言 max_tokens 翻倍）。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.bodies = []

    async def post(self, url, json=None, headers=None, timeout=None):
        self.bodies.append(copy.deepcopy(json))
        return self._responses[min(len(self.bodies) - 1, len(self._responses) - 1)]


def _run(client):
    app._api_semaphore = asyncio.Semaphore(8)     # 未经 startup，兜住 async with None
    return asyncio.run(app.vlm_recognize_page(client, b"\x89PNG", 1, 1, "fid-test"))


def test_think_block_is_stripped():
    c = _FakeClient([_FakeResp("<think>\n先看排版\n</think>\n# 标题\n正文一行")])
    assert _run(c) == "# 标题\n正文一行"


def test_numbered_bold_heading_survives_prefix_heuristic():
    """转写第一行常是「1. **公司概况**」——不得被当成推理剥掉。"""
    page = "1. **公司概况**\n本公司成立于 2001 年。"
    assert _run(_FakeClient([_FakeResp(page)])) == page


def test_empty_content_with_length_retries_with_doubled_max_tokens():
    """思考吃光额度：正文空 + finish_reason=length → 翻倍 max_tokens 再来一次。"""
    polluted = ""
    c = _FakeClient([_FakeResp(polluted, "length"), _FakeResp("# 第二次拿到的正文")])
    assert _run(c) == "# 第二次拿到的正文"
    assert len(c.bodies) == 2, "应当重试一次"
    assert c.bodies[1]["max_tokens"] == c.bodies[0]["max_tokens"] * 2


def test_doubling_happens_only_once():
    c = _FakeClient([_FakeResp("", "length")] * 5)
    _run(c)
    tokens = [b["max_tokens"] for b in c.bodies]
    assert tokens[1] == tokens[0] * 2
    assert len(set(tokens[1:])) == 1, f"max_tokens 只应翻倍一次，实际 {tokens}"


def test_empty_content_with_stop_does_not_retry():
    """正常结束却空正文不是额度问题，别白白多打一次。"""
    c = _FakeClient([_FakeResp("", "stop")])
    _run(c)
    assert len(c.bodies) == 1


def test_payload_carries_profile_and_no_hand_copied_flags():
    c = _FakeClient([_FakeResp("ok")])
    _run(c)
    body = c.bodies[0]
    assert body["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert "enable_thinking" not in body and "thinking" not in body


def test_http_error_still_returns_marker(monkeypatch):
    async def _no_sleep(_):
        return None

    monkeypatch.setattr(app.asyncio, "sleep", _no_sleep)
    c = _FakeClient([_FakeResp("", status_code=500)])
    out = _run(c)
    assert "识别失败" in out


class _TimeoutThenEmptyClient:
    """前两次超时，第三次（也是最后一次）返回空正文 + finish_reason=length。

    真机上这就是「网关抖两下、最后一次思考吃光额度」。翻倍的 continue 若不受
    attempt < max_retries 约束，第三次会 continue 出循环、掉到函数末尾返回 None——
    签名写的是 -> str，调用方随后在 None 上炸出一句与病因无关的「处理失败」。
    """

    def __init__(self):
        self.calls = 0

    async def post(self, url, json=None, headers=None, timeout=None):
        import httpx
        self.calls += 1
        if self.calls <= 2:
            raise httpx.TimeoutException("read timeout")
        return _FakeResp("", "length")


def test_timeouts_then_empty_length_never_returns_none(monkeypatch):
    async def _no_sleep(_):
        return None

    monkeypatch.setattr(app.asyncio, "sleep", _no_sleep)
    c = _TimeoutThenEmptyClient()
    out = _run(c)
    assert out is not None, "掉出重试循环返回了 None，-> str 的契约破了"
    assert isinstance(out, str) and out == ""
    assert c.calls == 3


def test_empty_length_on_last_attempt_does_not_double(monkeypatch):
    """最后一次尝试上翻倍没有意义（没有下一次了），别白改 body。"""
    async def _no_sleep(_):
        return None

    monkeypatch.setattr(app.asyncio, "sleep", _no_sleep)
    c = _TimeoutThenEmptyClient()
    _run(c)
    assert c.calls == 3


# ── F1：正文非空但被截断也要翻倍重试一次 ──────────────────────────────

def test_non_empty_truncated_page_is_retried_with_doubled_budget():
    """半页 Markdown 断在表格中间：正文非空，拼进文档后没人看得出是额度问题。
    只看「正文为空」会漏掉这种——实测下它比空正文更常见。"""
    half = "# 财务数据\n\n| 项目 | 金额 |\n| --- | -"
    c = _FakeClient([_FakeResp(half, "length"), _FakeResp("# 财务数据\n\n完整的表格。")])
    assert _run(c) == "# 财务数据\n\n完整的表格。"
    assert len(c.bodies) == 2
    assert c.bodies[1]["max_tokens"] == c.bodies[0]["max_tokens"] * 2


def test_still_truncated_after_doubling_returns_content_with_a_warning():
    """翻倍后还断就交出手里的内容——半页远好过一句错误标记；但要留一条告警。"""
    import logging
    records = []

    class _H(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                records.append(record.getMessage())

    c = _FakeClient([_FakeResp("# 第一次的半页", "length"),
                     _FakeResp("# 第二次还是半页", "length")])
    h = _H()
    app.log.addHandler(h)
    try:
        out = _run(c)
    finally:
        app.log.removeHandler(h)
    assert out == "# 第二次还是半页"
    assert len(c.bodies) == 2, "只该翻倍一次"
    assert any("翻倍后仍被截断" in m for m in records), records


def test_retry_returning_empty_falls_back_to_the_truncated_page():
    """重试拿回空正文时，翻倍前那份被截断的转写仍是手里最好的东西，别丢——
    但必须在产物里留下截断标记，否则缺了下半页的那页和完整页长得一模一样。"""
    c = _FakeClient([_FakeResp("# 半页内容", "length"), _FakeResp("", "stop")])
    out = _run(c)
    assert out.startswith("# 半页内容")
    assert "转写可能不完整（输出被截断）" in out


def test_failed_retry_falls_back_to_the_truncated_page(monkeypatch):
    """翻倍重试超时的时候，别用「识别超时」把已经拿到的半页盖掉。"""
    async def _no_sleep(_):
        return None

    monkeypatch.setattr(app.asyncio, "sleep", _no_sleep)

    class _TruncThenTimeout:
        def __init__(self):
            self.calls = 0

        async def post(self, url, json=None, headers=None, timeout=None):
            import httpx
            self.calls += 1
            if self.calls == 1:
                return _FakeResp("# 半页内容", "length")
            raise httpx.TimeoutException("read timeout")

    out = _run(_TruncThenTimeout())
    assert out.startswith("# 半页内容"), f"半页被错误标记盖掉了: {out!r}"
    assert "转写可能不完整（输出被截断）" in out, "回退的半页必须自带截断标记"
    assert "识别超时" not in out


def test_complete_page_is_never_retried():
    c = _FakeClient([_FakeResp("# 完整的一页\n\n正文。")])
    assert _run(c) == "# 完整的一页\n\n正文。"
    assert len(c.bodies) == 1


def test_last_attempt_truncation_does_not_claim_a_doubling_happened(monkeypatch):
    """超时、超时、最后一次才截断：一次都没翻倍过，别说「翻倍后仍被截断」。

    这条路径下 `attempt < max_retries` 已经不成立，翻倍分支进不去，日志却照旧说
    「翻倍后仍被截断（max_tokens=4096）」——4096 正是初始值，谁看谁困惑。
    """
    import logging
    records = []

    class _H(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                records.append(record.getMessage())

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(app.asyncio, "sleep", _no_sleep)

    class _TwoTimeoutsThenTruncated:
        def __init__(self):
            self.calls = 0

        async def post(self, url, json=None, headers=None, timeout=None):
            import httpx
            self.calls += 1
            if self.calls <= 2:
                raise httpx.TimeoutException("read timeout")
            return _FakeResp("# 最后一次拿到的半页", "length")

    h = _H()
    app.log.addHandler(h)
    try:
        out = _run(_TwoTimeoutsThenTruncated())
    finally:
        app.log.removeHandler(h)

    assert out == "# 最后一次拿到的半页"
    assert any("最后一次尝试上被截断" in m and "已无重试余量" in m for m in records), records
    assert not any("翻倍后仍被截断" in m for m in records), \
        f"一次都没翻倍过，却报了「翻倍后仍被截断」: {records}"


def test_doubled_then_still_truncated_says_so():
    """反向确认：真的翻倍过时，用的还是「翻倍后仍被截断」那句。"""
    import logging
    records = []

    class _H(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                records.append(record.getMessage())

    c = _FakeClient([_FakeResp("# 半页一", "length"), _FakeResp("# 半页二", "length")])
    h = _H()
    app.log.addHandler(h)
    try:
        out = _run(c)
    finally:
        app.log.removeHandler(h)
    assert out == "# 半页二" and len(c.bodies) == 2
    assert any("翻倍后仍被截断" in m for m in records), records
    assert not any("最后一次尝试上被截断" in m for m in records), records
