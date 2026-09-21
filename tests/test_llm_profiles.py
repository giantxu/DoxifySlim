"""模型档案 + 统一响应清洗（2026-09-05）。

私有化网关 http://192.0.2.10:4000 背后的 `kimi-2.6` 从 09-04 起实际是 GLM
（地址 / 模型名 / API Key 不变）。GLM 的思考关不掉，只能用 reasoning_effort 调深浅；
更糟的是旧的 Kimi 开关 `chat_template_kwargs={"thinking": False}` 是 GLM 模板不认识的
kwarg，降级后思考直接写进 `content`（中文 Markdown 列表「1. **拆解用户请求：**…」），
正文永远出不来。不带任何 kwarg 时思考被正确分到 `reasoning_content`。

本文件锁三件事：
① 档案机制：`LLM_PROFILE` 选档案，切模型改一行环境变量；未知档案立即 KeyError。
② 统一清洗：只取 content、剥 <think>、剥无标签推理前缀、忽略 reasoning_content。
③ 调用点扫描：app.py 不得再直接读 `["message"]["content"]`（流式 delta 路径豁免）。

运行: cd /Users/xuzheng/CodingProjects/Doxify && conda run -n mineru python -m pytest tests/test_llm_profiles.py -q
"""
import re
from pathlib import Path

import pytest


from llm_common import (PROFILES, extract_content, finish_reason, first_json,
                        llm_extra_body, llm_headers, llm_profile_name,
                        strip_thinking)


# ── ① 档案 ────────────────────────────────────────────────────────

def test_default_profile_is_glm(monkeypatch):
    monkeypatch.delenv("LLM_PROFILE", raising=False)
    assert llm_profile_name() == "glm53flash"
    assert llm_extra_body() == {"chat_template_kwargs": {"reasoning_effort": "low"}}


def test_switching_profile_is_one_env_line(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "kimi26")
    assert llm_profile_name() == "kimi26"
    assert llm_extra_body() == {"chat_template_kwargs": {"thinking": False}}


def test_unknown_profile_raises_immediately(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "no-such-model")
    with pytest.raises(KeyError):
        llm_extra_body()


def test_blank_profile_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "  ")
    assert llm_profile_name() == "glm53flash"


def test_extra_body_is_a_copy_not_the_profile_itself(monkeypatch):
    """调用方会把它 ** 展开进 payload；返回内部字典会让一次误改污染全进程。"""
    monkeypatch.setenv("LLM_PROFILE", "glm53flash")
    body = llm_extra_body()
    body["chat_template_kwargs"]["reasoning_effort"] = "high"
    assert PROFILES["glm53flash"] == {"chat_template_kwargs": {"reasoning_effort": "low"}}


def test_profiles_have_exactly_the_two_known_models():
    assert set(PROFILES) == {"kimi26", "glm53flash"}


def test_headers_carry_bearer_key(monkeypatch):
    monkeypatch.setenv("TARGET_API_KEY", "sk-test-doxify")
    h = llm_headers()
    assert h["Content-Type"] == "application/json"
    assert h["Authorization"] == "Bearer sk-test-doxify"


# ── ② 响应清洗 ────────────────────────────────────────────────────

CLEAN = "2026年9月4日，商务部批准延期申请。"


def _resp(content, fr="stop", reasoning=None):
    return {"choices": [{"finish_reason": fr,
                         "message": {"role": "assistant", "content": content,
                                     "reasoning_content": reasoning}}]}


def test_clean_content_passes_through():
    assert extract_content(_resp(CLEAN)) == CLEAN


def test_reasoning_content_is_ignored():
    assert extract_content(_resp(CLEAN, reasoning="1. **拆解用户请求**")) == CLEAN


def test_think_block_is_stripped():
    assert extract_content(_resp(f"<think>\nblah\n</think>\n{CLEAN}")) == CLEAN


def test_unclosed_think_returns_empty():
    assert extract_content(_resp("<think>\n1. 拆解", "length")) == ""


def test_pure_chinese_reasoning_without_body_returns_empty():
    """实测：200 token 全被思考吃光，content 里一个字正文都没有。"""
    polluted = ("1.  **拆解用户请求：**\n    *   **核心任务：**“用一句中文介绍你自己”。"
                "\n    *   **关键约束：**“一句”")
    assert extract_content(_resp(polluted, "length")) == ""


def test_reasoning_prefix_then_json_yields_json():
    got = extract_content(_resp(
        '1.  **拆解用户请求：**\n    *   核心任务是抽取字段。\n\n'
        '{"relevant": true, "cases": []}'))
    assert got == '{"relevant": true, "cases": []}'


def test_reasoning_prefix_then_chinese_body_yields_body():
    got = extract_content(_resp(
        "1.  **拆解用户请求：** 用户要一句话摘要。\n2. **头脑风暴：** 需要简洁。\n" + CLEAN))
    assert got == CLEAN


def test_english_reasoning_prefix_is_stripped():
    got = extract_content(_resp(
        "Let me analyze this. The user wants a summary. Good." + CLEAN))
    assert got == CLEAN


@pytest.mark.parametrize("legit", [
    '{"notices": [{"case_no": "A-570-104"}]}',
    "Ningbo Zhongjiang High Strength Bolts Co., Ltd.",
    "是",
    "47",
    "首先，公司应当披露。其次，应当公告。",
    "# 标题\n\n- 第一项\n- 第二项\n\n**加粗**正文。",
    "1. 公司基本情况\n2. 财务数据",
])
def test_legitimate_output_is_untouched(legit):
    assert extract_content(_resp(legit)) == legit


# v2（2026-09-05 终审后收紧口头禅）：合法译文 / 散文 / 目录回复不得被剥。
# v1 的「让我|我需要|我应该|用户(想要|要求|希望)」和裸的「\d+\.\s*\*\*」误伤了下面这些。
@pytest.mark.parametrize("legit", [
    "让我们来看这份合同的第三条。它规定了交付期限。",
    "1. **公司概况**\n\n本公司成立于 2001 年，主营紧固件产品。",
    "1. **甲公司**：主营化工。\n2. **乙公司**：主营地产。",
    "用户要求退款的，经营者应当在七日内退还。",
    "1. **公司概况**：5\n2. **中介机构**：8",
    "Let me know if the parties agree. The contract is valid.",
])
def test_v2_tightened_lead_keeps_legitimate_text(legit):
    assert extract_content(_resp(legit)) == legit


def test_v2_analysis_variant_is_still_cleared():
    """实测 GLM 的另一种开头：「1. **分析用户的请求：**」仍须清空。"""
    polluted = ("1.  **分析用户的请求：**\n    *   **核心任务：**“用一句中文介绍你自己”。")
    assert extract_content(_resp(polluted, "length")) == ""


def test_prefix_heuristic_off_keeps_numbered_bold_heading():
    """OCR 转写本身可能以「1. **公司概况**」开头，不能当推理剥掉。"""
    ocr = "1. **公司概况**\n本公司成立于 2001 年。"
    assert extract_content(_resp(ocr), prefix_heuristic=False) == ocr


def test_prefix_heuristic_off_still_strips_think_tags():
    assert extract_content(_resp(f"<think>x</think>{CLEAN}"), prefix_heuristic=False) == CLEAN


def test_sdk_object_response_works():
    class _M:
        content = "<think>x</think>ok"

    class _C:
        finish_reason = "stop"
        message = _M()

    class _R:
        choices = [_C()]

    assert extract_content(_R()) == "ok"


def test_malformed_responses_return_empty():
    assert extract_content({}) == ""
    assert extract_content({"choices": []}) == ""
    assert extract_content(_resp(None)) == ""


def test_strip_thinking_is_idempotent():
    assert strip_thinking(strip_thinking("<think>x</think>正文")) == "正文"


def test_finish_reason_reads_and_tolerates_garbage():
    assert finish_reason(_resp(CLEAN, "length")) == "length"
    assert finish_reason({}) is None


def test_first_json_finds_balanced_object():
    assert first_json('废话 {"a": {"b": 1}} 尾巴') == '{"a": {"b": 1}}'
    assert first_json("没有 JSON") is None


# ── ③ 调用点扫描 ──────────────────────────────────────────────────

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_app_never_reads_message_content_directly():
    """所有非流式读取必须走 extract_content()。流式走 delta，不在此列。"""
    src = APP.read_text(encoding="utf-8")
    assert not re.search(r'\["message"\]\s*\[\s*"content"\s*\]', src), \
        "app.py 仍直接读 message.content，须改走 extract_content()"
    assert not re.search(r"\.message\.content", src)
    assert "extract_content" in src, "app.py 未接入 extract_content"


def test_app_no_longer_hand_copies_thinking_flags():
    src = APP.read_text(encoding="utf-8")
    assert '"enable_thinking"' not in src, "手抄的 enable_thinking 应由档案取代"
    assert '"thinking": {"type": "disabled"}' not in src
    assert '"chat_template_kwargs"' not in src, "chat_template_kwargs 只应出现在 llm_common.PROFILES"


def test_app_payloads_use_llm_extra_body():
    """5 处 payload 组装点都要展开档案。"""
    src = APP.read_text(encoding="utf-8")
    assert src.count("**llm_extra_body()") == 5, \
        f"期望 5 处 **llm_extra_body()，实际 {src.count('**llm_extra_body()')} 处"
