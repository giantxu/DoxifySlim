"""模型档案 + 统一的 LLM 响应清洗——app.py 的所有出站调用只经这里取正文。

2026-09-05：私有化网关 kimi-2.6 背后换成 GLM（思考不可关闭）。三种污染形态：
  · 思考在 reasoning_content（正常，忽略）；
  · 思考以 <think>…</think> 内联在 content；
  · 无标签，content 开头是一段推理（中文「1. **拆解用户请求：**…」或英文
    「Let me analyze…」），末尾才是正文。旧的 Kimi 开关 {thinking:false} 会造成
    这种形态——GLM 模板不认识这个 kwarg，降级后思考直接写进 content。
第一防线是档案配置正确（reasoning_effort: low）；这里只是兜底。

app.py 已 5800 行，档案与清洗单独放这里，两边靠 `from llm_common import ...` 相连。
"""
from __future__ import annotations

import copy
import json
import os
import re

# ---------------------------------------------------------------------------
# 档案：切模型只改 .env 里的 LLM_PROFILE 一行
# ---------------------------------------------------------------------------
# 档案内容整体作为额外字段合并进 payload 顶层（本项目是裸 httpx，不是 SDK）。
# 千万别给 GLM 发 {"thinking": False}——它不认识这个 kwarg，降级后把思考写进 content。
PROFILES: dict[str, dict] = {
    "kimi26":     {"chat_template_kwargs": {"thinking": False}},
    "glm53flash": {"chat_template_kwargs": {"reasoning_effort": "low"}},
}
DEFAULT_PROFILE = "glm53flash"


def llm_profile_name() -> str:
    """当前档案名。未配 / 空串回落默认档案；名字本身不校验（交给 llm_extra_body）。"""
    return (os.getenv("LLM_PROFILE") or "").strip() or DEFAULT_PROFILE


def llm_extra_body() -> dict:
    """要合并进请求体顶层的额外字段。档案名不存在直接 KeyError（启动时就炸，
    好过每次调用悄悄发一个空 body 又被思考污染）。"""
    name = llm_profile_name()
    try:
        profile = PROFILES[name]
    except KeyError:
        raise KeyError(
            f"未知的 LLM_PROFILE={name!r}，可选：{sorted(PROFILES)}"
        ) from None
    return copy.deepcopy(profile)


def llm_headers() -> dict:
    """OpenAI 兼容端点的请求头。Key 每次从环境读——本模块在 app.py 的 import 段
    被载入，早于 load_dotenv()，模块级快照会拿到空串。绝不入日志。"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('TARGET_API_KEY', '')}",
    }


# ---------------------------------------------------------------------------
# 统一响应清洗
# ---------------------------------------------------------------------------
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.S | re.I)
_THINK_OPEN = re.compile(r"<think>", re.I)
_REASONING_LEAD = re.compile(
    r"^\s*(?:"
    r"let me|let's|the user|i need|i should|i will|i'll|first,|okay|ok,|"
    r"looking at|analyzing|key facts|the document|this (?:document|is a)|so the|"
    r"now,? |wait,|actually,|hmm|"
    # 「1.  **拆解用户请求：**」——编号+加粗且加粗内容像推理标题；「1. **公司概况**」
    # 这类正文标题不命中（2026-09-05 终审：译文/散文开头的合法标题曾被误剥）
    r"\d+\.\s*\*\*[^*\n]{0,20}(?:拆解|分析|理解|解析|请求|需求|任务|约束|思考|用户)|"
    r"\*\*?(?:拆解|分析|理解|解析)(?:用户)?(?:的)?(?:请求|需求|任务|问题)|"
    r"(?:拆解|分析|理解)(?:用户)?(?:的)?(?:请求|需求|任务|问题)|"
    r"头脑风暴|思考过程"
    r")", re.I)
# 推理段里的句子形态：Markdown 列表项 / 加粗标记（实测 GLM 思考是「*   **核心任务：**…」）
_REASONING_SHAPE = re.compile(r"^\s*(?:[*\-•]\s|\d+\.\s|\*\*)|\*\*")
_CJK = re.compile(r"[一-鿿]")
_SENT_SPLIT = re.compile(r"(?<=[\.。!?！？\n])\s*(?=[一-鿿A-Za-z0-9\"'“(（{\[*\-•])")


def _get(o, k):
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def finish_reason(resp) -> str | None:
    try:
        return _get(_get(resp, "choices")[0], "finish_reason")
    except Exception:
        return None


def first_json(text: str) -> str | None:
    """取文本里第一个平衡的 JSON 对象/数组（JSON 场景兜底）。"""
    dec = json.JSONDecoder()
    for m in re.finditer(r"[{\[]", text):
        try:
            _, end = dec.raw_decode(text[m.start():])
            return text[m.start():m.start() + end]
        except ValueError:
            continue
    return None


def strip_thinking(text: str, *, prefix_heuristic: bool = True) -> str:
    """剥 <think> 块与无标签推理前缀。幂等。<think> 未闭合返回 ''。

    prefix_heuristic=False 时只剥 <think>，不做「开头像推理就剥」的猜测——
    OCR 转写类输出本身可能以「1. **公司概况**」开头，会被误判。"""
    if not text:
        return ""
    s = _THINK_BLOCK.sub("", text)
    if _THINK_OPEN.search(s):
        return ""
    s = s.strip()
    if not prefix_heuristic or not s or not _REASONING_LEAD.match(s):
        return s
    if not _CJK.search(s):
        # 通篇无中文（纯英文译文 / 英文名）：不猜。英文口头禅在英文正文里太常见。
        return s
    js = first_json(s)
    if js and len(js) > 1:
        return js
    sents = [x for x in _SENT_SPLIT.split(s) if x.strip()]
    keep: list[str] = []
    for seg in reversed(sents):
        t = seg.strip()
        if (_REASONING_LEAD.match(t) or _REASONING_SHAPE.search(t)
                or (not _CJK.search(t) and _CJK.search(s))):
            break
        keep.append(seg)
    # 到这里文本确认以推理开头：末尾没有可保留的正文就返回 ''，让调用方走
    # 「空正文 → 翻倍重试 / 报错」，绝不把推理当正文交出去。
    return "".join(reversed(keep)).strip()


def extract_content(resp, *, prefix_heuristic: bool = True) -> str:
    """从 OpenAI 兼容响应（dict 或 SDK 对象）取清洗后的正文。畸形输入返回 ''。"""
    try:
        msg = _get(_get(resp, "choices")[0], "message")
        content = _get(msg, "content")
    except Exception:
        return ""
    if not isinstance(content, str):
        return ""
    return strip_thinking(content, prefix_heuristic=prefix_heuristic)
