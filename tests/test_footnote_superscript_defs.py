"""行首 Unicode 上标的脚注定义行。

运行: conda run -n mineru python -m pytest tests/test_footnote_superscript_defs.py -v

背景（2026-08-20，一份 2900 行的 VLM 产物）：同一篇 VLM 产物里两种脚注形态混杂——
115 条被模型直接吐成标准的 [^N]:，另外 30 组仍是 Unicode 上标，且**定义行**写成
"¹ [Notification of final sample]: The TRA collapsed..." 这种行首上标的形式。

三个识别器都不认这种定义：_FN_EXISTING_DEF_RE 找 [^N]:，_FN_ITEM_RE 找 "N. 正文"
且需在尾注区标题之下，而该文档没有 References 类标题（_find_footnote_section 返回
None）。于是形成死锁：安全闸门要求「编号有已知定义」才转换正文标记，而定义本身正是
未被识别的那一种——标记不转，定义也不转，整组原样留下。
"""
import app


def test_superscript_definition_line_is_recognised():
    """行首上标 + 空白 + 正文 = 脚注定义，编号必须进入 nums。"""
    md = "正文里引用了¹。\n\n¹ Applicant submission, pages 44-45.\n"
    assert 1 in app._collect_footnote_numbers(md)


def test_superscript_definition_and_marker_both_converted():
    md = "See the notice.¹\n\n¹ Applicant submission, pages 44-45.\n"
    out = app._normalize_footnotes(md)
    assert "See the notice.[^1]" in out, out
    assert "[^1]: Applicant submission, pages 44-45." in out, out
    assert "¹" not in out, "上标应被全部转换"


def test_multi_digit_superscript_definition():
    md = "参见此处¹²⁶。\n\n¹²⁶ Commission Staff Working Document, page 30.\n"
    out = app._normalize_footnotes(md)
    assert "[^126]" in out
    assert "[^126]: Commission Staff Working Document, page 30." in out


def test_definition_line_with_bracketed_link_body():
    """实际文档里的形态：定义正文本身以 [text](url) 开头，不能被误当成别的语法。"""
    md = "引用¹\n\n¹ [Notification of final sample](http://x): The TRA collapsed the group.\n"
    out = app._normalize_footnotes(md)
    assert "[^1]: [Notification of final sample](http://x): The TRA collapsed the group." in out, out


def test_does_not_touch_units_like_m2():
    """面积单位不是脚注定义：上标不在行首、且没有独立的定义行。"""
    md = "面积为 30m² 的房间。\n\n[^5]: 真正的定义\n"
    out = app._normalize_footnotes(md)
    assert "30m²" in out, "m² 必须原样保留"


def test_superscript_alone_on_line_is_not_a_definition():
    """孤零零一个上标、后面没有正文，不算定义（避免误判残留的页码上标）。"""
    md = "正文¹\n\n¹\n\n[^9]: 别的定义\n"
    nums = app._collect_footnote_numbers(md)
    assert 1 not in nums


def test_mixed_document_converts_both_styles():
    """混合形态文档：已有的 [^N]: 不受影响，上标那组补齐。"""
    md = (
        "First claim.[^2]\n"
        "Second claim.³\n\n"
        "[^2]: Already fine.\n"
        "³ Analysis on UK market share.\n"
    )
    out = app._normalize_footnotes(md)
    assert "[^2]: Already fine." in out
    assert "Second claim.[^3]" in out, out
    assert "[^3]: Analysis on UK market share." in out, out
