"""跨页段落被脚注块隔开时的接合。

运行: conda run -n mineru python -m pytest tests/test_paragraph_across_footnotes.py -v

背景（2026-08-21）：PDF 一页的版式是「正文 …（被截断）／页底脚注块」，下一页接着写
正文。按页拼接后顺序变成：

    173. …is a clear goal of its        ← 被截断的段落
    ---
    [^48]: …  [^49]: …  [^55]: …        ← 页底脚注块
    policy plans.56 As per industry…    ← 下一页正文，实为 173 段的下半句

_merge_broken_paragraphs 看到「上一块（[^55] 定义）不以句末标点结尾、下一块以小写
开头」，就把下一页的正文粘到了**脚注定义**上。实测该文档 58 条定义受污染。

正确做法有两条：脚注定义永不吸收后文；续行要回溯到脚注块**之前**那个被截断的段落。
回溯只允许跨过脚注定义块与紧邻它们的 ---，不得跨过真正的章节分隔。
"""
import app


FN_BLOCK = "[^48]: [Law on Commercial Banks](https://x)\n\n[^55]: [Agency Working Guidance](https://y)"


def test_continuation_rejoins_paragraph_not_footnote():
    md = (
        "173. The Ministry stated that \"accelerating the electrification\" is a clear goal of its\n\n"
        "---\n\n" + FN_BLOCK + "\n\n"
        "policy plans. As per industry publications, demand is surging.\n"
    )
    out = app._merge_broken_paragraphs(md)
    assert "is a clear goal of its policy plans. As per industry publications" in out, out
    assert "(https://y) policy plans" not in out, "正文不得被粘到脚注定义上"


def test_footnote_definition_never_absorbs_following_prose():
    md = "[^55]: [Agency Working Guidance](https://y)\n\npolicy plans continue here.\n"
    out = app._merge_broken_paragraphs(md)
    assert "(https://y) policy plans" not in out, out


def test_new_paragraph_after_footnotes_is_not_merged():
    """下一块以大写开头 = 新段落，不能接到上文。"""
    md = (
        "172. A complete sentence ends here.\n\n"
        "---\n\n" + FN_BLOCK + "\n\n"
        "174. Evidence stated in the paragraphs above indicates that policy has encouraged demand.\n"
    )
    out = app._merge_broken_paragraphs(md)
    assert "ends here. 174." not in out, out
    assert "174. Evidence stated" in out


def test_does_not_merge_across_a_real_section_break():
    """回溯不得跨过真正的章节分隔（--- 后面不是脚注块）。"""
    md = (
        "A sentence that was cut off\n\n"
        "---\n\n"
        "and this looks like a continuation but sits after a real break.\n"
    )
    out = app._merge_broken_paragraphs(md)
    assert "cut off and this looks" not in out, out


def test_footnote_block_stays_in_place():
    """脚注定义本身不得被移动或改写。"""
    md = (
        "173. cut off here\n\n" + FN_BLOCK + "\n\n"
        "continuation text follows.\n"
    )
    out = app._merge_broken_paragraphs(md)
    assert "[^48]: [Law on Commercial Banks](https://x)" in out
    assert "[^55]: [Agency Working Guidance](https://y)" in out


def test_ordinary_continuation_still_merges():
    """没有脚注块时的普通跨页续行不受影响。"""
    md = "A sentence cut off\n\nmid-way through.\n"
    out = app._merge_broken_paragraphs(md)
    assert "cut off mid-way through." in out, out


# ------------------------------------ 续行以行内强调标记开头（斜体引文跨页）


def test_continuation_starting_with_italic_marker():
    """整段斜体的引文跨页时，VLM 会在新页重新开一次斜体，续行以 * 开头。

    续行判据原本要求「小写字母/左括号/中文」开头，* 不在其中，于是这类段落接不回去。
    """
    md = (
        "142. Zhejiang commented: \"*the market is open. To the best of the Company's\n\n"
        "---\n\n"
        "[^30]: [Report](https://x)\n\n[^37]: Working Document, page 556.\n\n"
        " *knowledge, there are approximately 30 producers in China.*\n"
    )
    out = app._merge_broken_paragraphs(md)
    assert "To the best of the Company's *knowledge, there are approximately 30" in out, out


def test_continuation_starting_with_bold_marker():
    md = "A sentence cut off\n\n**mid-way** through.\n"
    out = app._merge_broken_paragraphs(md)
    assert "cut off **mid-way** through." in out, out


def test_bullet_list_after_cut_paragraph_is_not_merged():
    """"* 条目" 是列表项（星号后有空格），不是斜体续行，不能接上去。"""
    md = "A sentence cut off\n\n* first bullet item\n"
    out = app._merge_broken_paragraphs(md)
    assert "cut off * first bullet" not in out, out


def test_emphasis_marker_followed_by_uppercase_is_not_merged():
    """去掉强调标记后仍是大写开头 = 新句子，不合并。"""
    md = "A sentence cut off\n\n*The next sentence starts here.*\n"
    out = app._merge_broken_paragraphs(md)
    assert "cut off *The next" not in out, out


def test_merge_recognises_raw_footnote_shapes_after_normalisation():
    """脚注归一化必须先于段落合并。

    后处理原本是「先合并段落、再归一化脚注」。合并发生时脚注定义还是 "^37 正文"
    这类原始形态，_is_footnote_def_block 只认 [^N]:，认不出来 → 回溯失败 → 下一页
    的正文接不回被截断的段落。VLM 逐页形态不一致，于是同一篇文档里有的段落接回了、
    有的没有——第 173 段那页吐的是 [^48]:，第 142 段那页吐的是 ^30/^37。

    这里直接验证「先归一化、再合并」这条顺序能正确处理原始形态。
    """
    md = (
        "142. Zhejiang commented: \"*the market is open. To the best of the Company's*\n\n"
        "---\n\n"
        "^30 [Report](https://x)\n\n"
        "^37 Working Document, page 556.\n\n"
        " *knowledge, there are approximately 30 producers in China.*\n\n"
        "143. Next paragraph starts here with markers^30 and^37 in use.\n"
    )
    out = app._merge_broken_paragraphs(app._normalize_footnotes(md))
    assert "To the best of the Company's* *knowledge, there are approximately 30" in out, out
    assert "[^37]: Working Document, page 556." in out, out
