"""段落内部的硬换行：VLM 把 PDF 的视觉折行原样保留了。

运行: conda run -n mineru python -m pytest tests/test_unwrap_hard_linebreaks.py -v

背景（2026-08-21）：VLM 逐页识别时会把 PDF 里的折行当成真实换行输出，于是一个段落
在 Markdown 源码里是好几行。_merge_broken_paragraphs 处理的是「块」（空行分隔），
块内部的换行它不碰，所以一直留着。

危险在于块内换行有两种截然不同的含义：
  续行  —— "205. The applicant alleges that…" / "reflect non-commercial factors…"  应合并
  新条目 —— "a) …" / "b) …" / "- …" / 表格行 / 标题                                不可动
实测该文档 106 个多行块里，55 个是硬换行段落、51 个含条目。判据：**首行之后若还有
行看起来像新条目，整块不动**；否则把各行接成一段。
"""
import app


def test_wrapped_paragraph_is_joined():
    md = ("205. The applicant alleges that GoC intervention causes the price of materials to\n"
          "reflect non-commercial factors in the lifting equipment industry.")
    out = app._unwrap_hard_linebreaks(md)
    assert out == ("205. The applicant alleges that GoC intervention causes the price of materials "
                   "to reflect non-commercial factors in the lifting equipment industry.")


def test_lettered_list_is_untouched():
    md = ("a) the goods have been or are being dumped in the UK;\n"
          "b) the dumping has caused injury to UK industry;\n"
          "c) how the amounts have been calculated.")
    assert app._unwrap_hard_linebreaks(md) == md


def test_indented_lettered_list_is_untouched():
    md = ("    a) goods imported into the UK have been dumped\n"
          "    b) the dumping has caused injury")
    assert app._unwrap_hard_linebreaks(md) == md


def test_bullet_list_is_untouched():
    md = "- a summary of the facts considered\n- details of the analysis"
    assert app._unwrap_hard_linebreaks(md) == md


def test_numbered_list_is_untouched():
    md = "1. first item\n2. second item"
    assert app._unwrap_hard_linebreaks(md) == md


def test_roman_numeral_list_is_untouched():
    md = "i) first\nii) second\niii) third"
    assert app._unwrap_hard_linebreaks(md) == md


def test_table_is_untouched():
    md = "| A | B |\n|:---|:---|\n| 1 | 2 |"
    assert app._unwrap_hard_linebreaks(md) == md


def test_code_fence_is_untouched():
    md = "```python\nx = 1\ny = 2\n```"
    assert app._unwrap_hard_linebreaks(md) == md


def test_blockquote_is_untouched():
    md = "> first quoted line\n> second quoted line"
    assert app._unwrap_hard_linebreaks(md) == md


def test_hard_break_two_spaces_is_preserved():
    """行尾两个空格是 Markdown 显式硬换行，作者本意如此，不合并。"""
    md = "first line  \nsecond line"
    assert app._unwrap_hard_linebreaks(md) == md


def test_list_item_with_wrapped_text_joins_the_wrap_only():
    """条目自身的折行要接回该条目，但条目之间不合并。"""
    md = ("- a summary of the facts considered during the investigation and an\n"
          "  explanation of the findings\n"
          "- details of the analysis")
    out = app._unwrap_hard_linebreaks(md)
    assert "investigation and an explanation of the findings" in out, out
    assert "\n- details of the analysis" in out, out


def test_cjk_lines_join_without_space():
    md = "本段落在此处被折行\n后半句紧接着上文。"
    assert app._unwrap_hard_linebreaks(md) == "本段落在此处被折行后半句紧接着上文。"


def test_hyphenated_word_across_lines():
    md = "an inter-\nnational standard applies."
    assert app._unwrap_hard_linebreaks(md) == "an international standard applies."


def test_multiple_blocks_preserved():
    md = "205. first para wrapped\nhere.\n\n206. second para wrapped\nthere."
    out = app._unwrap_hard_linebreaks(md)
    assert out == "205. first para wrapped here.\n\n206. second para wrapped there."


def test_footnote_definition_block_joins_its_own_wrap():
    md = "[^88]: Commission Staff Working Document, Section 16,\n    page 458-496"
    out = app._unwrap_hard_linebreaks(md)
    assert out == "[^88]: Commission Staff Working Document, Section 16, page 458-496"
