"""页码兜底：VLM 偶尔不遵守「不要转录页码」这条提示词。

运行: conda run -n mineru python -m pytest tests/test_page_number_strip.py -v

背景（2026-08-21）：提示词第 10 条要求跳过页眉页脚的页码，125 页里 116 页遵守了，
9 页漏网。实测这 9 处**全部位于该页文本的末尾**，且数字与 PDF 页码完全相等（差值
集合为 {0}）。这是 VLM 的指令遵守波动，程序侧加一道兜底。

判据刻意窄：整行只有一个数字，且位于页首或页尾。正文里不可能出现这种行——表格行有
|，列表项有标记，脚注定义要求数字后跟正文，编号段落也带正文。
"""
import app


def test_strips_trailing_page_number():
    t = "正文最后一句。\n\n20"
    assert app._strip_page_number(t) == "正文最后一句。"


def test_strips_leading_page_number():
    t = "20\n\n正文第一句。"
    assert app._strip_page_number(t) == "正文第一句。"


def test_keeps_number_in_the_middle():
    """页中间孤立的数字不动——那可能是别的东西，兜底只管页首页尾。"""
    t = "前面一段。\n\n20\n\n后面一段。"
    assert app._strip_page_number(t) == t


def test_keeps_numbered_paragraph():
    t = "20. 这是编号段落，有正文。\n\n下一段。"
    assert app._strip_page_number(t) == t


def test_keeps_bare_footnote_definition():
    """"20 定义正文" 是裸数字脚注定义，不能当页码删掉。"""
    t = "正文。\n\n20 Commission Staff Working Document, page 671."
    assert app._strip_page_number(t) == t


def test_keeps_table_row_with_number():
    t = "| A | B |\n|:---|:---|\n| 20 | 30 |"
    assert app._strip_page_number(t) == t


def test_handles_multi_digit_and_whitespace():
    assert app._strip_page_number("正文\n\n  100  ") == "正文"
    assert app._strip_page_number("正文\n\n7") == "正文"


def test_empty_and_number_only_page():
    assert app._strip_page_number("") == ""
    assert app._strip_page_number("20") == "", "整页只有页码 → 变成空页"


def test_does_not_strip_four_plus_digit_values():
    """年份这类四位数留着——页码极少有四位，而 2024 独占一行更可能是别的内容。"""
    t = "正文\n\n2024"
    assert app._strip_page_number(t) == t


# ------------------------------------------- 页中间的页码：靠「等于页码」识别


def test_strips_mid_page_number_matching_page_index():
    """VLM 偶尔把页脚放在页中间（阅读顺序所致），位置判据够不着。

    此时用更紧的信号：整行数字恰好等于该页页码。实测该文档 9 处漏网页码与 PDF 页码
    差值全为 0，这条判据足够可靠，且孤立成行又恰好等于页码的正当内容几乎不存在。
    """
    t = "514. 前一段正文。\n\n96\n\n**Table 24: 标题**"
    assert app._strip_page_number(t, 96) == "514. 前一段正文。\n\n**Table 24: 标题**"


def test_mid_page_number_not_matching_page_index_is_kept():
    """页中间的数字若与页码不符，来路不明，保留。"""
    t = "前一段。\n\n42\n\n后一段。"
    assert app._strip_page_number(t, 96) == t


def test_page_index_rule_does_not_break_tables_or_footnotes():
    t = "| A |\n|:---|\n| 96 |"
    assert app._strip_page_number(t, 96) == t
    t2 = "正文。\n\n96 Commission Staff Working Document, page 671."
    assert app._strip_page_number(t2, 96) == t2
