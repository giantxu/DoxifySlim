"""裸数字脚注：VLM 把上标格式完全丢失时的形态。

运行: conda run -n mineru python -m pytest tests/test_footnote_bare_numbers.py -v

背景（2026-08-21）：同一篇文档的第三种形态——标记与定义都退化成**纯数字**，既没有
插入符也没有上标，定义行连 "N." 的点号都没有：

    正文: ...Raw Materials Information System76, which notes...
    定义: 72 重磅！2021年中国及31省市液压行业政策汇总...
          73 Acme Hydraulic 2026 Company Profile...

裸数字的歧义远大于前两种，必须极保守。判据是**连续编号的连排**：真正的脚注区是一串
编号递增、行距相近的定义行；孤立的一行几乎必然是别的东西。实测该文档的候选行里，
连排规则正确采信了 5 个脚注区（22 条），并拒绝了三个危险的假阳性——文档日期
"16 July 2026"、正文续行 "40 assemblies, and weldments…"、句子续行 "11 of the
Regulations."。

正文侧同样保守：只转换连排已确认的编号，且数字不得紧跟在数字或小数点之后
（挡住 "16.25%" 与章节号 "H.1.3.11"）。
"""
import app


def _defs(*lines):
    return "\n\n".join(lines) + "\n"


def test_consecutive_run_is_recognised():
    md = ("The system76 notes this. Overcapacity77 too. Response78 says so.\n\n"
          "76 RMIS - Lithium-based batteries supply chain challenges,\n\n"
          "77 Commission Staff Working Document, Page 671.\n\n"
          "78 Questionnaire response NON-con Link\n")
    out = app._normalize_footnotes(md)
    assert "system[^76] notes" in out, out
    assert "[^76]: RMIS - Lithium-based batteries supply chain challenges," in out
    assert "[^77]: Commission Staff Working Document, Page 671." in out
    assert "[^78]: Questionnaire response NON-con Link" in out


def test_page_number_inside_definition_is_not_touched():
    """定义正文里的 "Page 671." 不能被当成标记。"""
    md = ("A76 B77 C78\n\n76 x\n\n77 Commission Staff Working Document, Page 671.\n\n78 y\n")
    out = app._normalize_footnotes(md)
    assert "Page 671." in out, out


def test_isolated_number_line_is_not_a_definition():
    """孤立一行的 "16 July 2026" 是日期，绝不能当脚注定义。"""
    md = "16 July 2026\n\n正文引用了 something16 这里。\n"
    out = app._normalize_footnotes(md)
    assert out == md, out


def test_sentence_continuation_line_is_not_a_definition():
    md = "上一段结尾\n\n11 of the Regulations.\n\n另一段11 正文\n"
    out = app._normalize_footnotes(md)
    assert out == md, out


def test_decimal_and_section_numbers_are_never_converted():
    """连排确认了 25/26/27，但 "16.25%" 与 "H.1.3.26" 里的数字不能动。"""
    md = ("Value is 16.25% here. See H.1.3.26 below. marker25 and marker26 and marker27.\n\n"
          "25 first def\n\n26 second def\n\n27 third def\n")
    out = app._normalize_footnotes(md)
    assert "16.25%" in out, out
    assert "H.1.3.26" in out, out
    assert "marker[^25]" in out and "marker[^26]" in out, out


def test_run_shorter_than_three_is_rejected():
    md = "a40 b41\n\n40 first\n\n41 second\n"
    out = app._normalize_footnotes(md)
    assert out == md, "只有两条的连排不足以采信"


def test_ordinary_numbered_paragraphs_untouched():
    """编号段落（"12. 正文"）带点号，不属于裸数字形态，且本就有 _FN_ITEM_RE 的闸门。"""
    md = "12. 依据第 13(3) 段，TRA 建议采取措施。\n\n13. 另一段正文。\n\n14. 第三段。\n"
    out = app._normalize_footnotes(md)
    assert out == md, out


# ---------------------------------------------- 句末标记（紧跟在句点之后）


def test_marker_after_sentence_period():
    """"investigations.88" —— 句子以字母+句点结束，脚注号紧随其后。

    此前前置守卫 (?<![\\d.]) 写在整个分支组之前，把这条分支整个屏蔽了。
    """
    md = ("Trade Defence investigations.88 Whilst its report covers X. "
          "PRC plastics sector.90 And more.\n\n"
          "87 first def\n\n88 second def\n\n89 third def\n\n90 fourth def\n")
    out = app._normalize_footnotes(md)
    assert "investigations.[^88]" in out, out
    assert "sector.[^90]" in out, out


def test_marker_after_four_digit_year():
    """"from 2023 to 2024.89" —— 年份后的句点 + 脚注号，与小数同形但可区分。"""
    # 正文需为多数编号提供角标佐证，否则连排会被判为段落编号（见
    # test_bare_footnote_needs_citation.py）——真实文档实测佐证率 89%。
    md = ("revenue increased year on year from 2023 to 2024.89 The TRA found more.87 "
          "Later work88 and a review90 followed.\n\n"
          "87 a\n\n88 b\n\n89 c\n\n90 d\n")
    out = app._normalize_footnotes(md)
    assert "2024.[^89]" in out, out


def test_decimal_with_two_digits_before_period_is_safe():
    """"16.25%" 与 "16.25 million"：句点前只有两位，不是年份，绝不能动。"""
    md = ("Margin is 16.25% and revenue 16.25 million here. Also see D2.26 section.\n\n"
          "25 a\n\n26 b\n\n27 c\n")
    out = app._normalize_footnotes(md)
    assert "16.25%" in out and "16.25 million" in out, out
    assert "D2.26" in out, out


def test_section_numbers_never_converted():
    md = ("See G2.1.2 and D2.1 and H.1.3.26 below. Real marker here.26 "
          "Also system25 and review27 apply.\n\n"
          "25 a\n\n26 b\n\n27 c\n")
    out = app._normalize_footnotes(md)
    assert "G2.1.2" in out and "D2.1" in out and "H.1.3.26" in out, out
    assert "here.[^26]" in out, out
