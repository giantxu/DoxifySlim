"""行首脚注定义行的各种畸形写法。

运行: conda run -n mineru python -m pytest tests/test_footnote_def_shapes.py -v

背景（2026-08-21）：同一篇 VLM 产物里，脚注定义行至少出现三种形态——标准的 [^N]:、
行首 Unicode 上标（"¹ 正文"）、行首 ASCII 插入符（"^93 正文"）。上一轮只补了 Unicode
一种，ASCII 那 26 行仍然没转，连带 49 处正文标记也留在原地。教训是：定义行的识别必须
按「标记形态」统一处理，不能一种一种打补丁。

死锁成因：安全闸门要求「编号有已知定义」才转换正文标记，而定义本身正是没被认出的
那一种，于是标记与定义都原样留下。

交叉校验：行首标记只有在该编号**同时**作为正文标记出现过时才算定义。没有任何引用的
"定义"没有意义，这条能挡住代码块里恰好以 ^12 开头的行之类的误判。
"""
import app


def test_ascii_caret_definition_line():
    """这次漏掉的形态：^93 开头的定义行。"""
    md = "semiconductors,^93 such as lifting equipment.\n\n^93 [What is a Semiconductor?](https://x)\n"
    out = app._normalize_footnotes(md)
    assert "semiconductors,[^93] such as" in out, out
    assert "[^93]: [What is a Semiconductor?](https://x)" in out, out
    assert "^93 " not in out


def test_unicode_superscript_definition_line_still_works():
    """上一轮修好的形态不得回归。"""
    md = "See the notice.¹\n\n¹ Applicant submission, pages 44-45.\n"
    out = app._normalize_footnotes(md)
    assert "See the notice.[^1]" in out
    assert "[^1]: Applicant submission, pages 44-45." in out


def test_pandoc_superscript_definition_line():
    md = "参见此处^12^。\n\n^12^ 引用来源说明\n"
    out = app._normalize_footnotes(md)
    assert "[^12]" in out
    assert "[^12]: 引用来源说明" in out


def test_html_sup_definition_line():
    md = "参见此处<sup>7</sup>。\n\n<sup>7</sup> 引用来源说明\n"
    out = app._normalize_footnotes(md)
    assert "[^7]" in out
    assert "[^7]: 引用来源说明" in out


def test_multi_digit_ascii_caret():
    md = "正文^126 引用。\n\n^126 Commission Staff Working Document, page 30.\n"
    out = app._normalize_footnotes(md)
    assert "[^126]: Commission Staff Working Document, page 30." in out, out


def test_definition_without_any_reference_is_ignored():
    """没有任何正文引用的「定义」不算数——挡住代码块里以 ^12 开头的行之类的误判。"""
    md = "普通段落，没有任何脚注标记。\n\n^12 这行只是恰好以插入符开头\n"
    out = app._normalize_footnotes(md)
    assert out == md, out


def test_ordinary_document_untouched():
    for md in (
        "第一条 本合同自双方签字之日起生效。\n\n第二条 面积为 30m² 的场地。\n",
        "12. 依据第 13(3) 段，TRA 建议采取措施。\n\n13. 另一段正文。\n",
        "体积 5cm³，面积 2m²。\n",
    ):
        assert app._normalize_footnotes(md) == md, md


def test_mixed_shapes_in_one_document():
    """真实文档的样子：三种形态混在一起，全部要归一化。"""
    md = (
        "First.[^2]\nSecond.³\nThird.^93\n\n"
        "[^2]: Already fine.\n"
        "³ Analysis on market share.\n"
        "^93 [What is a Semiconductor?](https://x)\n"
    )
    out = app._normalize_footnotes(md)
    for expect in ("Second.[^3]", "Third.[^93]",
                   "[^3]: Analysis on market share.",
                   "[^93]: [What is a Semiconductor?](https://x)"):
        assert expect in out, (expect, out)
