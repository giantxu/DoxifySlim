"""PaddleOCR 的 LaTeX 上标残留，以及「引号+句点+编号」这种脚注标记。

运行: conda run -n mineru python -m pytest tests/test_latex_and_quote_footnote.py -v

背景（2026-08-21）：
1. _normalize_vlm_latex 只认 $^{N}$，认不出 PaddleOCR 实际吐出的另外两种形态——
   $ {}^{32} $（空基底的纯角标，实测 5 处）和 $ 15^{th} $（序数词，实测 1 处）。
   前者是脚注角标，认不出就等于丢一条脚注；后者渲染出来是一串乱码 LaTeX。
2. 句子以引号收尾时脚注角标落在句点之后：`decision-making”.27`。既有的三条裸数字
   规则都够不着它——规则 1 要求数字紧贴引号（这里隔着句点），规则 2 要求句点前是
   字母（这里是引号），规则 3 要求四位年份。

第 2 条**不能**靠放宽「连排确认」护栏来解决：实测放宽后 82 个章节号（G2.1 等）会被
误转，因为数字 1~6 恰好都有脚注定义。只能加一条位置上足够窄的规则——章节号永远
不会长成 `”.N`。
"""
import app


# ---------------------------------------------------------------- LaTeX

def test_empty_base_superscript_is_a_footnote_marker():
    assert app._normalize_vlm_latex("text $ {}^{32} $ more") == "text <sup>32</sup> more"


def test_plain_superscript_still_works():
    assert app._normalize_vlm_latex("text $ ^{63} $") == "text <sup>63</sup>"


def test_ordinal_keeps_its_base():
    """$ 15^{th} $ 是序数词，基底 15 必须留下，不能只剩上标。"""
    assert app._normalize_vlm_latex("In the $ 15^{th} $ FYP") == "In the 15<sup>th</sup> FYP"


def test_alphanumeric_base():
    assert app._normalize_vlm_latex("$ AD0012^{a} $") == "AD0012<sup>a</sup>"


def test_underline_still_works():
    assert app._normalize_vlm_latex(r"$\underline{\text{Prime Rate}}$") == "<u>Prime Rate</u>"


def test_subscript_still_works():
    assert app._normalize_vlm_latex("$_{2}$") == "<sub>2</sub>"


def test_real_math_untouched():
    """含运算符的是真公式，不能拆。"""
    src = "$a + b^{2} = c$"
    assert app._normalize_vlm_latex(src) == src


def test_document_without_latex_unchanged():
    src = "# Heading\n\nPlain prose with a $ sign.\n"
    assert app._normalize_vlm_latex(src) == src


# ---------------------------------------------- 引号 + 句点 + 编号

def _doc(body):
    """带一条脚注定义，好让 _normalize_footnotes 的安全门放行。"""
    return body + "\n\n[^27]: Some source\n\n[^2]: Another source"


def test_marker_after_closing_quote_and_period():
    out = app._normalize_footnotes(_doc('a mandate to “participate in decision-making”.27'))
    assert 'decision-making”.[^27]' in out


def test_straight_quote_variant():
    out = app._normalize_footnotes(_doc('he said "no".27'))
    assert '"no".[^27]' in out


def test_section_number_never_matches_this_shape():
    """这条规则的安全性全靠位置：章节号不可能长成 `”.N`。

    实测放宽护栏会误伤 82 个章节号，所以这里必须验证 G2.1 原样不动。
    """
    out = app._normalize_footnotes(_doc("See section G2.1 and D2.1 for detail."))
    body = out.split("\n\n[^")[0]          # 定义行本身含 [^N]，只看正文
    assert "G2.1" in body and "D2.1" in body
    assert "[^" not in body


def test_decimal_after_quote_is_not_converted():
    """引号后跟小数（罕见但可能）：小数点后跟着更多数字，规则用 (?![\\d%]) 挡住。"""
    out = app._normalize_footnotes(_doc('the price was “high”.275 per unit'))
    assert '“high”.275' in out


def test_number_without_definition_is_left_alone():
    """没有定义的编号不转——转了就是指向空处的引用。"""
    out = app._normalize_footnotes(_doc('a quote”.99 continues'))
    assert 'quote”.99' in out


def test_no_definitions_means_no_conversion():
    """安全门：文档没有任何脚注证据时整个函数不动手。"""
    src = 'a mandate to “participate”.27'
    assert app._normalize_footnotes(src) == src
