"""裸数字连排必须有正文角标佐证，否则那是段落编号而不是脚注。

运行: conda run -n mineru python -m pytest tests/test_bare_footnote_needs_citation.py -v

背景（2026-08-24）：一份编号段落的法律文书被彻底毁掉——304 个段落全部变成
`[^N]: ` 脚注定义，正文荡然无存，而 `[^N]` 引用是 0 个。

裸数字连排这个启发式本是为「VLM 把脚注的上标丢了、退化成纯数字」设计的，判据是
「相邻行上出现连续编号」。一份有 300 个编号段落的文书把这个判据压倒性地满足了：
连排长度 208，远超真实脚注文档的 18。连排长度本身分不开这两类。

分得开的是**引用关系**：脚注天生是被正文引用的，段落编号不是。实测——
  真脚注文档       18 条定义，16 条有正文角标佐证 = 89%
  答辩状（段落编号）208 条定义，33 条有佐证       = 16%
阈值取一半，两者干净分开。
"""
import app


def _numbered_paragraphs(n=12):
    """模拟法律文书：编号段落，正文里没有任何角标引用这些编号。"""
    return "\n\n".join(
        f"{i} This is the body of paragraph number {i}, which contains ordinary prose "
        f"and continues for a while without citing anything."
        for i in range(1, n + 1))


def _real_footnotes():
    """模拟真脚注：正文有角标，页脚有连排定义。"""
    body = ("The report on that system25 was published, and the later analysis26 "
            "confirmed it. A further review27 followed.\n\n")
    defs = "25 First source title.\n26 Ibid.\n27 Third source title."
    return body + defs


def test_numbered_paragraphs_are_not_footnotes():
    assert app._collect_bare_footnote_numbers(_numbered_paragraphs()) == set()


def test_numbered_paragraphs_survive_normalization():
    src = _numbered_paragraphs()
    out = app._normalize_footnotes(src)
    assert "[^" not in out, "段落编号被当成脚注定义，正文会被彻底毁掉"


def test_real_footnotes_still_detected():
    got = app._collect_bare_footnote_numbers(_real_footnotes())
    assert got == {25, 26, 27}


def test_real_footnotes_still_convert():
    out = app._normalize_footnotes(_real_footnotes())
    assert "[^25]: First source title." in out
    assert "system[^25]" in out


def test_partial_corroboration_above_half_is_kept():
    """真实文档里 VLM 常漏掉个别角标，所以判据是过半而非全部。"""
    body = ("The system25 and the later work26 are cited here.\n\n"
            "25 First source.\n26 Second source.\n27 Third source.")
    assert app._collect_bare_footnote_numbers(body) == {25, 26, 27}


def test_no_corroboration_at_all_is_rejected():
    body = "Plain prose with no citation markers at all in it.\n\n25 A.\n26 B.\n27 C."
    assert app._collect_bare_footnote_numbers(body) == set()


def test_definition_line_itself_is_not_its_own_citation():
    """定义行开头的数字不能算作对自己的引用，否则永远 100% 佐证。"""
    body = "Body prose without markers.\n\n25 Source one.\n26 Source two.\n27 Source three."
    assert app._collect_bare_footnote_numbers(body) == set()
