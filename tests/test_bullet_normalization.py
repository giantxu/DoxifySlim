"""字面 bullet 字符归一化成 Markdown 列表项。

运行: conda run -n mineru python -m pytest tests/test_bullet_normalization.py -v

背景（2026-08-21）：MinerU 把项目符号原样输出成 U+2022 圆点（实测一份 125 页文档 94 行），
而 `•` 不是 Markdown 语法——渲染出来是段落里的一个字符，不是列表项。更要命的是
_NEW_ITEM_RE / _LIST_ITEM_RE 都不认它，于是这些行被当成普通散文，会被
_unwrap_hard_linebreaks 和 _merge_broken_paragraphs 并进上一段。

必须排在那两个 pass 之前：结构没还原，它们就会拿错误的前提做判断。
"""
import app


def test_bullet_becomes_dash():
    assert app._bullet_chars_to_markdown("• details of the analysis") == "- details of the analysis"


def test_indented_bullet_keeps_indent():
    assert app._bullet_chars_to_markdown("    • nested") == "    - nested"


def test_various_bullet_glyphs():
    for ch in "•‣▪▫◦●○■□":
        assert app._bullet_chars_to_markdown(f"{ch} item") == "- item", ch


def test_bullet_mid_line_untouched():
    """句中的圆点不是项目符号，动它就是篡改正文。"""
    src = "The list uses • as its marker."
    assert app._bullet_chars_to_markdown(src) == src


def test_middle_dot_not_converted():
    """U+00B7 在中文人名里当间隔号用（约翰·史密斯），绝不能当 bullet。"""
    src = "· not a bullet"
    assert app._bullet_chars_to_markdown(src) == src


def test_dash_bullet_not_touched():
    """连字符/破折号开头的行可能是真列表，也可能是破折号引语；已有 - 的不动。"""
    src = "- already markdown"
    assert app._bullet_chars_to_markdown(src) == src


def test_bullet_without_following_space_untouched():
    """没有分隔空格的更像正文里的符号。"""
    src = "•nospace"
    assert app._bullet_chars_to_markdown(src) == src


def test_multiline_document():
    src = ("Intro line\n\n"
           "• first item\n"
           "• second item\n\n"
           "Closing line")
    out = app._bullet_chars_to_markdown(src)
    assert out.splitlines()[2:4] == ["- first item", "- second item"]
    assert out.startswith("Intro line") and out.endswith("Closing line")


def test_converted_bullets_are_recognised_as_list_items():
    """转换的全部意义：转完之后，两个列表正则必须认得它。"""
    line = app._bullet_chars_to_markdown("• details")
    assert app._NEW_ITEM_RE.match(line)
    assert app._LIST_ITEM_RE.match(line)


def test_document_without_bullets_unchanged():
    src = "# Heading\n\nJust prose.\n"
    assert app._bullet_chars_to_markdown(src) == src


def test_trailing_hard_break_preserved():
    """行尾硬换行是 Markdown 语义，转换 bullet 时不得顺手抹掉。"""
    assert app._bullet_chars_to_markdown("• item   \n") == "- item   \n"
