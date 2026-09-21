"""脚注转换不得污染链接/图片路径、行内代码和 URL。

运行: conda run -n mineru python -m pytest tests/test_footnote_protects_urls.py -v

背景（2026-08-24）：一份实际文档的 59 个图片引用全部损坏——
`![图](images/p003_1.jpeg)` 变成了 `![图](images/p[^003]_1.jpeg)`，图片全部无法显示。

根因：_normalize_footnotes 在纯文本上做正则替换，不知道 Markdown 链接目标的存在。
裸数字规则里的「数字紧贴字母」一条（本意匹配 `System76` 这种正文角标）在
`images/p003_1.jpeg` 里命中了紧贴 `p` 的 `003`。

这类位置的共同点是：内容是机器路径而非人写的散文，任何脚注标记都不该出现在里面。
"""
import app


# 连排确认还要求多数编号在正文里有角标佐证，否则会被判为段落编号
# （见 test_bare_footnote_needs_citation.py）。这一句提供佐证，断言时剔除。
_CITES = "Cited as system1 and work2 and review3 and note4 here."


def _doc(body):
    """带一段连排裸脚注定义 + 佐证句，让 1~4 进入「连排确认」集合。"""
    return (body + "\n\n" + _CITES
            + "\n\n1 第一条来源说明文字。\n2 第二条来源说明文字。"
              "\n3 第三条来源说明文字。\n4 第四条来源说明文字。\n")


def _body(out):
    return out.split("\n\nCited as ")[0]


def test_image_path_is_not_corrupted():
    out = app._normalize_footnotes(_doc("![图](images/p003_1.jpeg)"))
    assert "![图](images/p003_1.jpeg)" in out


def test_all_page_numbered_images_survive():
    refs = "\n\n".join(f"![图](images/p{n:03d}_1.png)" for n in (3, 9, 10, 23, 41))
    out = app._normalize_footnotes(_doc(refs))
    for n in (3, 9, 10, 23, 41):
        assert f"images/p{n:03d}_1.png" in out, n
    assert "[^" not in _body(out)


def test_markdown_link_target_is_not_corrupted():
    out = app._normalize_footnotes(_doc("见[附件](files/doc2.pdf)说明"))
    assert "(files/doc2.pdf)" in out


def test_inline_code_is_not_corrupted():
    out = app._normalize_footnotes(_doc("变量 `img2` 的取值"))
    assert "`img2`" in out


def test_bare_url_is_not_corrupted():
    out = app._normalize_footnotes(_doc("来源 https://example.com/a/v2/page3 备注"))
    assert "https://example.com/a/v2/page3" in out


def test_html_img_tag_is_not_corrupted():
    out = app._normalize_footnotes(_doc('<img src="images/p003_1.jpeg" alt="x">'))
    assert 'src="images/p003_1.jpeg"' in out


def test_real_inline_marker_still_converts():
    """保护不能矫枉过正：正文里真正的角标必须照转。"""
    out = app._normalize_footnotes(_doc("参见 Information System3，其中提到"))
    assert "System[^3]" in out


def test_marker_adjacent_to_a_link_still_converts():
    """紧挨着链接的正文角标不受保护范围影响。"""
    out = app._normalize_footnotes(_doc("见[附件](files/doc2.pdf)。资料3 表明"))
    assert "(files/doc2.pdf)" in out


def test_sup_marker_inside_text_still_converts():
    out = app._normalize_footnotes(_doc("正文<sup>3</sup>继续"))
    assert "[^3]" in out
    assert "<sup>" not in _body(out)


def test_document_without_links_behaves_as_before():
    src = _doc("普通正文，引用了资料3。")
    out = app._normalize_footnotes(src)
    assert "[^3]" in out
