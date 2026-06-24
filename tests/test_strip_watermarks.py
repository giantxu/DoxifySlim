"""单元测试：_strip_watermarks 行为验证。

运行: cd /Users/xuzheng/CodingProjects/Doxify && python -m pytest tests/ -v
"""
from app import _strip_watermarks


def test_removes_header_barcode_line():
    md = (
        "Barcode: 4918520-01 A-570-139 REV - Admin Review 4/1/24 - 3/31/25\n"
        "\n"
        "# 正文标题\n"
        "正文内容。\n"
    )
    out = _strip_watermarks(md, enabled=True)
    assert "Barcode:" not in out
    assert "# 正文标题" in out
    assert "正文内容。" in out


def test_removes_footer_filed_by_line():
    md = (
        "正文段落一。\n"
        "\n"
        "Filed By: counsel@example.com, Filed Date: 5/5/26 2:15 PM, Submission Status: Approved\n"
        "\n"
        "正文段落二。\n"
    )
    out = _strip_watermarks(md, enabled=True)
    assert "Filed By:" not in out
    assert "Submission Status:" not in out
    assert "正文段落一。" in out
    assert "正文段落二。" in out


def test_removes_multi_page_watermarks():
    page = (
        "Barcode: 4918520-01 X-999-000 REV - Admin Review 1/1/25 - 12/31/25\n"
        "# 第 N 页标题\n"
        "页内容。\n"
        "Filed By: a@b.com, Filed Date: 1/1/26 10:00 AM, Submission Status: Approved\n"
    )
    md = page + "\n\n---\n\n" + page
    out = _strip_watermarks(md, enabled=True)
    assert out.count("Barcode:") == 0
    assert out.count("Filed By:") == 0
    assert out.count("# 第 N 页标题") == 2


def test_case_insensitive_match():
    md = (
        "BARCODE: 999-01 X-1-1 REV - admin review 1/1 - 2/2\n"
        "正文。\n"
        "filed by: x@y, FILED DATE: now, submission status: Pending\n"
    )
    out = _strip_watermarks(md, enabled=True)
    assert "BARCODE:" not in out
    assert "filed by:" not in out.lower()
    assert "x@y" not in out
    assert "正文。" in out


def test_disabled_returns_input_unchanged():
    md = (
        "Barcode: 4918520-01 A-570-139 REV - Admin Review 4/1/24 - 3/31/25\n"
        "正文。\n"
    )
    out = _strip_watermarks(md, enabled=False)
    assert out == md


def test_empty_input_returns_empty():
    assert _strip_watermarks("", enabled=True) == ""


def test_does_not_match_word_barcode_in_prose():
    """Lines starting with 'Barcode:' but lacking the DOC case-number
    structure should NOT be treated as watermarks.
    """
    md = (
        "本节讨论 Barcode 的格式与含义，详见附录。\n"
        "Barcode: 见附录 A。\n"  # 行首 "Barcode:" 但缺少案号结构
    )
    out = _strip_watermarks(md, enabled=True)
    assert "Barcode" in out
    assert "见附录 A" in out


def test_does_not_match_filed_by_without_full_triplet():
    md = "案件中 Filed By: 律师事务所，详情见上文。\n"
    out = _strip_watermarks(md, enabled=True)
    assert "律师事务所" in out


def test_collapses_blank_runs_after_strip():
    md = "段落一。\n\nBarcode: 1234-01 Y-1-1 REV - Admin Review 1 - 2\n\n段落二。\n"
    out = _strip_watermarks(md, enabled=True)
    assert "\n\n\n" not in out


def test_header_tolerates_newline_between_admin_and_review():
    """A watermark line broken by OCR between 'Admin' and 'Review' is still
    stripped because the case-number anchor matches the first line on its own.
    The 'Review ...' continuation line is left in place (it looks like prose),
    but the Barcode line itself is correctly removed.
    """
    md = (
        "Barcode: 12345-01 X-1-1 REV - Admin\n"
        "Review 1/1/24 - 12/31/24\n"
        "正文内容。\n"
    )
    out = _strip_watermarks(md, enabled=True)
    assert "Barcode:" not in out
    assert "12345-01" not in out
    assert "正文内容。" in out


def test_removes_investigation_stage_watermark():
    """Real-world INV - Investigation variant (was missed by the
    original Admin-Review-only regex).
    """
    md = (
        "Barcode:4657790-01 A-570-182 INV - Investigation -\n"
        "\n"
        "Parties should be aware that...\n"
    )
    out = _strip_watermarks(md, enabled=True)
    assert "Barcode:" not in out
    assert "Parties should be aware that..." in out


def test_removes_multiple_case_stage_variants():
    """Header regex must cover all CBP case stages, not just Admin Review."""
    samples = [
        "Barcode: 4918520-01 A-570-139 REV - Admin Review 4/1/24 - 3/31/25",
        "Barcode:4657790-01 A-570-182 INV - Investigation -",
        "Barcode: 1234567-89 C-570-100 SUN - Sunset Review",
        "Barcode: 9999-99 A-570-001 NSR - New Shipper Review",
    ]
    for s in samples:
        md = f"{s}\n正文。\n"
        out = _strip_watermarks(md, enabled=True)
        assert "Barcode" not in out, f"Failed to strip: {s}"
        assert "正文。" in out
