"""矢量簇里的「表格框线」不应被当成图表提取。

运行: conda run -n mineru python -m pytest tests/test_figure_table_filter.py -v

背景（2026-08-20，一份 125 页的贸易救济裁定书）：PDF 里表格的框线本身就是矢量绘图，
cluster_drawings() 会把它聚成一个簇，而 _keep() 只按尺寸过滤，于是整张表被裁剪渲染成
PNG。VLM 另一边已把表格正确转成 Markdown，结果同一张表既是表格又是图片；更糟的是表格
跨页时，多余图片被 _inject_figures 的兜底追加到页尾，正好插在表格两半之间，把表格劈开。

实测该文档 40 个符合尺寸条件的簇中，39 个表格的曲线数与斜线数**全为 0**，唯一的真折线图
有 80 条曲线、12 条斜线。判据即由此而来：只有横平竖直的线/矩形且含文字 → 表格 → 丢弃；
出现曲线或斜线 → 真图表 → 保留。
"""
import fitz
import pytest

import app


def _table_page(doc, shaded=True):
    p = doc.new_page(width=595, height=842)
    x0, y0, w, h, rows, cols = 60, 100, 460, 22, 5, 4
    if shaded:
        p.draw_rect(fitz.Rect(x0, y0, x0 + w, y0 + h), color=(0, 0, 0),
                    fill=(0.85, 0.85, 0.85), width=0.8)
    for r in range(rows + 1):
        y = y0 + r * h
        p.draw_line(fitz.Point(x0, y), fitz.Point(x0 + w, y), color=(0, 0, 0), width=0.8)
    for c in range(cols + 1):
        x = x0 + c * (w / cols)
        p.draw_line(fitz.Point(x, y0), fitz.Point(x, y0 + rows * h), color=(0, 0, 0), width=0.8)
    for r in range(rows):
        for c in range(cols):
            p.insert_text((x0 + c * (w / cols) + 6, y0 + r * h + 15), f"Cell {r}-{c}", fontsize=9)
    return p


def _line_chart_page(doc):
    """折线图：轴线是轴对齐的，但数据线是斜的、还有贝塞尔曲线。标签放在簇内。"""
    p = doc.new_page(width=595, height=842)
    ax0, ay0, aw, ah = 80, 150, 400, 250
    p.draw_line(fitz.Point(ax0, ay0 + ah), fitz.Point(ax0 + aw, ay0 + ah), color=(0, 0, 0), width=1)
    p.draw_line(fitz.Point(ax0, ay0), fitz.Point(ax0, ay0 + ah), color=(0, 0, 0), width=1)
    pts = [fitz.Point(ax0 + i * 80, ay0 + ah - 40 - i * 35) for i in range(5)]
    for a, b in zip(pts, pts[1:]):
        p.draw_line(a, b, color=(0.8, 0, 0), width=2)
    p.draw_bezier(pts[0], pts[1], pts[2], pts[3], color=(0, 0, 0.8), width=1.5)
    for i in range(5):
        p.insert_text((ax0 + 10 + i * 80, ay0 + ah - 12), f"Year {i}", fontsize=8)
    return p


def _pdf(*builders) -> bytes:
    doc = fitz.open()
    for b in builders:
        b(doc)
    data = doc.tobytes()
    doc.close()
    return data


def test_ruled_table_is_not_extracted_as_figure():
    figs = app._extract_figures(_pdf(_table_page))
    assert figs == {}, f"表格框线被当成图提取了: {figs}"


def test_unshaded_table_also_dropped():
    figs = app._extract_figures(_pdf(lambda d: _table_page(d, shaded=False)))
    assert figs == {}


def test_line_chart_is_still_extracted():
    """真图表必须保留——这是 VLM 转不了的内容。"""
    figs = app._extract_figures(_pdf(_line_chart_page))
    assert 1 in figs and len(figs[1]) == 1, f"折线图被误删了: {figs}"
    assert figs[1][0]["ext"] == "png"


def test_mixed_document_keeps_only_the_chart():
    figs = app._extract_figures(_pdf(_table_page, _line_chart_page))
    assert 1 not in figs, "第 1 页是表格，不该有图"
    assert 2 in figs and len(figs[2]) == 1, "第 2 页的折线图必须保留"


def test_predicate_reports_curves_and_diagonals():
    """判别函数本身：表格无曲线无斜线，折线图两者都有。"""
    data = _pdf(_table_page, _line_chart_page)
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        tbl_page, chart_page = doc[0], doc[1]
        tbl_rect = fitz.Rect(tbl_page.cluster_drawings()[0])
        chart_rect = fitz.Rect(chart_page.cluster_drawings()[0])
        assert app._is_ruled_table(tbl_page, tbl_rect) is True
        assert app._is_ruled_table(chart_page, chart_rect) is False
    finally:
        doc.close()


def test_graphic_without_text_is_kept():
    """无文字的纯图形即使全是轴对齐线条也保留——宁可多留，不可误删。"""
    def build(doc):
        p = doc.new_page(width=595, height=842)
        for i in range(6):
            p.draw_rect(fitz.Rect(80 + i * 5, 150 + i * 5, 400 - i * 5, 380 - i * 5),
                        color=(0, 0, 0), width=1)
    figs = app._extract_figures(_pdf(build))
    assert 1 in figs, "没有文字就无从判断 VLM 能否转录，应保留"
