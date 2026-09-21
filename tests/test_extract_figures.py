"""单元测试:_extract_figures 图表提取与过滤规则。

运行: cd /Users/xuzheng/CodingProjects/Doxify && conda run -n mineru python -m pytest tests/test_extract_figures.py -v
"""
import fitz
import pytest

from app import _extract_figures


def _solid_png(w: int, h: int, color: tuple = (255, 0, 0)) -> bytes:
    """生成 w×h 纯色 PNG 字节。"""
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, w, h))
    pix.set_rect(pix.irect, color)
    return pix.tobytes("png")


def _new_a4(doc: fitz.Document) -> fitz.Page:
    return doc.new_page(width=595, height=842)


def test_extracts_embedded_image():
    doc = fitz.open()
    page = _new_a4(doc)
    page.insert_image(fitz.Rect(100, 100, 400, 300), stream=_solid_png(300, 200))
    data = doc.tobytes()
    doc.close()

    result = _extract_figures(data)
    assert list(result.keys()) == [1]
    assert len(result[1]) == 1
    fig = result[1][0]
    assert len(fig["bytes"]) > 0
    assert fig["ext"]  # png/jpeg 均可,非空即可


def test_filters_tiny_image():
    # 20×20pt:短边 < 40pt 且面积 < 页面 1.5%,应被过滤
    doc = fitz.open()
    page = _new_a4(doc)
    page.insert_image(fitz.Rect(100, 100, 120, 120), stream=_solid_png(50, 50))
    data = doc.tobytes()
    doc.close()

    assert _extract_figures(data) == {}


def test_filters_full_page_background():
    # 覆盖 > 90% 页面:视为背景,应被过滤
    doc = fitz.open()
    page = _new_a4(doc)
    page.insert_image(fitz.Rect(0, 0, 595, 842), stream=_solid_png(600, 850))
    data = doc.tobytes()
    doc.close()

    assert _extract_figures(data) == {}


def test_filters_repeated_xref_logo():
    # 同一 xref 出现在 5 页:视为页眉 logo,应全部过滤
    doc = fitz.open()
    page = _new_a4(doc)
    xref = page.insert_image(fitz.Rect(100, 100, 300, 250), stream=_solid_png(200, 150))
    for _ in range(4):
        p = _new_a4(doc)
        p.insert_image(fitz.Rect(100, 100, 300, 250), xref=xref)
    data = doc.tobytes()
    doc.close()

    assert _extract_figures(data) == {}


def test_extracts_vector_drawing_cluster():
    # 矢量绘图(柱状图形态):应裁剪渲染为 PNG。
    # 注意 cluster_drawings 默认容差仅 3pt,柱条彼此不相邻,
    # 需通过一条横轴线把所有柱条连通成一个簇。
    doc = fitz.open()
    page = _new_a4(doc)
    page.draw_line(fitz.Point(100, 300), fitz.Point(400, 300), color=(0, 0, 0))
    for i in range(5):
        x = 110 + i * 55
        page.draw_rect(fitz.Rect(x, 300 - (i + 1) * 30, x + 40, 300),
                       color=(0, 0, 1), fill=(0.2, 0.4, 0.8))
    data = doc.tobytes()
    doc.close()

    result = _extract_figures(data)
    assert list(result.keys()) == [1]
    assert len(result[1]) == 1
    assert result[1][0]["ext"] == "png"
    assert len(result[1][0]["bytes"]) > 0


def test_figures_sorted_by_vertical_position():
    doc = fitz.open()
    page = _new_a4(doc)
    page.insert_image(fitz.Rect(100, 500, 400, 700), stream=_solid_png(300, 200, (0, 255, 0)))
    page.insert_image(fitz.Rect(100, 100, 400, 300), stream=_solid_png(300, 200, (255, 0, 0)))
    data = doc.tobytes()
    doc.close()

    result = _extract_figures(data)
    figs = result[1]
    assert len(figs) == 2
    assert figs[0]["bbox"].y0 < figs[1]["bbox"].y0


def test_invalid_pdf_raises():
    # 契约:异常向上抛,由调用方捕获降级
    with pytest.raises(Exception):
        _extract_figures(b"not a pdf at all")


def test_multi_page_figures():
    doc = fitz.open()
    for _ in range(2):
        page = _new_a4(doc)
        page.insert_image(fitz.Rect(100, 100, 400, 300), stream=_solid_png(300, 200))
    data = doc.tobytes()
    doc.close()

    result = _extract_figures(data)
    assert sorted(result.keys()) == [1, 2]
    assert all(len(v) == 1 for v in result.values())
