"""单元测试:_inject_figures 占位符回填三分支。

运行: cd /Users/xuzheng/CodingProjects/Doxify && conda run -n mineru python -m pytest tests/test_inject_figures.py -v
"""
from app import _inject_figures


def test_equal_placeholders_and_images():
    md = "第一段。\n\n[[FIGURE]]\n\n第二段。\n\n[[FIGURE]]\n\n第三段。"
    out = _inject_figures(md, ["images/p001_1.png", "images/p001_2.png"])
    assert "[[FIGURE]]" not in out
    # 顺序对应:第一个占位符 → 第一张图
    assert out.index("images/p001_1.png") < out.index("images/p001_2.png")
    assert "![图](images/p001_1.png)" in out
    assert "![图](images/p001_2.png)" in out


def test_more_images_than_placeholders():
    md = "第一段。\n\n[[FIGURE]]\n\n第二段。"
    out = _inject_figures(md, ["images/p001_1.png", "images/p001_2.png"])
    assert "![图](images/p001_1.png)" in out
    # 多出的图追加到页尾
    assert out.rstrip().endswith("![图](images/p001_2.png)")


def test_more_placeholders_than_images():
    md = "第一段。\n\n[[FIGURE]]\n\n第二段。\n\n[[FIGURE]]\n\n第三段。"
    out = _inject_figures(md, ["images/p001_1.png"])
    assert "![图](images/p001_1.png)" in out
    assert "[[FIGURE]]" not in out  # 多余占位符被删除
    assert "第三段。" in out


def test_no_placeholders_no_images():
    md = "纯文本页。"
    assert _inject_figures(md, []) == md


def test_images_but_no_placeholders():
    md = "纯文本页。"
    out = _inject_figures(md, ["images/p002_1.png"])
    assert out.startswith("纯文本页。")
    assert out.rstrip().endswith("![图](images/p002_1.png)")
