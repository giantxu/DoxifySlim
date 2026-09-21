"""HTML 表格转 Markdown。

运行: conda run -n mineru python -m pytest tests/test_html_table_to_markdown.py -v

背景（2026-08-21）：MinerU 与 PaddleOCR-VL 的表格识别模型输出的是 HTML 而非 Markdown
（MinerU 34 张、Paddle 37 张）。HTML 表格在 Obsidian 里能渲染，但不是 Markdown 语法：
无法被 _table_continuation 跨页合并，diff 不可读，也没法手工编辑。

转换的底线是无损：含合并单元格（rowspan/colspan >= 2）的表 Markdown 表达不了，原样保留
HTML。实测该文档 MinerU 34 张里 32 张可无损转换，Paddle 37 张里 31 张。
"""
import app


def test_simple_table_becomes_markdown():
    html = ("<table><tr><td>Injury Period</td><td>Year</td></tr>"
            "<tr><td>1 Oct 2021 - 30 Sep 2022</td><td>Year 1</td></tr></table>")
    out = app._html_tables_to_markdown(html)
    assert out.splitlines() == [
        "| Injury Period | Year |",
        "| --- | --- |",
        "| 1 Oct 2021 - 30 Sep 2022 | Year 1 |",
    ]


def test_mineru_rowspan1_colspan1_attrs_are_not_merges():
    """MinerU 给每个单元格都写 rowspan=1 colspan=1；那不是合并，必须照转。"""
    html = ("<table><tr><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>B</td></tr>"
            "<tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td></tr></table>")
    out = app._html_tables_to_markdown(html)
    assert "| A | B |" in out and "| 1 | 2 |" in out
    assert "<table" not in out


def test_paddle_inline_styles_are_stripped():
    html = ("<table border=1 style='margin: auto;'>"
            "<tr><td style='text-align: center; word-wrap: break-word;'>A</td>"
            "<td style='text-align: center;'>B</td></tr>"
            "<tr><td style='x'>1</td><td style='y'>2</td></tr></table>")
    out = app._html_tables_to_markdown(html)
    assert "| A | B |" in out and "style" not in out


def test_merged_cells_stay_html():
    """Markdown 表达不了合并单元格；转了就是丢信息，宁可留 HTML。"""
    html = '<table><tr><td colspan="2">Spans two</td></tr><tr><td>a</td><td>b</td></tr></table>'
    assert app._html_tables_to_markdown(html) == html


def test_rowspan_merge_stays_html():
    html = '<table><tr><td rowspan="3">Tall</td><td>x</td></tr><tr><td>y</td></tr></table>'
    assert app._html_tables_to_markdown(html) == html


def test_th_header_cells_supported():
    html = "<table><tr><th>H1</th><th>H2</th></tr><tr><td>a</td><td>b</td></tr></table>"
    out = app._html_tables_to_markdown(html)
    assert out.startswith("| H1 | H2 |")


def test_pipe_in_cell_is_escaped():
    """单元格里的竖线不转义会把这一行劈成多列。"""
    html = "<table><tr><td>a|b</td><td>c</td></tr><tr><td>1</td><td>2</td></tr></table>"
    out = app._html_tables_to_markdown(html)
    assert r"a\|b" in out
    # 未转义的竖线才是列分隔符：两列 -> 3 个
    import re as _re
    assert len(_re.findall(r"(?<!\\)\|", out.splitlines()[0])) == 3


def test_ragged_rows_are_padded_not_dropped():
    """列数不齐时补空格，绝不丢内容——丢了就是无声的数据损失。"""
    html = "<table><tr><td>A</td><td>B</td></tr><tr><td>only one</td></tr></table>"
    out = app._html_tables_to_markdown(html)
    assert "| only one |  |" in out


def test_inner_tags_inside_cell_are_unwrapped():
    html = "<table><tr><td><b>Bold</b></td><td>x</td></tr><tr><td>1</td><td>2</td></tr></table>"
    out = app._html_tables_to_markdown(html)
    assert "| Bold | x |" in out


def test_br_becomes_space():
    html = "<table><tr><td>a<br>b</td><td>c</td></tr><tr><td>1</td><td>2</td></tr></table>"
    out = app._html_tables_to_markdown(html)
    assert "| a b | c |" in out


def test_single_row_table_gets_empty_body():
    """只有一行时它就是表头；Markdown 必须有分隔行才渲染成表。"""
    html = "<table><tr><td>Only</td><td>Row</td></tr></table>"
    out = app._html_tables_to_markdown(html)
    assert out.splitlines() == ["| Only | Row |", "| --- | --- |"]


def test_empty_table_left_alone():
    assert app._html_tables_to_markdown("<table></table>") == "<table></table>"


def test_surrounding_text_preserved():
    md = "Before.\n\n<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>\n\nAfter."
    out = app._html_tables_to_markdown(md)
    assert out.startswith("Before.\n\n") and out.endswith("\n\nAfter.")
    assert "| A | B |" in out


def test_multiple_tables_all_converted():
    t = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
    out = app._html_tables_to_markdown(t + "\n\ntext\n\n" + t)
    assert out.count("| A | B |") == 2 and "<table" not in out


def test_document_without_tables_is_unchanged():
    md = "# Heading\n\nJust prose with a < sign and 2 > 1.\n"
    assert app._html_tables_to_markdown(md) == md


def test_markdown_table_untouched():
    """vlm 模式本来就产 Markdown 表，转换器对它必须是空操作。"""
    md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    assert app._html_tables_to_markdown(md) == md


# ------------------------------------------------- 前后必须留空行（GFM 硬要求）

def test_caption_on_same_line_is_split_off():
    """MinerU 实测把表题和表格放在同一行：`Table 1: Amounts <table>…`。

    HTML 表跟在文字后面照样渲染，Markdown 表不行——GFM 要求表格自成块，否则整块退化
    成普通段落。实测该文档 34 张表里 29 张是这个形状，不补空行就等于把表全毁了。
    """
    md = ("Table 1: Amounts <table><tr><td>A</td><td>B</td></tr>"
          "<tr><td>1</td><td>2</td></tr></table>\n\nAfter.")
    out = app._html_tables_to_markdown(md)
    assert "Table 1: Amounts\n\n| A | B |" in out
    assert out.endswith("\n\nAfter.")


def test_text_immediately_after_table_gets_blank_line():
    md = ("<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
          "Trailing prose.")
    out = app._html_tables_to_markdown(md)
    assert "| 1 | 2 |\n\nTrailing prose." in out


def test_single_newline_before_table_is_widened():
    md = "Caption line\n<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
    out = app._html_tables_to_markdown(md)
    assert "Caption line\n\n| A | B |" in out


def test_existing_blank_lines_not_doubled():
    md = "Before.\n\n<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>\n\nAfter."
    out = app._html_tables_to_markdown(md)
    assert "\n\n\n" not in out


def test_merged_table_left_html_keeps_its_context():
    """未转换的表不该被挪动位置。"""
    md = 'Caption <table><tr><td colspan="2">X</td></tr><tr><td>a</td><td>b</td></tr></table>'
    assert app._html_tables_to_markdown(md) == md


# ---------------------------------- 回归：不得破坏文档别处的 Markdown 硬换行

def test_hard_line_breaks_elsewhere_survive():
    """两个以上行尾空格是 Markdown 的硬换行。

    2026-08-21 回归：为清掉表题行的行尾空格，我在函数末尾加了全文范围的
    re.sub(r"[ \t]+\n", "\n", out)，把整篇 39 处硬换行全抹了。目录每行本来靠硬换行
    分行，护栏一失，_unwrap_hard_linebreaks 把整个目录粘成了一段。
    清理只能作用在表格接缝处。
    """
    md = ("Section A: Introduction . . 3   \n"
          "Section B: Preliminary findings.. 4   \n"
          "Section C: Next steps.. . 6\n\n"
          "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>")
    out = app._html_tables_to_markdown(md)
    assert "Introduction . . 3   \n" in out, "目录行的硬换行被抹掉了"
    assert "findings.. 4   \n" in out
    assert "| A | B |" in out


def test_no_table_document_keeps_trailing_spaces():
    md = "line one   \nline two   \n"
    assert app._html_tables_to_markdown(md) == md
