"""页眉页脚水印：新包装形态 + 必须剥在段落合并之前。

运行: conda run -n mineru python -m pytest tests/test_watermark_shapes_and_order.py -v

背景（2026-09-15）：用户勾选了去页眉页脚，产物里却仍有污染，且页脚行把下一页的
正文粘在了自己后面。两个叠加的缺陷：

1. 现在网关背后的模型输出页眉页脚时会加包装——反引号（`Barcode:...`）、中文标注
   前缀（*页眉：* Barcode:... *页脚：* Filed By:...）——旧正则只认行首裸文本，全部放过。
2. _strip_watermarks 排在 _merge_broken_paragraphs 之后。页眉页脚位于页边界，正是
   合并器工作的地方：页脚行不以句末标点结尾，合并器把下一页正文粘上去（实测第 122
   行），粘连后再剥就会带走正文。剥离必须在一切合并类 pass 之前——页脚剥掉后，被它
   隔断的段落两半相邻，合并器正好能接回。
"""
import app

H = "Barcode:4672356-01 C-570-161 INV - Investigation -"
F = "Filed By: counsel@example.com, Filed Date: 12/2/24 9:13 AM, Submission Status: Approved"


def _strip(md):
    return app._strip_watermarks(md, True, "test")


# ---------------------------------------------------------------- 新形态

def test_bare_lines_still_stripped():
    assert _strip(f"before\n\n{H}\n\n{F}\n\nafter") == "before\n\nafter"


def test_backtick_wrapped_lines_stripped():
    md = f"before\n\n`{H}`\n\n`{F}`\n\nafter"
    assert _strip(md) == "before\n\nafter"


def test_labeled_single_line_stripped():
    """模型把页眉页脚合并成一行并加中文标注：整行都是版面家具，整行删。"""
    md = f"before\n\n*页眉：* {H} *页脚：* {F}\n\nafter"
    assert _strip(md) == "before\n\nafter"


def test_labeled_footer_alone_stripped():
    md = f"before\n\n*页脚：* {F}\n\nafter"
    assert _strip(md) == "before\n\nafter"


def test_bold_wrapped_stripped():
    md = f"before\n\n**{H}**\n\nafter"
    assert _strip(md) == "before\n\nafter"


def test_lookalike_prose_untouched():
    """正文里引用这些词不能误删：没有编号格式/三件套就不是水印。"""
    md = "The report was Filed By counsel and discusses the barcode system."
    assert _strip(md) == md


def test_prose_mentioning_barcode_number_format_but_inline():
    """句中提及（非行首锚定包装）不删。"""
    md = "见文件 (Barcode:4672356-01 C-570-161 INV) 所载内容与其他材料。"
    assert _strip(md) == md


# ---------------------------------------------------------------- 顺序

def test_strip_runs_before_paragraph_merge_pin():
    """钉住调用点顺序：三条流水线里 _strip_watermarks 必须在
    _merge_broken_paragraphs 之前。直接检查源码顺序——运行时无从观测。"""
    import inspect, re as _re
    src = inspect.getsource(app)
    # 每个调用点：找同一函数体内两个调用的相对位置
    for fn in ("parse_pdf_streaming",):          # DoxifySlim 只有 VLM 一条流水线
        body = _re.search(rf"(?s)(?:async )?def {fn}\(.*?(?=\n(?:async )?def [a-zA-Z_])", src)
        assert body, fn
        b = body.group(0)
        i_strip = b.find("_strip_watermarks(")
        i_merge = b.find("_merge_broken_paragraphs(")
        assert i_strip != -1 and i_merge != -1, fn
        assert i_strip < i_merge, f"{fn}: 水印剥离必须在段落合并之前"


def test_footer_between_pages_no_longer_blocks_merge():
    """端到端语义：页脚行隔断的段落，剥掉后能接回一段。

    模拟组装后的形态：上一页段落中断（无句末标点）→ 页脚行 → 下一页续行小写开头。
    正确结果 = 页脚消失 + 两半合为一段。
    """
    md = ("The bank stated that the figures appear in the\n\n"
          f"`{F}`\n\n"
          "section of the annual reports of EXIM Bank.")
    out = app._strip_watermarks(md, True, "t")
    out = app._merge_broken_paragraphs(out)
    assert "Filed By" not in out
    assert "appear in the section of the annual reports" in out
