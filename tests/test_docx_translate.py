"""DOCX 翻译引擎：lxml 原位改文本节点，格式/修订/批注全保留。

运行: conda run -n mineru python -m pytest tests/test_docx_translate.py -v

设计约束（2026-08-24，样例为一份含 43 处插入/14 处删除/22 条批注的葡语合同）：
- 只改 w:t / w:delText 的文本内容，XML 树一个节点不增不删——格式、修订、批注、
  书签、域、图片因此天然保留，不依赖任何高层库的「理解」。
- 分段按修订语境切开（normal / ins / del 各自成段）：把删除文本和替换它的插入
  文本拼在一起翻译会得到交错的胡话，且回填会把译文错归到别人的修订名下。
- 段内 加粗/斜体/下划线 用 **·**/*·*/<u>·</u> 标记随文送给 LLM（实测 158/358 个
  段落存在段内格式差异，整段合并会抹掉合同里的加粗定义词）；标记解析失败时
  退化为整段填入首个 run——宁可丢局部格式也绝不丢文字。
- 制表符/换行符是分段边界：跨过它们拼接翻译会把译文错铺到另一个单元格/行里。
"""
import asyncio
import io
import os
import re
import zipfile

import pytest
from lxml import etree

import app

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def _p(inner: str) -> etree._Element:
    """构造一个 <w:p>，inner 为其子元素 XML。"""
    xml = f'<w:p xmlns:w="{W}">{inner}</w:p>'
    return etree.fromstring(xml)


def _r(text: str, rpr: str = "") -> str:
    pr = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
    return f'<w:r>{pr}<w:t xml:space="preserve">{text}</w:t></w:r>'


def _texts(el) -> list[str]:
    return [t.text or "" for t in el.iter(f"{{{W}}}t", f"{{{W}}}delText")]


# ---------------------------------------------------------------- 分段

def test_plain_paragraph_is_one_segment():
    p = _p(_r("Hello ") + _r("world."))
    segs = app._docx_collect_segments(p)
    assert len(segs) == 1
    assert app._docx_seg_text(segs[0]) == "Hello world."


def test_ins_del_normal_are_separate_segments():
    p = _p(
        _r("kept ")
        + f'<w:del w:id="1" w:author="A"><w:r><w:delText>old</w:delText></w:r></w:del>'
        + f'<w:ins w:id="2" w:author="A">{_r("new")}</w:ins>'
    )
    segs = app._docx_collect_segments(p)
    assert [s["ctx"] for s in segs] == ["normal", "del", "ins"]
    assert [app._docx_seg_text(s) for s in segs] == ["kept ", "old", "new"]


def test_tab_breaks_segment():
    p = _p(_r("Item") + "<w:r><w:tab/></w:r>" + _r("Value"))
    segs = app._docx_collect_segments(p)
    assert [app._docx_seg_text(s) for s in segs] == ["Item", "Value"]


def test_nested_txbx_paragraph_collected_exactly_once():
    """文本框里的 w:p 嵌在外层 w:p 里；文本必须恰好收集一次——外层轮次跳过它，
    内层 w:p 自己的轮次收下它。收集两次意味着翻译两次、回填互相覆盖。"""
    inner = f'<w:p xmlns:w="{W}">{_r("inside")}</w:p>'
    p = _p(_r("outside") + f"<w:r><w:pict>{inner}</w:pict></w:r>")
    segs = app._docx_collect_segments(p)
    assert [app._docx_seg_text(s) for s in segs] == ["outside", "inside"]


# ---------------------------------------------------------------- 标记

def test_marked_text_wraps_bold():
    p = _p(_r("plain ") + _r("bold", "<w:b/>") + _r(" tail"))
    seg = app._docx_collect_segments(p)[0]
    assert app._docx_seg_marked(seg) == "plain **bold** tail"


def test_marked_text_bold_italic_underline():
    p = _p(_r("x", "<w:b/><w:i/>") + _r("y", '<w:u w:val="single"/>'))
    seg = app._docx_collect_segments(p)[0]
    assert app._docx_seg_marked(seg) == "***x***<u>y</u>"


def test_parse_marked_roundtrip():
    assert app._docx_parse_marked("plain **bold** tail") == [
        ("", "plain "), ("b", "bold"), ("", " tail")]
    assert app._docx_parse_marked("***x***<u>y</u>") == [("bi", "x"), ("u", "y")]


def test_parse_marked_unbalanced_returns_none():
    assert app._docx_parse_marked("oops **unclosed") is None


# ---------------------------------------------------------------- 回填

def test_apply_plain_translation():
    p = _p(_r("Hello ") + _r("world."))
    seg = app._docx_collect_segments(p)[0]
    app._docx_apply_translation(seg, "你好，世界。")
    assert _texts(p) == ["你好，世界。", ""]
    t0 = p.iter(f"{{{W}}}t").__next__()
    assert t0.get("{http://www.w3.org/XML/1998/namespace}space") == "preserve"


def test_apply_translation_keeps_format_span():
    p = _p(_r("the ") + _r("Contractor", "<w:b/>") + _r(" shall pay"))
    seg = app._docx_collect_segments(p)[0]
    app._docx_apply_translation(seg, "**承包商**应支付")
    assert _texts(p) == ["", "承包商", "应支付"]


def test_apply_translation_marker_lost_falls_back_to_first_run():
    """LLM 丢了标记时：整段填入第一个 run，其余清空——丢局部格式不丢文字。"""
    p = _p(_r("the ") + _r("Contractor", "<w:b/>") + _r(" shall pay"))
    seg = app._docx_collect_segments(p)[0]
    app._docx_apply_translation(seg, "承包商应支付")
    assert _texts(p) == ["承包商应支付", "", ""]


def test_apply_translation_del_text():
    p = _p('<w:del w:id="1" w:author="A"><w:r><w:delText>3.351.428,55</w:delText></w:r></w:del>')
    seg = app._docx_collect_segments(p)[0]
    assert seg["ctx"] == "del"
    app._docx_apply_translation(seg, "3.351.428,55（旧值）")
    assert _texts(p) == ["3.351.428,55（旧值）"]
    # 节点必须仍是 delText——修订语义不能变
    assert p.find(f".//{{{W}}}delText") is not None


def test_empty_translation_keeps_original():
    p = _p(_r("keep me"))
    seg = app._docx_collect_segments(p)[0]
    app._docx_apply_translation(seg, "")
    assert _texts(p) == ["keep me"]


# ---------------------------------------------------------------- 跳过规则

def test_short_and_symbol_segments_are_skipped():
    assert app._docx_should_translate("Pelo presente instrumento") is True
    assert app._docx_should_translate("(") is False          # 子词/符号碎片
    assert app._docx_should_translate("a") is False
    assert app._docx_should_translate("RG:") is False        # <=3 字符
    assert app._docx_should_translate("2026") is False       # 无字母
    assert app._docx_should_translate("   ") is False


# ---------------------------------------------------------------- 打包

def test_pack_batches_respects_limits():
    items = ["x" * 900] * 7
    batches = app._docx_pack_batches(items, max_chars=2000, max_items=3)
    assert all(len(b) <= 3 for b in batches)
    assert all(sum(len(items[i]) for i in b) <= 2000 or len(b) == 1 for b in batches)
    assert sorted(i for b in batches for i in b) == list(range(7))


# ---------------------------------------------------------------- 端到端（假翻译器）
# 测试自带一份合成 docx（含修订/批注/加粗/tab/页眉），不依赖任何真实文件。
# 想在真实文件上跑同一组断言：DOXIFY_DOCX_E2E=/path/to/file.docx pytest -k real_file

CT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/comments.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="comments.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

DOC = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W}"><w:body>
<w:p><w:r><w:t>O contrato entre as </w:t></w:r><w:r><w:rPr><w:b/></w:rPr><w:t>partes</w:t></w:r><w:r><w:t> estabelece o seguinte.</w:t></w:r></w:p>
<w:p><w:commentRangeStart w:id="0"/><w:r><w:t xml:space="preserve">O valor total dos servicos era </w:t></w:r>
<w:del w:id="11" w:author="Revisor A" w:date="2026-06-01T10:00:00Z"><w:r><w:delText>1.000,00</w:delText></w:r></w:del>
<w:ins w:id="12" w:author="Revisor A" w:date="2026-06-01T10:00:00Z"><w:r><w:t>2.000,00</w:t></w:r></w:ins>
<w:commentRangeEnd w:id="0"/><w:r><w:commentReference w:id="0"/></w:r></w:p>
<w:p><w:r><w:t>Item</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>Descricao completa do item</w:t></w:r></w:p>
</w:body></w:document>"""

COMMENTS = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:comments xmlns:w="{W}">
<w:comment w:id="0" w:author="Revisor A" w:date="2026-06-01T10:00:00Z" w:initials="RA">
<w:p><w:r><w:t>Este valor precisa de confirmacao formal.</w:t></w:r></w:p>
</w:comment></w:comments>"""

STYLES = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="{W}"/>'


def _build_sample_docx() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CT)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/_rels/document.xml.rels", DOC_RELS)
        z.writestr("word/document.xml", DOC)
        z.writestr("word/comments.xml", COMMENTS)
        z.writestr("word/styles.xml", STYLES)
    return buf.getvalue()


async def _fake_batch(texts):
    """伪翻译：给每项加【】，保留标记结构可验证。"""
    return [f"【{t}】" for t in texts]


@pytest.fixture(scope="module")
def translated_sample():
    data = _build_sample_docx()
    out, stats = asyncio.run(app.translate_docx_bytes(data, _fake_batch))
    return data, out, stats


def _part(data, name):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return z.read(name)


def test_e2e_output_is_valid_zip_with_same_parts(translated_sample):
    src, out, _ = translated_sample
    with zipfile.ZipFile(io.BytesIO(src)) as a, zipfile.ZipFile(io.BytesIO(out)) as b:
        assert [i.filename for i in a.infolist()] == [i.filename for i in b.infolist()]


def test_e2e_untouched_parts_byte_identical(translated_sample):
    src, out, _ = translated_sample
    for name in ("word/styles.xml", "_rels/.rels", "[Content_Types].xml"):
        assert _part(src, name) == _part(out, name), name


def test_e2e_revision_and_comment_structure_preserved(translated_sample):
    src, out, _ = translated_sample
    for name in ("word/document.xml", "word/comments.xml"):
        a, b = _part(src, name).decode("utf-8"), _part(out, name).decode("utf-8")
        for tag in ("w:ins ", "w:del ", "w:delText", "w:commentRangeStart",
                    "w:commentReference", "w:tab/"):
            assert len(re.findall(f"<{tag}", a)) == len(re.findall(f"<{tag}", b)), (name, tag)
        assert sorted(re.findall(r'w:author="[^"]+"', a)) == \
            sorted(re.findall(r'w:author="[^"]+"', b))


def test_e2e_all_streams_translated(translated_sample):
    _, out, stats = translated_sample
    doc = _part(out, "word/document.xml").decode("utf-8")
    assert "【O contrato entre as " in doc          # 正文
    assert "【2.000,00】" not in doc                 # 纯数字分段跳过翻译
    assert "<w:delText" in doc                      # 删除节点仍是删除节点
    com = _part(out, "word/comments.xml").decode("utf-8")
    assert "【" in com                               # 批注翻译了
    assert 'w:author="Revisor A"' in com            # 作者没动
    assert stats["translated"] >= 4


def test_e2e_bold_span_survives(translated_sample):
    _, out, _ = translated_sample
    doc = _part(out, "word/document.xml").decode("utf-8")
    # 加粗 run 仍然存在且非空（假翻译器保留了 ** 标记 → 回填到加粗 span）
    m = re.search(r"<w:rPr><w:b/></w:rPr><w:t[^>]*>([^<]*)</w:t>", doc)
    assert m and m.group(1), doc[:400]


def test_e2e_output_reparses_and_opens(translated_sample):
    _, out, _ = translated_sample
    for name in ("word/document.xml", "word/comments.xml"):
        etree.fromstring(_part(out, name))
    import docx as _docx
    d = _docx.Document(io.BytesIO(out))
    assert len(d.paragraphs) == 3


REAL = os.environ.get("DOXIFY_DOCX_E2E", "")


@pytest.mark.skipif(not (REAL and os.path.exists(REAL)), reason="设 DOXIFY_DOCX_E2E 指向真实 docx 后启用")
def test_e2e_real_file():
    import pathlib
    data = pathlib.Path(REAL).read_bytes()
    out, stats = asyncio.run(app.translate_docx_bytes(data, _fake_batch))
    with zipfile.ZipFile(io.BytesIO(data)) as a, zipfile.ZipFile(io.BytesIO(out)) as b:
        assert [i.filename for i in a.infolist()] == [i.filename for i in b.infolist()]
    assert stats["translated"] > 0


# ---------------------------------------------------------------- LLM 响应解析

def test_parse_llm_json_array_strict():
    assert app._docx_parse_llm_array('["甲", "乙"]', 2) == ["甲", "乙"]


def test_parse_llm_json_array_with_fences_and_prose():
    s = '好的，以下是翻译：\n```json\n["甲", "乙"]\n```'
    assert app._docx_parse_llm_array(s, 2) == ["甲", "乙"]


def test_parse_llm_json_array_with_unescaped_quotes_repaired():
    """真机实测（2026-08-24）：LLM 在 JSON 字符串里输出未转义引号，
    json.loads 抛 Expecting ',' delimiter，整批退化成逐条调用。json_repair 能救。"""
    s = '["合同"金额"为一亿", "乙方"]'
    out = app._docx_parse_llm_array(s, 2)
    assert isinstance(out, list) and len(out) == 2


def test_parse_llm_json_array_wrong_length_raises():
    import pytest as _pt
    with _pt.raises(ValueError):
        app._docx_parse_llm_array('["只有一条"]', 2)


def test_uniform_segment_keeps_literal_asterisks():
    """格式统一的分段没送过标记，译文里的 * 就是正文字符，不得被当标记剥掉。"""
    p = _p(_r("rated 5* hotel"))
    seg = app._docx_collect_segments(p)[0]
    app._docx_apply_translation(seg, "五星（5*）酒店")
    assert _texts(p) == ["五星（5*）酒店"]


# ---------------------------------------------------------------- 语言校验

def test_lang_ok_accepts_chinese():
    assert app._docx_lang_ok("本合同经双方协商一致签订", "中文") is True


def test_lang_ok_rejects_english_when_target_is_chinese():
    """真机实测（2026-08-24）：一整批被译成英文——JSON 形状合法，模型只是选错了
    目标语言。9/194 个长段落中招，全在文档开头。"""
    assert app._docx_lang_ok("CONTRACT FOR SERVICES WITH SUPPLY OF MATERIALS", "中文") is False


def test_lang_ok_accepts_pure_proper_noun():
    """纯专有名词按提示词就该保留原文，零 CJK 是合法的——短拉丁文本放行。"""
    assert app._docx_lang_ok("ANEXO II", "中文") is True


def test_lang_ok_mixed_chinese_with_names_passes():
    assert app._docx_lang_ok("由 EMPRESA EXEMPLO LTDA 承担全部责任", "中文") is True


def test_lang_ok_not_enforced_for_non_cjk_targets():
    assert app._docx_lang_ok("The contract shall be governed by...", "English") is True


class _FakeResp:
    """假响应。finish_reason 是必须的：调用点用它区分「模型说完了但没内容」和
    「思考吃光 max_tokens 被截断」——后者要翻倍额度重来一次（2026-09-05 GLM 事故）。"""
    def __init__(self, content, finish_reason="stop"):
        self._c = content
        self._fr = finish_reason
        self.status_code = 200
    def raise_for_status(self): pass
    def json(self):
        return {"choices": [{"finish_reason": self._fr,
                             "message": {"content": self._c}}]}


class _LangDriftClient:
    """第一次整批调用返回英文（跑偏），单条重译返回中文。"""
    def __init__(self):
        self.calls = 0
    async def post(self, url, json=None, headers=None, timeout=None):
        import json as _j
        self.calls += 1
        items = _j.loads(json["messages"][1]["content"])
        if len(items) > 1:
            return _FakeResp(_j.dumps(["English drift number one", "English drift number two"]))
        return _FakeResp(_j.dumps(["中文译文"], ensure_ascii=False))


def test_language_drift_triggers_per_item_retry():
    import asyncio as _a
    app._api_semaphore = _a.Semaphore(8)          # 未经 startup，兜住 async with None
    app._translate_semaphore = _a.Semaphore(3)
    client = _LangDriftClient()
    out = _a.run(app._translate_docx_batch_llm(
        client, ["texto português número um", "texto português número dois"], "中文"))
    assert out == ["中文译文", "中文译文"]


def test_uniform_segment_strips_model_added_markers():
    """真机实测：提示词教了标记语法，模型在没收到标记的分段里也会自作主张加
    **甲方**。格式统一的分段没送过标记，译文里成对的标记只能是模型加的，剥掉；
    孤立的 *（如 5*）不是成对标记，保留（见上一个测试）。"""
    p = _p(_r("CONTRATANTE poderá adquirir"))
    seg = app._docx_collect_segments(p)[0]
    app._docx_apply_translation(seg, "**甲方**可采购")
    assert _texts(p) == ["甲方可采购"]


class _StubbornEnglishClient:
    """整批和普通单条重译都返回英文；只有强化提示词的重译才给中文。

    同事实测（2026-08-27）：两段长英文在中文译稿里原样残留。假设的失败链：模型
    拒译/回显 → 语言护栏拦截 → 单条重译仍英文 → 静默保留原文。这个夹具复现该链，
    验证强化重译能救回来。
    """
    async def post(self, url, json=None, headers=None, timeout=None):
        import json as _j
        items = _j.loads(json["messages"][1]["content"])
        strict = "不得保留" in json["messages"][0]["content"]
        if strict:
            return _FakeResp(_j.dumps(["中文译文"] * len(items), ensure_ascii=False))
        return _FakeResp(_j.dumps(["The parties shall negotiate all remaining matters"] * len(items)))


def test_stubborn_english_rescued_by_strict_retry():
    import asyncio as _a
    app._api_semaphore = _a.Semaphore(8)
    app._translate_semaphore = _a.Semaphore(3)
    out = _a.run(app._translate_docx_batch_llm(
        _StubbornEnglishClient(), ["The supplier shall deliver the goods within thirty days"], "中文"))
    assert out == ["中文译文"]


class _HopelessEnglishClient:
    """无论怎么问都返回英文——最终必须保留原文且可被统计。"""
    async def post(self, url, json=None, headers=None, timeout=None):
        import json as _j
        items = _j.loads(json["messages"][1]["content"])
        return _FakeResp(_j.dumps(["Still English no matter what"] * len(items)))


def test_hopeless_english_keeps_original_and_is_counted():
    import asyncio as _a
    app._api_semaphore = _a.Semaphore(8)
    app._translate_semaphore = _a.Semaphore(3)
    src = "This clause shall survive termination of the agreement"
    out = _a.run(app._translate_docx_batch_llm(
        _HopelessEnglishClient(), [src], "中文"))
    assert out == [src], "彻底失败时必须保留原文，不能塞进英文『译文』"


def test_engine_stats_count_kept_original():
    """引擎统计保留原文的分段数——同事那次静默遗漏了两段，没人知道。"""
    import asyncio as _a
    p_xml = _build_sample_docx()

    async def _refuse(texts):
        return list(texts)          # 全部原样返回 = 全部未译

    _, stats = _a.run(app.translate_docx_bytes(p_xml, _refuse, target_lang="中文"))
    assert stats["kept_original"] >= 3
    # 假翻译器正常翻译时该计数应为 0
    _, stats2 = _a.run(app.translate_docx_bytes(p_xml, _fake_batch, target_lang="中文"))
    assert stats2["kept_original"] == 0


# ---------------------------------------------------------------- 字体补齐

def test_font_pass_adds_eastasia_to_bare_run():
    """英文文档的 run 通常没有任何 rFonts；译成中文后必须补东亚字体，
    否则 Word 走回退瞎猜（同事实测：宋体/等线混杂）。"""
    p = _p(_r("Contract terms"))
    seg = app._docx_collect_segments(p)[0]
    app._docx_apply_translation(seg, "合同条款")
    app._docx_apply_fonts(seg, cjk_font="宋体", latin_font="")
    r = p.find(f"{{{W}}}r")
    rpr = r.find(f"{{{W}}}rPr")
    assert rpr is not None and rpr is r[0], "rPr 必须是 run 的第一个子元素"
    fonts = rpr.find(f"{{{W}}}rFonts")
    assert fonts is not None and fonts.get(f"{{{W}}}eastAsia") == "宋体"


def test_font_pass_preserves_existing_latin_font():
    """拉丁轨原样不动：残留的数字、公司名仍用原字体，中西文各归其位。"""
    p = _p(_r("Amount due", '<w:rFonts w:ascii="Arial" w:hAnsi="Arial"/>'))
    seg = app._docx_collect_segments(p)[0]
    app._docx_apply_translation(seg, "应付金额 100")
    app._docx_apply_fonts(seg, cjk_font="宋体", latin_font="")
    fonts = p.find(f".//{{{W}}}rFonts")
    assert fonts.get(f"{{{W}}}ascii") == "Arial"
    assert fonts.get(f"{{{W}}}hAnsi") == "Arial"
    assert fonts.get(f"{{{W}}}eastAsia") == "宋体"


def test_font_pass_skips_runs_without_cjk():
    """译后不含中文的 run（纯数字/编号）不碰——没有需要管的字符。"""
    p = _p(_r("Reference 12345"))
    seg = app._docx_collect_segments(p)[0]
    app._docx_apply_translation(seg, "REF-12345")
    app._docx_apply_fonts(seg, cjk_font="宋体", latin_font="")
    assert p.find(f".//{{{W}}}rFonts") is None


def test_font_pass_disabled_when_empty():
    p = _p(_r("Contract terms"))
    seg = app._docx_collect_segments(p)[0]
    app._docx_apply_translation(seg, "合同条款")
    app._docx_apply_fonts(seg, cjk_font="", latin_font="")
    assert p.find(f".//{{{W}}}rFonts") is None


def test_font_pass_latin_mirror_for_cn2en():
    """中译英镜像：中文文档把 ascii 也设成宋体，英文用宋体拉丁字形很难看。
    换 ascii/hAnsi 为选定拉丁字体，eastAsia 不动（残留中文仍用原字体）。"""
    p = _p(_r("合同条款", '<w:rFonts w:ascii="宋体" w:eastAsia="宋体" w:hAnsi="宋体"/>'))
    seg = app._docx_collect_segments(p)[0]
    app._docx_apply_translation(seg, "Terms of the Contract")
    app._docx_apply_fonts(seg, cjk_font="", latin_font="Times New Roman")
    fonts = p.find(f".//{{{W}}}rFonts")
    assert fonts.get(f"{{{W}}}ascii") == "Times New Roman"
    assert fonts.get(f"{{{W}}}hAnsi") == "Times New Roman"
    assert fonts.get(f"{{{W}}}eastAsia") == "宋体", "东亚轨不动"


def test_font_pass_respects_rstyle_order():
    """rFonts 必须排在 rStyle 之后——OOXML 的 rPr 子元素有序。"""
    p = _p('<w:r><w:rPr><w:rStyle w:val="Emphasis"/></w:rPr><w:t>text</w:t></w:r>')
    seg = app._docx_collect_segments(p)[0]
    app._docx_apply_translation(seg, "文本")
    app._docx_apply_fonts(seg, cjk_font="宋体", latin_font="")
    rpr = p.find(f".//{{{W}}}rPr")
    assert rpr[0].tag == f"{{{W}}}rStyle"
    assert rpr[1].tag == f"{{{W}}}rFonts"


def test_e2e_font_pass_via_engine():
    """引擎级：cjk_font 传入后，译出的中文 run 都带 eastAsia；styles.xml 不动。"""
    import asyncio as _a
    data = _build_sample_docx()

    async def _zh(texts):
        return ["中文译文" + t[:0] for t in texts]

    out, _ = _a.run(app.translate_docx_bytes(data, _zh, target_lang="中文", cjk_font="宋体"))
    doc = _part(out, "word/document.xml").decode("utf-8")
    assert 'w:eastAsia="宋体"' in doc
    assert _part(data, "word/styles.xml") == _part(out, "word/styles.xml")


# ---------------------------------------------------------------- cs 轨（泰文等复杂文种）

def test_font_pass_thai_uses_cs_track():
    """泰文在 OOXML 里归「复杂文种」，字体走 rFonts 的 w:cs 属性——
    eastAsia/ascii 两轨都管不到它。"""
    p = _p(_r("Payment terms"))
    seg = app._docx_collect_segments(p)[0]
    app._docx_apply_translation(seg, "เงื่อนไขการชำระเงิน")
    app._docx_apply_fonts(seg, cjk_font="", latin_font="", cs_font="TH Sarabun New")
    fonts = p.find(f".//{{{W}}}rFonts")
    assert fonts is not None
    assert fonts.get(f"{{{W}}}cs") == "TH Sarabun New"
    assert fonts.get(f"{{{W}}}eastAsia") is None
    assert fonts.get(f"{{{W}}}ascii") is None


def test_font_pass_cs_not_set_without_thai_text():
    p = _p(_r("Plain latin"))
    seg = app._docx_collect_segments(p)[0]
    app._docx_apply_translation(seg, "Still latin text here")
    app._docx_apply_fonts(seg, cjk_font="", latin_font="", cs_font="TH Sarabun New")
    assert p.find(f".//{{{W}}}rFonts") is None


# ---------------------------------------------------------------- 语言→字体轨

def test_office_track_known_languages():
    assert app._office_font_track("中文") == "eastAsia"
    assert app._office_font_track("日本語") == "eastAsia"
    assert app._office_font_track("English") == "latin"
    assert app._office_font_track("葡萄牙语") == "latin"
    assert app._office_font_track("西班牙语") == "latin"
    assert app._office_font_track("越南语") == "latin"
    assert app._office_font_track("马来西亚语") == "latin"
    assert app._office_font_track("泰语") == "cs"


def test_office_track_explicit_override_wins():
    assert app._office_font_track("随便什么语", form_track="cs") == "cs"


def test_office_track_unknown_falls_back_latin():
    assert app._office_font_track("斯瓦希里语") == "latin"


def test_office_font_map_shape():
    """每个内置语言都有轨道和 1~3 个候选字体——前端菜单直接吃这份数据。"""
    for lang, (track, fonts) in app._OFFICE_LANG_FONTS.items():
        assert track in ("eastAsia", "latin", "cs"), lang
        assert 1 <= len(fonts) <= 3, lang


class _EmptyThenGoodClient:
    """第一次被思考吃光额度（空正文 + length），翻倍后才吐出 JSON 数组。"""
    def __init__(self):
        self.max_tokens_seen = []
    async def post(self, url, json=None, headers=None, timeout=None):
        import json as _j
        self.max_tokens_seen.append(json["max_tokens"])
        if len(self.max_tokens_seen) == 1:
            return _FakeResp("", finish_reason="length")
        items = _j.loads(json["messages"][1]["content"])
        return _FakeResp(_j.dumps(["中文译文"] * len(items), ensure_ascii=False))


def test_empty_content_with_length_doubles_max_tokens():
    import asyncio as _a
    app._api_semaphore = _a.Semaphore(8)
    app._translate_semaphore = _a.Semaphore(3)
    c = _EmptyThenGoodClient()
    out = _a.run(app._translate_docx_batch_llm(c, ["hello"], "中文"))
    assert out == ["中文译文"]
    assert c.max_tokens_seen[1] == c.max_tokens_seen[0] * 2


class _ThinkPollutedClient:
    """译文前挂了个 <think> 块——清洗后才是合法 JSON 数组。"""
    async def post(self, url, json=None, headers=None, timeout=None):
        import json as _j
        items = _j.loads(json["messages"][1]["content"])
        body = _j.dumps(["中文译文"] * len(items), ensure_ascii=False)
        return _FakeResp("<think>\n先看看有几项\n</think>\n" + body)


def test_think_block_is_stripped_before_json_parse():
    import asyncio as _a
    app._api_semaphore = _a.Semaphore(8)
    app._translate_semaphore = _a.Semaphore(3)
    out = _a.run(app._translate_docx_batch_llm(_ThinkPollutedClient(), ["hello"], "中文"))
    assert out == ["中文译文"]


def test_docx_payload_carries_profile():
    """载荷里只能有档案给的开关，不能再有手抄的 thinking:false。"""
    import asyncio as _a
    seen = {}

    class _C:
        async def post(self, url, json=None, headers=None, timeout=None):
            import json as _j
            seen.update(json)
            items = _j.loads(json["messages"][1]["content"])
            return _FakeResp(_j.dumps(["中文译文"] * len(items), ensure_ascii=False))

    app._api_semaphore = _a.Semaphore(8)
    app._translate_semaphore = _a.Semaphore(3)
    _a.run(app._translate_docx_batch_llm(_C(), ["hello"], "中文"))
    assert seen["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert "enable_thinking" not in seen and "thinking" not in seen


def test_shared_budget_retry_helper_doubles_once():
    """三处非流式调用点共用的 _llm_post_with_budget_retry：正文空 + length →
    翻倍一次，第二次的正文原样返回；翻倍只发生一次。"""
    import asyncio as _a
    seen = []

    class _C:
        async def post(self, url, json=None, headers=None, timeout=None):
            seen.append((json["max_tokens"], timeout))
            if len(seen) == 1:
                return _FakeResp("", finish_reason="length")
            return _FakeResp("<think>x</think>干净的正文")

    app._api_semaphore = _a.Semaphore(8)
    app._translate_semaphore = _a.Semaphore(3)
    body = {"max_tokens": 300}
    out = _a.run(app._llm_post_with_budget_retry(
        _C(), body, {}, timeout=60, what="[test] 探针"))
    assert out == "干净的正文"
    assert [m for m, _ in seen] == [300, 600], seen
    assert body["max_tokens"] == 600, "翻倍必须落在同一个 body 上"
    # F2：额度翻倍意味着要多写一倍的字，超时同步翻倍，否则原超时会把这次掐死
    assert [t for _, t in seen] == [60, 120], seen


def test_shared_budget_retry_helper_does_not_retry_on_stop():
    """finish_reason=stop 的空正文不是额度问题，别白打一次。"""
    import asyncio as _a
    calls = []

    class _C:
        async def post(self, url, json=None, headers=None, timeout=None):
            calls.append(1)
            return _FakeResp("", finish_reason="stop")

    app._api_semaphore = _a.Semaphore(8)
    out = _a.run(app._llm_post_with_budget_retry(
        _C(), {"max_tokens": 300}, {}, timeout=60, what="[test] 探针"))
    assert out == "" and len(calls) == 1


def test_shared_budget_retry_helper_honours_prefix_heuristic():
    """OCR 那类调用要能关掉前缀启发式（编号加粗标题是合法正文）。"""
    import asyncio as _a
    page = "1. **公司概况**\n本公司成立于 2001 年。"

    class _C:
        async def post(self, url, json=None, headers=None, timeout=None):
            return _FakeResp(page)

    app._api_semaphore = _a.Semaphore(8)
    assert _a.run(app._llm_post_with_budget_retry(
        _C(), {"max_tokens": 300}, {}, timeout=60, what="x",
        prefix_heuristic=False)) == page


def test_shared_helper_warns_when_cleaner_trims_content(caplog):
    """清洗器把正文剪短（剥掉推理前缀）时必须留一行 WARNING——此前只有剪成空才报，
    剥掉一段前缀再交出剩下半截是完全静默的。"""
    import asyncio as _a
    import logging

    records = []

    class _H(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                records.append(record.getMessage())

    class _C:
        async def post(self, url, json=None, headers=None, timeout=None):
            return _FakeResp("1.  **拆解用户请求：** 用户要一句话摘要。\n"
                             "2. **头脑风暴：** 需要简洁。\n商务部批准了延期申请。")

    app._api_semaphore = _a.Semaphore(8)
    h = _H()
    app.log.addHandler(h)
    try:
        out = _a.run(app._llm_post_with_budget_retry(
            _C(), {"max_tokens": 300}, {}, timeout=60, what="[test] 探针"))
    finally:
        app.log.removeHandler(h)

    assert out == "商务部批准了延期申请。"
    assert any("清洗器剥掉了" in m and "[test] 探针" in m for m in records), records


def test_shared_helper_silent_when_nothing_is_trimmed():
    """正文没被动过就别刷告警——噪音会让真正的泄漏被忽略。"""
    import asyncio as _a
    import logging

    records = []

    class _H(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                records.append(record.getMessage())

    class _C:
        async def post(self, url, json=None, headers=None, timeout=None):
            return _FakeResp("商务部批准了延期申请。")

    app._api_semaphore = _a.Semaphore(8)
    h = _H()
    app.log.addHandler(h)
    try:
        _a.run(app._llm_post_with_budget_retry(
            _C(), {"max_tokens": 300}, {}, timeout=60, what="[test] 探针"))
    finally:
        app.log.removeHandler(h)
    assert not records, records


class _HeadingResidualClient:
    """残留英文修正：原样回显（只把英文词换掉），首行标题必须完好回来。"""
    async def post(self, url, json=None, headers=None, timeout=None):
        fixed = json["messages"][1]["content"].replace("elegant", "优雅")
        return _FakeResp(fixed)


def test_residual_fix_keeps_a_leading_numbered_bold_heading():
    """整块译文以「1. **任务分工**」开头时，修正回合不得把首行标题吃掉——
    它会直接进最终译文与下载文件。"""
    import asyncio as _a
    src = "1. **任务分工**\n甲方负责设计，乙方负责 elegant 的施工方案。"
    app._api_semaphore = _a.Semaphore(8)
    app._translate_semaphore = _a.Semaphore(3)
    out = _a.run(app._fix_residual_english(
        _HeadingResidualClient(), src, ["elegant"], "中文"))
    assert out.startswith("1. **任务分工**"), out
    assert "优雅" in out


# ── F1：finish_reason=length 且正文非空的截断也要翻倍重试一次 ──────────

class _TruncatedThenCompleteClient:
    """第一次正文非空但被截断（length），翻倍后才写完整。

    终审指出：`reasoning_effort: low` 下更常见的不是「思考吃光额度、正文为空」，
    而是「JSON 写了一半被截断」——正文非空、解析炸掉，被当成格式错误，真正的病因
    （额度不够）反而看不见。所以翻倍的触发条件不看正文是否为空。
    """

    def __init__(self):
        self.calls = []

    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((json["max_tokens"], timeout))
        if len(self.calls) == 1:
            return _FakeResp('{"a": 1, "b": ', finish_reason="length")
        return _FakeResp('{"a": 1, "b": 2}')


def test_non_empty_truncation_also_triggers_the_doubled_retry():
    import asyncio as _a
    app._api_semaphore = _a.Semaphore(8)
    c = _TruncatedThenCompleteClient()
    out = _a.run(app._llm_post_with_budget_retry(
        c, {"max_tokens": 8000}, {}, timeout=300, what="[test] 截断"))
    assert out == '{"a": 1, "b": 2}', "应当返回重试那次的完整内容"
    assert c.calls == [(8000, 300), (16000, 600)], c.calls


def test_still_truncated_after_doubling_warns_and_returns_content():
    """翻倍后还截断就别再重试了——告警一句，把拿到的内容交出去，由调用方决定。"""
    import asyncio as _a
    import logging

    records = []

    class _H(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                records.append(record.getMessage())

    calls = []

    class _C:
        async def post(self, url, json=None, headers=None, timeout=None):
            calls.append((json["max_tokens"], timeout))
            return _FakeResp("写到一半就断了的正文", finish_reason="length")

    app._api_semaphore = _a.Semaphore(8)
    h = _H()
    app.log.addHandler(h)
    try:
        out = _a.run(app._llm_post_with_budget_retry(
            _C(), {"max_tokens": 300}, {}, timeout=60, what="[test] 截断"))
    finally:
        app.log.removeHandler(h)

    assert out == "写到一半就断了的正文"
    assert len(calls) == 2, f"只该重试一次: {calls}"
    assert any("翻倍后仍被截断" in m and "结果可能不完整" in m for m in records), records


def test_finish_reason_stop_never_retries():
    """正常收尾不许多打一次——这是最常见的路径，翻倍条件放宽后尤其要钉住。"""
    import asyncio as _a
    calls = []

    class _C:
        async def post(self, url, json=None, headers=None, timeout=None):
            calls.append(timeout)
            return _FakeResp("完整的正文。")

    app._api_semaphore = _a.Semaphore(8)
    out = _a.run(app._llm_post_with_budget_retry(
        _C(), {"max_tokens": 300}, {}, timeout=60, what="[test] 正常"))
    assert out == "完整的正文。" and calls == [60]


def test_no_max_tokens_means_no_doubling():
    """没设 max_tokens 就没有「额度不够」可言，别凭空往 body 里塞一个。"""
    import asyncio as _a
    calls = []

    class _C:
        async def post(self, url, json=None, headers=None, timeout=None):
            calls.append(1)
            return _FakeResp("内容", finish_reason="length")

    app._api_semaphore = _a.Semaphore(8)
    body = {}
    out = _a.run(app._llm_post_with_budget_retry(
        _C(), body, {}, timeout=60, what="[test] 无额度"))
    assert out == "内容" and len(calls) == 1 and "max_tokens" not in body


def test_doubled_timeout_leaves_non_numeric_timeouts_alone():
    import httpx as _h
    assert app._doubled_timeout(60) == 120
    assert app._doubled_timeout(1.5) == 3.0
    t = _h.Timeout(30.0, connect=5.0)
    assert app._doubled_timeout(t) is t
