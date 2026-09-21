"""页级缓存 / 断点续跑。

运行: cd /Users/xuzheng/CodingProjects/Doxify && conda run -n mineru python -m pytest tests/test_page_cache.py -v

背景：2026-08-06 一批 5931 页的作业跑到 48.7% 时因外出中断，结果只存在于浏览器内存里。
即便逐页落盘（见 test_streaming_runtime），重跑仍会把已识别的页再花钱识别一遍——
落盘解决的是「不丢」，续跑解决的是「不重做」，是两件事。

缓存键必须按 PDF 内容哈希，不能用 file_id：每次上传都是新 uuid，天然不可能命中。
"""
import asyncio

import pytest

import app


PDF_A = b"%PDF-1.4 fake content A"
PDF_B = b"%PDF-1.4 fake content B"


# ---------------------------------------------------------------- 缓存键


def test_cache_key_is_stable_for_same_input():
    assert app._page_cache_key(PDF_A, 200, "m1") == app._page_cache_key(PDF_A, 200, "m1")


def test_cache_key_varies_with_content_dpi_and_model():
    base = app._page_cache_key(PDF_A, 200, "m1")
    assert app._page_cache_key(PDF_B, 200, "m1") != base, "内容变了必须换键"
    assert app._page_cache_key(PDF_A, 300, "m1") != base, "DPI 变了渲染出的图不同，必须换键"
    assert app._page_cache_key(PDF_A, 200, "m2") != base, "换模型后旧结果不该复用"


# ---------------------------------------------------------------- 读写


def test_write_then_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    key = app._page_cache_key(PDF_A, 200, "m1")
    assert app._read_cached_page(key, 5) is None
    app._write_cached_page(key, 5, "第五页正文")
    assert app._read_cached_page(key, 5) == "第五页正文"


@pytest.mark.parametrize("bad", [
    "[第 7 页识别超时]",
    "[第 7 页识别失败：HTTP 502]",
    "[第 7 页识别异常：Server disconnected without sending a response.]",
    "[第 7 页处理失败：render exploded]",
    "",
    "   \n ",
])
def test_failures_are_never_cached(tmp_path, monkeypatch, bad):
    """缓存失败占位符会让重跑永远跳过那一页，坏页就此固化——必须拒绝。"""
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    key = app._page_cache_key(PDF_A, 200, "m1")
    app._write_cached_page(key, 7, bad)
    assert app._read_cached_page(key, 7) is None, f"不该缓存: {bad!r}"


def test_normal_text_mentioning_brackets_is_still_cached(tmp_path, monkeypatch):
    """别把正常正文误判成失败标记——正文里出现方括号是常事。"""
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    key = app._page_cache_key(PDF_A, 200, "m1")
    text = "见 [第 3 页识别结果] 的说明，另有 [?] 无法辨认处。"
    app._write_cached_page(key, 9, text)
    assert app._read_cached_page(key, 9) == text


# ---------------------------------------------------------------- 命中行为


def _fake_renderer(record):
    class R:
        async def render(self, page_num):
            record.append(page_num)
            return b"png"
    return R()


def test_cache_hit_skips_render_and_api(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    key = app._page_cache_key(PDF_A, 200, "m1")
    app._write_cached_page(key, 2, "缓存的第二页")

    called = []

    async def fake_vlm(client, img, page_num, total, file_id=""):
        called.append(page_num)
        return f"新识别 {page_num}"

    monkeypatch.setattr(app, "vlm_recognize_page", fake_vlm)
    rendered = []

    async def run():
        results, pq = {}, asyncio.Queue()
        await app._process_group(None, _fake_renderer(rendered), [1, 2, 3], 3,
                                 results, pq, {}, "fid-1", cache_key=key)
        return results

    results = asyncio.run(run())

    assert results[2] == "缓存的第二页"
    assert 2 not in called, "命中缓存不得再调 API"
    assert 2 not in rendered, "命中缓存不得再渲染该页"
    assert called == [1, 3] and rendered == [1, 3]


def test_cache_hit_does_not_reinject_figures(tmp_path, monkeypatch):
    """缓存里存的是已回填过图片引用的文本，再注入一次会把图片引用变成两份。"""
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    key = app._page_cache_key(PDF_A, 200, "m1")
    app._write_cached_page(key, 1, "正文\n\n![图](images/p001_1.png)")

    async def fake_vlm(client, img, page_num, total, file_id=""):
        raise AssertionError("不该被调用")

    monkeypatch.setattr(app, "vlm_recognize_page", fake_vlm)

    async def run():
        results, pq = {}, asyncio.Queue()
        await app._process_group(None, _fake_renderer([]), [1], 1, results, pq,
                                 {1: ["images/p001_1.png"]}, "fid-2", cache_key=key)
        return results

    results = asyncio.run(run())
    assert results[1].count("images/p001_1.png") == 1


def test_cache_miss_writes_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    key = app._page_cache_key(PDF_A, 200, "m1")

    async def fake_vlm(client, img, page_num, total, file_id=""):
        return f"识别结果 {page_num}"

    monkeypatch.setattr(app, "vlm_recognize_page", fake_vlm)

    async def run():
        results, pq = {}, asyncio.Queue()
        await app._process_group(None, _fake_renderer([]), [4], 4, results, pq,
                                 {}, "fid-3", cache_key=key)

    asyncio.run(run())
    assert app._read_cached_page(key, 4) == "识别结果 4"


def test_count_cached_pages_reports_upfront(tmp_path, monkeypatch):
    """必须能在开跑前就数出命中数。

    原先只在 asyncio.gather 之后统计，500 页的文件要全部跑完才打出「断点续跑」，
    用户在头一个小时里完全看不出续跑是否生效——和「日志只在出错时写」同一类错误。
    """
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    key = app._page_cache_key(PDF_A, 200, "m1")
    for n in (1, 2, 5):
        app._write_cached_page(key, n, f"P{n}")
    app._write_cached_page(key, 3, "[第 3 页识别超时]")  # 失败页不该被计入

    assert app._count_cached_pages(key, 10) == 3
    assert app._count_cached_pages("", 10) == 0
    assert app._count_cached_pages(key, 2) == 2, "超出总页数的缓存页不该计入"


def test_no_cache_key_disables_caching(tmp_path, monkeypatch):
    """cache_key 为空时行为退回原样，不建任何缓存目录。"""
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)

    async def fake_vlm(client, img, page_num, total, file_id=""):
        return "x"

    monkeypatch.setattr(app, "vlm_recognize_page", fake_vlm)

    async def run():
        results, pq = {}, asyncio.Queue()
        await app._process_group(None, _fake_renderer([]), [1], 1, results, pq,
                                 {}, "fid-4", cache_key="")

    asyncio.run(run())
    assert not (tmp_path / "_page_cache").exists()
