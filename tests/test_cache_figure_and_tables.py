"""两项修复：页缓存与图片回填解耦、跨页表格接合。

运行: conda run -n mineru python -m pytest tests/test_cache_figure_and_tables.py -v

背景（2026-08-20）：
1. 页缓存存的是「已回填图片引用」的文本，命中缓存会跳过 _inject_figures。图表提取
   逻辑一改（例如不再把表格当图），缓存里的旧引用照样复现——而那些图片已不再生成，
   变成坏链。实测重跑一份已缓存文档：提取端只出 3 张图，markdown 里仍是 42 个引用。
   修法：缓存存**未回填**的原文（含 [[FIGURE]] 占位符），每次运行都重新回填；读取时
   把历史缓存里的图片引用还原成占位符，实现无痛迁移，不必重调 VLM。
2. 页与页之间用空行拼接，而空行会截断 Markdown 表格，于是跨页表格断成两截。
   _merge_broken_paragraphs 明确规避表格（怕误伤），所以一直没人管。
"""
import asyncio

import app


# ---------------------------------------------------------------- 缓存 ↔ 回填


def test_cache_stores_uninjected_text(tmp_path, monkeypatch):
    """写入缓存的必须是带占位符的原文，而不是回填后的文本。"""
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    key = app._page_cache_key(b"pdf", 200, "m")

    async def fake_vlm(client, img, page_num, total, file_id=""):
        return "正文\n\n[[FIGURE]]\n\n更多正文"

    monkeypatch.setattr(app, "vlm_recognize_page", fake_vlm)

    class R:
        async def render(self, n):
            return b"png"

    async def run():
        results, pq = {}, asyncio.Queue()
        await app._process_group(None, R(), [1], 1, results, pq,
                                 {1: ["images/p001_1.png"]}, "fid", cache_key=key)
        return results[1]

    text = asyncio.run(run())
    assert "![图](images/p001_1.png)" in text, "返回给调用方的应是已回填的文本"
    cached = app._read_cached_page(key, 1)
    assert "[[FIGURE]]" in cached, f"缓存里应保留占位符，实得: {cached!r}"
    assert "images/p001_1.png" not in cached, "缓存不得包含已回填的图片引用"


def test_cache_hit_reinjects_with_current_figures(tmp_path, monkeypatch):
    """命中缓存时按**当前**的图片列表回填——图表逻辑变了要立刻生效。"""
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    key = app._page_cache_key(b"pdf", 200, "m")
    app._write_cached_page(key, 1, "正文\n\n[[FIGURE]]\n\n更多正文")

    async def boom(*a, **k):
        raise AssertionError("命中缓存不该调用 VLM")

    monkeypatch.setattr(app, "vlm_recognize_page", boom)

    class R:
        async def render(self, n):
            raise AssertionError("命中缓存不该渲染")

    async def run(refs):
        results, pq = {}, asyncio.Queue()
        await app._process_group(None, R(), [1], 1, results, pq, refs, "fid", cache_key=key)
        return results[1]

    # 这一版仍认为该页有图
    assert "images/new.png" in asyncio.run(run({1: ["images/new.png"]}))
    # 下一版判定那是表格、不再出图 → 占位符应被删掉，不留残迹
    out = asyncio.run(run({}))
    assert "images/" not in out and "[[FIGURE]]" not in out, out


def test_legacy_cache_with_injected_refs_is_migrated(tmp_path, monkeypatch):
    """历史缓存里存的是已回填的文本，读取时要还原成占位符，否则旧引用会复活。"""
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    key = app._page_cache_key(b"pdf", 200, "m")
    d = app._page_cache_dir(key)
    d.mkdir(parents=True, exist_ok=True)
    (d / "p0001.md").write_text("正文\n\n![图](images/p001_1.png)\n\n更多正文", encoding="utf-8")

    restored = app._read_cached_page(key, 1)
    assert "[[FIGURE]]" in restored, f"历史引用应还原为占位符: {restored!r}"
    assert "images/p001_1.png" not in restored


# ---------------------------------------------------------------- 跨页表格


def test_table_split_across_pages_is_rejoined():
    md = (
        "| Exporter | Margin |\n|:---|---:|\n| Alpha | 16.25% |\n\n"
        " | Bravo | 64.25% |\n| Others | 71.74% |\n"
    )
    out = app._merge_broken_paragraphs(md)
    assert "| Alpha | 16.25% |\n| Bravo | 64.25% |" in out, out
    assert "\n\n | Bravo" not in out, "行首多余空格也要一并去掉"


def test_new_table_after_a_table_is_not_merged():
    """下一块自带表头分隔行 = 另起一张表，不能接上去。"""
    md = (
        "| A | B |\n|:---|---:|\n| 1 | 2 |\n\n"
        "| C | D |\n|:---|---:|\n| 3 | 4 |\n"
    )
    out = app._merge_broken_paragraphs(md)
    assert "| 1 | 2 |\n\n| C | D |" in out, out


def test_paragraph_after_table_is_not_merged():
    md = "| A | B |\n|:---|---:|\n| 1 | 2 |\n\n后续正文段落。\n"
    out = app._merge_broken_paragraphs(md)
    assert "| 1 | 2 |\n\n后续正文段落。" in out, out


def test_table_after_paragraph_is_not_merged():
    md = "一段正文。\n\n| A | B |\n|:---|---:|\n| 1 | 2 |\n"
    out = app._merge_broken_paragraphs(md)
    assert "一段正文。\n\n| A | B |" in out, out


# ------------------------------------------- 续表带重复分隔行（VLM 在新页重起语法）


def test_continuation_with_empty_header_and_separator_is_merged():
    """VLM 常在新页顶部重起表格语法，带一个空表头 + 分隔行。列数一致即视为续表。"""
    md = (
        "| A | B | C |\n|:---|:---|:---|\n| 1 | 2 | 3 |\n\n"
        " | | | |\n|:---|:---|:---|\n| 4 | 5 | 6 |\n"
    )
    out = app._merge_broken_paragraphs(md)
    assert "| 1 | 2 | 3 |\n| 4 | 5 | 6 |" in out, out
    assert out.count(":---") == 3, "重复的分隔行与空表头都该去掉，只保留原表的"


def test_continuation_row_carrying_split_cell_text_is_merged():
    """跨页把单元格文字切断的情形：续块首行是残句而非表头。"""
    md = (
        "| Company | Role | Doc |\n|:---|:---|:---|\n| Beta Equipment (East) | producer | form |\n\n"
        " | Manufacturing (West) Co Ltd | | |\n|:---|:---|:---|\n| Gamma UK | importer | form |\n"
    )
    out = app._merge_broken_paragraphs(md)
    assert "| Manufacturing (West) Co Ltd | | |" in out
    assert "\n\n | Manufacturing" not in out, out
    assert out.count(":---") == 3


def test_adjacent_table_with_different_column_count_stays_separate():
    """列数不同 = 确实是另一张表，不能合并。"""
    md = (
        "| A | B |\n|:---|:---|\n| 1 | 2 |\n\n"
        "| C | D | E |\n|:---|:---|:---|\n| 3 | 4 | 5 |\n"
    )
    out = app._merge_broken_paragraphs(md)
    assert "| 1 | 2 |\n\n| C | D | E |" in out, out
