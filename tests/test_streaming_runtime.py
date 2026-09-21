"""VLM 解析路径的运行时行为：惰性渲染、每页落盘、进度可观测性。

运行: cd /Users/xuzheng/CodingProjects/Doxify && conda run -n mineru python -m pytest tests/test_streaming_runtime.py -v

背景（2026-08-06 现场诊断）：
- 12 个 500 页 PDF 并行解析时进程 RSS 到 6.8 GB —— pdf_to_images() 把整本书的
  PNG 全量常驻内存；且它是同步函数被直接调用在协程里，渲染 12 个文件的 31 分钟内
  事件循环完全冻结，一次 VLM 调用都发不出去。
- 日志只在出错时才写，成功的页一行不打，跑了 1.7 小时的作业从日志看像是死了。
- VLM 模式仅当抽到图表时才落盘 .md，其余全靠 SSE 推给前端，断线即全部作废。
"""
import asyncio
import re
import time

import fitz
import pytest

import app


def _make_pdf(n_pages: int = 3) -> bytes:
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"page {i + 1}")
    data = doc.tobytes()
    doc.close()
    return data


# ---------------------------------------------------------------- 惰性渲染


def test_renderer_reports_page_count_without_rendering():
    r = app._PageRenderer(_make_pdf(4), dpi=72)
    try:
        assert r.page_count == 4
    finally:
        r.close()


def test_renderer_output_matches_eager_pdf_to_images():
    """惰性逐页渲染必须与原来的全量渲染字节级一致，否则是行为回归。"""
    data = _make_pdf(3)
    eager = app.pdf_to_images(data, dpi=100)

    r = app._PageRenderer(data, dpi=100)
    try:
        lazy = [r.render_sync(i) for i in range(1, 4)]
    finally:
        r.close()

    assert lazy == eager


def test_renderer_render_is_awaitable_and_returns_png():
    data = _make_pdf(2)
    r = app._PageRenderer(data, dpi=100)
    try:
        png = asyncio.run(r.render(2))
    finally:
        r.close()
    assert png.startswith(b"\x89PNG")


def test_renderer_render_does_not_block_the_event_loop():
    """render() 必须把同步渲染丢进 executor：渲染期间事件循环要能继续调度别的协程。

    用一个确定性的阻塞 stub 代替真渲染——真实空白页只要不到 1ms，快到测不出差别，
    而现场真正的病灶恰恰是「同步渲染长时间霸占事件循环」这一条。
    """
    r = app._PageRenderer(_make_pdf(1), dpi=72)
    r.render_sync = lambda page_num: (time.sleep(0.2), b"\x89PNG-stub")[1]

    async def run():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.005)
                ticks += 1

        t = asyncio.create_task(ticker())
        png = await r.render(1)
        t.cancel()
        return png, ticks

    try:
        png, ticks = asyncio.run(run())
    finally:
        r.close()

    assert png == b"\x89PNG-stub"
    assert ticks >= 5, f"渲染的 0.2s 里事件循环只跑了 {ticks} 次，说明没有丢进 executor"


def test_renderer_raises_on_corrupt_pdf():
    """损坏 PDF 必须在构造时抛异常，交由 _run_parse_task 兜成 file_error。"""
    with pytest.raises(Exception):
        app._PageRenderer(b"not a pdf at all", dpi=72)


# ---------------------------------------------------------------- 每页落盘


def test_persist_page_writes_recoverable_file(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    app._persist_page_sync("fid-x", 7, "第七页内容")

    p = tmp_path / "fid-x" / "_pages" / "p0007.md"
    assert p.read_text(encoding="utf-8") == "第七页内容"


def test_persist_page_sorts_lexicographically_by_page(tmp_path, monkeypatch):
    """页文件名必须零填充，否则 p10 会排在 p2 前面，手工 cat 恢复时顺序错乱。"""
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    for n in (1, 2, 10, 100):
        app._persist_page_sync("fid-y", n, f"P{n}")

    names = sorted(q.name for q in (tmp_path / "fid-y" / "_pages").iterdir())
    assert names == ["p0001.md", "p0002.md", "p0010.md", "p0100.md"]


def test_persist_page_failure_does_not_raise(tmp_path, monkeypatch):
    """写盘失败只应告警，不能让整个文件的解析任务崩掉。"""
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path / "nonexistent" / "\x00bad")
    app._persist_page_sync("fid-z", 1, "x")  # 不抛即通过


# ---------------------------------------------------------------- 可观测性


def test_fmt_eta_renders_hours_and_minutes():
    assert app._fmt_eta(0) == "0m00s"
    assert app._fmt_eta(95) == "1m35s"
    assert app._fmt_eta(3600 + 120) == "1h02m"
    assert app._fmt_eta(float("inf")) == "--"
    assert app._fmt_eta(-1) == "--"


def test_eta_uses_recent_rate_not_lifetime_average():
    """ETA 必须基于近期速率。

    断点续跑时缓存会在一个周期内回放上千页，累计平均速率被拉高近一个数量级，
    用它算出的 ETA 会把 3.4 小时报成 25 分钟（2026-08-06 实测）。
    """
    now = time.monotonic()
    # 500 页的文件，其中 240 页已在缓存里
    jobs = {"a": {"filename": "f.pdf", "total": 500, "cached": 240,
                  "done": 240, "cached_done": 240,
                  "started": now - 120, "last_done": 0}}

    # 第一周期：缓存在一个周期内全部回放完
    app._heartbeat_lines(jobs, now)

    # 第二周期：进入真实识别，本周期完成 10 页
    jobs["a"]["done"] = 250
    line = app._heartbeat_lines(jobs, now + app.HEARTBEAT_SEC)[0]

    assert "+10" in line and "缓存240" in line, line
    m = re.search(r"剩余~(?:(\d+)h)?(\d+)m", line)
    assert m, line
    minutes = int(m.group(1) or 0) * 60 + int(m.group(2))
    # 剩余 250 页真识别、每周期 10 页 → 应在数十分钟量级，不能被缓存爆发拉成几分钟
    assert minutes > 15, f"ETA {minutes}m 明显偏低，说明缓存回放仍被计入速率: {line}"


def test_heartbeat_lines_report_progress_and_stall():
    """心跳必须能区分「在动」和「卡住」：卡住的文件本周期增量为 0。"""
    now = time.monotonic()
    jobs = {
        "aaaaaaaa": {"filename": "moving.pdf", "total": 100, "done": 40,
                     "started": now - 600, "last_done": 30},
        "bbbbbbbb": {"filename": "stuck.pdf", "total": 100, "done": 5,
                     "started": now - 600, "last_done": 5},
    }
    lines = app._heartbeat_lines(jobs, now)
    joined = " | ".join(lines)

    assert "moving.pdf" in joined and "40/100" in joined
    assert "+10" in joined, "应报告本周期新增页数"
    assert "stuck.pdf" in joined and "+0" in joined

    # 快照后 last_done 必须推进，否则下个周期的增量是累计值而非增量
    assert jobs["aaaaaaaa"]["last_done"] == 40
    assert jobs["bbbbbbbb"]["last_done"] == 5


def test_heartbeat_lines_empty_when_no_jobs():
    assert app._heartbeat_lines({}, time.monotonic()) == []


def test_pages_dir_cleared_only_after_final_md_lands(tmp_path, monkeypatch):
    """最终 .md 落盘后中间页就是冗余，可清；但写失败时必须原样保留——那正是要恢复的场景。"""
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    for n in (1, 2):
        app._persist_page_sync("fid-c", n, f"P{n}")
    pages = tmp_path / "fid-c" / "_pages"
    assert pages.is_dir()

    app._clear_persisted_pages("fid-c", final_md_written=False)
    assert pages.is_dir(), "最终产物没写成时不得删除中间页"

    app._clear_persisted_pages("fid-c", final_md_written=True)
    assert not pages.exists()


# ---------------------------------------------------------------- worker 容错


def test_process_group_survives_a_failing_page(tmp_path, monkeypatch):
    """单页失败不得中断整组。

    收集循环是 `while completed < total: await progress_queue.get()`——worker 一旦
    抛异常就再也不会投递剩余页的事件，该文件的解析会永久挂起（连 file_error 都发不出）。
    因此每页必须自带兜底：失败页回退成占位文本，照常计数。
    """
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)

    class BoomOnPage2:
        def __init__(self):
            self.rendered = []

        async def render(self, page_num):
            self.rendered.append(page_num)
            if page_num == 2:
                raise RuntimeError("render exploded")
            return b"png"

    async def fake_vlm(client, img, page_num, total, file_id=""):
        assert file_id == "fid-boom", "必须把 file_id 透传下去，否则告警无法归属到文件"
        return f"text {page_num}"

    monkeypatch.setattr(app, "vlm_recognize_page", fake_vlm)

    async def run():
        results, pq = {}, asyncio.Queue()
        r = BoomOnPage2()
        await app._process_group(None, r, [1, 2, 3], 3, results, pq, {}, "fid-boom")
        evts = []
        while not pq.empty():
            evts.append(pq.get_nowait())
        return results, evts, r.rendered

    results, evts, rendered = asyncio.run(run())

    assert rendered == [1, 2, 3], "第 2 页失败后必须继续渲染第 3 页"
    assert len(evts) == 3, f"三页都必须投递事件，否则收集循环挂死；实得 {len(evts)}"
    assert set(results) == {1, 2, 3}, "results 必须齐全，否则最终合并时 KeyError"
    assert "render exploded" in results[2] or "失败" in results[2]
    assert results[1] == "text 1" and results[3] == "text 3"


def test_cancelling_parse_stops_vlm_workers(tmp_path, monkeypatch):
    """取消解析作业必须真正停掉 worker，不能留下孤儿继续跑。

    worker 是独立 task，parse_pdf_streaming 被取消时它们不会跟着死：在途的 VLM
    调用会跑完（继续烧配额），随后剩余页面对着已关闭的渲染 executor 逐页快速失败，
    把「[第 N 页处理失败：…]」写进 output/<file_id>/_pages/。这违背了取消按钮
    「能真正停止作业」的验收标准。
    """
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(app, "_extract_and_save_figures_sync", lambda *a, **k: {})
    monkeypatch.setattr(app, "_page_cache_key", lambda *a, **k: "")
    monkeypatch.setattr(app, "CONCURRENCY_THRESHOLD", 1)   # 强制并发模式，多个 worker

    in_flight = []

    async def slow_vlm(client, img, page_num, total, file_id):
        in_flight.append(page_num)
        await asyncio.sleep(30)          # 永远不返回：模拟在途的 VLM 调用
        return f"page {page_num}"

    monkeypatch.setattr(app, "vlm_recognize_page", slow_vlm)

    async def run():
        app._api_semaphore = asyncio.Semaphore(8)
        q: asyncio.Queue = asyncio.Queue()
        t = asyncio.create_task(
            app.parse_pdf_streaming(_make_pdf(6), "a.pdf", "fid-cancel", q))
        await asyncio.sleep(0.5)         # 让 worker 真正进到 VLM 调用里
        assert in_flight, "前提不成立：worker 还没开始调用 VLM"
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
        await asyncio.sleep(0.3)         # 留出孤儿 worker 继续写占位符的时间
        alive = [x for x in asyncio.all_tasks()
                 if "_process_group" in str(x.get_coro()) and not x.done()]
        pages = sorted((tmp_path / "fid-cancel" / "_pages").glob("*.md")) \
            if (tmp_path / "fid-cancel" / "_pages").exists() else []
        return alive, [p.read_text(encoding="utf-8") for p in pages]

    alive, page_texts = asyncio.run(run())
    assert not alive, f"取消后仍有 {len(alive)} 个 worker 存活"
    assert not any("处理失败" in t for t in page_texts), \
        f"孤儿 worker 写入了失败占位符: {page_texts}"
