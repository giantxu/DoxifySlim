"""
Doxify — PDF → Markdown 解析 + Markdown 翻译工具

功能：
1. PDF 解析：远程多模态 LLM（当前为 GLM-5.3-Flash）逐页识别（DoxifySlim 仅保留此路径），
   带逐页进度，输出 Markdown + 图片 ZIP 打包下载
2. Markdown 翻译：分块并行流式翻译，含残留英文自动检测与修正

启动方式：python app.py
"""

import os
import base64
import asyncio
import contextlib
import hashlib
import io
import logging
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from pathlib import Path
from urllib.parse import quote

import fitz  # PyMuPDF
import httpx
import uvicorn
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import (StreamingResponse, JSONResponse, HTMLResponse,
                               FileResponse)
from dotenv import load_dotenv

from llm_common import (PROFILES, extract_content, finish_reason,
                        llm_extra_body, llm_headers, llm_profile_name,
                        strip_thinking)

load_dotenv()

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
TARGET_API_URL = os.getenv("TARGET_API_URL", "")
# 没有 TARGET_API_KEY 常量：出站请求头一律由 llm_common.llm_headers() 组装，
# 它自己从环境读 key。留一份模块级副本只会诱使人再手抄 Bearer 头。
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "4000"))
ACTUAL_MODEL_NAME = os.getenv("ACTUAL_MODEL_NAME", "kimi26")
# 模型档案：决定发给网关的思考开关（见 llm_common.PROFILES）。切模型改 .env 一行。
LLM_PROFILE = llm_profile_name()
try:
    _ = llm_extra_body()          # 启动即校验 LLM_PROFILE，配错立刻报错
except KeyError as _e:
    raise SystemExit(
        f"Doxify 启动失败：.env 里的 LLM_PROFILE={LLM_PROFILE!r} 不是已知档案。"
        f"可选：{sorted(PROFILES)}。配错档案会让网关把思考写进正文，"
        "所以这里直接拒绝启动，而不是带病运行。"
    ) from _e
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))
PAGE_TIMEOUT = int(os.getenv("PAGE_TIMEOUT", "120"))
PDF_DPI = int(os.getenv("PDF_DPI", "200"))
# 单文件内并发 Worker 数
CONCURRENCY = int(os.getenv("CONCURRENCY", "3"))
# 启用并发的最小页数
CONCURRENCY_THRESHOLD = int(os.getenv("CONCURRENCY_THRESHOLD", "10"))
# 全局最大同时发出的 API 请求数（多文件×多Worker）
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))  # 事务所网关上限 3
# 翻译：同时在途的流式请求上限。与 MAX_CONCURRENT_REQUESTS 是两层约束——后者管
# 所有出站请求的发起，这一层专管翻译，且覆盖整个流的生命周期（见 translate_chunk_stream）。
TRANSLATE_CONCURRENCY = int(os.getenv("TRANSLATE_CONCURRENCY", "3"))
# 翻译：每块最大字符数
TRANSLATE_CHUNK_CHARS = int(os.getenv("TRANSLATE_CHUNK_CHARS", "3000"))
# 翻译：默认目标语言
TRANSLATE_TARGET_LANG = os.getenv("TRANSLATE_TARGET_LANG", "中文")
# 翻译：首段缓冲期间显示的占位文本（见 _process_chunk）
_CHUNK_PLACEHOLDER = "翻译进行中，请稍后……"
# 翻译：单块流式输出的 max_tokens。整块译文为空时用它的两倍重发一次。
TRANSLATE_MAX_TOKENS = int(os.getenv("TRANSLATE_MAX_TOKENS", "6000"))
# 翻译：首段缓冲长度。无标签推理没有标签可认，只能先攒一段再整体判定（见
# translate_chunk_stream）。太小判不准，太大首字延迟明显；240 字符 ≈ 一两句话。
STREAM_HEAD_CHARS = int(os.getenv("STREAM_HEAD_CHARS", "240"))
# 是否丢弃「只是表格框线」的矢量簇（VLM 会把表格转成 Markdown，无需再留图）。
# 设为 0 可退回旧行为——某类文档若出现真图表被误删，这是应急开关。
FIGURE_DROP_TABLES = os.getenv("FIGURE_DROP_TABLES", "1") == "1"
# 心跳日志间隔（秒）：无论是否有页完成都按期打一行，长作业不再看起来像死了。<=0 关闭。
HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "60"))
# 已结束作业的保留上限。解析作业的事件日志量级参考：5929 页 × 约 1.5 KB ≈ 9 MB。
# 翻译作业完全不是这个量级——chunk_token 是**每个 token 一条事件**，同样字数下
# 条数高出两三个数量级，长文档单作业事件日志可达数十 MB。
JOB_HISTORY_MAX = int(os.getenv("JOB_HISTORY_MAX", "20"))
# 单个订阅者队列上限。取值需明显大于一次重放的规模：补发 backlog 期间产生的实时
# 事件会堆在队列里，溢出就会被 _publish 踢掉、要重连补一次大 backlog。溢出本身
# 不再是死局——生成器发现自己被踢会主动收流（见 job_events._gen），客户端按既有
# 路径带 cursor 重连补齐；调大只是为了少走这条弯路。
SUBSCRIBER_QUEUE_MAX = int(os.getenv("SUBSCRIBER_QUEUE_MAX", "10000"))




# 解析输出根目录：每个 file_id 一个子目录，存 .md + images/，供 ZIP 下载
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# DOXIFY_LOG_FILE 可覆写日志路径。测试必须设它（见 tests/conftest.py）：否则
# `import app` 就会往生产 gateway.log 里追加测试造的假 file_id 和栈回溯，把正在跑的
# 长作业的日志搅浑。
LOG_FILE = os.getenv("DOXIFY_LOG_FILE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "gateway.log")

# 直接在 gateway logger 上挂 handlers，不依赖根 logger。
# PaddlePaddle 初始化时会覆盖根 logger 的 handlers，导致 basicConfig 方式失效。
_log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gateway")
log.setLevel(logging.INFO)
log.propagate = False  # 隔离根 logger，防止 PaddlePaddle 干扰
if not log.handlers:
    _sh = logging.StreamHandler()
    _sh.setFormatter(_log_fmt)
    log.addHandler(_sh)
    _fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _fh.setFormatter(_log_fmt)
    log.addHandler(_fh)

app = FastAPI(title="Doxify", version="1.0.0")

# 静态资产（JT&N 品牌 logo 等），供 WebUI 引用：/static/jtn-logo.png 等
from fastapi.staticfiles import StaticFiles
_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# 全局信号量：限制同时发出的 VLM API 请求数
_api_semaphore: asyncio.Semaphore | None = None
# 翻译专用信号量，见 TRANSLATE_CONCURRENCY
_translate_semaphore: asyncio.Semaphore | None = None

# 全局 HTTP 客户端：过去每个文件各建一个 AsyncClient，12 个文件就是 12 个独立连接池，
# 各自按 httpx 默认值保留至多 20 条 keepalive 连接。网关侧主动断开空闲连接后这些
# fd 会滞留在 CLOSE_WAIT（现场实测常驻 5-9 个）。共用一个池并按信号量收紧上限，
# 连接数与实际并发对齐，空闲连接 30s 内回收。
_http_client: httpx.AsyncClient | None = None

# 活跃解析作业表：file_id -> {filename, total, done, started, last_done}
# 供心跳日志用，不参与业务逻辑。
_active_jobs: dict[str, dict] = {}
_heartbeat_task: asyncio.Task | None = None


def _new_http_client() -> httpx.AsyncClient:
    # trust_env=False：不读系统代理。macOS 下 httpx 会拿到系统代理却不认其
    # 例外列表（如内网网段），导致内网 LLM 网关请求被代理吞掉返回 502。
    return httpx.AsyncClient(
        trust_env=False,
        limits=httpx.Limits(
            max_connections=MAX_CONCURRENT_REQUESTS + 2,
            max_keepalive_connections=MAX_CONCURRENT_REQUESTS,
            keepalive_expiry=30.0,
        ),
        timeout=httpx.Timeout(PAGE_TIMEOUT, connect=10.0),
    )


def _get_http_client() -> httpx.AsyncClient:
    """返回进程级共享客户端；未经 startup（如单测直接调用）时惰性建一个。"""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = _new_http_client()
    return _http_client


def _fmt_eta(seconds: float) -> str:
    """把剩余秒数格式化成 1h02m / 3m20s；无法估算时返回 --。"""
    if seconds != seconds or seconds in (float("inf"), float("-inf")) or seconds < 0:
        return "--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _heartbeat_lines(jobs: dict[str, dict], now: float) -> list[str]:
    """为每个活跃作业生成一行进度摘要，并推进其 last_done 快照。

    带上「本周期新增页数」是关键：卡住的文件会显示 +0，一眼能和「在动但慢」区分开。
    有副作用（更新 last_done），因为增量只有在打印那一刻才有意义。
    """
    lines = []
    for fid, j in jobs.items():
        done, total = j["done"], j["total"]
        delta = done - j.get("last_done", 0)
        j["last_done"] = done

        # 速率与 ETA 只按「需要真识别的页」算。断点续跑时缓存会在一个周期内回放上千页，
        # 把它计入速率会让累计平均和 EMA 都被拉高近一个数量级——2026-08-06 实测把
        # 3.4 小时报成了 25 分钟。缓存页是已完成的工作，剩余的全是真识别，两者不可混算。
        cached_done = j.get("cached_done", 0)
        real_done = done - cached_done
        real_total = max(total - j.get("cached", 0), 0)
        delta_real = real_done - j.get("last_real_done", 0)
        j["last_real_done"] = real_done

        recent = delta_real / max(HEARTBEAT_SEC, 1)  # 页/秒
        prev = j.get("ema_rate")
        ema = recent if prev is None else 0.5 * recent + 0.5 * prev
        j["ema_rate"] = ema

        eta = (real_total - real_done) / ema if ema > 1e-9 else float("inf")
        pct = (100.0 * done / total) if total else 0.0
        cache_note = f" 缓存{cached_done}" if cached_done else ""
        lines.append(
            f"{j['filename']} {done}/{total} ({pct:.1f}%) "
            f"+{delta}/{HEARTBEAT_SEC}s {ema * 60:.1f}页/分 剩余~{_fmt_eta(eta)}{cache_note} [{fid[:8]}]"
        )
    return lines


async def _heartbeat_loop() -> None:
    """定时打印所有活跃作业的进度。

    存在的理由：其余日志都是事件驱动的（出错、完成），一个健康但缓慢的长作业
    可以几小时不写一行，从日志上完全无法与「进程挂了」区分。心跳无条件按期输出。
    """
    while True:
        try:
            await asyncio.sleep(HEARTBEAT_SEC)
            lines = _heartbeat_lines(_active_jobs, time.monotonic())
            if lines:
                log.info("心跳: %d 个文件处理中 | %s", len(lines), " | ".join(lines))
        except asyncio.CancelledError:
            raise
        except Exception:  # 心跳永远不能拖垮主流程
            log.exception("心跳日志异常")


@app.on_event("startup")
async def _init_runtime():
    global _api_semaphore, _heartbeat_task, _translate_semaphore
    _api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    _translate_semaphore = asyncio.Semaphore(TRANSLATE_CONCURRENCY)
    _get_http_client()
    if HEARTBEAT_SEC > 0:
        _heartbeat_task = asyncio.create_task(_heartbeat_loop())
    log.info("并发配置: workers=%d, threshold=%d, max_requests=%d, translate=%d, heartbeat=%s",
             CONCURRENCY, CONCURRENCY_THRESHOLD, MAX_CONCURRENT_REQUESTS,
             TRANSLATE_CONCURRENCY, f"{HEARTBEAT_SEC}s" if HEARTBEAT_SEC > 0 else "off")


@app.on_event("shutdown")
async def _close_runtime():
    if _heartbeat_task is not None:
        _heartbeat_task.cancel()
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()

# ---------------------------------------------------------------------------
# 作业注册表：把长任务从 HTTP 请求的生命周期里剥离出来
# ---------------------------------------------------------------------------
# 此前 asyncio.create_task 建在 SSE 生成器内部，客户端一断（关标签页/刷新/休眠）
# 事件就没人接收、任务变成孤儿，前端也无从重连。注册表让任务独立于请求存活，
# 并保留完整事件日志供重连时按 cursor 补发。
#
# 注册表对事件内容一无所知——kind 只是给前端看的标签。这是解析与翻译共用同一套
# 机制的结构保证。


@dataclass
class Job:
    id: str
    kind: str                                   # "parse" | "translate"
    status: str = "running"                     # running | done | error | cancelled
    created_at: float = 0.0
    events: list[dict] = field(default_factory=list)
    subscribers: set = field(default_factory=set)
    task: asyncio.Task | None = None
    error: str | None = None


_jobs: dict[str, Job] = {}


def _publish(job: Job, evt: dict) -> None:
    """把事件写入日志并推给所有订阅者。事件自带序号 i，供前端推进 cursor。"""
    evt = {**evt, "i": len(job.events)}
    job.events.append(evt)
    for q in list(job.subscribers):
        try:
            q.put_nowait(evt)
        except asyncio.QueueFull:
            # 死掉或过慢的订阅者：踢掉即可，它重连后会用 cursor 补齐，无数据损失
            job.subscribers.discard(q)


def _subscribe(job: Job, cursor: int) -> tuple[asyncio.Queue, list[dict]]:
    """登记订阅者并返回它错过的事件。

    「加订阅」与「取快照」之间**绝不能插入 await**：asyncio 单线程模型下这两步
    之间不可能运行 _publish，因此既不漏也不重。若顺序颠倒（先快照后订阅），
    中间就存在丢事件的窗口，而且只会偶发、极难排查。
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_MAX)
    job.subscribers.add(q)
    cursor = max(0, min(cursor, len(job.events)))
    backlog = job.events[cursor:]
    return q, backlog


def _evict_old_jobs() -> None:
    """只淘汰已结束的作业，运行中的永不淘汰。淘汰时整个事件日志一并释放。"""
    finished = [j for j in _jobs.values() if j.status != "running"]
    if len(finished) <= JOB_HISTORY_MAX:
        return
    finished.sort(key=lambda j: j.created_at)
    for j in finished[: len(finished) - JOB_HISTORY_MAX]:
        _jobs.pop(j.id, None)


async def _run_job(job: Job, runner) -> None:
    """驱动作业并保证每条退出路径都落到终态。

    「卡在 running 永不结束」是本设计最危险的失败模式——前端会无限重连，
    观感上与程序挂死无异。因此 job_end 放在 finally 里，取消路径也照发。
    """
    try:
        await runner(job)
        job.status = "done"
    except asyncio.CancelledError:
        job.status = "cancelled"
        raise
    except Exception as e:
        log.exception("[%s] 作业失败", job.id)
        job.status, job.error = "error", str(e)
    finally:
        _publish(job, {"type": "job_end", "status": job.status, "error": job.error})
        _evict_old_jobs()


def _start_job(kind: str, runner) -> Job:
    """建作业并后台运行。runner 是 async def runner(job) -> None，只负责发事件。

    这层抽象让注册表能脱离 LLM 与 HTTP 测试——测试注入一个假 runner 即可。
    """
    job = Job(id=uuid.uuid4().hex, kind=kind, created_at=time.time())
    _jobs[job.id] = job
    job.task = asyncio.create_task(_run_job(job, runner))
    return job


def _sse(evt: dict) -> str:
    return f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"


@app.get("/jobs")
async def list_jobs():
    """列出作业。一期只提供端点、不做界面，用于 localStorage 丢失时人工找回。"""
    return {"jobs": [
        {"id": j.id, "kind": j.kind, "status": j.status, "created_at": j.created_at}
        for j in sorted(_jobs.values(), key=lambda x: x.created_at, reverse=True)
    ]}


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str, cursor: int = 0):
    """订阅作业事件：先补发 events[cursor:]，再转实时流。"""
    job = _jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "not found"}, status_code=404)

    async def _gen():
        q, backlog = _subscribe(job, cursor)
        try:
            # 传输层元信息：前端据此在补发期间抑制 DOM 写入。刷新页面会重放数千条
            # page_done，逐条更新 DOM 会直接卡死。created_at 同时给出耗时的起算点。
            yield _sse({"type": "_replay", "count": len(backlog),
                        "created_at": job.created_at, "status": job.status})
            for evt in backlog:
                yield _sse(evt)
                if evt.get("type") == "job_end":
                    return
            if job.status != "running":
                # 作业已结束就不能再 await：它永远不会有新事件。但必须先把队列里
                # 已排队的事件放完——_publish 是「先入队，再返回」，作业可能恰好在
                # 上面 yield 挂起的窗口里结束，此时 job_end 已经在队列里，直接 return
                # 会把它丢掉，前端就会误判为断线、白做一次重连。
                while not q.empty():
                    evt = q.get_nowait()
                    yield _sse(evt)
                    if evt.get("type") == "job_end":
                        return
                return
            while True:
                evt = await q.get()
                yield _sse(evt)
                if evt.get("type") == "job_end":
                    return
                if q not in job.subscribers:
                    # 被 _publish 当作慢订阅者踢掉了（队列满）：主动收流，客户端会带
                    # cursor 重连补齐，无数据丢失。丢弃只可能发生在队列满时，所以这里
                    # 保证还有至少一次循环迭代能察觉到。不收流就会永远停在 await q.get()
                    # ——这条队列已无人投递，客户端看到的是一条永不结束的静默流：
                    # 不出字节、不关闭、不报错，重连机制结构上无法启动。
                    log.warning("[%s] 订阅者队列溢出被踢，主动收流待其重连", job.id)
                    return
        finally:
            job.subscribers.discard(q)

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    if job.status == "running" and job.task is not None:
        job.task.cancel()
    return {"status": "cancelling"}

# ---------------------------------------------------------------------------
# 核心：PDF → 图片 → VLM → Markdown
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Markdown 水印剥离（针对 CBP / EAPA 风格的 Barcode 头与 Filed By 脚水印）
# ---------------------------------------------------------------------------

# 头部：行首 "Barcode:" + 标准 DOC 案号结构（<digits>-<digits> <letter>-<digits>-<digits>）。
# 这个锚点覆盖 Investigation/Admin Review/NSR/Sunset Review 等各种案件阶段。
# 行首允许的包装：引用符 / 强调星号下划线 / 反引号 / 波浪线 / 可选的「页眉：/页脚：」
# 中文标注——现网关背后的模型会把版面家具包成 `Barcode:...` 或
# "*页眉：* Barcode:... *页脚：* Filed By:..."（2026-09-15 实测三种形态全绕过旧正则）。
# 判定仍锚定内容本身（Barcode 编号格式 / Filed 三件套），包装只是外皮；
# 匹配到就整行删——能命中内容锚点的行整行都是家具。
_WM_WRAP = r"[>*_`~\s]*"
_WM_LABEL = r"(?:\*?页[眉脚]：?\*?\s*)?"
_WATERMARK_HEADER_RE = re.compile(
    rf"(?im)^{_WM_WRAP}{_WM_LABEL}`?Barcode:\s*\d+-\d+\s+[A-Z]-\d+-\d+[^\n]*$"
)
# 脚部："Filed By:" + 同一行内同时含 "Filed Date:" 与 "Submission Status:" 三件套
_WATERMARK_FOOTER_RE = re.compile(
    rf"(?im)^{_WM_WRAP}{_WM_LABEL}`?Filed\s*By:\s*\S+[^\n]*?Filed\s*Date:[^\n]*?Submission\s*Status:[^\n]*$"
)
# 收敛连续空行
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _strip_watermarks(md: str, enabled: bool, file_id: str | None = None) -> str:
    """从 Markdown 中剥离 EAPA 风格的页眉/页脚水印行。

    - enabled=False 或 md 为空时直接返回原值（短路）
    - 命中数 > 0 时记一条 INFO 日志，含 file_id 便于回溯
    """
    if not enabled or not md:
        return md
    md2, n_h = _WATERMARK_HEADER_RE.subn("", md)
    md2, n_f = _WATERMARK_FOOTER_RE.subn("", md2)
    md2 = _BLANK_RUN_RE.sub("\n\n", md2).strip()
    if (n_h + n_f) > 0:
        tag = f"[{file_id}] " if file_id else ""
        log.info("%s去除水印: header=%d, footer=%d", tag, n_h, n_f)
    return md2


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> list[bytes]:
    """将 PDF 每页转为 PNG 图片字节列表。

    全量渲染，内存随页数线性增长。VLM 解析路径已改用 _PageRenderer 惰性逐页渲染，
    这里保留给需要一次性拿到全部页的调用方（及测试基准）。
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


class _PageRenderer:
    """按需逐页把 PDF 渲染成 PNG，用完即弃。

    替代 VLM 路径上的 pdf_to_images()，一次修掉两个问题：

    1. 内存。全量渲染把整本书的位图常驻内存，500 页 @200DPI 约 0.5-1 GB；多文件
       并行时叠加成数 GB（2026-08-06 现场：12 个文件 → RSS 6.8 GB）。惰性渲染下
       同时在内存里的图片数等于并发 worker 数，与总页数无关。
    2. 阻塞。pdf_to_images() 是同步函数，过去直接在协程里调用，渲染期间整个事件
       循环冻结——现场 12 个文件串行渲染了 31 分钟，其间 SSE 心跳、其他文件、
       /health 全部停摆，一次 VLM 调用都发不出去。这里把渲染丢进 executor。

    fitz.Document 非线程安全，故每个渲染器独占一个单线程 executor，页与页之间
    天然串行；文件之间互不影响。渲染发生在 _api_semaphore 之外，不占用 API 配额。
    """

    def __init__(self, pdf_bytes: bytes, dpi: int = 200) -> None:
        self._doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        self._matrix = fitz.Matrix(dpi / 72, dpi / 72)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf-render")
        self.page_count: int = self._doc.page_count

    def render_sync(self, page_num: int) -> bytes:
        """渲染 1-based 页码，返回 PNG 字节。"""
        pix = self._doc[page_num - 1].get_pixmap(matrix=self._matrix)
        return pix.tobytes("png")

    async def render(self, page_num: int) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.render_sync, page_num)

    def close(self) -> None:
        # wait=True：executor 里可能还有一次渲染在跑，先让它结束再关闭 doc，
        # 否则底层 MuPDF 在已释放的文档上取页会段错误。
        self._executor.shutdown(wait=True)
        try:
            self._doc.close()
        except Exception:
            pass


# 整行只有一个 1-3 位数字 —— 页眉/页脚里的页码
_LONE_PAGE_NUM_RE = re.compile(r"^\s*\d{1,3}\s*$")


def _strip_page_number(text: str, page_num: int | None = None) -> str:
    """删掉页首/页尾那种「整行只有一个数字」的页码行。

    提示词第 10 条已要求 VLM 跳过页码，但它的遵守并不稳定——实测一份 125 页文档有
    9 页漏网，且全部出现在该页文本末尾、数字与页码完全相等。这里做一道兜底。

    判据刻意收窄到页首页尾：正文里不会出现这种行（表格行有 |，列表项有标记，裸数字
    脚注定义要求数字后面跟正文，编号段落也带正文），而页中间孤立的数字来路不明，
    宁可留着。四位以上的数字也不动——页码极少有四位，独占一行的 2024 更可能是别的东西。
    """
    blocks = [b for b in text.split("\n")]
    # 找到首尾的非空行下标
    idx = [i for i, l in enumerate(blocks) if l.strip()]
    if not idx:
        return text
    targets = list(dict.fromkeys((idx[0], idx[-1])))  # 整页只有一行时首尾同一下标
    if page_num is not None:
        # VLM 偶尔把页脚放在页中间（阅读顺序所致），位置判据够不着。此时用更紧的
        # 信号：整行数字恰好等于该页页码。孤立成行又正好等于页码的正当内容几乎不存在。
        targets += [i for i in idx
                    if i not in targets and blocks[i].strip() == str(page_num)]
    changed = False
    for i in targets:
        if _LONE_PAGE_NUM_RE.match(blocks[i]):
            blocks[i] = None
            changed = True
    if not changed:
        return text
    out = "\n".join(b for b in blocks if b is not None)
    return _BLANK_RUN_RE.sub("\n\n", out).strip("\n")   # 删行后收敛多余空行


def _persist_page_sync(file_id: str, page_num: int, text: str) -> None:
    """把单页结果写到 output/<file_id>/_pages/pNNNN.md。

    VLM 模式过去只在抽到图表时才落盘最终 .md，其余全靠 SSE 推给浏览器；前端一断线
    （关标签页、休眠、网络抖动），几小时的识别结果全部作废。逐页落盘让任何时刻的
    崩溃/断线都只损失在途的那几页，剩下的可直接 `cat _pages/*.md` 手工拼回。
    文件名零填充到 4 位，保证字典序即页序。

    写盘失败只告警不抛：持久化是兜底手段，不该反过来打断正在成功的解析。
    """
    try:
        d = OUTPUT_DIR / file_id / "_pages"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"p{page_num:04d}.md").write_text(text, encoding="utf-8")
    except (OSError, ValueError) as e:
        log.warning("[%s] 第 %d 页落盘失败: %s", file_id, page_num, e)


# 识别失败的占位符：这些文本绝不能进缓存，否则重跑会永远跳过那一页、把坏页固化下来。
# 锚定行首 + 整体成行，避免把正文里恰好出现的方括号误判成失败标记。
_PAGE_FAILURE_RE = re.compile(r"^\[第 \d+ 页(识别失败|识别超时|识别异常|处理失败)[^\]]*\]$")


def _page_cache_key(pdf_bytes: bytes, dpi: int, model: str) -> str:
    """页级缓存的键。

    必须按 PDF **内容**哈希：file_id 每次上传都是新 uuid，用它做键永远命不中。
    dpi 和 model 一并计入——换 DPI 送进 VLM 的图就变了，换模型旧结果也不该复用。
    """
    h = hashlib.sha256()
    h.update(pdf_bytes)
    h.update(f"|dpi={dpi}|model={model}".encode())
    return h.hexdigest()[:32]


def _page_cache_dir(cache_key: str) -> Path:
    return OUTPUT_DIR / "_page_cache" / cache_key


# 历史缓存里存的是「已回填图片引用」的文本。图表提取逻辑一改（例如不再把表格当图），
# 那些引用会原样复现、指向已不再生成的文件。读取时还原成占位符即可无痛迁移，
# 不必让用户重跑 VLM。新写入的缓存本就是占位符形式，这一步是空操作。
_INJECTED_IMG_RE = re.compile(r"!\[[^\]]*\]\(images/[^)]+\)")


def _restore_figure_placeholders(text: str) -> str:
    return _INJECTED_IMG_RE.sub("[[FIGURE]]", text)


def _read_cached_page(cache_key: str, page_num: int) -> str | None:
    """返回缓存的页文本（图片引用已还原为占位符）；未命中或不可用时返回 None。"""
    if not cache_key:
        return None
    p = _page_cache_dir(cache_key) / f"p{page_num:04d}.md"
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    return _restore_figure_placeholders(text) if _is_cacheable_page(text) else None


def _is_cacheable_page(text: str) -> bool:
    return bool(text and text.strip() and not _PAGE_FAILURE_RE.match(text.strip()))


def _count_cached_pages(cache_key: str, total: int) -> int:
    """开跑前数出有多少页已在缓存中。

    必须能提前报出来：否则要等整个文件跑完才知道续跑生效没有，500 页的文件就是
    一个多小时的盲区——这正是 2026-08-06 那次让人误以为「又从头开始了」的原因。
    """
    if not cache_key:
        return 0
    return sum(1 for n in range(1, total + 1) if _read_cached_page(cache_key, n) is not None)


def _write_cached_page(cache_key: str, page_num: int, text: str) -> None:
    """写入页级缓存；失败占位符和空内容直接拒绝。"""
    if not cache_key or not _is_cacheable_page(text):
        return
    try:
        d = _page_cache_dir(cache_key)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"p{page_num:04d}.md").write_text(text, encoding="utf-8")
    except (OSError, ValueError) as e:
        log.warning("页缓存写入失败 %s p%d: %s", cache_key[:8], page_num, e)


def _clear_persisted_pages(file_id: str, final_md_written: bool) -> None:
    """最终 .md 成功落盘后清掉 _pages/ 中间产物。

    最终产物没写成时保留——那恰恰是需要靠 _pages/ 手工恢复的场景。
    （download_zip 本就跳过 `_` 开头的目录，_pages 不会混进用户下载的 ZIP。）
    """
    if not final_md_written:
        return
    try:
        shutil.rmtree(OUTPUT_DIR / file_id / "_pages", ignore_errors=True)
    except OSError as e:
        log.warning("[%s] 清理 _pages 失败: %s", file_id, e)


def _is_ruled_table(page, rect) -> bool:
    """矢量簇是否只是表格框线（而非 VLM 转不了的真图表）。

    PDF 里表格的框线本身就是矢量绘图，cluster_drawings() 会把它聚成一个簇，而尺寸
    过滤无法与真图表区分，于是整张表被裁剪渲染成 PNG——可 VLM 那边已经把它转成了
    Markdown 表格。结果同一张表既是表格又是图片；表格跨页时多余图片还会被
    _inject_figures 的兜底追加到页尾，正好插在表格两半之间，把表格劈开。

    判据来自实测：一份 125 页裁定书里 40 个符合尺寸条件的簇，39 个表格的曲线数与
    斜线数全为 0，唯一的真折线图有 80 条曲线、12 条斜线。所以——
      出现曲线或斜线 → 真图表（折线、饼图、示意图…）→ 保留
      只有横平竖直的线/矩形，且区域内有可抽取文字 → 表格 → 丢弃

    要求「有文字」是保守起见：没有文字就无从判断 VLM 能否转录它，宁可多留一张图，
    也不能把真内容删掉。
    """
    if not FIGURE_DROP_TABLES:
        return False
    for d in page.get_drawings():
        if not (fitz.Rect(d["rect"]) & rect):
            continue
        for it in d["items"]:
            kind = it[0]
            if kind == "c":                      # 贝塞尔曲线
                return False
            if kind == "l":                      # 斜线（两个方向都有位移）
                if abs(it[1].x - it[2].x) > 1 and abs(it[1].y - it[2].y) > 1:
                    return False
    # 纯图形（无文字）无从判断，保留
    return len(page.get_text("words", clip=rect)) >= 3


def _extract_figures(pdf_bytes: bytes) -> dict[int, list[dict]]:
    """提取每页图表(嵌入位图 + 矢量绘图簇),返回 {页码(1-based): [figure, ...]}。

    figure: {"bytes": bytes, "ext": str, "bbox": fitz.Rect},按 (y0, x0) 排序。
    过滤规则(见设计稿):同一 xref 出现 > 3 页(页眉 logo/水印)、覆盖 > 90% 页面(背景)、
    面积 < 页面 1.5% 或短边 < 40pt(装饰元素)。
    位图 bbox 与矢量簇重叠(相交面积/较小者面积 > 0.5)时并入簇,整簇裁剪渲染,
    避免混合图表被拆成两半。异常向上抛,由调用方捕获降级。
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        # 第一遍:统计每个位图 xref 出现的页数
        xref_pages: dict[int, int] = {}
        page_infos: list[list[dict]] = []
        for page in doc:
            infos = page.get_image_info(xrefs=True)
            page_infos.append(infos)
            for info in infos:
                x = info.get("xref", 0)
                if x:
                    xref_pages[x] = xref_pages.get(x, 0) + 1

        zoom = PDF_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        result: dict[int, list[dict]] = {}

        for page_index, page in enumerate(doc):
            page_area = abs(page.rect)
            if page_area <= 0:
                continue

            def _keep(rect: fitz.Rect) -> bool:
                if abs(rect) > page_area * 0.9:
                    return False  # 整页背景
                if abs(rect) < page_area * 0.015:
                    return False  # 过小装饰
                if min(rect.width, rect.height) < 40:
                    return False  # 细长装饰线/小图标
                return True

            clusters = [fitz.Rect(r) for r in page.cluster_drawings()]
            figures: list[tuple[float, float, dict]] = []

            # 嵌入位图:与矢量簇重叠的并入簇,其余单独按原图提取
            for info in page_infos[page_index]:
                xref = info.get("xref", 0)
                if xref and xref_pages.get(xref, 0) > 3:
                    continue  # 页眉 logo / 重复装饰
                rect = fitz.Rect(info["bbox"])
                merged = False
                for ci, cr in enumerate(clusters):
                    inter = rect & cr
                    smaller = min(abs(rect), abs(cr))
                    if smaller > 0 and abs(inter) / smaller > 0.5:
                        clusters[ci] = cr | rect  # 并集扩簇
                        merged = True
                        break
                if merged or not xref or not _keep(rect):
                    continue
                img = doc.extract_image(xref)
                figures.append((rect.y0, rect.x0,
                                {"bytes": img["image"], "ext": img["ext"], "bbox": rect}))

            # 矢量簇(含并入的位图)裁剪渲染为 PNG
            for cr in clusters:
                if not _keep(cr):
                    continue
                if _is_ruled_table(page, cr):
                    continue  # 表格 VLM 会转成 Markdown，再留张图既冗余又会劈开表格
                pix = page.get_pixmap(matrix=matrix, clip=cr)
                figures.append((cr.y0, cr.x0,
                                {"bytes": pix.tobytes("png"), "ext": "png", "bbox": cr}))

            if figures:
                figures.sort(key=lambda t: (t[0], t[1]))
                result[page_index + 1] = [f for _, _, f in figures]
        return result
    finally:
        doc.close()


# VLM 按提示词在图表原位输出的占位符
_FIGURE_PLACEHOLDER_RE = re.compile(r"\[\[FIGURE\]\]")


def _inject_figures(page_md: str, image_refs: list[str]) -> str:
    """把本页提取的图片引用按序回填到 [[FIGURE]] 占位符处。

    兜底:图片多于占位符 → 剩余图片追加页尾;占位符多于图片 → 多余占位符删除。
    任何情况下 image_refs 中的图片全部落入返回的 Markdown。
    """
    refs = list(image_refs)

    def _sub(_m: re.Match) -> str:
        return f"![图]({refs.pop(0)})" if refs else ""

    out = _FIGURE_PLACEHOLDER_RE.sub(_sub, page_md)
    if refs:
        tail = "\n\n".join(f"![图]({r})" for r in refs)
        out = out.rstrip() + "\n\n" + tail
    return out


def _extract_and_save_figures_sync(pdf_bytes: bytes, file_id: str) -> dict[int, list[str]]:
    """提取图表并写盘(阻塞),返回 {页码: [相对引用路径, ...]}。供 run_in_executor 调用。"""
    # 本地提取图表并落盘;失败仅降级为纯文本,不阻断解析
    try:
        figures_by_page = _extract_figures(pdf_bytes)
    except Exception as e:
        log.warning("[%s] 图表提取失败,降级为纯文本: %s", file_id, e)
        figures_by_page = {}

    refs_by_page: dict[int, list[str]] = {}
    if figures_by_page:
        img_dir = OUTPUT_DIR / file_id / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for pno, figs in figures_by_page.items():
            for i, fig in enumerate(figs, 1):
                name = f"p{pno:03d}_{i}.{fig['ext']}"
                try:
                    (img_dir / name).write_bytes(fig["bytes"])
                except OSError as e:
                    log.warning("[%s] 图片写盘失败 %s: %s", file_id, name, e)
                    continue
                refs_by_page.setdefault(pno, []).append(f"images/{name}")
        log.info("[%s] 提取图表 %d 张", file_id,
                 sum(len(v) for v in refs_by_page.values()))

    return refs_by_page


def _doubled_timeout(timeout):
    """重试那次的超时。额度翻倍意味着模型要多写一倍的字，原超时会把本来能成的
    那次掐死。非数值的超时对象（httpx.Timeout）原样返回——别猜它各字段的语义。"""
    return timeout * 2 if isinstance(timeout, (int, float)) else timeout


def _log_cleaner_trim(what: str, data, cleaned: str) -> None:
    """清洗器把正文剪短了就留一行 WARNING（剪成空由调用方的翻倍分支单独报）。

    剪短是「模型确实吐了思考、或者启发式误伤了正文」二选一，两种都必须看得见：
    前者说明档案配错，后者说明清洗器要收紧。此前只有「剪成空」会报，剥掉一段前缀
    再交出剩下半截是完全静默的——正文丢了也没人知道。
    """
    try:
        raw = _get_raw_content(data)
    except Exception:
        return
    if not raw or not cleaned or len(cleaned) >= len(raw.strip()):
        return
    log.warning("%s：清洗器剥掉了 %d 字符（原始 %d → 清洗后 %d，档案=%s）。"
                "要么服务端把思考写进了正文，要么启发式误伤了正文，两种都该查。",
                what, len(raw.strip()) - len(cleaned), len(raw.strip()), len(cleaned),
                LLM_PROFILE)
    log.debug("%s：被剥掉的开头 = %r", what, raw.strip()[:len(raw.strip()) - len(cleaned)][:400])


def _get_raw_content(data) -> str:
    """取未经清洗的 message.content（只给诊断日志用，不作正文）。"""
    msg = data["choices"][0]["message"] if isinstance(data, dict) \
        else data.choices[0].message
    raw = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    return raw if isinstance(raw, str) else ""


async def _llm_post_with_budget_retry(
    client,
    body: dict,
    headers: dict,
    *,
    timeout,
    what: str,
    prefix_heuristic: bool = True,
    gate: bool = False,
) -> str:
    """发一次非流式请求，返回清洗后的正文；`finish_reason == "length"` 时把
    max_tokens 与超时都翻倍，重试一次。

    翻倍的触发条件**不看正文是否为空**：思考吃光额度会得到空正文，但在
    `reasoning_effort: low` 下更常见的是「JSON 写了一半被截断」——正文非空、解析失败，
    被当成格式错误处理，真正的病因（额度不够）反而看不见。超时同步翻倍是因为额度
    翻倍就意味着模型要多写一倍的字，原超时会把本来能成的那次掐死。

    三处非流式调用点（DOCX 批量、残留英文修正、字体建议）此前各抄了一份一模一样的
    `_post()` + 翻倍块，改动一处就会漂移，收敛到这里。

    gate=True 把整次调用计入翻译闸门：非流式 post() 返回即完整响应，包住它就等于
    包住整次调用（流式那条路不一样，见 translate_chunk_stream 的手动 acquire）。
    body 会被就地改写 max_tokens，调用方不要跨次复用同一个 dict。
    OCR 的 vlm_recognize_page 不走这里：它有自己的三次重试循环（超时递增、HTTP 分支、
    逐次异常兜底），翻倍逻辑接在那个循环里，硬塞进来只会两头都别扭。
    """
    async def _post(call_timeout) -> tuple[str, dict]:
        outer = _get_translate_semaphore() if gate else contextlib.nullcontext()
        async with outer:
            async with _api_semaphore:
                resp = await client.post(TARGET_API_URL, json=body, headers=headers,
                                         timeout=call_timeout)
        resp.raise_for_status()
        data = resp.json()
        cleaned = extract_content(data, prefix_heuristic=prefix_heuristic)
        _log_cleaner_trim(what, data, cleaned)
        return cleaned, data

    content, data = await _post(timeout)
    if finish_reason(data) == "length" and body.get("max_tokens"):
        body["max_tokens"] *= 2
        retry_timeout = _doubled_timeout(timeout)
        log.warning("%s：finish_reason=length（%s，档案=%s），max_tokens 翻倍至 %d、"
                    "超时翻倍至 %s，重试一次",
                    what, "正文为空" if not content else "正文非空但被截断",
                    LLM_PROFILE, body["max_tokens"], retry_timeout)
        content, data = await _post(retry_timeout)
        if finish_reason(data) == "length":
            log.warning("%s：翻倍后仍被截断（max_tokens=%s），结果可能不完整",
                        what, body["max_tokens"])
    return content


async def vlm_recognize_page(
    client: httpx.AsyncClient,
    image_bytes: bytes,
    page_num: int,
    total_pages: int,
    file_id: str = "",
) -> str:
    """调用远程 VLM 识别单页图片，返回 Markdown 文本。

    file_id 只用于日志归属：多文件并行时，页码在各文件间重复，不带 file_id 的
    超时/异常告警无法判断出自哪个文件（2026-08-06 排查现场就卡在这一点上）。
    """
    tag = f"[{file_id}] " if file_id else ""
    b64_image = base64.b64encode(image_bytes).decode()

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个专业的文档 OCR 助手。请将图片中的所有文字内容完整、准确地转录为 Markdown 格式。"
                "要求：\n"
                "1. 保持原文的层级结构，使用合适的标题级别\n"
                "2. 表格必须转为 Markdown 表格格式\n"
                "3. 手写文字也要尽力识别\n"
                "4. 保留原文内容，不要添加解释或总结\n"
                "5. 文件中如果存在脚注，则以Markdown格式进行识别和记录\n"
                "6. 文件中的公式，分子式等信息，需要以Markdown格式进行准确识别和记录\n"
                "7. 如果页面空白，则直接输入说明“空白页”，不要自动生成内容\n"
                "8. 如果有无法识别的文字，用 [?] 标记\n"
                "9. 页面中的图表、插图、照片：在其所在位置单独一行输出占位符 [[FIGURE]]，"
                "不要用文字描述或转录图片内部内容；表格不是图片，仍须转为 Markdown 表格\n"
                "10. 不要转录页眉、页脚中的页码（如“12”“第 12 页”“- 12 -”），正文中的页码引用不受影响"
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_image}",
                    },
                },
                {
                    "type": "text",
                    "text": f"请将这张图片（第 {page_num}/{total_pages} 页）中的所有内容转录为 Markdown。",
                },
            ],
        },
    ]

    headers = llm_headers()

    body = {
        "model": ACTUAL_MODEL_NAME,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.1,
        # 思考开关由档案决定（llm_common.PROFILES）：GLM 用 reasoning_effort:low，
        # Kimi 用 thinking:false。给 GLM 发 thinking:false 会把思考写进 content。
        **llm_extra_body(),
    }

    max_retries = 3
    doubled = False          # 额度翻倍只做一次，别把重试次数全耗在这上面
    # 翻倍重试前拿到的那份「被截断但非空」的转写。重试可能超时/报错，那时它是
    # 我们手里最好的东西——半页正文远好过一句「识别超时」。
    truncated = ""

    def _fallback(reason: str) -> str | None:
        """最终失败时回退到翻倍前那份被截断的转写；没有就返回 None，让调用处
        用自己的错误标记。半页 Markdown 远好过一句「识别超时」，但**必须**在产物里
        留下痕迹——不留标记的话，缺了下半页的那一页和完整页在文档里长得一模一样。"""
        if not truncated:
            return None
        log.warning("%sVLM 第 %d/%d 页最终失败（%s），回退到翻倍前那份被截断的转写",
                    tag, page_num, total_pages, reason)
        return truncated + f"\n\n[第 {page_num} 页转写可能不完整（输出被截断）]"

    for attempt in range(1, max_retries + 1):
        # 每次重试超时递增 50%
        timeout_sec = PAGE_TIMEOUT * (1.5 ** (attempt - 1))
        async with _api_semaphore:
            try:
                resp = await client.post(
                    TARGET_API_URL,
                    json=body,
                    headers=headers,
                    timeout=httpx.Timeout(timeout_sec, connect=10.0),
                )
                if resp.status_code != 200:
                    log.error("%sVLM 第 %d/%d 页返回 HTTP %d (尝试 %d/%d): %s",
                              tag, page_num, total_pages, resp.status_code,
                              attempt, max_retries, resp.text[:300])
                    if attempt < max_retries:
                        await asyncio.sleep(2 * attempt)
                        continue
                    return (_fallback("HTTP 错误")
                            or f"[第 {page_num} 页识别失败：HTTP {resp.status_code}]")

                result = resp.json()
                # prefix_heuristic=False：转写正文本身可能以「1. **公司概况**」这类
                # 编号加粗标题开头，正好撞上推理前缀启发式，会被整段误剥。
                content = extract_content(result, prefix_heuristic=False)

                cut = finish_reason(result) == "length"
                if cut and not doubled and attempt < max_retries:
                    # 额度不够：正文为空（思考吃光）或写到一半被截断，两种都翻倍重来。
                    # 只看「正文为空」会漏掉更常见的那种——半页 Markdown 断在表格中间，
                    # 拼进文档后没人看得出这是额度问题。
                    # `attempt < max_retries` 不能省：前两次超时、第三次才截断时，
                    # 无条件 continue 会走完循环掉到函数末尾，返回 None——签名是 -> str，
                    # 调用方随后在 None 上炸出一句莫名其妙的「处理失败」。
                    doubled = True
                    truncated = content or truncated
                    body["max_tokens"] *= 2
                    log.warning("%sVLM 第 %d/%d 页 finish_reason=length（%s），"
                                "max_tokens 翻倍至 %d 重试（档案=%s）",
                                tag, page_num, total_pages,
                                "正文为空" if not content else "正文非空但被截断",
                                body["max_tokens"], LLM_PROFILE)
                    continue
                if cut and doubled:
                    log.warning("%sVLM 第 %d/%d 页翻倍后仍被截断（max_tokens=%d），"
                                "该页转写可能不完整",
                                tag, page_num, total_pages, body["max_tokens"])
                elif cut:
                    # 到这里 cut 成立却没翻倍过，只可能是「前面几次超时/报错、最后一次
                    # 才拿到截断结果」——没有重试余量了，别谎称翻倍过。
                    log.warning("%sVLM 第 %d/%d 页最后一次尝试上被截断（max_tokens=%d），"
                                "已无重试余量，该页转写可能不完整",
                                tag, page_num, total_pages, body["max_tokens"])
                if not content:
                    log.error("%sVLM 第 %d/%d 页正文为空（finish_reason=%s，档案=%s）："
                              "思考可能吃光 max_tokens，或档案配置不对",
                              tag, page_num, total_pages, finish_reason(result), LLM_PROFILE)

                # 重试拿回空正文时，翻倍前那份截断的转写仍是手里最好的东西
                return content or _fallback("重试后正文为空") or ""

            except httpx.TimeoutException:
                log.warning("%sVLM 第 %d/%d 页超时 (尝试 %d/%d, 超时 %.0fs)",
                            tag, page_num, total_pages, attempt, max_retries, timeout_sec)
                if attempt < max_retries:
                    await asyncio.sleep(2 * attempt)
                    continue
                return _fallback("超时") or f"[第 {page_num} 页识别超时]"
            except Exception as e:
                log.warning("%sVLM 第 %d/%d 页异常 (尝试 %d/%d): %s",
                            tag, page_num, total_pages, attempt, max_retries, e)
                if attempt < max_retries:
                    await asyncio.sleep(2 * attempt)
                    continue
                log.exception("%sVLM 第 %d 页最终失败", tag, page_num)
                return _fallback("异常") or f"[第 {page_num} 页识别异常：{e}]"

    # 循环正常走完（唯一路径：最后一次尝试翻倍后仍空正文）。必须返回字符串——
    # 函数签名是 -> str，掉出去的 None 会在调用方变成一句无关的「处理失败」。
    return _fallback("重试后正文为空") or ""


def _split_into_groups(total: int, n_groups: int) -> list[list[int]]:
    """将 1..total 的页码均匀分成 n_groups 组，余数分给前几组。"""
    groups = []
    base, remainder = divmod(total, n_groups)
    start = 1
    for i in range(n_groups):
        size = base + (1 if i < remainder else 0)
        if size > 0:
            groups.append(list(range(start, start + size)))
        start += size
    return groups


async def _process_group(
    client: httpx.AsyncClient,
    renderer: "_PageRenderer",
    page_indices: list[int],  # 1-based 页码
    total: int,
    results: dict[int, str],
    progress_queue: asyncio.Queue,
    refs_by_page: dict[int, list[str]],
    file_id: str,
    cache_key: str = "",
) -> None:
    """顺序处理一组页面，每页完成后将结果写入 results 并通知 queue。

    图片在这里按需渲染、用完即弃：整个进程同时持有的页面位图数等于并发 worker 数，
    而不是总页数。渲染在 _api_semaphore 之外完成，不占用 API 并发配额。

    cache_key 非空时启用断点续跑：命中的页既不渲染也不调 API。缓存里存的是**已经
    回填过图片引用**的文本，所以命中分支绝不能再走一次 _inject_figures，否则同一张
    图会被引用两次。
    """
    loop = asyncio.get_running_loop()
    for page_num in page_indices:
        cached = _read_cached_page(cache_key, page_num)
        if cached is not None:
            # 缓存里是未回填的原文，这里按**当前**的图片列表重新回填：图表提取逻辑
            # 变更后立刻生效，且无需重调 VLM。此前缓存存的是已回填文本、命中即跳过
            # 回填，导致旧引用复活并指向不再生成的文件。
            text = _inject_figures(_strip_page_number(cached, page_num),
                                   refs_by_page.get(page_num, []))
            results[page_num] = text
            await loop.run_in_executor(None, _persist_page_sync, file_id, page_num, text)
            await progress_queue.put({"type": "page_done", "page": page_num,
                                      "text": text, "cached": True})
            continue

        # 每页自带兜底：worker 一旦抛异常就不再投递剩余页的事件，而收集循环是
        # `while completed < total: await progress_queue.get()`——那会让整个文件
        # 永久挂起，连 file_error 都发不出去。失败页退化成占位文本，照常计数。
        raw = ""
        try:
            img = await renderer.render(page_num)
            raw = await vlm_recognize_page(client, img, page_num, total, file_id)
            del img  # 尽早释放，避免整组处理期间多留一份位图
            raw = _strip_page_number(raw, page_num)
            text = _inject_figures(raw, refs_by_page.get(page_num, []))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("[%s] 第 %d 页处理失败", file_id, page_num)
            raw = text = f"[第 {page_num} 页处理失败：{e}]"
        # 缓存存**未回填**的原文（含 [[FIGURE]] 占位符），回填每次运行都重做一遍。
        # 失败占位符不会被写入（见 _write_cached_page），重跑时会重新识别。
        await loop.run_in_executor(None, _write_cached_page, cache_key, page_num, raw)
        results[page_num] = text
        # 逐页落盘：断线/崩溃只损失在途的几页，其余可从 _pages/ 恢复
        await loop.run_in_executor(None, _persist_page_sync, file_id, page_num, text)
        await progress_queue.put({"type": "page_done", "page": page_num, "text": text})


async def parse_pdf_streaming(
    pdf_bytes: bytes,
    filename: str,
    file_id: str,
    queue: asyncio.Queue,
    strip_watermark: bool = True,
    page_markers: bool = True,
    footnote_fix: bool = True,
) -> None:
    """
    解析单个 PDF，将进度事件放入 queue。
    事件格式均携带 file_id 以区分多文件场景。
    """
    log.info("[%s] 开始解析: %s (%.1f MB)", file_id, filename, len(pdf_bytes) / 1024 / 1024)

    # 惰性渲染器：只在每页即将送 VLM 时才渲染它。构造只解析文档结构，不渲染像素，
    # 因此这里不会像旧的 pdf_to_images() 那样阻塞事件循环几分钟。
    renderer = _PageRenderer(pdf_bytes, dpi=PDF_DPI)
    total = renderer.page_count
    log.info("[%s] 共 %d 页 (DPI=%d, 惰性渲染)", file_id, total, PDF_DPI)

    loop = asyncio.get_running_loop()
    started = time.monotonic()
    # 内容哈希做断点续跑的键。109 MB 的 PDF 算一次 sha256 约 0.3s，丢进 executor 免得堵事件循环。
    cache_key = await loop.run_in_executor(
        None, _page_cache_key, pdf_bytes, PDF_DPI, ACTUAL_MODEL_NAME)
    # 先置空：finally 要取消它，而它的赋值点在 try 里好几个 await 之后——
    # 抽图/统计缓存阶段出错或被取消时，finally 会先撞上 NameError。
    worker_tasks: list[asyncio.Task] = []
    try:
        refs_by_page = await loop.run_in_executor(
            None, _extract_and_save_figures_sync, pdf_bytes, file_id)

        # 先报出续跑规模，再开跑。等 gather 之后才统计等于要跑完 500 页才知道
        # 续跑生效没有，中间是一个多小时的盲区。
        cached_ahead = await loop.run_in_executor(
            None, _count_cached_pages, cache_key, total)
        if cached_ahead:
            log.info("[%s] 断点续跑: %d/%d 页已在缓存中，仅需识别 %d 页",
                     file_id, cached_ahead, total, total - cached_ahead)

        await queue.put({"type": "file_start", "file_id": file_id,
                         "filename": filename, "total": total,
                         "cached": cached_ahead})

        results: dict[int, str] = {}
        progress_queue: asyncio.Queue = asyncio.Queue()

        _active_jobs[file_id] = {"filename": filename, "total": total,
                                 "done": 0, "started": started, "last_done": 0,
                                 "cached": cached_ahead, "cached_done": 0}

        client = _get_http_client()
        # 顺序模式 = 只有一组的并发模式。走同一条 _process_group 路径，避免缓存、
        # 逐页落盘、单页兜底这三件事在两个分支里各写一遍（历史上就是这么漏掉的）。
        if total > CONCURRENCY_THRESHOLD:
            groups = _split_into_groups(total, CONCURRENCY)
            log.info("[%s] 并发模式: %d 组 %s",
                     file_id, len(groups), [len(g) for g in groups])
        else:
            groups = [list(range(1, total + 1))]
            log.info("[%s] 顺序模式", file_id)

        worker_tasks[:] = [
            asyncio.create_task(
                _process_group(client, renderer, g, total, results, progress_queue,
                               refs_by_page, file_id, cache_key))
            for g in groups
        ]
        # 启动所有 worker，同时从 progress_queue 收集结果转发到主 queue
        completed = cached_hits = 0
        while completed < total:
            evt = await progress_queue.get()
            completed += 1
            if evt.pop("cached", False):
                cached_hits += 1
                _active_jobs[file_id]["cached_done"] = cached_hits
            _active_jobs[file_id]["done"] = completed
            await queue.put({**evt, "file_id": file_id,
                             "done": completed, "total": total})
        await asyncio.gather(*worker_tasks)
        if cached_hits:
            log.info("[%s] 断点续跑完成: 实际命中 %d/%d 页", file_id, cached_hits, total)
    finally:
        _active_jobs.pop(file_id, None)
        # 作业被取消时，本协程会在收集循环处抛 CancelledError，但 worker 是独立 task，
        # 不会跟着死：它们会跑完在途的 VLM 调用（继续烧配额），然后对着下面刚关掉的
        # 渲染 executor 逐页快速失败，把「[第 N 页处理失败：…]」占位符写进
        # output/<file_id>/_pages/。必须显式取消，且要排在 renderer.close() 之前——
        # 先关渲染器就等于亲手给还活着的 worker 制造那一串失败页。
        for t in worker_tasks:
            t.cancel()
        renderer.close()

    elapsed = time.monotonic() - started

    # 按页码顺序合并（page_markers 控制是否插入分页标识）
    if page_markers:
        full_md = "\n\n---\n\n".join(
            f"<!-- 第 {i} 页 -->\n\n{results[i]}"
            for i in range(1, total + 1)
        )
    else:
        full_md = "\n\n".join(results[i] for i in range(1, total + 1))
    # 顺序要紧：先归一化脚注，再合并段落。合并要能认出页底脚注块，才不会把下一页的
    # 正文粘到定义上、并能回溯到被截断的段落；而 VLM 吐出的定义形态五花八门
    # （"^37 正文" / "¹ 正文" / 裸数字），只有归一化之后才统一成 [^N]: 可被识别。
    # 水印剥离必须在一切合并类 pass 之前：页眉页脚位于页边界，正是合并器工作的
    # 地方；页脚行不带句末标点，合并器会把下一页正文粘上去，粘连后再剥就会带走
    # 正文（2026-09-15 实测）。先剥掉，被隔断的段落两半相邻，合并器正好能接回。
    full_md = _strip_watermarks(full_md, strip_watermark, file_id)
    if footnote_fix:
        full_md = _normalize_footnotes(full_md)
    # 先把段落内部的硬换行接回一行，再做跨块合并：合并靠「块尾是否句末标点」判断，
    # 而块尾若还是折行的半句，这个判断就不准。
    # 引用块还原成列表要排在合并之前：结构没还原，跨页的两块就对不上
    full_md = _bullet_chars_to_markdown(full_md)
    full_md = _unquote_list_blocks(full_md)
    full_md = _unwrap_hard_linebreaks(full_md)
    if not page_markers:
        # 不插分页标识时，把被分页截断的段落接回一段
        full_md = _merge_broken_paragraphs(full_md)
    has_images = bool(refs_by_page)
    # 无条件落盘：过去只有抽到图表的文件才写 .md，纯文字文档的结果只存在于 SSE 流里，
    # 前端一断就全丢。现在始终留一份最终产物在 output/<file_id>/ 下。
    md_path = OUTPUT_DIR / file_id / f"{_safe_doc_stem(filename)}.md"
    md_written = False
    try:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(full_md, encoding="utf-8")
        md_written = True
    except OSError as e:
        log.warning("[%s] Markdown 写盘失败: %s", file_id, e)
    _clear_persisted_pages(file_id, md_written)
    log.info("[%s] 解析完成: %s | %d 页, %d 字符, 耗时 %s (%.1f 页/分)",
             file_id, filename, total, len(full_md), _fmt_eta(elapsed),
             total / elapsed * 60 if elapsed > 0 else 0.0)
    await queue.put({"type": "file_done", "file_id": file_id,
                     "filename": filename, "pages": total,
                     "chars": len(full_md), "markdown": full_md,
                     "has_images": has_images})



# Markdown 图片引用正则：![alt](path.ext) ，path 必须以常见图片扩展名结尾
_IMG_EXTS = ("jpg", "jpeg", "png", "gif", "webp", "bmp", "tif", "tiff", "svg")
_IMG_REF_RE = re.compile(
    r"!\[([^\]]*)\]\(([^)\s]+?\.(?:" + "|".join(_IMG_EXTS) + r"))\)",
    re.IGNORECASE,
)
# PaddleOCR-VL 1.6 用 HTML <img src="imgs/xxx.jpg"> 输出图片（非 markdown 语法），
# 需单独重写 src。捕获 src= 前缀、引号、路径，便于原样保留标签其余部分。
_HTML_IMG_SRC_RE = re.compile(
    r"(<img\b[^>]*?\bsrc\s*=\s*)([\"'])(.*?)\2",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# HTML 表格转 Markdown
# ---------------------------------------------------------------------------

# 吞掉表格两侧的水平空白：接缝处的空格要跟着表一起被替换掉，否则表题行会留下行尾
# 空格。绝不能改成全文范围清理——两个以上行尾空格是 Markdown 的硬换行，抹掉会把
# 目录之类靠硬换行分行的内容粘成一段（2026-08-21 实测踩过）。
_HTML_TABLE_RE = re.compile(r"[ \t]*<table\b[^>]*>.*?</table>[ \t]*", re.S | re.I)
_HTML_TR_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
_HTML_TD_RE = re.compile(r"<(t[dh])\b([^>]*)>(.*?)</\1>", re.S | re.I)
# 只有 >= 2 才是真合并。有的引擎给每个单元格都写 rowspan=1 colspan=1，那不是合并。
_HTML_MERGE_RE = re.compile(r"\b(?:rowspan|colspan)\s*=\s*[\"']?([2-9]|\d\d+)", re.I)


def _cell_text(html: str) -> str:
    """取单元格纯文本：<br> 变空格，其余标签剥掉，竖线转义。"""
    t = re.sub(r"<br\s*/?>", " ", html, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    t = unescape(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.replace("|", r"\|")


def _one_table_to_markdown(html: str) -> str:
    """单张 HTML 表转 Markdown；无法无损表达时原样返回 HTML。"""
    if _HTML_MERGE_RE.search(html):
        return html                      # 合并单元格，Markdown 表达不了
    rows: list[list[str]] = []
    for tr in _HTML_TR_RE.findall(html):
        cells = [_cell_text(m.group(3)) for m in _HTML_TD_RE.finditer(tr)]
        if cells:
            rows.append(cells)
    if not rows:
        return html
    width = max(len(r) for r in rows)
    # 列数不齐时补空，绝不截断——截断就是无声的数据损失
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |",
           "| " + " | ".join(["---"] * width) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _html_tables_to_markdown(md: str) -> str:
    """把 HTML 表格转成 Markdown 表格（VLM 偶尔也会以 HTML 输出复杂表格）。

    两者的表格识别模型输出的都是 HTML。HTML 表在 Obsidian 里能渲染，但不是 Markdown
    语法：_table_continuation 认不出来因而无法跨页合并，diff 不可读，也没法手工编辑。

    含合并单元格的表原样保留 HTML——Markdown 表达不了跨行跨列，硬转就是丢信息。
    对本来就产 Markdown 表的 vlm 模式，这个函数是空操作。
    """
    if "<table" not in md.lower():
        return md

    def _sub(m: "re.Match[str]") -> str:
        conv = _one_table_to_markdown(m.group(0))
        if conv.lstrip().startswith("<table"):
            return conv                  # 未转换的表原样留在原位
        # GFM 要求表格自成块：前后没有空行，整块会退化成普通段落。实测有引擎把表题
        # 和表格放在同一行（34 张里 29 张），不补空行等于把表全毁了。
        before, after = md[:m.start()], md[m.end():]
        if not before.strip():
            pre = ""
        elif before.endswith("\n\n"):
            pre = ""
        elif before.endswith("\n"):
            pre = "\n"
        else:
            pre = "\n\n"
        if not after.strip():
            post = ""
        elif after.startswith("\n\n"):
            post = ""
        elif after.startswith("\n"):
            post = "\n"
        else:
            post = "\n\n"
        return pre + conv + post

    return _HTML_TABLE_RE.sub(_sub, md)





def _safe_doc_stem(filename: str) -> str:
    """从用户上传的文件名取一个用于落盘的 stem（保留中文，剔除路径分隔符）。"""
    stem = Path(filename).stem or "document"
    # 防止用户文件名里有路径字符
    return re.sub(r'[\\/:*?"<>|]', "_", stem)


def _normalize_vlm_latex(md: str) -> str:
    """把 PaddleOCR-VL（尤其 1.6 模型）用行内 LaTeX 表达的排版语义转成通用 HTML 标签，
    避免在不支持数学公式的 Markdown 查看器里显示成原始 LaTeX。

    - $\\underline{\\text{X}}$ / $\\underline{X}$ → <u>X</u>（下划线文本）
    - $^{N}$ → <sup>N</sup>（上标，如脚注角标）
    - $_{N}$ → <sub>N</sub>（下标）

    仅处理这几种纯排版用法；真正的数学公式（含运算符等）不受影响。
    """
    md = re.sub(r"\$\s*\\underline\{\\text\{(.*?)\}\}\s*\$", r"<u>\1</u>", md)
    md = re.sub(r"\$\s*\\underline\{(.*?)\}\s*\$", r"<u>\1</u>", md)
    # $ {}^{32} $ —— 空基底的纯角标，PaddleOCR 用它表示脚注引用；认不出就丢一条脚注
    md = re.sub(r"\$\s*\{\}\s*\^\{(.*?)\}\s*\$", r"<sup>\1</sup>", md)
    md = re.sub(r"\$\s*\^\{(.*?)\}\s*\$", r"<sup>\1</sup>", md)
    # $ 15^{th} $ —— 序数词等「基底 + 上标」，基底必须留下。限定基底为纯字母数字，
    # 含运算符的真公式不会命中。
    md = re.sub(r"\$\s*([A-Za-z0-9]{1,12})\s*\^\{([A-Za-z0-9]{1,8})\}\s*\$",
                r"\1<sup>\2</sup>", md)
    md = re.sub(r"\$\s*_\{(.*?)\}\s*\$", r"<sub>\1</sub>", md)
    return md


# ── 跨页/跨块被切断的段落重新接回一段 ──
# PDF 分页处会把一句话拦腰截断；各模式按页/块拼接时用空行连接，而 Markdown 里
# 空行 = 新段落，于是同一句话被断成两段。下面在“不插入分页标识”时做一次后处理，
# 把明显是续行的块用一个空格接回去。

# 句末标点（含中英文），允许后跟引号/括号收尾
_SENTENCE_END_RE = re.compile(r'[.!?;:。！？；：]["\'”’)\]】》]*$')
# 结构性块前缀：标题/表格/引用/代码围栏/分隔线
_STRUCT_PREFIXES = ("#", "|", ">", "```", "---", "***", "___")
# 块级 HTML（PaddleOCR-VL 会输出 <div><img ...></div>）；行内标签如 <u> 不算结构块
_BLOCK_HTML_RE = re.compile(
    r"^<\s*(div|img|table|figure|p|br|hr|h[1-6]|ul|ol|li|pre|blockquote)\b", re.I)
# 新的列表项 / 编号条目（如 "- x" "1. x" "(2) x"）
# 列表条目的行首标记。此前只认 - * + 与数字，字母/罗马数字序号（a. / (b) / iv.）
# 一概不认，于是这类条目会被 _is_paragraph_continuation 当成段落续行并进上文。
_LIST_ITEM_RE = re.compile(
    r"^(?:[-*+]\s"
    r"|\(?\d{1,3}[.)]\s"
    r"|\(?[a-zA-Z][.)]\s"
    r"|\(?[ivxlcIVXLC]{1,5}[.)]\s"
    r")")
# 中日韩统一表意文字（中文续行拼接时不插空格）
_CJK_RE = re.compile(r"[一-鿿]")


def _is_paragraph_continuation(prev: str, nxt: str) -> bool:
    """判断 nxt 是否是 prev 被分页截断后的续行。"""
    prev_s, nxt_s = prev.rstrip(), nxt.lstrip()
    if not prev_s or not nxt_s:
        return False
    last_line = prev_s.rsplit("\n", 1)[-1].strip()
    first_line = nxt_s.split("\n", 1)[0].strip()
    # 结构性块（标题/表格/代码/分隔线/块级 HTML/图片）两侧都不合并
    if last_line.startswith(_STRUCT_PREFIXES) or first_line.startswith(_STRUCT_PREFIXES):
        return False
    if _BLOCK_HTML_RE.match(last_line) or _BLOCK_HTML_RE.match(first_line):
        return False
    if first_line.startswith("!["):
        return False
    # 下一块是新的列表项/编号 → 是新条目，不是续行
    if _LIST_ITEM_RE.match(first_line):
        return False
    # 上一块已用句末标点收尾 → 是完整段落，不合并
    if _SENTENCE_END_RE.search(prev_s):
        return False
    # 跨过行内强调标记再判断：整段斜体的引文跨页时，VLM 会在新页重新开一次斜体，
    # 续行以 "*knowledge…" 这样的形式出现。星号后紧跟空格的是列表项，上面
    # _LIST_ITEM_RE 已经拦掉，所以这里剥掉的只可能是强调标记。
    probe = re.sub(r"^(?:\*{1,3}|_{1,3})(?=\S)", "", first_line)
    # 续行判据：下一块以小写字母、左括号或中文字符开头
    return bool(re.match(r"^[a-z(]", probe) or re.match(r"^[一-鿿]", probe))


# 表格分隔行："|:---|---:|"。必须含至少一个 '-'，否则空表头行 "| | | |" 也会命中。
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]*-[\s:|-]*\|\s*$")


def _is_table_row(line: str) -> bool:
    return line.lstrip().startswith("|")


def _table_cols(line: str) -> int:
    """表格行的列数（按 | 切分，去掉首尾空段）。"""
    parts = line.strip().strip("|").split("|")
    return len(parts)


def _table_continuation(prev: str, nxt: str) -> list[str] | None:
    """下一块若是上一块表格的续表，返回应接上去的行；否则 None。

    页与页之间用空行拼接，而空行会截断 Markdown 表格，跨页的长表因此断成两截。
    _merge_broken_paragraphs 原本明确规避表格（怕误伤），于是一直没人接合它们。

    两种续表形态（都在实测文档里出现过）：
      1. 纯数据行续上——最简单，直接接。
      2. VLM 在新页顶部把表格语法重起了一遍：一行伪表头（有时整行为空，如 "| | | |"，
         有时是被切断的单元格残句）+ 一行分隔行。此时要丢掉那行多余的分隔行，
         把伪表头当成数据行接上去。

    判别「续表」还是「另起一张表」用列数：跨页续表的列数必然与原表一致。列数不同
    即视为新表。两张表紧挨着、中间不隔任何文字或标题的情况，在 PDF 转出的文档里
    几乎只可能是分页造成的。
    """
    prev_lines = prev.rstrip().split("\n")
    nxt_lines = nxt.strip("\n").split("\n")
    if not prev_lines or not nxt_lines:
        return None
    if not _is_table_row(prev_lines[-1]) or not _is_table_row(nxt_lines[0]):
        return None
    if _TABLE_SEP_RE.match(prev_lines[-1]):
        return None                       # 上一块以分隔行结尾，形态异常，不碰
    if _TABLE_SEP_RE.match(nxt_lines[0]):
        return None                       # 下一块直接就是分隔行，形态异常，不碰
    if _table_cols(prev_lines[-1]) != _table_cols(nxt_lines[0]):
        return None                       # 列数不同 → 确实是另一张表

    rows = [l.lstrip() if _is_table_row(l) else l for l in nxt_lines]
    if len(rows) > 1 and _TABLE_SEP_RE.match(rows[1]):
        # 形态 2：伪表头 + 多余分隔行。但要先排除「真的另起了一张表」——
        # 判据是伪表头里有没有空单元格：分页残留的伪表头总是残缺的（整行皆空的
        # "| | | |"，或只填了被切断的那一格），而真表头每一格都有内容。
        cells = [c.strip() for c in rows[0].strip().strip("|").split("|")]
        if all(cells):
            return None                   # 每格都有内容 → 是新表的表头，不合并
        del rows[1]                       # 丢掉多余的分隔行
        if not any(cells):
            del rows[0]                   # 整行皆空的伪表头也丢掉
    return rows


_FN_DEF_BLOCK_RE = re.compile(r"^\[\^\d{1,3}\]:")


def _is_footnote_def_block(blk: str) -> bool:
    return bool(_FN_DEF_BLOCK_RE.match(blk.lstrip()))


def _continuation_target(out: list[str]) -> int | None:
    """跨过页底脚注块，回溯到那个被分页截断的段落，返回它在 out 中的下标。

    PDF 一页的版式是「正文（可能被截断）／页底脚注块」，下一页接着写正文。按页拼接
    后，下一页的正文紧跟在脚注块之后，于是被误当成脚注定义的续行粘了上去——实测该
    文档 58 条定义因此混进了正文。

    回溯只允许跨过脚注定义块，以及紧邻它们、用来分隔页底注释的 ---；不得跨过真正的
    章节分隔，否则会把两段无关的文字接到一起。
    """
    i = len(out) - 1
    seen_fn = False
    while i >= 0:
        b = out[i].strip()
        if _is_footnote_def_block(b):
            seen_fn = True
            i -= 1
            continue
        if b in ("---", "***", "___"):
            # 分隔线只有在「它与当前位置之间全是脚注定义」时才可跨过
            if not seen_fn:
                return None
            i -= 1
            continue
        break
    return i if (seen_fn and i >= 0) else None


# 「新条目」的行首标记：- * + / 1. 1) / a) (a) / i) (iv). —— 这些行是各自独立的条目，
# 不是上一行的折行续写
# 字面项目符号字形。刻意不含 U+00B7 间隔号（中文人名「约翰·史密斯」用它）和各种
# 破折号（可能是破折号引语），那两类误伤的代价远大于收益。
_BULLET_LINE_RE = re.compile(r"^([ \t]*)[•‣▪▫◦●○■□][ \t]+")


def _bullet_chars_to_markdown(md: str) -> str:
    """把行首的字面 bullet 字符换成 Markdown 的 `- `。

    有的引擎把项目符号原样输出成 U+2022（实测 125 页文档 94 行），而 `•` 不是 Markdown
    语法：渲染成段落里的一个字符，不是列表项。更麻烦的是 _NEW_ITEM_RE / _LIST_ITEM_RE
    都不认它，这些行会被 _unwrap_hard_linebreaks 和 _merge_broken_paragraphs 当成普通
    散文并进上一段——所以必须排在那两个 pass 之前。

    只认「行首（可缩进）+ 符号 + 空白」：句中的圆点是正文，动它就是篡改。
    """
    if not any(ch in md for ch in "•‣▪▫◦●○■□"):
        return md
    return "\n".join(_BULLET_LINE_RE.sub(r"\1- ", ln) for ln in md.split("\n"))


_NEW_ITEM_RE = re.compile(
    r"^\s*(?:[-*+]\s"
    r"|\d{1,3}[.)]\s"
    r"|\(?[a-zA-Z][.)]\s"
    r"|\(?[ivxlcIVXLC]{1,5}[.)]\s"
    r")")
# 结构性行首：表格 / 标题 / 引用 / 代码围栏 / 图片 / 脚注定义 / 分隔线
_STRUCT_LINE_RE = re.compile(r"^\s*(?:\||#|>|```|!\[|\[\^\d{1,3}\]:|---|\*\*\*|___)")


# 引用块里的列表条目行："> a. steel;" / "> 1. one" / "> - one"
_QUOTED_ITEM_RE = re.compile(
    r"^\s*>\s*(?:[-*+]\s|\d{1,3}[.)]\s|\(?[a-zA-Z][.)]\s|\(?[ivxlcIVXLC]{1,5}[.)]\s)")


def _unquote_list_blocks(md: str) -> str:
    """把「每一行都是列表条目」的引用块还原成列表。

    VLM 会把相对正文缩进的列表误当成引用，输出 "> a. steel;"。实测该文档 3 个引用块
    里 2 个其实是列表，只有 1 个是真引用。误标的后果不只是样式：它让列表在页边界断开
    ——第 60 页整张列表带 >，第 61 页续写的 "h. other" 不带，两块结构不同就接不起来。

    判据要求**每一行**都是条目才动手，真引用（哪怕内部含条目）不受影响。
    """
    out = []
    for blk in re.split(r"(\n{2,})", md):
        if not blk or blk.startswith("\n") or not blk.lstrip().startswith(">"):
            out.append(blk)
            continue
        lines = [l for l in blk.split("\n") if l.strip()]
        if lines and all(_QUOTED_ITEM_RE.match(l) for l in lines):
            out.append("\n".join(re.sub(r"^\s*>\s?", "", l) for l in lines))
        else:
            out.append(blk)
    return "".join(out)


# 字母序号条目的行首标记，用于跨块接合
_LETTER_ITEM_RE = re.compile(r"^\s*([a-z])[.)]\s")


def _letter_list_continuation(prev: str, nxt: str) -> bool:
    """下一块是否是上一块字母列表的续写（被分页切断）。

    **只认字母序号**。曾用「编号连续」做判据，结果把全文所有编号段落（1. 2. … 314.）
    都当成了断裂列表——那些本来就该各自成段。限定字母后，实测全文只剩 1 处命中，
    正是真正断裂的那个。
    """
    pl = [l for l in prev.split("\n") if l.strip()]
    nl = [l for l in nxt.split("\n") if l.strip()]
    if not pl or not nl:
        return False
    a, b = _LETTER_ITEM_RE.match(pl[-1]), _LETTER_ITEM_RE.match(nl[0])
    return bool(a and b and ord(b.group(1)) == ord(a.group(1)) + 1)


def _unwrap_hard_linebreaks(md: str) -> str:
    """把段落内部的硬换行接回一行。

    VLM 逐页识别时会把 PDF 的视觉折行当成真实换行输出，一个段落在 Markdown 源码里
    因此是好几行。_merge_broken_paragraphs 处理的是空行分隔的「块」，块内换行它不碰。

    块内换行有两种含义，必须逐行区分：
      续行   "205. The applicant alleges…" + "reflect non-commercial factors…"  → 接回
      新条目 "a) …" / "- …" / 表格行 / 标题                                     → 不接
    因此判断落在**每一对相邻行**上，而不是整块一刀切——列表条目自身的折行要接回该
    条目，条目与条目之间则不接。实测该文档 106 个多行块里 55 个是硬换行段落。

    三种情况不接：下一行是新条目或结构行；上一行是标题/分隔线/表格行（接上去会毁掉
    结构）；上一行以两个空格结尾（Markdown 显式硬换行，作者本意如此）。
    """
    # 接上去会毁掉结构的行首：标题、表格、分隔线、代码围栏
    no_absorb = re.compile(r"^\s*(?:#|\||```|---|\*\*\*|___)")
    out = []
    for blk in re.split(r"(\n{2,})", md):
        if not blk or blk.startswith("\n") or "```" in blk:
            out.append(blk)
            continue
        def _keep(l: str) -> str:
            # 行尾两个空格是显式硬换行，rstrip 会把它抹掉、后面就检测不到了
            return l if l.endswith("  ") else l.rstrip()

        merged: list[str] = []
        for line in blk.split("\n"):
            if not line.strip():
                continue
            if not merged:
                merged.append(_keep(line))
                continue
            prev = merged[-1]
            if (_NEW_ITEM_RE.match(line) or _STRUCT_LINE_RE.match(line)
                    or no_absorb.match(prev) or prev.endswith("  ")):
                merged.append(_keep(line))
                continue
            m, n = prev.rstrip(), line.strip()
            if m.endswith("-"):
                merged[-1] = m[:-1] + n                    # 连字符断词
            elif m and n and _CJK_RE.match(m[-1]) and _CJK_RE.match(n[0]):
                merged[-1] = m + n                         # 中文之间不加空格
            else:
                merged[-1] = m + " " + n
        out.append("\n".join(merged))
    return "".join(out)


def _merge_broken_paragraphs(md: str) -> str:
    """把被分页/分块切断的段落接回一段（仅在不插入分页标识时调用）。

    连字符断词（"inter-" + "national"）去掉连字符直接拼接；其余用一个空格接续。
    代码围栏内部不做任何合并。
    """
    blocks = re.split(r"\n{2,}", md)
    out: list[str] = []
    in_fence = False
    for blk in blocks:
        b = blk.strip("\n")
        if not b.strip():
            continue
        fence_toggles = b.count("```") % 2 == 1
        if in_fence:
            out.append(b)
            if fence_toggles:
                in_fence = False
            continue
        if fence_toggles:
            in_fence = True
            out.append(b)
            continue
        if _is_footnote_def_block(b):
            out.append(b)          # 脚注定义永不吸收后文
            continue

        # 页底脚注块把段落隔开时，回溯到脚注块之前那个被截断的段落
        tgt = _continuation_target(out)
        if tgt is not None and _is_paragraph_continuation(out[tgt], b):
            prev, nxt = out[tgt].rstrip(), b.lstrip()
            if prev.endswith("-"):
                out[tgt] = prev[:-1] + nxt
            elif _CJK_RE.match(prev[-1]) and _CJK_RE.match(nxt[0]):
                out[tgt] = prev + nxt
            else:
                out[tgt] = prev + " " + nxt
            continue

        if out and _letter_list_continuation(out[-1], b):
            # 字母列表被分页切断：接回同一块，行首多余空格一并去掉
            out[-1] = out[-1].rstrip() + "\n" + "\n".join(
                l.strip() for l in b.split("\n") if l.strip())
            continue

        cont = _table_continuation(out[-1], b) if out else None
        if cont:
            # 续表：用单个换行接回去。行首多余空格也要去掉——模型常在页首多吐一个
            # 空格，" | Bravo" 这样的行会让 Markdown 认不出它是表格行。
            out[-1] = out[-1].rstrip() + "\n" + "\n".join(cont)
        elif (out and not _is_footnote_def_block(out[-1])
              and _is_paragraph_continuation(out[-1], b)):
            prev, nxt = out[-1].rstrip(), b.lstrip()
            if prev.endswith("-"):          # 连字符断词：inter- + national
                out[-1] = prev[:-1] + nxt
            elif _CJK_RE.match(prev[-1]) and _CJK_RE.match(nxt[0]):
                out[-1] = prev + nxt        # 中文之间不插空格
            else:
                out[-1] = prev + " " + nxt
        else:
            out.append(b)
    return "\n\n".join(out)


# ── 脚注/尾注标记归一化 ──
# PDF→MD 转换后，同一份文档里的脚注标记常出现多种畸形写法：
#   ^12^（Pandoc 上标）、 ^12（半截插入符，无闭合）、¹²（Unicode 上标）、<sup>12</sup>
# 而文末的尾注区往往是普通有序列表（"12. 内容"）而非 Markdown 脚注定义。
# 这里统一转成标准语法： 正文 [^12] ，文末 [^12]: 内容
#
# 安全阀：只有当文档确实存在尾注区（Works cited / 参考文献 …）或已有 [^N]: 定义时才动手，
# 且只转换“编号能对上定义”的标记。普通合同/表格文档完全不受影响。
_FN_SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_FN_SUP2D = {c: str(i) for i, c in enumerate(_FN_SUP)}
_FN_REF_HEADING_RE = re.compile(
    r"^#{1,6}\s*(?:works\s+cited|references|bibliography|注释|脚注|尾注|参考文献|参考资料|引用的著作)\s*$",
    re.I)
# 尾注条目行，允许最多 3 个前导空格（PDF→MD 常出现 " 25. xxx"）
_FN_ITEM_RE = re.compile(r"^ {0,3}(\d{1,3})\.\s+(.*)$")
_FN_EXISTING_DEF_RE = re.compile(r"^\[\^(\d{1,3})\]:", re.M)
# 行首「畸形标记 + 空白 + 正文」= 脚注定义行。VLM 逐页产出并不一致：同一篇文档里
# 一部分脚注被直接吐成 [^N]:，另一部分连定义行都写成行首标记，且形态不止一种——
# 实测同一份产物里同时出现 "¹ 正文"（Unicode 上标）与 "^93 正文"（ASCII 插入符）。
# 此前的识别器都不认这些形态，于是形成死锁：安全闸门要求「编号有已知定义」才转换
# 正文标记，而定义本身正是没被认出的那一种，标记与定义都原样留下。
#
# 四种标记写法统一在这里识别，避免以后再一种一种打补丁：
#   ¹²  Unicode 上标 ／ ^12^ Pandoc 上标 ／ ^12 半截插入符 ／ <sup>12</sup>
_FN_MARKER_ALT = (
    rf"[{_FN_SUP}]+"                 # Unicode 上标
    r"|\^\d{1,3}\^"                  # ^12^
    r"|\^\d{1,3}"                     # ^12
    r"|<sup>\s*\d{1,3}\s*</sup>"      # <sup>12</sup>
)
# 行首（≤3 前导空格）+ 标记 + 空白 + 非空正文。要求后面有正文，避免把孤立的页码
# 上标误判成定义。
_FN_LINE_DEF_RE = re.compile(rf"^ {{0,3}}({_FN_MARKER_ALT})[ \t]+(?=\S)", re.M)
# 同样的标记，但出现在正文中（不在行首）——用于交叉校验
_FN_INLINE_MARKER_RE = re.compile(rf"(?<!\[)(?<!\n)({_FN_MARKER_ALT})")


# ── 裸数字脚注 ──
# 第三种形态：VLM 把上标格式完全丢失，标记与定义都退化成纯数字，定义行连 "N." 的
# 点号都没有（"72 重磅！2021年…"、正文里 "…Information System76, which notes"）。
# 歧义远大于带标记的形态，因此判据要求**连续编号的连排**：真正的脚注区是一串编号
# 递增、行距相近的定义行，孤立一行几乎必然是别的东西。实测被这条挡住的假阳性包括
# 文档日期 "16 July 2026"、正文续行 "40 assemblies, and weldments…"、句子续行
# "11 of the Regulations."。
_FN_BARE_DEF_RE = re.compile(r"^ {0,3}(\d{1,3})[ \t]+(?![.\d])(?=\S)", re.M)
# 连排的最小长度与相邻定义行的最大行距
_FN_BARE_RUN_MIN = 3
_FN_BARE_RUN_GAP = 6


# 正文角标的三种位置（紧贴字母/引号/右括号、句末字母+句点、四位年份+句点）。
# _collect_bare_footnote_numbers 用它做「是否被引用」的佐证统计，替换时也用同一套位置。
_FN_BARE_INLINE_RE = re.compile(
    r"(?:"
    r"(?<=[A-Za-z\"”’)\]])"
    r"|(?<=[A-Za-z]\.)"
    r"|(?<=\d{4}\.)(?<!\d{5}\.)"
    r")(\d{1,3})(?![\d%])")


def _collect_bare_footnote_numbers(md: str) -> set[int]:
    """返回「裸数字连排」确认的脚注编号。"""
    lines = md.split("\n")
    cands = [(i, int(m.group(1)))
             for i, l in enumerate(lines) if (m := _FN_BARE_DEF_RE.match(l))]
    runs, cur = [], []
    for ln, n in cands:
        if cur and ln - cur[-1][0] <= _FN_BARE_RUN_GAP and n == cur[-1][1] + 1:
            cur.append((ln, n))
        else:
            if cur:
                runs.append(cur)
            cur = [(ln, n)]
    if cur:
        runs.append(cur)
    confirmed = {n for r in runs if len(r) >= _FN_BARE_RUN_MIN for _, n in r}
    if not confirmed:
        return confirmed

    # 连排长度分不开「真脚注」和「编号段落」——一份 300 段的法律文书连排长度 208，
    # 远超真实脚注文档的 18，把整篇正文都变成了 [^N]: 定义（实测 2026-08-24）。
    # 分得开的是引用关系：脚注天生被正文引用，段落编号不是。实测佐证率
    # 真脚注文档 89%、编号段落文书 16%，阈值取一半即可干净分开。
    def_lines = {i for i, l in enumerate(lines)
                 if (m := _FN_BARE_DEF_RE.match(l)) and int(m.group(1)) in confirmed}
    body = "\n".join(l for i, l in enumerate(lines) if i not in def_lines)
    cited = {int(x) for x in _FN_BARE_INLINE_RE.findall(body)}
    if len(confirmed & cited) * 2 < len(confirmed):
        return set()
    return confirmed


def _marker_to_int(tok: str) -> int | None:
    """把任意一种标记写法解析成编号。"""
    tok = tok.strip()
    if tok and tok[0] in _FN_SUP:
        try:
            return int("".join(_FN_SUP2D[c] for c in tok))
        except KeyError:
            return None
    digits = re.sub(r"\D", "", tok)
    return int(digits) if digits else None
# 面积/体积单位（m² cm³ …）：这些上标不是脚注
_FN_UNIT_TAIL_RE = re.compile(r"(?:^|[\s\d(])(?:m|cm|mm|km|ft|in|yd)$", re.I)


def _find_footnote_section(lines: list[str]) -> int | None:
    """返回尾注区正文起始行号（标题的下一行），没有则 None。"""
    for i, ln in enumerate(lines):
        if _FN_REF_HEADING_RE.match(ln.strip()):
            return i + 1
    return None


def _collect_footnote_numbers(md: str) -> set[int]:
    """收集文档中“确实存在定义”的脚注编号。"""
    nums = {int(n) for n in _FN_EXISTING_DEF_RE.findall(md)}
    # 行首标记写法的定义行同样算「确实存在定义」，否则它对应的正文标记永远转不了。
    #
    # 交叉校验的作用是**防止在完全没有脚注的文档上误开闸门**（例如代码块里恰好有
    # 一行以 ^12 开头）。所以只在文档尚无其他脚注证据时才逐个编号设卡：一旦已经
    # 存在 [^N]: 定义或尾注区，脚注的存在性已经确证，再对每个编号单独要求「必须有
    # 正文引用」就过严了——VLM 时常漏写引用，那些定义行会因此留着裸露的 ^133。
    strict = not nums
    inline = set()
    if strict:
        for line in md.split("\n"):
            body = _FN_LINE_DEF_RE.sub("", line, count=1) if _FN_LINE_DEF_RE.match(line) else line
            for m in _FN_INLINE_MARKER_RE.finditer(body):
                n = _marker_to_int(m.group(1))
                if n is not None:
                    inline.add(n)
    for m in _FN_LINE_DEF_RE.finditer(md):
        n = _marker_to_int(m.group(1))
        if n is not None and (not strict or n in inline):
            nums.add(n)
    nums |= _collect_bare_footnote_numbers(md)
    lines = md.split("\n")
    start = _find_footnote_section(lines)
    if start is not None:
        for ln in lines[start:]:
            m = _FN_ITEM_RE.match(ln)
            if m:
                nums.add(int(m.group(1)))
    return nums


# 脚注替换必须绕开的区域：内容是机器路径/代码而非人写的散文，任何脚注标记都不该
# 出现在里面。2026-08-24 实测：一份文档的 59 个图片引用全部损坏——
# `![图](images/p003_1.jpeg)` 里紧贴 `p` 的 `003` 命中了「数字紧贴字母」这条本该匹配
# 正文角标（System76）的规则，变成 `images/p[^003]_1.jpeg`，图片全部无法显示。
# 只匹配不跨行的构造：替换成单字符哨兵后行结构不变，尾注区的按行处理不受影响。
_FN_PROTECT_RE = re.compile(
    r"`[^`\n]+`"                            # 行内代码
    r"|(?<=\])\([^)\n]*\)"                  # Markdown 链接/图片目标
    r"|<(?:img|a)\b[^>\n]*>"                # HTML 图片/链接标签（<sup> 不在其列）
    r"|(?<![\w/])https?://[^\s)>\]\n]+",    # 裸 URL
    re.I)

# 哨兵刻意不含数字和字母：含数字会被脚注规则命中，字母+数字组合会造出新的假标记。
_FN_SENTINEL = "\x00"


def _mask_protected(md: str) -> tuple[str, list[str]]:
    saved: list[str] = []

    def _take(m: "re.Match[str]") -> str:
        saved.append(m.group(0))
        return _FN_SENTINEL

    return _FN_PROTECT_RE.sub(_take, md), saved


def _unmask_protected(md: str, saved: list[str]) -> str:
    it = iter(saved)
    return re.sub(_FN_SENTINEL, lambda _m: next(it).replace("\\", "\\\\"), md)


def _normalize_footnotes(md: str) -> str:
    """把畸形脚注标记归一化为 [^N]，并把尾注列表转成 [^N]: 定义。

    链接/图片路径、行内代码和 URL 在替换期间被换成哨兵，结束后原样还回——
    见 _FN_PROTECT_RE 的说明。
    """
    nums = _collect_footnote_numbers(md)
    if not nums:
        return md  # 文档没有尾注区 → 原样返回

    md, _protected = _mask_protected(md)

    def _keep(n: str) -> bool:
        return int(n) in nums

    # 先处理行首上标的定义行，再做正文标记替换：顺序反了的话，通用的上标替换会把
    # 定义行的行首上标也变成引用（"[^1] 正文"），定义就永久丢失了。
    def _def_line(m: re.Match) -> str:
        n = _marker_to_int(m.group(1))
        return f"[^{n}]: " if n is not None and n in nums else m.group(0)

    md = _FN_LINE_DEF_RE.sub(_def_line, md)

    # 裸数字：只处理连排确认过的编号
    bare = _collect_bare_footnote_numbers(md)
    if bare:
        md = _FN_BARE_DEF_RE.sub(
            lambda m: f"[^{m.group(1)}]: " if int(m.group(1)) in bare else m.group(0), md)
        # 正文标记的三种位置，全部只对连排确认过的编号生效：
        #   1. 紧贴字母/引号/右括号  "System76"、'oversupply."77'
        #   2. 句末的字母 + 句点     "investigations.88"
        #   3. 年份 + 句点           "from 2023 to 2024.89"
        # 第 3 条限定句点前恰为 4 位数，这样年份放行而小数被挡（"16.25 million" 的
        # 句点前只有两位）。章节号（D2.1、G2.1.2）的编号是 1、2，从不会进连排集合，
        # 因此不必额外设防——实测该文档这三条规则在连排集合内的假阳性为 0。
        bare_inline = re.compile(
            r"(?:"
            r"(?<=[A-Za-z\"”’)\]])"        # 紧贴在字母/引号/右括号之后
            r"|(?<=[A-Za-z]\.)"            # 句末：字母 + 句点 + 编号
            r"|(?<=\d{4}\.)(?<!\d{5}\.)"   # 年份 + 句点 + 编号（2024.89）
            r")(\d{1,3})(?![\d%])")
        md = bare_inline.sub(
            lambda m: f"[^{m.group(1)}]" if int(m.group(1)) in bare else m.group(0), md)

    # 句子以引号收尾时，角标落在句点之后：`decision-making”.27`。上面三条裸数字规则
    # 都够不着（规则 1 要求紧贴引号，这里隔着句点；规则 2 要求句点前是字母；规则 3
    # 要求四位年份）。这条**不**受连排护栏约束，只要该编号有定义即可——安全性来自
    # 位置本身：章节号（G2.1）永远不会长成 `”.N`。实测放宽连排护栏会误伤 82 个章节号，
    # 所以只能这样窄着做。
    md = re.sub(r'([”"’])\.(\d{1,3})(?![\d%])',
                lambda m: (f"{m.group(1)}.[^{m.group(2)}]" if _keep(m.group(2))
                           else m.group(0)), md)

    # <sup>N</sup>（_normalize_vlm_latex 会把 $^{N}$ 转成这个形式）
    md = re.sub(r"<sup>\s*(\d{1,3})\s*</sup>",
                lambda m: f"[^{m.group(1)}]" if _keep(m.group(1)) else m.group(0), md)
    # ^N^ （Pandoc 上标，吸收前面可选空白）
    md = re.sub(r"(?<!\[)[ \t]*\^(\d{1,3})\^",
                lambda m: f"[^{m.group(1)}]" if _keep(m.group(1)) else m.group(0), md)
    # ^N （半截插入符；前面不能是 '['，后面不能再跟 '^'）
    md = re.sub(r"(?<!\[)[ \t]*\^(\d{1,3})(?!\^)",
                lambda m: f"[^{m.group(1)}]" if _keep(m.group(1)) else m.group(0), md)

    # Unicode 上标（可多位：²³ → 23）
    def _uni(m: re.Match) -> str:
        n = int("".join(_FN_SUP2D[c] for c in m.group(0)))
        if n not in nums:
            return m.group(0)
        prev = m.string[max(0, m.start() - 4):m.start()]
        if len(m.group(0)) == 1 and m.group(0) in "²³" and _FN_UNIT_TAIL_RE.search(prev):
            return m.group(0)  # m² / cm³ 等单位，不是脚注
        return f"[^{n}]"
    md = re.sub(f"[{_FN_SUP}]+", _uni, md)

    # 尾注区：有序列表 → 脚注定义（续行缩进 4 空格）
    lines = md.split("\n")
    start = _find_footnote_section(lines)
    if start is None:
        return _unmask_protected(md, _protected)
    out = lines[:start]
    in_def = False
    for ln in lines[start:]:
        m = _FN_ITEM_RE.match(ln)
        if m:
            out.append(f"[^{int(m.group(1))}]: {m.group(2).rstrip()}")
            in_def = True
        elif not ln.strip():
            out.append("")
        elif ln.lstrip().startswith(("#", "---", "***", "<!--", "|", "```")):
            # 分页标识 / 分隔线 / 新标题：不是定义的续行
            out.append(ln)
            in_def = False
        elif in_def:
            out.append("    " + ln.strip())  # 定义的续行
        else:
            out.append(ln)
    return _unmask_protected("\n".join(out), _protected)




# ---------------------------------------------------------------------------
# 本地处理：PaddleOCR-VL（VLM 文档解析，含表格/公式/图表）
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Web 界面（支持 SSE 实时进度）
# ---------------------------------------------------------------------------

UPLOAD_PAGE_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>金诚同达 · 文档解析</title>
<link rel="icon" type="image/png" href="/static/jtn-icon.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@200;300;400;500;700;900&display=swap" rel="stylesheet">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  :root{
    --red:rgb(188,27,41); --red-strong:rgb(166,22,35); --red-tint:rgba(188,27,41,.05);
    --gray:rgb(102,100,100); --ink:#2c2b2b;
    --line:rgba(102,100,100,.22); --line-soft:rgba(102,100,100,.12);
    --muted:rgba(102,100,100,.66); --bg:#fff; --panel:#fafafa;
  }
  html{ -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility; }
  body{ font-family:'Noto Sans SC',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    background:var(--bg); color:var(--gray); min-height:100vh; font-weight:400;
    display:flex; flex-direction:column; }
  body::before{ content:""; position:fixed; top:0; left:0; right:0; height:3px;
    background:var(--red); z-index:60; }

  /* ── 品牌页眉 ── */
  .nav{ width:100%; background:#fff; border-bottom:1px solid var(--line);
    display:flex; align-items:center; padding:0 40px; height:74px; gap:30px; }
  .nav-logo{ height:34px; width:auto; display:block; }
  .nav-sep{ flex:1; }
  .nav a{ color:var(--gray); text-decoration:none; font-size:13px; font-weight:500;
    letter-spacing:.04em; transition:color .2s; }
  .nav a:hover{ color:var(--ink); }
  .nav a.active{ color:var(--red); }
  .nav a.active::before{ content:""; display:inline-block; width:7px; height:7px;
    background:var(--red); margin-right:9px; vertical-align:1px; }

  .main{ flex:1; display:flex; flex-direction:column; align-items:stretch;
    padding:60px 24px 24px; width:100%; max-width:880px; margin:0 auto; }

  /* ── Hero ── */
  .hero{ position:relative; margin-bottom:46px; }
  .overline{ display:flex; align-items:center; gap:11px; margin-bottom:18px; }
  .overline .sq{ width:9px; height:9px; background:var(--red); flex:none; }
  .overline em{ font-style:normal; font-size:12px; letter-spacing:.30em;
    color:var(--red); font-weight:700; text-transform:uppercase; }
  h1{ font-size:36px; font-weight:300; color:var(--ink); letter-spacing:.01em;
    line-height:1.18; margin-bottom:14px; }
  h1 b{ font-weight:700; }
  .subtitle{ color:var(--muted); font-size:14px; line-height:1.8; max-width:560px; }
  .dotmatrix{ position:absolute; top:4px; right:0; display:grid;
    grid-template-columns:repeat(5,7px); grid-auto-rows:7px; gap:8px; }
  .dotmatrix i{ background:var(--red); opacity:.15; }
  @media(max-width:700px){ .dotmatrix{ display:none; } h1{ font-size:28px; } }

  .section-label{ display:flex; align-items:center; gap:10px; margin:0 0 16px;
    font-size:12px; font-weight:700; color:var(--ink); letter-spacing:.08em; }
  .section-label::before{ content:""; width:7px; height:7px; background:var(--red); flex:none; }

  /* ── 模式选择 ── */
  .mode-grid{ display:grid; grid-template-columns:1fr 1fr; gap:1px;
    background:var(--line); border:1px solid var(--line); margin-bottom:34px; }
  .mode-card{ background:#fff; padding:22px 24px; cursor:pointer;
    transition:background .18s; position:relative; }
  .mode-card:hover{ background:var(--panel); }
  .mode-card.selected{ background:var(--red-tint); }
  .mode-card.selected::before{ content:""; position:absolute; top:0; left:0; bottom:0;
    width:3px; background:var(--red); }
  .mode-card.selected::after{ content:""; position:absolute; top:16px; right:16px;
    width:8px; height:8px; background:var(--red); }
  .mode-card input[type=radio]{ position:absolute; opacity:0; pointer-events:none; }
  .mode-icon{ display:none; }
  .mode-title{ font-size:15px; font-weight:700; color:var(--ink); margin-bottom:9px;
    letter-spacing:.01em; }
  .mode-badge{ display:inline-block; font-size:10px; letter-spacing:.14em;
    text-transform:uppercase; color:var(--muted); margin-bottom:10px; font-weight:500; }
  .mode-card.selected .mode-badge{ color:var(--red); }
  .mode-desc{ font-size:12px; color:var(--muted); line-height:1.75; }
  .mode-opt{ display:flex; align-items:center; gap:9px; margin-top:14px; padding-top:13px;
    border-top:1px solid var(--line-soft); font-size:12px; color:var(--gray); cursor:pointer; }
  .mode-opt input[type=checkbox]{ appearance:none; -webkit-appearance:none; width:15px; height:15px;
    border:1.5px solid var(--line); background:#fff; cursor:pointer; position:relative; flex:none; transition:all .15s; }
  .mode-opt input[type=checkbox]:checked{ background:var(--red); border-color:var(--red); }
  .mode-opt input[type=checkbox]:checked::after{ content:""; position:absolute; left:4px; top:1px;
    width:4px; height:8px; border:solid #fff; border-width:0 2px 2px 0; transform:rotate(45deg); }
  .mode-opt span{ cursor:pointer; }

  /* ── 选项行（复选框）── */
  .lang-row{ display:flex; align-items:center; gap:10px; width:100%;
    margin-bottom:12px; padding:15px 18px; background:#fff; border:1px solid var(--line); }
  .lang-row label{ font-size:13px; color:var(--gray); display:flex; align-items:center;
    gap:11px; cursor:pointer; user-select:none; }
  .lang-row input[type=checkbox]{ appearance:none; -webkit-appearance:none;
    width:16px; height:16px; border:1.5px solid var(--line); background:#fff;
    cursor:pointer; position:relative; flex:none; transition:all .15s; }
  .lang-row input[type=checkbox]:checked{ background:var(--red); border-color:var(--red); }
  .lang-row input[type=checkbox]:checked::after{ content:""; position:absolute;
    left:5px; top:1px; width:4px; height:8px; border:solid #fff;
    border-width:0 2px 2px 0; transform:rotate(45deg); }
  .lang-row select{ background:#fff; border:1px solid var(--line); color:var(--gray);
    padding:6px 10px; font-size:13px; outline:none; font-family:inherit; }
  .lang-row select:focus{ border-color:var(--red); }

  /* ── 拖放区域 ── */
  .drop-zone{ width:100%; border:1.5px dashed var(--line); background:#fff;
    padding:58px 20px; text-align:center; cursor:pointer; transition:all .2s; margin-top:6px; }
  .drop-zone:hover, .drop-zone.drag-over{ border-color:var(--red); background:var(--red-tint); }
  .drop-zone input{ display:none; }
  .drop-zone p{ font-size:15px; color:var(--ink); font-weight:500; }
  .drop-zone small{ font-size:12px; color:var(--muted); margin-top:9px; display:block; }

  #fileCards{ width:100%; margin-top:26px; }

  /* ── 文件卡片 ── */
  .file-card{ background:#fff; border:1px solid var(--line); border-left:3px solid var(--red);
    padding:18px 22px; margin-bottom:14px; }
  .file-card-header{ display:flex; justify-content:space-between; align-items:center;
    margin-bottom:12px; gap:12px; }
  .file-name{ font-size:14px; font-weight:700; color:var(--ink);
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:66%; }
  .file-status{ font-size:12px; color:var(--red); font-weight:500; white-space:nowrap; }
  .file-status.done{ color:var(--gray); }
  .file-status.error{ color:var(--red-strong); }

  .progress-bar-bg{ width:100%; height:3px; background:var(--line-soft);
    margin-bottom:12px; overflow:hidden; }
  .progress-bar{ height:100%; background:var(--red); transition:width .3s; width:0%; }
  .progress-bar.done{ background:var(--red); }
  .progress-bar.indeterminate{ width:40% !important;
    animation:indeterminate 1.4s ease-in-out infinite; }
  @keyframes indeterminate{ 0%{ margin-left:-40%; } 100%{ margin-left:100%; } }

  .result-row{ display:flex; gap:8px; margin-top:6px; flex-wrap:wrap; }
  .btn{ padding:8px 16px; border:1px solid transparent; cursor:pointer; font-size:12px;
    font-weight:500; font-family:inherit; letter-spacing:.03em; transition:all .18s; }
  .btn-copy{ background:var(--red); color:#fff; }
  .btn-copy:hover{ background:var(--red-strong); }
  .btn-copy.copied{ background:var(--gray); }
  .btn-download{ background:#fff; color:var(--gray); border-color:var(--line); }
  .btn-download:hover{ border-color:var(--gray); color:var(--ink); }
  .btn-zip{ background:#fff; color:var(--red); border-color:var(--red); }
  .btn-zip:hover{ background:var(--red-tint); }
  .file-info{ font-size:11px; color:var(--muted); margin-top:8px; letter-spacing:.02em; }

  .spinner{ display:inline-block; width:11px; height:11px; border:2px solid var(--red);
    border-top-color:transparent; border-radius:50%; animation:spin .8s linear infinite;
    vertical-align:middle; margin-right:7px; }
  @keyframes spin{ to{ transform:rotate(360deg); } }

  /* ── 页脚 ── */
  .foot{ width:100%; border-top:1px solid var(--line); margin-top:auto;
    padding:24px 40px; display:flex; align-items:center; justify-content:space-between; gap:16px; }
  .foot img{ height:22px; width:auto; }
  .foot span{ font-size:11px; color:var(--muted); letter-spacing:.05em; }
</style>
</head>
<body>
<nav class="nav">
  <img class="nav-logo" src="/static/jtn-logo.png" alt="JT&N 金诚同达">
  <span class="nav-sep"></span>
  <a href="/" class="active">文档解析</a>
  <a href="/translate">Markdown 翻译</a>
  <a href="/office">Office 文档翻译</a>
</nav>
<div class="main">
  <div class="hero">
    <div class="dotmatrix"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
    <div class="overline"><span class="sq"></span><em>Document Parsing · 文档解析</em></div>
    <h1>PDF <b>智能解析</b></h1>
    <p class="subtitle">拖入 PDF，GLM-5.3-Flash 逐页识别，实时输出结构化 Markdown；图表原样保留，支持多文件并行处理。</p>
  </div>

  <div class="section-label">处理选项</div>
  <!-- 处理选项 -->
  <div class="lang-row">
    <label>
      <input type="checkbox" id="stripWatermark" checked>
      去除页眉/页脚水印（Barcode / Filed By 行）
    </label>
  </div>

  <div class="lang-row">
    <label>
      <input type="checkbox" id="pageMarkers" checked>
      在 Markdown 中插入分页标识（页码注释 / 分隔线）
    </label>
  </div>

  <div class="lang-row">
    <label>
      <input type="checkbox" id="footnoteFix" checked>
      规范化脚注/尾注（^12^、¹² 等 → <code>[^12]</code>，尾注列表 → 脚注定义）· 仅当文档有尾注区时生效
    </label>
  </div>

  <div class="section-label">上传文档</div>
  <div class="drop-zone" id="dropZone">
    <p>将 PDF 文件拖到这里，或点击选择文件</p>
    <small>支持同时上传多个 PDF，自动并行处理</small>
    <input type="file" id="fileInput" accept=".pdf" multiple>
  </div>

  <div id="jobBar" style="display:none;margin:12px 0;text-align:right">
    <span id="jobHint" style="margin-right:12px;color:#888;font-size:13px"></span>
    <button class="btn" id="cancelJobBtn" onclick="cancelCurrentJob()">取消解析</button>
  </div>
  <div id="fileCards"></div>
</div>

<footer class="foot">
  <img src="/static/jtn-logo.png" alt="JT&N 金诚同达">
  <span>金诚同达律师事务所　·　Doxify 文档智能工具</span>
</footer>

<script src="/static/job-client.js"></script>
<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileCards = document.getElementById('fileCards');
const fileState = {};


dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const files = [...e.dataTransfer.files].filter(f => f.name.toLowerCase().endsWith('.pdf'));
  if (files.length) uploadFiles(files);
});
fileInput.addEventListener('change', () => {
  const files = [...fileInput.files].filter(f => f.name.toLowerCase().endsWith('.pdf'));
  if (files.length) uploadFiles(files);
  fileInput.value = '';
});

function createCard(fileId, filename) {
  fileState[fileId] = { md: '', filename, pages: 0, pageTexts: {}, indeterminate: false };
  const card = document.createElement('div');
  card.className = 'file-card';
  card.id = 'card-' + fileId;
  card.innerHTML = `
    <div class="file-card-header">
      <span class="file-name" title="${filename}">${filename}</span>
      <span class="file-status" id="status-${fileId}"><span class="spinner"></span>排队中...</span>
    </div>
    <div class="progress-bar-bg">
      <div class="progress-bar" id="bar-${fileId}"></div>
    </div>
    <div class="result-row" id="btns-${fileId}" style="display:none">
      <button class="btn btn-copy" onclick="copyFile('${fileId}')">复制 Markdown</button>
      <button class="btn btn-download" onclick="downloadFile('${fileId}')">下载 .md</button>
      <button class="btn btn-zip" id="zip-${fileId}" onclick="downloadZip('${fileId}')" style="display:none">下载 ZIP (含图片)</button>
    </div>
    <div class="file-info" id="info-${fileId}"></div>
  `;
  fileCards.prepend(card);
}

const PARSE_JOB_KEY = 'doxify.job.parse';
let currentJobId = null;
let suppressDom = false;      // 重放期间不写 DOM：刷新页面会重放数千条 page_done
let jobStartTime = Date.now();
let uploadInFlight = false;   // 与 currentJobId 并列声明

function refreshAllCards() {
  for (const fid of Object.keys(fileState)) {
    const st = fileState[fid];
    if (st.finished) {
      // 进度条/按钮已经在各自分支画对了（那两行不受 suppressDom 限制），但状态文案那行
      // 用的是 setStatus，重放期间会被短路——file_done/file_error 触发时 suppressDom
      // 还没来得及在 onReplayDone 里清掉。这里把文案补回来，否则绿色完成条配一句
      // 卡死的「处理中...」。
      if (st.finished === 'done') {
        setStatus(fid, '完成', true);
      } else if (st.finished === 'error' || st.finished === 'cancelled') {
        setStatus(fid, st.finished === 'cancelled' ? '已取消' : '处理失败', false);
        const el = document.getElementById('status-' + fid);
        if (el) el.className = 'file-status error';
      }
      continue;
    }
    const bar = document.getElementById('bar-' + fid);
    if (!bar) continue;
    if (st.indeterminate) {
      // 不定进度模式：file_start 收到 total === 0 时就置了这个标记，没有逐页
      // 进度可算，重放结束时仍未完成就还原为不定进度态。
      bar.classList.add('indeterminate');
      setStatus(fid, '<span class="spinner"></span>本地处理中（请稍候）...', false);
    } else if (!st.pages) {
      continue;   // 尚未收到 file_start，仍在排队，createCard 的「排队中...」不要动它
    } else {
      const done = st.done || 0;
      bar.style.width = Math.round((done / st.pages) * 100) + '%';
      const hint = st.cached ? `（含 ${st.cached} 页缓存）` : '';
      setStatus(fid, `<span class="spinner"></span>识别中 (${done}/${st.pages})${hint}`, false);
    }
  }
}

function setUploadDisabled(disabled) {
  // 一期只支持单作业，上传区在作业进行期间禁用，让约束可见而不是只靠 alert 拦截。
  dropZone.style.pointerEvents = disabled ? 'none' : '';
  dropZone.style.opacity = disabled ? '0.5' : '';
  fileInput.disabled = disabled;
}

function cancelCurrentJob() {
  if (currentJobId) JobClient.cancel(currentJobId);
}

function markUnfinishedCards(status, error) {
  // job_end 不带 file_id，handleEvent 在 fid 判断处直接 return，任何卡片都不会被触碰。
  // 不在这里收尾，被取消/失败作业的在途卡片会永远停在「识别中 (3/50)」的旋转态——
  // 看上去和还在跑一模一样，job.error 也永远不会出现在界面上。
  const label = status === 'cancelled' ? '已取消' : '失败';
  for (const fid of Object.keys(fileState)) {
    const st = fileState[fid];
    if (st.finished) continue;               // 已经画成完成/失败的不动
    st.finished = status === 'cancelled' ? 'cancelled' : 'error';
    const bar = document.getElementById('bar-' + fid);
    if (bar) {
      bar.classList.remove('indeterminate');
      bar.style.width = '100%';
      bar.style.background = '#e74c3c';
    }
    setStatus(fid, label, false);
    const el = document.getElementById('status-' + fid);
    if (el) el.className = 'file-status error';
    const info = document.getElementById('info-' + fid);
    if (info && error) info.textContent = error;
  }
}

function attachParseJob(jobId, tmpCards) {
  currentJobId = jobId;
  document.getElementById('jobBar').style.display = '';
  // onGiveUp 会把取消按钮藏掉（那时它已成空操作），新作业接上时必须还原
  document.getElementById('cancelJobBtn').style.display = '';
  setUploadDisabled(true);
  return JobClient.attach(PARSE_JOB_KEY, jobId, {
    onMeta: (m) => {
      jobStartTime = m.created_at * 1000;
      suppressDom = m.count > 0;
      if (m.count > 0) {
        document.getElementById('jobHint').textContent = `正在恢复 ${m.count} 条进度...`;
      }
    },
    onEvent: (evt) => handleEvent(evt, tmpCards, jobStartTime),
    onReplayDone: () => {
      suppressDom = false;
      document.getElementById('jobHint').textContent = '';
      refreshAllCards();
    },
    onEnd: (status, error) => {
      currentJobId = null;
      // 终态处理器都要清 suppressDom：重放中途断线、重连遇 404（服务重启过——正是
      // 本功能针对的场景）会让 suppressDom 停在 true，setStatus 在该页面剩余生命周期
      // 内全部静默失效，包括下一个作业。
      suppressDom = false;
      document.getElementById('jobBar').style.display = 'none';
      document.getElementById('jobHint').textContent = '';
      setUploadDisabled(false);
      if (status && status !== 'done') markUnfinishedCards(status, error);
    },
    onGone: () => {
      currentJobId = null;
      suppressDom = false;   // 理由同 onEnd
      document.getElementById('jobBar').style.display = 'none';
      document.getElementById('jobHint').textContent = '';
      setUploadDisabled(false);
    },
    onReconnecting: (n) => {
      document.getElementById('jobHint').textContent = `连接中断，重连中 (${n})...`;
    },
    onGiveUp: () => {
      // 必须说清「作业可能还在跑、刷新是接回不是重开」：否则用户很容易再传一次，
      // 在第一个作业没被取消的情况下起出第二个，白烧一倍网关配额。
      document.getElementById('jobHint').textContent =
        '重连失败，作业可能仍在运行，刷新页面可接回';
      // 放弃重连时可能还卡在重放中途，不清掉 suppressDom 界面就永久冻结。
      suppressDom = false;
      currentJobId = null;
      // 取消按钮此刻已是空操作（currentJobId 已清空），留着可见只会误导用户以为
      // 还能停掉作业。提示语就在 #jobBar 内部，所以只能藏按钮、不能藏整条 bar。
      document.getElementById('cancelJobBtn').style.display = 'none';
      setUploadDisabled(false);
    },
  });
}

async function uploadFiles(files) {
  // 一期只支持单作业：并发作业会让 currentJobId / suppressDom / jobStartTime 这几个
  // 模块级状态互相踩——尤其是前一个作业重放结束会把 suppressDom 清掉，而后一个作业
  // 可能正在重放几千条事件。完整的多作业界面属于二期。
  //
  // currentJobId 要等 POST 返回后才置位，大文件上传期间它仍是 null——这段窗口
  // 恰恰是并发最可能发生的时刻，必须用一个同步置位的标志堵住。
  if (currentJobId || uploadInFlight) {
    alert('已有解析任务在进行中，请等待完成或点「取消解析」后再上传。');
    return;
  }
  uploadInFlight = true;
  setUploadDisabled(true);

  const formData = new FormData();
  for (const f of files) formData.append('files', f);
  const strip = document.getElementById('stripWatermark');
  formData.append('strip_watermark', (strip && strip.checked) ? '1' : '0');
  const pm = document.getElementById('pageMarkers');
  formData.append('page_markers', (pm && pm.checked) ? '1' : '0');
  const fnf = document.getElementById('footnoteFix');
  formData.append('footnote_fix', (fnf && fnf.checked) ? '1' : '0');

  const tmpCards = [];
  for (const f of files) {
    const tmpId = 'tmp_' + Math.random().toString(36).slice(2);
    createCard(tmpId, f.name);
    tmpCards.push(document.getElementById('card-' + tmpId));
  }

  try {
    const resp = await fetch('/jobs/parse', { method: 'POST', body: formData });
    const body = await resp.json();
    if (!body.job_id) {
      tmpCards.forEach(c => c.remove());
      alert(body.error || '启动失败');
      setUploadDisabled(false);
      return;
    }
    JobClient.save(PARSE_JOB_KEY, body.job_id);
    await attachParseJob(body.job_id, tmpCards);
  } catch (e) {
    console.error('上传失败', e);
    tmpCards.forEach(c => c.remove());
    setUploadDisabled(false);
  } finally {
    uploadInFlight = false;
  }
}

function handleEvent(evt, tmpCards, startTime) {
  if (evt.type === 'init') {
    tmpCards.forEach(c => c.remove());
    for (const {file_id, filename} of evt.files) {
      createCard(file_id, filename);
      document.getElementById('status-' + file_id).innerHTML =
        '<span class="spinner"></span>处理中...';
    }
    return;
  }

  const fid = evt.file_id;
  if (!fid || !fileState[fid]) return;
  const st = fileState[fid];

  if (evt.type === 'file_start') {
    st.pages = evt.total;
    if (evt.total === 0) {
      // 不定进度（total === 0）
      st.indeterminate = true;
      document.getElementById('bar-' + fid).classList.add('indeterminate');
      setStatus(fid, '<span class="spinner"></span>本地处理中（请稍候）...', false);
    } else {
      // 续跑时必须明说，否则进度条从 0 爬起来和「从头重跑」看不出区别
      st.cached = evt.cached || 0;
      const hint = st.cached ? `（续跑：${st.cached} 页已缓存，需识别 ${evt.total - st.cached} 页）` : '';
      setStatus(fid, `<span class="spinner"></span>识别中 (0/${evt.total})${hint}`, false);
    }
  } else if (evt.type === 'page_done') {
    const { page, text, done, total } = evt;
    st.pageTexts[page] = text;
    st.done = done;
    const pct = Math.round((done / total) * 100);
    if (!suppressDom) document.getElementById('bar-' + fid).style.width = pct + '%';
    const hint = st.cached ? `（含 ${st.cached} 页缓存）` : '';
    setStatus(fid, `<span class="spinner"></span>识别中 (${done}/${total})${hint}`, false);
  } else if (evt.type === 'file_progress') {
    // 文本进度：把每行输出展示为状态文字
    const safe = String(evt.message || '').replace(/[<>&"]/g, c => ({
      '<':'&lt;', '>':'&gt;', '&':'&amp;', '"':'&quot;'
    }[c]));
    setStatus(fid, '<span class="spinner"></span>' + safe, false);
  } else if (evt.type === 'file_done') {
    st.finished = 'done';   // 告诉 refreshAllCards 这张卡片已经画对了，重放收尾时别覆盖
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    const bar = document.getElementById('bar-' + fid);
    bar.classList.remove('indeterminate');
    bar.style.width = '100%';
    bar.classList.add('done');
    setStatus(fid, '完成', true);
    const pagesStr = evt.pages > 0 ? `${evt.pages} 页 | ` : '';
    document.getElementById('info-' + fid).textContent =
      `${pagesStr}${evt.chars} 字符 | 耗时 ${elapsed} 秒`;
    document.getElementById('btns-' + fid).style.display = 'flex';

    // 优先使用服务端发来的 markdown，否则从 pageTexts 重建
    if (evt.markdown) {
      st.md = evt.markdown;
    } else {
      let md = '';
      for (let i = 1; i <= evt.pages; i++) {
        const t = st.pageTexts[i] || '';
        md += (md ? '\\n\\n---\\n\\n' : '') + `<!-- 第 ${i} 页 -->\\n\\n${t}`;
      }
      st.md = md;
    }
    // 含图片（VLM 模式抽出的图表）时显示 ZIP 下载按钮
    if (evt.has_images) {
      const zb = document.getElementById('zip-' + fid);
      if (zb) zb.style.display = '';
    }
  } else if (evt.type === 'file_error') {
    st.finished = 'error';   // 同上，refreshAllCards 不应把失败态覆盖回「识别中」
    const bar = document.getElementById('bar-' + fid);
    bar.classList.remove('indeterminate');
    bar.style.width = '100%';
    bar.style.background = '#e74c3c';
    setStatus(fid, '处理失败', false);
    document.getElementById('status-' + fid).className = 'file-status error';
    document.getElementById('info-' + fid).textContent = evt.error || '未知错误';
  }
}

function setStatus(fid, html, done) {
  if (suppressDom) return;
  const el = document.getElementById('status-' + fid);
  if (!el) return;
  el.innerHTML = html;
  el.className = 'file-status' + (done ? ' done' : '');
}

function copyFile(fid) {
  navigator.clipboard.writeText(fileState[fid].md || '');
  const btn = event.target;
  btn.textContent = '已复制!'; btn.classList.add('copied');
  setTimeout(() => { btn.textContent = '复制 Markdown'; btn.classList.remove('copied'); }, 2000);
}

function downloadFile(fid) {
  const name = fileState[fid].filename.replace(/\\.pdf$/i, '') + '.md';
  const blob = new Blob([fileState[fid].md || ''], { type: 'text/markdown' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
}

function downloadZip(fid) {
  // 直接跳到下载端点，浏览器自动按 Content-Disposition 处理文件名
  window.location.href = '/download_zip/' + encodeURIComponent(fid);
}

// 刷新/重开标签页后自动接回正在跑的作业。cursor 从 0 开始，init 事件会重放、
// 卡片自动重建，handleEvent 无需特判（init 分支对空 tmpCards 数组天然安全）。
window.addEventListener('DOMContentLoaded', () => {
  const jobId = JobClient.restore(PARSE_JOB_KEY);
  // attach 理论上不该再抛出未捕获的异常了，但这里补一层 .catch 兜底记录，
  // 不能让它安静地变成 unhandled rejection——出问题至少要能在控制台看到。
  if (jobId) attachParseJob(jobId, []).catch(e => console.error('重连失败', e));
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# DOCX 翻译：lxml 原位改文本节点，格式/修订/批注全保留
# ---------------------------------------------------------------------------
# 只改 w:t / w:delText 的文本内容，XML 树一个节点不增不删——格式、修订（w:ins/w:del）、
# 批注、书签、域、图片因此天然保留。python-docx 这类高层库对修订/批注支持不完整，
# 重建文档必丢东西，所以全程 lxml 直改。
#
# 分段规则（决定翻译质量与修订语义的正确性）：
# - 修订语境切换处必断：normal / ins / del 各自成段。把删除文本和替换它的插入文本
#   拼在一起翻译得到交错的胡话，回填还会把译文错归到别人的修订名下。
# - w:tab / w:br / w:cr 处必断：跨过它们拼接会把译文错铺到另一个单元格/行里。
# - 段内 加粗/斜体/下划线/上下标 差异用 **·**/*·*/<u>·</u>/<sup>·</sup> 标记随文送给
#   LLM（实测 158/358 个段落存在段内格式差异）；标记解析失败退化为整段填入首个
#   run——宁可丢局部格式绝不丢文字。

from lxml import etree as _lxml_etree

_DOCX_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def _docx_fmt_key(t_node) -> str:
    """文本节点所在 run 的格式键：b/i/u/^(上标)/_(下标) 的组合，无格式为空串。"""
    r = t_node.getparent()
    if r is None:
        return ""
    rpr = r.find(f"{_DOCX_W}rPr")
    if rpr is None:
        return ""
    key = ""
    b = rpr.find(f"{_DOCX_W}b")
    if b is not None and b.get(f"{_DOCX_W}val") not in ("0", "false", "none"):
        key += "b"
    i = rpr.find(f"{_DOCX_W}i")
    if i is not None and i.get(f"{_DOCX_W}val") not in ("0", "false", "none"):
        key += "i"
    u = rpr.find(f"{_DOCX_W}u")
    if u is not None and u.get(f"{_DOCX_W}val") != "none":
        key += "u"
    va = rpr.find(f"{_DOCX_W}vertAlign")
    if va is not None:
        v = va.get(f"{_DOCX_W}val")
        if v == "superscript":
            key += "^"
        elif v == "subscript":
            key += "_"
    return key


def _docx_rev_context(node, stop) -> str:
    """向上找最近的 ins/del 祖先（到 stop 为止）。"""
    p = node.getparent()
    while p is not None and p is not stop:
        if p.tag == f"{_DOCX_W}del":
            return "del"
        if p.tag == f"{_DOCX_W}ins":
            return "ins"
        p = p.getparent()
    return "normal"


def _docx_nearest_p(node):
    p = node.getparent()
    while p is not None:
        if p.tag == f"{_DOCX_W}p":
            return p
        p = p.getparent()
    return None


_DOCX_BREAKERS = None  # 惰性构造，避免模块级 f-string 噪音


def _docx_collect_segments(root) -> list[dict]:
    """收集 root 下所有段落的可翻译分段。

    segment: {"ctx": "normal|ins|del", "spans": [(fmt_key, [text_nodes])]}
    嵌套段落（文本框 w:txbxContent 里的 w:p）由其自身的迭代轮次处理，外层跳过，
    否则同一段文本会被收集两次、翻译两次、回填互相覆盖。
    """
    global _DOCX_BREAKERS
    if _DOCX_BREAKERS is None:
        _DOCX_BREAKERS = {f"{_DOCX_W}tab", f"{_DOCX_W}br", f"{_DOCX_W}cr"}
    text_tags = {f"{_DOCX_W}t", f"{_DOCX_W}delText"}
    segs: list[dict] = []
    for para in root.iter(f"{_DOCX_W}p"):
        cur_ctx = None
        cur_spans: list[tuple[str, list]] = []

        def _flush():
            nonlocal cur_spans, cur_ctx
            if cur_spans:
                segs.append({"ctx": cur_ctx, "spans": cur_spans})
            cur_spans = []

        for node in para.iter():
            if node is para:
                continue
            if node.tag in _DOCX_BREAKERS:
                _flush()
                continue
            if node.tag not in text_tags:
                continue
            if _docx_nearest_p(node) is not para:
                continue  # 嵌套段落的文本，归它自己那轮
            ctx = _docx_rev_context(node, para)
            if ctx != cur_ctx:
                _flush()
                cur_ctx = ctx
            key = _docx_fmt_key(node)
            if cur_spans and cur_spans[-1][0] == key:
                cur_spans[-1][1].append(node)
            else:
                cur_spans.append((key, [node]))
        _flush()
    return segs


def _docx_seg_text(seg) -> str:
    return "".join((n.text or "") for _, nodes in seg["spans"] for n in nodes)


def _docx_wrap(key: str, text: str) -> str:
    if "^" in key:
        text = f"<sup>{text}</sup>"
    if "_" in key:
        text = f"<sub>{text}</sub>"
    if "u" in key:
        text = f"<u>{text}</u>"
    if "i" in key:
        text = f"*{text}*"
    if "b" in key:
        text = f"**{text}**"
    return text


def _docx_seg_marked(seg) -> str:
    """段文本，格式差异用标记表达。整段格式一致时不加标记——没有差异就没有信息。"""
    keys = {k for k, _ in seg["spans"]}
    if len(keys) == 1:
        return _docx_seg_text(seg)
    out = []
    for key, nodes in seg["spans"]:
        t = "".join((n.text or "") for n in nodes)
        out.append(_docx_wrap(key, t) if t else "")
    return "".join(out)


_DOCX_MARK_OPEN = [("***", "bi"), ("**", "b"), ("*", "i"),
                   ("<u>", "u"), ("<sup>", "^"), ("<sub>", "_")]
_DOCX_MARK_CLOSE = {"***": "***", "**": "**", "*": "*",
                    "<u>": "</u>", "<sup>": "</sup>", "<sub>": "</sub>"}


def _docx_parse_marked(s: str) -> list[tuple[str, str]] | None:
    """把带标记的译文拆成 [(fmt_key, text)]；标记不配对返回 None（触发兜底）。"""
    out: list[tuple[str, str]] = []
    i, plain = 0, ""

    def _push_plain():
        nonlocal plain
        if plain:
            out.append(("", plain))
            plain = ""

    while i < len(s):
        matched = False
        for op, key in _DOCX_MARK_OPEN:
            if s.startswith(op, i):
                cl = _DOCX_MARK_CLOSE[op]
                end = s.find(cl, i + len(op))
                if end == -1:
                    return None
                inner = s[i + len(op):end]
                # 内层可能还有一层标记（如 **<u>x</u>**）
                if any(inner.startswith(o) and inner.endswith(_DOCX_MARK_CLOSE[o])
                       and len(inner) >= len(o) + len(_DOCX_MARK_CLOSE[o])
                       for o, _ in _DOCX_MARK_OPEN):
                    sub = _docx_parse_marked(inner)
                    if sub is None:
                        return None
                    _push_plain()
                    for k2, t2 in sub:
                        out.append(("".join(sorted(set(key + k2))), t2))
                else:
                    _push_plain()
                    out.append((key, inner))
                i = end + len(cl)
                matched = True
                break
        if not matched:
            plain += s[i]
            i += 1
    _push_plain()
    # 归一化 key 的字符顺序，与 _docx_fmt_key 的产出对齐（b<i<u<^<_ 按构造顺序）
    order = "biu^_"
    return [("".join(c for c in order if c in k), t) for k, t in out]


def _docx_set_text(node, text: str) -> None:
    node.text = text
    if text != text.strip() or not text:
        node.set(_XML_SPACE, "preserve")


def _docx_apply_translation(seg, translated: str) -> None:
    """把译文写回分段。空译文保留原文——静默丢字是最坏的失败模式。"""
    if not translated:
        return
    spans = seg["spans"]
    multi = len({k for k, _ in spans}) > 1
    if not multi:
        # 格式统一的分段没送过标记。译文里**成对**的标记只能是模型自作主张加的
        # （提示词教了它标记语法），剥掉；孤立的 *（如「5*」）解析不成对，原样保留。
        stray = _docx_parse_marked(translated)
        if stray and any(k for k, _ in stray):
            translated = "".join(t for _, t in stray)
        first = spans[0][1][0]
        _docx_set_text(first, translated)
        for _k, nodes in spans:
            for n in nodes:
                if n is not first:
                    _docx_set_text(n, "")
        return
    pieces = _docx_parse_marked(translated)
    assign: list[str] = [""] * len(spans)
    if pieces is not None:
        # 按格式键贪心配对：每个译文片段找第一个未占用的同键 span；找不到就并入上一段
        used = [False] * len(spans)
        last = -1
        ok = True
        for key, text in pieces:
            # 先在 last 之后找同键 span（维持文档顺序），找不到再回绕从头找
            # ——译文可能重排格式片段的先后
            hit = next((j for j in range(last + 1, len(spans))
                        if not used[j] and spans[j][0] == key), None)
            if hit is None:
                hit = next((j for j in range(len(spans))
                            if not used[j] and spans[j][0] == key), None)
            if hit is None:
                if last >= 0:
                    assign[last] += text
                else:
                    ok = False
                    break
            else:
                used[hit] = True
                assign[hit] += text
                last = hit
        if not ok:
            pieces = None
    if pieces is None:
        # 兜底：标记没了/对不上——整段填入首个 span，其余清空
        plain = translated
        for op, _k in _DOCX_MARK_OPEN:
            plain = plain.replace(op, "").replace(_DOCX_MARK_CLOSE[op], "")
        assign = [""] * len(spans)
        assign[0] = plain
    for (key, nodes), text in zip(spans, assign):
        _docx_set_text(nodes[0], text)
        for n in nodes[1:]:
            _docx_set_text(n, "")


_DOCX_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def _docx_should_translate(text: str) -> bool:
    """超短片段（子词修订碎片如 'rea'/'is'、符号、纯数字）不送翻译，原样保留。

    误译一个被拆开的子词碎片会破坏它所在的单词；这类碎片本来就没有独立语义。
    """
    t = text.strip()
    if len(t) <= 3:
        return False
    return bool(_DOCX_LETTER_RE.search(t))


def _docx_pack_batches(items: list[str], max_chars: int = 2400,
                       max_items: int = 24) -> list[list[int]]:
    """把待翻译文本按索引打包成批，控制单批字符量与条数。"""
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_chars = 0
    for i, t in enumerate(items):
        if cur and (cur_chars + len(t) > max_chars or len(cur) >= max_items):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(i)
        cur_chars += len(t)
    if cur:
        batches.append(cur)
    return batches


# 泰文区块 U+0E00–U+0E7F。OOXML 把它归「复杂文种」，字体走 rFonts 的 w:cs 属性，
# eastAsia/ascii 两轨都管不到。老挝文/高棉文等同理，暂只覆盖泰文（当前目标语言集）。
_DOCX_THAI_RE = re.compile(r"[\u0e00-\u0e7f]")


def _docx_apply_fonts(seg, cjk_font: str, latin_font: str, cs_font: str = "") -> None:
    """给分段里被翻译改过的 run 补字体声明。只在 run 级动手，零波及。

    Word 的字体按字符类别分轨：ascii/hAnsi 管拉丁、eastAsia 管汉字，互不干扰。
    英文文档的 run 通常只声明拉丁轨（或什么都不声明，继承 Times New Roman 之类），
    译出的汉字没人管，Word 回退瞎猜——实测同一文档里宋体/等线混杂。补 eastAsia
    而不动拉丁轨，残留的数字、公司名保持原字体，中西文各归其位。
    中译英是镜像：中文文档常把 ascii 也设成宋体，英文用宋体拉丁字形很难看，
    换 ascii/hAnsi、不动 eastAsia。
    """
    if not cjk_font and not latin_font and not cs_font:
        return
    for _key, nodes in seg["spans"]:
        for node in nodes:
            text = node.text or ""
            if not text:
                continue
            has_cjk = bool(_DOCX_CJK_RE.search(text))
            has_lat = bool(_DOCX_LATIN_RE.search(text))
            has_cs = bool(_DOCX_THAI_RE.search(text))
            want_ea = cjk_font if has_cjk else ""
            want_lat = latin_font if has_lat else ""
            want_cs = cs_font if has_cs else ""
            if not want_ea and not want_lat and not want_cs:
                continue
            run = node.getparent()
            if run is None:
                continue
            rpr = run.find(f"{_DOCX_W}rPr")
            if rpr is None:
                rpr = run.makeelement(f"{_DOCX_W}rPr", {})
                run.insert(0, rpr)          # rPr 必须是 run 的第一个子元素
            fonts = rpr.find(f"{_DOCX_W}rFonts")
            if fonts is None:
                fonts = rpr.makeelement(f"{_DOCX_W}rFonts", {})
                # OOXML 的 rPr 子元素有序：rFonts 排在 rStyle 之后、其余之前
                style = rpr.find(f"{_DOCX_W}rStyle")
                rpr.insert(rpr.index(style) + 1 if style is not None else 0, fonts)
            if want_ea:
                fonts.set(f"{_DOCX_W}eastAsia", want_ea)
            if want_lat:
                fonts.set(f"{_DOCX_W}ascii", want_lat)
                fonts.set(f"{_DOCX_W}hAnsi", want_lat)
            if want_cs:
                fonts.set(f"{_DOCX_W}cs", want_cs)


# 内置目标语言 → (字体轨, 常用字体候选)。前端菜单与后端轨道判定共用这一份。
# 拉丁文种（葡/西/越/马来）常用字体与英文一致：Times New Roman 正式文书通用，
# 且四种语言的扩展字符都齐；马来官方公文惯用 Arial。泰文首选泰国官方标准
# TH Sarabun New，Windows 内置的 Angsana New 兜底。日文取新旧 Word 的两代默认。
_OFFICE_LANG_FONTS: dict[str, tuple[str, list[str]]] = {
    "中文":      ("eastAsia", ["宋体", "等线", "微软雅黑"]),
    "English":  ("latin",    ["Times New Roman", "Calibri"]),
    "日本語":     ("eastAsia", ["ＭＳ 明朝", "游明朝"]),
    "葡萄牙语":    ("latin",    ["Times New Roman", "Calibri"]),
    "西班牙语":    ("latin",    ["Times New Roman", "Calibri"]),
    "越南语":     ("latin",    ["Times New Roman", "Calibri"]),
    "泰语":      ("cs",       ["TH Sarabun New", "Angsana New"]),
    "马来西亚语":   ("latin",    ["Arial", "Times New Roman"]),
}


def _office_font_track(lang: str, form_track: str = "") -> str:
    """决定所选字体写入哪条轨。显式传入的轨道（自定义语言经 LLM 判定）优先；
    内置语言查表；未知语言回退拉丁轨——世界上大多数文字系统在 OOXML 里走
    ascii/hAnsi，猜错的代价也只是字体声明落错轨、渲染回退，不损内容。"""
    if form_track in ("eastAsia", "latin", "cs"):
        return form_track
    if lang in _OFFICE_LANG_FONTS:
        return _OFFICE_LANG_FONTS[lang][0]
    return "latin"


_DOCX_PART_RE = re.compile(
    r"^word/(document|comments|footnotes|endnotes|header\d+|footer\d+)\.xml$")


async def translate_docx_bytes(data: bytes, translate_batch,
                               progress_cb=None, on_plan=None,
                               target_lang: str = "", cjk_font: str = "",
                               latin_font: str = "", cs_font: str = "") -> tuple[bytes, dict]:
    """翻译一份 DOCX：返回 (新文件字节, 统计)。

    translate_batch: async (list[str]) -> list[str]，与 LLM 解耦，测试注入伪翻译器。
    progress_cb(done, total)：批次粒度进度。
    """
    import io as _io
    import zipfile as _zip

    src = _zip.ZipFile(_io.BytesIO(data))
    part_roots: dict[str, object] = {}
    all_segs: list[dict] = []
    for name in src.namelist():
        if _DOCX_PART_RE.match(name):
            root = _lxml_etree.fromstring(src.read(name))
            part_roots[name] = root
            all_segs.extend(_docx_collect_segments(root))

    todo = [(i, _docx_seg_marked(s)) for i, s in enumerate(all_segs)
            if _docx_should_translate(_docx_seg_text(s))]
    texts = [t for _, t in todo]
    batches = _docx_pack_batches(texts)
    stats = {"segments": len(all_segs), "translated": len(todo),
             "batches": len(batches), "chars": sum(len(t) for t in texts),
             "sample": "\n".join(texts)[:2000],   # 供源语言检测
             "kept_original": 0}                   # 保留原文（未译成）的分段数
    if on_plan is not None:
        on_plan(stats)

    for bi, batch in enumerate(batches):
        chunk = [texts[j] for j in batch]
        try:
            results = await translate_batch(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[docx] 批 %d/%d 翻译失败，该批保留原文: %s",
                        bi + 1, len(batches), e)
            results = chunk
        if not isinstance(results, list) or len(results) != len(chunk):
            log.warning("[docx] 批 %d/%d 返回条数不符（%s != %d），该批保留原文",
                        bi + 1, len(batches),
                        len(results) if isinstance(results, list) else type(results).__name__,
                        len(chunk))
            results = chunk
        for j, translated in zip(batch, results):
            seg = all_segs[todo[j][0]]
            if isinstance(translated, str) and translated.strip() and translated != texts[j]:
                _docx_apply_translation(seg, translated.strip())
                _docx_apply_fonts(seg, cjk_font, latin_font, cs_font)
            elif target_lang and not _docx_lang_ok(texts[j], target_lang):
                # 原文语言本就不是目标语言、译文却与原文相同 → 这段没译成。
                # 纯专有名词分段（lang_ok 放行原文）不算：识别不了「没译」和「不必译」。
                stats["kept_original"] += 1
                log.warning("[docx] 分段保留原文（未译成）: %.60s…", texts[j])
        if progress_cb is not None:
            progress_cb(bi + 1, len(batches))

    out_buf = _io.BytesIO()
    with _zip.ZipFile(out_buf, "w", _zip.ZIP_DEFLATED) as z:
        for info in src.infolist():
            if info.filename in part_roots:
                payload = _lxml_etree.tostring(
                    part_roots[info.filename], xml_declaration=True,
                    encoding="UTF-8", standalone=True)
            else:
                payload = src.read(info.filename)
            z.writestr(info, payload)
    src.close()
    return out_buf.getvalue(), stats


_DOCX_CJK_RE = re.compile(r"[一-鿿]")
_DOCX_LATIN_RE = re.compile(r"[a-zA-Z]")


def _docx_lang_ok(text: str, target_lang: str) -> bool:
    """目标语言校验：目标含中文时，译文不该是长篇纯拉丁文。

    真机实测：模型偶尔把一整批葡语译成英语——JSON 形状完全合法，长度校验拦不住，
    只有看内容才能发现。短拉丁文本放行：纯专有名词（公司名、ANEXO II）按提示词
    本就保留原文，零 CJK 是合法的。
    """
    if "中" not in target_lang:
        return True
    if _DOCX_CJK_RE.search(text):
        return True
    return len(_DOCX_LATIN_RE.findall(text)) <= 15


def _docx_parse_llm_array(content: str, expect: int) -> list[str]:
    """从 LLM 响应中取出同长度的字符串数组。

    真机实测：模型会在 JSON 字符串里输出未转义的引号（译文含「"」时必现），
    json.loads 直接炸。先严格解析，失败再用 json_repair 修——它就是为这类
    LLM 产出的破 JSON 而生的（requirements.txt 显式依赖）。
    """
    content = content.strip()
    start, stop = content.find("["), content.rfind("]")
    if start == -1 or stop <= start:
        raise ValueError("响应里没有 JSON 数组")
    blob = content[start:stop + 1]
    try:
        parsed = json.loads(blob)
    except ValueError:
        import json_repair
        parsed = json_repair.loads(blob)
    if not isinstance(parsed, list) or len(parsed) != expect \
            or not all(isinstance(x, str) for x in parsed):
        raise ValueError(f"JSON 数组形状不符：{type(parsed).__name__} len="
                         f"{len(parsed) if isinstance(parsed, list) else '-'}")
    return parsed


async def _translate_docx_batch_llm(client, texts: list[str],
                                    target_lang: str) -> list[str]:
    """把一批分段文本译为目标语言。JSON 数组进出；两次整批重试后逐条兜底。

    单条失败返回原文——引擎侧「原文保留」是 docx 翻译唯一可接受的失败模式，
    静默丢字或错位都比留一段原文严重得多。
    """
    sys_prompt = (
        f"你是专业法律/商务文档翻译。用户给出一个 JSON 字符串数组，逐项译为{target_lang}，"
        "返回同长度的 JSON 字符串数组，除 JSON 外不输出任何内容。规则：\n"
        "- 每项独立翻译；项内是句子片段时按片段直译，不补全成完整句\n"
        "- 保留 **粗体**、*斜体*、<u>下划线</u>、<sup>上标</sup>、<sub>下标</sub> 标记，"
        "让它们包裹译文中对应的内容\n"
        "- 数字、金额、日期、条款编号、公司名、人名、地址、邮箱、URL 保持原样\n"
        "- 空字符串原样返回空字符串"
    )
    headers = llm_headers()

    async def _call(items: list[str], extra: str = "") -> list[str]:
        body = {
            "model": ACTUAL_MODEL_NAME,
            "messages": [
                {"role": "system", "content": sys_prompt + extra},
                {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
            ],
            "stream": False,
            "temperature": 0.1,
            "max_tokens": 8000,
            **llm_extra_body(),          # 思考开关由档案决定
        }

        # 翻倍重试后仍为空，就让 _docx_parse_llm_array 抛出去走原有的逐条兜底
        content = await _llm_post_with_budget_retry(
            client, body, headers, timeout=LLM_TIMEOUT, gate=True,
            what="[docx] 整批译文")
        # 剥代码围栏，取最外层的 JSON 数组
        return _docx_parse_llm_array(content, len(items))

    # 同事实测（2026-08-27）：两段长英文在中文译稿里原样残留。失败链是模型拒译/
    # 回显 → 护栏拦截 → 普通重译仍英文 → 静默保留原文。所以单条重译分两档：普通
    # 一次，语言仍不对再来一次强化提示词的；彻底失败必须留下日志，不能再静默。
    _STRICT = (f"\n注意：本次输出必须是{target_lang}译文；"
               "除专有名词、编号、代码外不得保留原文句子。")

    async def _one(t: str) -> str:
        """单条翻译；失败保留原文。"""
        try:
            r = (await _call([t]))[0]
            if _docx_lang_ok(r, target_lang):
                return r
            r = (await _call([t], extra=_STRICT))[0]
            if _docx_lang_ok(r, target_lang):
                return r
            log.warning("[docx] 强化重译后译文语言仍不对，保留原文: %.60s…", t)
            return t
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[docx] 单条翻译失败，保留原文（%d 字符）: %s", len(t), e)
            return t

    results: list[str] | None = None
    for attempt in (1, 2):
        try:
            results = await _call(texts)
            break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[docx] 整批翻译第 %d 次失败（%d 条）: %s", attempt, len(texts), e)
    if results is None:
        return [await _one(t) for t in texts]

    # 目标语言校验：整批 JSON 合法但译成了别的语言（真机见过整批葡译英）。
    # 跑偏条目逐条重译；重译仍跑偏就保留原文——decisions 同引擎侧：原文保留是
    # 唯一可接受的失败模式。
    for i, (src, dst) in enumerate(zip(texts, results)):
        if not _docx_lang_ok(dst, target_lang):
            log.info("[docx] 第 %d 条译文语言跑偏（%.40s…），逐条重译", i + 1, dst)
            results[i] = await _one(src)
    return results


# ---------------------------------------------------------------------------
# Markdown 翻译：分块 + 流式翻译
# ---------------------------------------------------------------------------

def split_markdown_chunks(text: str, max_chars: int = 3000) -> list[str]:
    """
    将 Markdown 文本按语义块切分，每块不超过 max_chars 字符。
    规则：
    - 代码块 (```) 和表格整块不切分
    - 以空行为边界合并相邻小段落
    """
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    in_code_block = False

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        # 检测代码块开始/结束
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        # 检测表格行（以 | 开头或包含 |---|）
        is_table = line.strip().startswith("|")

        if in_code_block or is_table:
            # 代码块/表格：收集完整块后再考虑是否需要换块
            block_lines = [line]
            i += 1
            if in_code_block:
                # 收集到代码块结束
                while i < len(lines):
                    block_lines.append(lines[i])
                    if lines[i].strip().startswith("```") and len(lines[i].strip()) <= 4:
                        in_code_block = False
                        i += 1
                        break
                    i += 1
            else:
                # 收集连续表格行
                while i < len(lines) and lines[i].strip().startswith("|"):
                    block_lines.append(lines[i])
                    i += 1

            block_text = "\n".join(block_lines)
            block_len = len(block_text)

            # 超大块（表格或代码块）：按行切分，确保每块不超过 max_chars
            if block_len > max_chars and len(block_lines) > 2:
                flush()
                if is_table:
                    # 表格：每组复制表头（标题行 + 分隔行）
                    header = block_lines[:2]
                    header_len = len("\n".join(header))
                    group: list[str] = list(header)
                    group_len = header_len
                    for row in block_lines[2:]:
                        row_len = len(row)
                        if group_len + row_len + 1 > max_chars and len(group) > 2:
                            chunks.append("\n".join(group))
                            group = list(header)
                            group_len = header_len
                        group.append(row)
                        group_len += row_len + 1
                    if len(group) > 2:
                        chunks.append("\n".join(group))
                else:
                    # 代码块：每段保留原始开头 ``` 标记和结尾 ```
                    opening = block_lines[0]   # e.g. ```python
                    closing = block_lines[-1]  # ```
                    overhead = len(opening) + len(closing) + 2  # 两行换行
                    group_lines: list[str] = [opening]
                    group_len = len(opening)
                    for cl in block_lines[1:-1]:
                        cl_len = len(cl)
                        if group_len + cl_len + overhead + 1 > max_chars and len(group_lines) > 1:
                            group_lines.append(closing)
                            chunks.append("\n".join(group_lines))
                            group_lines = [opening]
                            group_len = len(opening)
                        group_lines.append(cl)
                        group_len += cl_len + 1
                    group_lines.append(closing)
                    chunks.append("\n".join(group_lines))
                continue

            if current_len + block_len > max_chars and current:
                flush()
            current.extend(block_lines)
            current_len += block_len
            continue

        # 普通行
        line_len = len(line)
        if current_len + line_len > max_chars and current:
            flush()

        current.append(line)
        current_len += line_len
        i += 1

    flush()
    # 过滤空块
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# 残留英文检测 & 二次修正（针对模型偶尔保留英文形容词的行为）
# ---------------------------------------------------------------------------

# 允许在中文译文中保留原样的英文词（小写匹配；去掉连字符/撇号后比较）
_ALLOWED_ENGLISH: set[str] = {
    # 协议 / 格式 / 通用技术缩写
    "api", "sdk", "http", "https", "url", "uri", "json", "xml", "yaml", "toml",
    "html", "css", "js", "ts", "sql", "graphql", "rest", "rpc", "grpc",
    "cpu", "gpu", "ram", "rom", "ssd", "hdd", "usb", "hdmi", "wifi", "bluetooth",
    "pdf", "png", "jpg", "jpeg", "gif", "svg", "webp", "bmp", "tiff", "ico",
    "mp3", "mp4", "wav", "avi", "mkv", "mov", "flac", "ogg", "zip", "rar", "tar", "gz",
    "ai", "ml", "llm", "nlp", "ocr", "cv", "vr", "xr", "ui", "ux", "cli", "gui",
    "ide", "ssh", "ftp", "tcp", "udp", "ip", "dns", "cdn", "vpn",
    "oauth", "jwt", "cors", "csp", "dom", "sso",
    "saas", "paas", "iaas", "crud", "mvp", "poc", "kpi", "roi", "rag", "moe", "lora",
    "os", "macos", "ios", "ipados", "tvos", "watchos", "android", "windows", "linux",
    "unix", "ubuntu", "debian", "centos", "fedora", "arch", "redhat",
    "docker", "kubernetes", "aws", "gcp", "azure",
    "csv", "tsv", "epub", "docx", "pptx", "xlsx",
    # 公司 / 品牌 / 产品 / 模型
    "google", "apple", "microsoft", "meta", "facebook", "amazon", "netflix",
    "tesla", "nvidia", "intel", "amd", "arm", "qualcomm", "huawei", "xiaomi",
    "openai", "anthropic", "claude", "gpt", "chatgpt", "kimi", "qwen", "gemini",
    "deepseek", "llama", "mistral", "mixtral", "yi", "grok", "phi",
    # 编程语言
    "python", "javascript", "typescript", "java", "rust", "golang", "ruby",
    "php", "perl", "scala", "kotlin", "swift", "dart", "lua", "julia",
    "haskell", "elixir", "erlang", "clojure", "matlab",
    # 前端框架
    "react", "vue", "angular", "svelte", "solid", "preact", "ember", "jquery",
    "nextjs", "nuxt", "remix", "gatsby", "astro", "sveltekit",
    # 后端框架
    "express", "koa", "fastify", "nestjs", "django", "flask", "fastapi",
    "tornado", "sanic", "rails", "laravel", "symfony", "spring", "springboot",
    "nodejs", "node", "deno", "bun",
    # 工具 / 仓库 / 包管理
    "github", "gitlab", "bitbucket", "gitea",
    "npm", "pnpm", "yarn", "pip", "poetry", "cargo", "gem", "composer",
    "gradle", "maven", "bazel", "make", "cmake", "ninja",
    # 数据库
    "mysql", "postgresql", "postgres", "redis", "mongodb", "sqlite", "oracle",
    "mssql", "cassandra", "dynamodb", "elasticsearch", "clickhouse", "duckdb", "snowflake",
    # 文档格式 / 工具
    "markdown", "latex", "asciidoc",
    "vscode", "vim", "emacs", "sublime", "intellij", "pycharm", "webstorm", "xcode",
    "obsidian", "notion", "logseq",
    # 项目相关
    "mineru", "paddleocr", "paddlepaddle", "pytorch", "tensorflow", "jax",
    "transformers", "huggingface", "langchain", "llamaindex",
    "pymupdf", "fitz", "pillow", "numpy", "pandas", "scipy", "scikit",
    "matplotlib", "seaborn", "plotly",
    "uvicorn", "gunicorn", "nginx", "apache", "caddy", "traefik",
    # Cherry Studio 这种带空格的：分别成词
    "cherry", "studio", "code",
    # 苹果硬件
    "iphone", "ipad", "ipod", "imac", "macbook", "airpods",
}

# 英文词正则：首字母为字母，后跟字母/连字符/撇号，最小总长 3
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")

# CJK 字符后紧跟括号的「中文术语（English 原文）」注释模式：括号内容应保留，不算残留。
# 同时覆盖全角 （） 与半角 ()。允许括号内含字母、数字、空格、连字符、斜杠、逗号、句点。
_PAREN_ANNOTATION_RE = re.compile(
    r"([一-鿿㐀-䶿])\s*[（(][A-Za-z0-9'\-\s/,.À-ɏ]+?[)）]"
)


def _detect_residual_english(text: str) -> list[str]:
    """扫描译文，返回疑似残留未翻译的英文词列表（去重，按出现顺序）。

    判别策略：
      - 剔除代码块、行内代码、URL、HTML 标签后再扫描
      - 命中白名单（不区分大小写）→ 跳过
      - 全大写 2-6 字母（缩写惯例）→ 跳过
      - 首字母大写非全大写（专有名词惯例）→ 跳过
      - 剩余全小写常规英文词 = 疑似残留
      - 若目标语言本来就是英文族（输出非 ASCII 占比 < 20%），则不检测
    """
    if not text:
        return []
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / max(len(text), 1) < 0.2:
        return []

    cleaned = text
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    cleaned = re.sub(r"`[^`\n]*`", " ", cleaned)
    # Markdown 链接/图片：保留显示文字，丢弃 URL 部分
    cleaned = re.sub(r"!?\[([^\]]*)\]\([^)]+\)", r" \1 ", cleaned)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    # 中文术语后紧跟括号注释「中文（English）」：括号内是英文原文标注，不算残留
    cleaned = _PAREN_ANNOTATION_RE.sub(r"\1", cleaned)

    residuals: list[str] = []
    seen: set[str] = set()
    for w in _ENGLISH_WORD_RE.findall(cleaned):
        key = w.lower()
        if key in seen:
            continue
        bare = w.replace("-", "").replace("'", "")
        if bare.lower() in _ALLOWED_ENGLISH:
            seen.add(key)
            continue
        if bare.isupper() and 2 <= len(bare) <= 6:
            seen.add(key)
            continue
        if bare[0].isupper() and not bare.isupper():
            # 首字母大写：推测为专有名词
            seen.add(key)
            continue
        residuals.append(w)
        seen.add(key)
    return residuals


async def _fix_residual_english(
    client: httpx.AsyncClient,
    bad_translation: str,
    residuals: list[str],
    target_lang: str,
) -> str:
    """二次调用 LLM 修正残留英文。非流式，一次拿到完整修正稿。"""
    sample = residuals[:20]
    sample_text = "、".join(sample)
    if len(residuals) > 20:
        sample_text += f" 等共 {len(residuals)} 个"
    fix_prompt = (
        f"以下是机器翻译的中文/{target_lang}稿，但残留了未翻译的英文词：{sample_text}。\n"
        f"请把这些英文词翻译为{target_lang}，其它内容**严格保持原样不动**，包括：\n"
        "  - Markdown 格式、代码块、URL\n"
        "  - 已存在的专有名词与技术缩写\n"
        "  - 中文术语后括号内作为原文标注的英文，如「数据空间（dataspace）」"
        "「抽象（abstraction）」——这是学术写作惯例，括号内的英文必须保留，"
        "**禁止把它再翻译一遍**，否则会出现「数据空间（数据空间）」的错误重复\n"
        "直接输出修正后的完整文本，不要添加任何解释或前后缀。"
    )
    body = {
        "model": ACTUAL_MODEL_NAME,
        "messages": [
            {"role": "system", "content": fix_prompt},
            {"role": "user", "content": bad_translation},
        ],
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 8000,
        **llm_extra_body(),          # 思考开关由档案决定
    }
    headers = llm_headers()

    # gate=True：残留英文修正也是翻译路径的出站调用，同样计入翻译闸门。
    # prefix_heuristic=False：输入输出都是整块译文散文，可能以「1. **任务分工**」
    # 这类合法标题开头；这一步的输入本身已是清洗过的译文，再冒出推理前缀近乎不可能，
    # 而误剥掉首行标题会直接进最终译文与下载文件——两害相权，取不猜。
    content = await _llm_post_with_budget_retry(
        client, body, headers, timeout=LLM_TIMEOUT, gate=True,
        prefix_heuristic=False, what="残留英文修正")
    return content.strip()


def _get_translate_semaphore() -> asyncio.Semaphore:
    """翻译并发闸门；未经 startup（如单测直接调用）时惰性建一个。"""
    global _translate_semaphore
    if _translate_semaphore is None:
        _translate_semaphore = asyncio.Semaphore(TRANSLATE_CONCURRENCY)
    return _translate_semaphore


def _should_trim_leading_ws(chunk: str) -> bool:
    """源块自身不以空白开头时，才抑制译文的前导空白。

    Markdown 的缩进有语义（4 空格代码块、嵌套列表项），而 split_markdown_chunks
    用 "\\n".join(lines) 拼块，首行带缩进时块本身就以空白开头——那种前导空白必须
    原样保留，不能一律 lstrip。
    """
    return bool(chunk) and not chunk[:1].isspace()


def _trim_leading_ws(token: str, trimming: bool) -> tuple[str, bool]:
    """抑制模型在译文最前面多吐的空白，返回 (应输出的 token, 是否仍需抑制)。

    模型几乎总在第一个 token 前带一个空格——实测翻译 "Hi Sheng Tao —" 时首个
    token 就是 ' 嗨'。这是 SentencePiece 类分词器的固有行为（词首空格属于 token
    本身），不是本项目加的：前端拼接首块时没有任何前缀。空格一路透传到界面上，
    表现为译文第一行永远缩进一格。

    只吃掉最前面的连续空白，token 内部与末尾的空格原样保留。
    """
    if not trimming:
        return token, False
    stripped = token.lstrip()
    if stripped:
        return stripped, False
    return "", True


# 源块自身就以「1. **加粗标题**」/「**加粗**」开头时，译文也会——这正好是
# strip_thinking 推理前缀启发式的形状。对这种块必须关掉前缀猜测，否则译文的第一行
# 标题会被当成思考剥掉（静默丢内容比漏一段思考更糟）。
_SOURCE_LIST_LEAD = re.compile(r"^\s*(?:\d+\.\s*\*\*|\*\*)")


def _pending_tag_tail(s: str, tag: str) -> int:
    """s 末尾可能是被 token 边界切开的 tag 前缀，返回该前缀长度（没有则 0）。

    流式分片不保证在标签边界上切：`</think>` 可能拆成 `</th` + `ink>`。不留住这
    半截，`in_think` 状态机要么把它当正文发出去，要么把后面的全部译文当思考丢掉。
    """
    for k in range(min(len(tag) - 1, len(s)), 0, -1):
        if s.endswith(tag[:k]):
            return k
    return 0


async def translate_chunk_stream(
    client: httpx.AsyncClient,
    chunk: str,
    chunk_idx: int,
    total_chunks: int,
    target_lang: str,
    max_tokens: int = TRANSLATE_MAX_TOKENS,
):
    """
    流式翻译单个 Markdown 块。
    逐 token yield 字符串，以 None 表示结束。
    """
    system_prompt = (
        f"你是专业翻译。请将用户提供的 Markdown 内容译为{target_lang}。/no_think\n"
        "\n"
        "【核心翻译规则】\n"
        f"所有英文单词都必须译为{target_lang}，包括形容词、副词、复合修饰语。"
        "不允许在译文中保留任何「形容性 / 描述性」的英文词。\n"
        "仅以下情况可以保留英文：\n"
        "  (1) 代码块（```...```）与行内代码（`...`）内的代码本体\n"
        "  (2) URL 链接地址\n"
        "  (3) 通用技术缩写：API、SDK、HTTP、HTTPS、URL、JSON、XML、HTML、CSS、JS、"
        "SQL、CPU、GPU、RAM、ROM、USB、PDF、AI、ML、LLM、UI、UX、IDE、CLI、IP、DNS、"
        "TCP、UDP、OCR、RAG 等\n"
        "  (4) 公司 / 产品 / 技术专有名词：Google、Apple、Microsoft、OpenAI、Kimi、"
        "Qwen、Linux、macOS、Python、JavaScript、React、FastAPI、MinerU、PaddleOCR、"
        "GitHub、PyTorch 等\n"
        "  (5) 中文译名后括号内的英文原文标注（学术惯例），如「数据空间（dataspace）」"
        "「抽象（abstraction）」——括号内的英文必须保留，**禁止再翻译一遍**\n"
        "\n"
        "【反例对照（必须遵守）】\n"
        "❌「取得了 spectacular 的成果」  → ✅「取得了惊人的成果」\n"
        "❌「rapidly-expanding 的需求」    → ✅「快速扩张的需求」\n"
        "❌「这是一个 elegant 的方案」     → ✅「这是一个优雅的方案」\n"
        "❌「在 cutting-edge 领域」        → ✅「在前沿领域」\n"
        "❌「具备 groundbreaking 意义」    → ✅「具有开创性意义」\n"
        "\n"
        "【格式规则】\n"
        "1. 保留所有 Markdown 格式符号（#、**、*、`、|、>、- 等）\n"
        "2. 代码块（```...```）内的代码不翻译，只翻译注释\n"
        "3. 链接格式 [文字](url) 中只翻译文字，保留 url 原样\n"
        "4. 表格结构保持不变，只翻译单元格内容\n"
        "5. 直接输出翻译结果，不要添加任何解释"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chunk},
    ]
    headers = llm_headers()
    # max_tokens：输入最多 3000 字，翻译后一般不超过 2x，6000 token 足够；
    # 不设上限时模型可能进入循环生成，导致单块耗时 10+ 分钟。整块译文为空时
    # 调用方（_process_chunk）会带翻倍的 max_tokens 重发一次。
    body = {
        "model": ACTUAL_MODEL_NAME,
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
        "max_tokens": max_tokens,
        **llm_extra_body(),          # 思考开关由档案决定
    }

    in_think = False   # 过滤 <think>...</think> 块（Kimi / Qwen / GLM 思维链输出）
    think_buf = ""
    # 模型几乎总在译文最前面多吐一个空格，界面上表现为第一行缩进一格。
    # 源块自身带缩进时（代码块/嵌套列表）不动它，见 _should_trim_leading_ws。
    trimming = _should_trim_leading_ws(chunk)
    # 思考泄漏诊断计数器：>0 即说明服务端未按档案关掉/压低思考
    think_chars_stripped = 0   # 被剥掉的 <think>...</think>（含标签自身）字符数
    reasoning_chars = 0        # delta.reasoning_content 字段累计长度（正常形态，忽略即可）
    head_discards = 0          # 首段缓冲被判定为推理、整段丢弃的次数
    head_discarded_chars = 0
    head_partial_strips = 0    # 首段缓冲被剥掉了前缀但仍有正文的次数
    head_partial_chars = 0
    stream_finish_reason = None   # 末条 SSE 事件的 choices[0].finish_reason
    warned_length = False         # 截断告警每块只发一次（_finalize 可能被走到两次）
    leak_logged = False
    # 首段缓冲：无标签的中文推理（「1. **拆解用户请求：**…」）没有任何标签可认，
    # 逐 token 流出去就再也收不回来。先攒够 STREAM_HEAD_CHARS 个真实字符（或等到流
    # 结束）再用 strip_thinking 判一次：判为推理就整段丢弃并告警，判为正文就整段发出，
    # 此后逐 token 直通——延迟只付一次，代价由 _process_chunk 的占位符补上。
    head_buf = ""
    head_flushed = False
    head_heuristic = not _SOURCE_LIST_LEAD.match(chunk)

    def _log_thinking_leak() -> None:
        """告警只从 _finalize() 出去一次——流的五个出口（[DONE]、自然结束、流内
        错误事件、超时、通用异常）都经过它，任何一个都可能刚丢掉一整段推理。"""
        nonlocal leak_logged
        # 阈值只用于 <think> 统计（标签自身 15 字符 + 一点容差）；首段缓冲被判为
        # 推理是整段内容被丢掉，多短都要报——长度不该成为沉默的理由。
        if leak_logged or not (head_discards or head_partial_strips
                               or think_chars_stripped > 20 or reasoning_chars > 0):
            return
        leak_logged = True
        log.warning(
            "第 %d/%d 块检测到思考泄漏: 丢弃推理首段 %d 次（%d 字符）, "
            "首段剥前缀 %d 次（%d 字符）, <think> 段剥离 %d 字符, "
            "reasoning_content 累计 %d 字符。"
            "档案 %s 可能与服务端实际模型不符，请检查 LLM_PROFILE。",
            chunk_idx, total_chunks, head_discards, head_discarded_chars,
            head_partial_strips, head_partial_chars,
            think_chars_stripped, reasoning_chars, LLM_PROFILE,
        )

    def _flush_head() -> str:
        """结算首段缓冲，返回应当下发的文本（判为推理则返回空串）。"""
        nonlocal head_buf, head_flushed, head_discards, head_discarded_chars
        nonlocal head_partial_strips, head_partial_chars
        head_flushed = True
        raw, head_buf = head_buf, ""
        if not raw:
            return ""
        cleaned = strip_thinking(raw, prefix_heuristic=head_heuristic)
        if not cleaned:
            if not raw.strip():
                return raw          # 纯空白不是推理，原样发出，别谎报泄漏
            head_discards += 1
            head_discarded_chars += len(raw)
            return ""
        if cleaned == raw.strip():
            # 没命中推理前缀，只差两端空白——原样发出，别动 Markdown 的首尾换行
            return raw
        # 剥掉了前缀但还剩正文：也是泄漏（或误伤），同样要进告警，不能只有整段丢弃才报
        head_partial_strips += 1
        head_partial_chars += len(raw.strip()) - len(cleaned)
        log.debug("第 %d/%d 块首段剥掉的前缀 = %r", chunk_idx, total_chunks,
                  raw.strip()[:len(raw.strip()) - len(cleaned)][:400])
        return cleaned

    def _accept(text: str) -> str:
        """把一段清洗后的正文送进首段缓冲，返回应当下发的文本。"""
        nonlocal head_buf
        if head_flushed:
            return text
        head_buf += text
        if len(head_buf) < STREAM_HEAD_CHARS:
            return ""
        return _flush_head()

    def _finalize() -> str:
        """流结束/中断时收尾：残留的 think_buf + 未结算的首段缓冲，并出泄漏告警。

        五个出口都调它，所以告警放在这里就一定发得出去——首段缓冲判为推理被丢掉
        的那些块，此前只有正常结束的两条路会告警，超时/异常路径是彻底静默的。"""
        nonlocal think_buf, think_chars_stripped, trimming, warned_length
        out = ""
        if think_buf:
            if in_think:
                think_chars_stripped += len(think_buf)
            else:
                text, trimming = _trim_leading_ws(think_buf, trimming)
                if text:
                    out += _accept(text)
            think_buf = ""
        if not head_flushed:
            out += _flush_head()
        if stream_finish_reason == "length" and not warned_length:
            warned_length = True
            # 流式不重跑：已经流给用户的 token 收不回来，重跑只会让同一块出现两遍。
            # 只留一条告警，让人知道这块译文是被 max_tokens 掐断的，不是模型偷懒。
            log.warning("第 %d/%d 块 finish_reason=length（max_tokens=%d），译文可能不完整。"
                        "整块为空时 _process_chunk 会按翻倍额度重发；非空截断不重发，"
                        "因为已下发的 token 收不回来。",
                        chunk_idx, total_chunks, max_tokens)
        _log_thinking_leak()
        return out

    # 翻译闸门必须罩住整个流，而不只是发起请求：stream=True 时 send() 一拿到响应头就
    # 返回，正文读取发生在其后。此前信号量只包住 send，限的是「发起」的瞬时并发——
    # 实测上限设 3 时同时在途仍有 8 个流。这里手动 acquire/release 而不用 async with，
    # 是为了不必把下面整段重新缩进；生成器被提前关闭时 finally 同样会执行。
    _sem = _get_translate_semaphore()
    await _sem.acquire()
    try:
        # 网关（尤其上游 vLLM 过载时）会间歇性返回 502，通常几秒后自愈：重试兜底
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            async with _api_semaphore:
                req = client.build_request("POST", TARGET_API_URL, json=body, headers=headers)
                resp = await client.send(req, stream=True)
            if resp.status_code == 200:
                break
            err_body = (await resp.aread()).decode("utf-8", errors="replace")[:300]
            await resp.aclose()
            log.warning("翻译块 %d/%d: LLM 返回 HTTP %d (尝试 %d/%d): %s",
                        chunk_idx, total_chunks, resp.status_code,
                        attempt, max_retries, err_body)
            if attempt < max_retries:
                await asyncio.sleep(2 * attempt)
                continue
            log.error("翻译块 %d/%d: 重试 %d 次后仍失败 (HTTP %d)",
                      chunk_idx, total_chunks, max_retries, resp.status_code)
            yield f"\n[第 {chunk_idx}/{total_chunks} 块翻译失败: HTTP {resp.status_code} {err_body}]"
            return

        buffer = ""

        async for raw in resp.aiter_bytes():
            buffer += raw.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    tail = _finalize()
                    if tail:
                        yield tail
                    return
                try:
                    obj = json.loads(data)
                    if "error" in obj and "choices" not in obj:
                        err_msg = str(obj["error"])[:300]
                        log.error("翻译块 %d/%d: 流内错误事件: %s",
                                  chunk_idx, total_chunks, err_msg)
                        tail = _finalize()      # 已攒下的正文先发出，别连同错误一起丢
                        if tail:
                            yield tail
                        yield f"\n[第 {chunk_idx}/{total_chunks} 块翻译失败: {err_msg}]"
                        await resp.aclose()
                        return
                    # 末条事件带 finish_reason、delta 为空（实测网关确实发这个字段）。
                    # 必须在下面 `if not token: continue` 之前读，否则永远读不到。
                    fr = obj["choices"][0].get("finish_reason")
                    if fr:
                        stream_finish_reason = fr
                    delta = obj["choices"][0].get("delta", {})
                    reasoning = delta.get("reasoning_content") or ""
                    if reasoning:
                        reasoning_chars += len(reasoning)
                    token = delta.get("content", "")
                    if not token:
                        continue
                    # 过滤 <think>...</think>，同时统计被剥字符数。标签可能被 token
                    # 边界切开（"</th" + "ink>"），所以找不到时要把末尾那截可能是
                    # 半个标签的字符留在 think_buf 里等下一个 token——旧实现直接丢弃/
                    # 直接下发，正好在 8 字符一片的流上把整块译文吃掉。
                    think_buf += token
                    while True:
                        if in_think:
                            end = think_buf.find("</think>")
                            if end == -1:
                                hold = _pending_tag_tail(think_buf, "</think>")
                                think_chars_stripped += len(think_buf) - hold
                                think_buf = think_buf[len(think_buf) - hold:]
                                break
                            think_chars_stripped += end + len("</think>")
                            think_buf = think_buf[end + len("</think>"):]
                            in_think = False
                        else:
                            start = think_buf.find("<think>")
                            if start == -1:
                                hold = _pending_tag_tail(think_buf, "<think>")
                                ready = think_buf[:len(think_buf) - hold]
                                think_buf = think_buf[len(think_buf) - hold:]
                                out, trimming = _trim_leading_ws(ready, trimming)
                                if out:
                                    out = _accept(out)
                                    if out:
                                        yield out
                                break
                            if start > 0:
                                out, trimming = _trim_leading_ws(think_buf[:start], trimming)
                                if out:
                                    out = _accept(out)
                                    if out:
                                        yield out
                            think_chars_stripped += len("<think>")
                            think_buf = think_buf[start + len("<think>"):]
                            in_think = True
                except Exception:
                    pass
        await resp.aclose()
        tail = _finalize()
        if tail:
            yield tail

    except httpx.TimeoutException:
        tail = _finalize()
        if tail:
            yield tail
        yield f"\n[第 {chunk_idx}/{total_chunks} 块翻译超时]"
    except Exception as e:
        log.exception("翻译块 %d 异常", chunk_idx)
        tail = _finalize()
        if tail:
            yield tail
        yield f"\n[第 {chunk_idx}/{total_chunks} 块翻译失败: {e}]"
    finally:
        _sem.release()


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------

OFFICE_PAGE_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>金诚同达 · Office 文档翻译</title>
<link rel="icon" type="image/png" href="/static/jtn-icon.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@200;300;400;500;700;900&display=swap" rel="stylesheet">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  :root{
    --red:rgb(188,27,41); --red-strong:rgb(166,22,35); --red-tint:rgba(188,27,41,.05);
    --gray:rgb(102,100,100); --ink:#2c2b2b;
    --line:rgba(102,100,100,.22); --line-soft:rgba(102,100,100,.12);
    --muted:rgba(102,100,100,.66); --bg:#fff;
  }
  html{ -webkit-font-smoothing:antialiased; }
  body{ font-family:'Noto Sans SC',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    background:var(--bg); color:var(--gray); min-height:100vh; display:flex; flex-direction:column; }
  body::before{ content:""; position:fixed; top:0; left:0; right:0; height:3px; background:var(--red); z-index:60; }
  .nav{ background:#fff; border-bottom:1px solid var(--line);
    display:flex; align-items:center; padding:0 40px; height:74px; gap:30px; }
  .nav-logo{ height:34px; width:auto; display:block; }
  .nav-sep{ flex:1; }
  .nav a{ color:var(--gray); text-decoration:none; font-size:13px; font-weight:500; letter-spacing:.04em; transition:color .2s; }
  .nav a:hover{ color:var(--ink); }
  .nav a.active{ color:var(--red); }
  .nav a.active::before{ content:""; display:inline-block; width:7px; height:7px; background:var(--red); margin-right:9px; vertical-align:1px; }
  .container{ flex:1; max-width:1200px; width:100%; margin:0 auto; padding:48px 40px; }
  .hero{ margin-bottom:32px; }
  .overline{ display:flex; align-items:center; gap:11px; margin-bottom:16px; }
  .overline .sq{ width:9px; height:9px; background:var(--red); flex:none; }
  .overline em{ font-style:normal; font-size:12px; letter-spacing:.30em; color:var(--red); font-weight:700; text-transform:uppercase; }
  h1{ font-size:32px; font-weight:300; color:var(--ink); line-height:1.2; margin-bottom:12px; }
  h1 b{ font-weight:700; }
  .subtitle{ color:var(--muted); font-size:14px; line-height:1.8; }
  .toolbar{ display:flex; gap:14px; align-items:center; margin-bottom:22px; flex-wrap:wrap; }
  .toolbar label{ font-size:13px; color:var(--gray); }
  select, input[type=text]{ background:#fff; border:1px solid var(--line); color:var(--gray);
    padding:8px 12px; font-size:13px; outline:none; font-family:inherit; }
  select:focus, input[type=text]:focus{ border-color:var(--red); }
  #customLangWrap{ display:none; }
  #customLang{ width:140px; }
  #fontHint{ font-size:12px; color:var(--muted); }
  .btn{ padding:9px 22px; border:1px solid transparent; cursor:pointer; font-size:13px;
    font-weight:500; font-family:inherit; letter-spacing:.03em; transition:all .18s; }
  .btn-primary{ background:var(--red); color:#fff; }
  .btn-primary:hover{ background:var(--red-strong); }
  .btn-primary:disabled{ background:var(--line); color:var(--muted); cursor:not-allowed; }
  .btn-secondary{ background:#fff; color:var(--gray); border-color:var(--line); }
  .btn-secondary:hover{ border-color:var(--gray); color:var(--ink); }
  .drop-zone{ border:1.5px dashed var(--line); background:#fff; padding:48px 20px;
    text-align:center; cursor:pointer; transition:all .2s; margin-bottom:18px; }
  .drop-zone:hover, .drop-zone.drag-over{ border-color:var(--red); background:var(--red-tint); }
  .drop-zone input{ display:none; }
  .drop-zone p{ font-size:15px; color:var(--ink); font-weight:500; }
  .drop-zone small{ font-size:12px; color:var(--muted); margin-top:8px; display:block; }
  .file-queue{ background:#fff; border:1px solid var(--line); padding:14px 18px; margin-bottom:18px; }
  .queue-header{ font-size:12px; color:var(--muted); margin-bottom:10px; letter-spacing:.06em; }
  .queue-item{ display:flex; align-items:center; gap:10px; padding:7px 0; border-bottom:1px solid var(--line-soft); }
  .queue-item:last-child{ border-bottom:none; }
  .queue-name{ flex:1; font-size:13px; color:var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .queue-size{ font-size:11px; color:var(--muted); white-space:nowrap; }
  .btn-remove{ background:none; border:none; color:var(--muted); cursor:pointer; font-size:18px; line-height:1; padding:0 4px; }
  .btn-remove:hover{ color:var(--red); }
  .file-card{ background:#fff; border:1px solid var(--line); border-left:3px solid var(--red); padding:18px 22px; margin-bottom:14px; }
  .file-card-header{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; gap:12px; }
  .file-name{ font-size:14px; font-weight:700; color:var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:68%; }
  .file-status{ font-size:12px; color:var(--red); font-weight:500; }
  .file-status.done{ color:var(--gray); }
  .card-progress-bg{ width:100%; height:3px; background:var(--line-soft); margin-bottom:12px; }
  .card-progress{ height:100%; background:var(--red); transition:width .3s; width:0%; }
  .btn-dl{ background:#fff; color:var(--gray); padding:8px 16px; font-size:12px;
    border:1px solid var(--line); cursor:pointer; font-family:inherit; }
  .btn-dl:hover{ border-color:var(--gray); color:var(--ink); }
  .spinner{ display:inline-block; width:11px; height:11px; border:2px solid var(--red);
    border-top-color:transparent; border-radius:50%; animation:spin .8s linear infinite;
    vertical-align:middle; margin-right:7px; }
  @keyframes spin{ to{ transform:rotate(360deg); } }
  .foot{ border-top:1px solid var(--line); padding:24px 40px; display:flex;
    align-items:center; justify-content:space-between; gap:16px; }
  .foot img{ height:22px; width:auto; }
  .foot span{ font-size:11px; color:var(--muted); letter-spacing:.05em; }
</style>
</head>
<body>

<nav class="nav">
  <img class="nav-logo" src="/static/jtn-logo.png" alt="JT&N 金诚同达">
  <span class="nav-sep"></span>
  <a href="/">文档解析</a>
  <a href="/translate">Markdown 翻译</a>
  <a href="/office" class="active">Office 文档翻译</a>
</nav>

<div class="container">
  <div class="hero">
    <div class="overline"><span class="sq"></span><em>Office Translation · 办公文档翻译</em></div>
    <h1>Office <b>文档翻译</b></h1>
    <p class="subtitle">上传 Word 文档（.docx），保留全部格式、修订（track changes）与批注，译毕以 .docx 下载。PPTX 与 XLSX 支持即将加入。</p>
  </div>

  <div class="toolbar">
    <label>目标语言</label>
    <select id="langSelect" onchange="onLangChange()">
      <option value="中文">中文</option>
      <option value="English">English</option>
      <option value="日本語">日本語</option>
      <option value="葡萄牙语">葡萄牙语</option>
      <option value="西班牙语">西班牙语</option>
      <option value="越南语">越南语</option>
      <option value="泰语">泰语</option>
      <option value="马来西亚语">马来西亚语</option>
      <option value="custom">自定义...</option>
    </select>
    <span id="customLangWrap">
      <input type="text" id="customLang" placeholder="输入语言名称" oninput="onCustomLangInput()">
    </span>
    <label id="fontLabel">译文字体</label>
    <select id="fontSelect"></select>
    <span id="fontHint"></span>
    <button class="btn btn-primary" id="goBtn" onclick="startOffice()">开始翻译</button>
    <button class="btn btn-secondary" id="cancelBtn" style="display:none" onclick="cancelJob()">取消</button>
    <button class="btn btn-secondary" onclick="clearAll()">清空</button>
  </div>

  <div class="drop-zone" id="dropZone">
    <p>将 .docx 文件拖到这里，或点击选择</p>
    <small>支持同时选择多个文件；修订、批注、页眉页脚、脚注尾注一并翻译并原样保留</small>
    <input type="file" id="docxInput" accept=".docx" multiple>
  </div>
  <div class="file-queue" id="fileQueue" style="display:none">
    <div class="queue-header" id="queueHeader"></div>
    <div id="queueList"></div>
  </div>
  <div id="fileCards"></div>
</div>

<footer class="foot">
  <img src="/static/jtn-logo.png" alt="JT&N 金诚同达">
  <span>金诚同达律师事务所　·　Doxify 文档智能工具</span>
</footer>

<script src="/static/job-client.js"></script>
<script>
// 内置语言 → { track, fonts }（与后端 _OFFICE_LANG_FONTS 同源注入）
const FONT_MAP = __FONT_MAP__;
const OFFICE_JOB_KEY = 'doxify_office_job';
let selectedFiles = [];
let fileStates = {};
let currentJobId = null;
let isRunning = false;
let customTrack = 'latin';        // 自定义语言经 /office/font_suggest 判定的轨道
let customQuerySeq = 0;           // 防过期响应覆盖新输入

function fillFontSelect(fonts, keep) {
  const sel = document.getElementById('fontSelect');
  const prev = keep ? sel.value : null;
  sel.innerHTML = '';
  for (const f of fonts.concat(['不调整'])) {
    const o = document.createElement('option');
    o.value = f; o.textContent = f;
    sel.appendChild(o);
  }
  if (prev && [...sel.options].some(o => o.value === prev)) sel.value = prev;
  sel.disabled = false;
}

function onLangChange() {
  const v = document.getElementById('langSelect').value;
  const hint = document.getElementById('fontHint');
  document.getElementById('customLangWrap').style.display = (v === 'custom') ? 'inline' : 'none';
  hint.textContent = '';
  if (v === 'custom') {
    fillFontSelect([]);
    onCustomLangInput();
  } else {
    fillFontSelect(FONT_MAP[v].fonts);
  }
}

let customTimer = null;
function onCustomLangInput() {
  clearTimeout(customTimer);
  const lang = document.getElementById('customLang').value.trim();
  if (!lang) { fillFontSelect([]); return; }
  customTimer = setTimeout(async () => {
    const seq = ++customQuerySeq;
    const sel = document.getElementById('fontSelect');
    sel.innerHTML = '<option>查询字体中…</option>';
    sel.disabled = true;
    document.getElementById('fontHint').textContent = '';
    try {
      const fd = new FormData(); fd.append('lang', lang);
      const r = await (await fetch('/office/font_suggest', { method: 'POST', body: fd })).json();
      if (seq !== customQuerySeq) return;   // 输入已变化，丢弃过期结果
      customTrack = r.track || 'latin';
      fillFontSelect(r.fonts && r.fonts.length ? r.fonts : ['Times New Roman']);
      document.getElementById('fontHint').textContent = `已按「${lang}」推荐字体`;
    } catch (e) {
      if (seq !== customQuerySeq) return;
      customTrack = 'latin';
      fillFontSelect(['Times New Roman']);
      document.getElementById('fontHint').textContent = '字体查询失败，已给通用回退';
    }
  }, 600);
}

function getTargetLang() {
  const v = document.getElementById('langSelect').value;
  if (v === 'custom') return document.getElementById('customLang').value.trim();
  return v;
}

// ── 文件选择 ──
const dropZone = document.getElementById('dropZone');
const docxInput = document.getElementById('docxInput');
dropZone.addEventListener('click', () => docxInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  addFiles([...e.dataTransfer.files]);
});
docxInput.addEventListener('change', () => { addFiles([...docxInput.files]); docxInput.value = ''; });

function addFiles(list) {
  for (const f of list) {
    if (/\\.docx$/i.test(f.name) &&
        !selectedFiles.find(sf => sf.name === f.name && sf.size === f.size)) {
      selectedFiles.push(f);
    }
  }
  renderQueue();
}
function removeFile(i) { selectedFiles.splice(i, 1); renderQueue(); }
function renderQueue() {
  const q = document.getElementById('fileQueue');
  if (!selectedFiles.length) { q.style.display = 'none'; return; }
  q.style.display = '';
  document.getElementById('queueHeader').textContent = `待翻译 ${selectedFiles.length} 个文件`;
  document.getElementById('queueList').innerHTML = selectedFiles.map((f, i) => `
    <div class="queue-item">
      <span class="queue-name">${f.name}</span>
      <span class="queue-size">${(f.size/1024).toFixed(0)} KB</span>
      <button class="btn-remove" onclick="removeFile(${i})">×</button>
    </div>`).join('');
}

// ── 卡片 ──
function createCard(fileId, filename) {
  fileStates[fileId] = { filename, total: 1, done: 0 };
  const card = document.createElement('div');
  card.className = 'file-card'; card.id = 'card-' + fileId;
  card.innerHTML = `
    <div class="file-card-header">
      <span class="file-name" title="${filename}">${filename}</span>
      <span class="file-status" id="status-${fileId}"><span class="spinner"></span>排队中...</span>
    </div>
    <div class="card-progress-bg"><div class="card-progress" id="bar-${fileId}"></div></div>
    <div id="btns-${fileId}" style="display:none">
      <button class="btn-dl" onclick="downloadCard('${fileId}')">下载 .docx</button>
    </div>`;
  document.getElementById('fileCards').appendChild(card);
}
function downloadCard(fileId) {
  const st = fileStates[fileId];
  if (st?.downloadUrl) window.location.href = st.downloadUrl;
}

function handleEvent(evt) {
  if (evt.type === 'init') {
    for (const f of evt.files) createCard(f.file_id, f.filename);
  } else if (evt.type === 'file_start') {
    const st = fileStates[evt.file_id]; if (!st) return;
    st.total = evt.total_chunks || 1;
    const sta = document.getElementById('status-' + evt.file_id);
    if (sta) sta.innerHTML = `<span class="spinner"></span>0/${st.total} 批`;
  } else if (evt.type === 'chunk_done') {
    const st = fileStates[evt.file_id]; if (!st) return;
    st.done++;
    const pct = Math.round(st.done / st.total * 100);
    const bar = document.getElementById('bar-' + evt.file_id);
    const sta = document.getElementById('status-' + evt.file_id);
    if (bar) bar.style.width = pct + '%';
    if (sta) sta.innerHTML = `<span class="spinner"></span>${st.done}/${st.total} 批`;
  } else if (evt.type === 'file_done') {
    const st = fileStates[evt.file_id]; if (!st) return;
    st.downloadUrl = evt.download_url;
    const bar = document.getElementById('bar-' + evt.file_id);
    const sta = document.getElementById('status-' + evt.file_id);
    const bts = document.getElementById('btns-' + evt.file_id);
    if (bar) bar.style.width = '100%';
    if (sta) {
      sta.textContent = (evt.kept_original > 0)
        ? `完成（${evt.kept_original} 段未译，保留原文）` : '完成';
      sta.className = 'file-status done';
    }
    if (bts) bts.style.display = '';
  } else if (evt.type === 'file_error') {
    const sta = document.getElementById('status-' + evt.file_id);
    const bar = document.getElementById('bar-' + evt.file_id);
    if (sta) { sta.textContent = '失败: ' + (evt.error || '未知错误'); }
    if (bar) bar.style.background = '#b91c1c';
  }
}

function attachOfficeJob(jobId) {
  currentJobId = jobId;
  document.getElementById('cancelBtn').style.display = '';
  const btn = document.getElementById('goBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>翻译中...';
  function reset(msg) {
    currentJobId = null; isRunning = false;
    document.getElementById('cancelBtn').style.display = 'none';
    btn.disabled = false; btn.textContent = msg || '开始翻译';
  }
  return JobClient.attach(OFFICE_JOB_KEY, jobId, {
    onEvent: handleEvent,
    onEnd: (status) => {
      reset();
      if (status && status !== 'done') {
        for (const fid in fileStates) {
          const sta = document.getElementById('status-' + fid);
          if (sta && !sta.classList.contains('done') && !sta.textContent.startsWith('失败'))
            sta.textContent = (status === 'cancelled') ? '已取消' : '失败';
        }
      }
    },
    onGone: () => reset(),
    onGiveUp: () => reset('连接失败，请重试'),
  });
}

async function startOffice() {
  if (isRunning || currentJobId) return;
  const lang = getTargetLang();
  if (!lang) { alert('请输入目标语言'); return; }
  if (!selectedFiles.length) { alert('请先选择 .docx 文件'); return; }
  const sel = document.getElementById('langSelect').value;
  const track = (sel === 'custom') ? customTrack : FONT_MAP[sel].track;
  const fd = new FormData();
  fd.append('target_lang', lang);
  fd.append('docx_font', document.getElementById('fontSelect').value || '不调整');
  fd.append('docx_font_track', track);
  for (const f of selectedFiles) fd.append('files', f);
  document.getElementById('fileCards').innerHTML = '';
  fileStates = {};
  document.getElementById('fileQueue').style.display = 'none';
  isRunning = true;
  try {
    const resp = await fetch('/jobs/translate', { method: 'POST', body: fd });
    const body = await resp.json();
    if (!body.job_id) { alert(body.error || '启动失败'); isRunning = false; return; }
    JobClient.save(OFFICE_JOB_KEY, body.job_id);
    selectedFiles = [];
    await attachOfficeJob(body.job_id);
  } catch (e) {
    alert('请求失败: ' + e.message); isRunning = false;
  }
}

function cancelJob() { if (currentJobId) JobClient.cancel(currentJobId); }
function clearAll() {
  if (currentJobId) return;
  selectedFiles = []; fileStates = {};
  renderQueue();
  document.getElementById('fileCards').innerHTML = '';
}

document.addEventListener('DOMContentLoaded', () => {
  onLangChange();
  const jobId = JobClient.restore(OFFICE_JOB_KEY);
  if (jobId) attachOfficeJob(jobId);
});
</script>
</body>
</html>
"""


TRANSLATE_PAGE_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>金诚同达 · Markdown 翻译</title>
<link rel="icon" type="image/png" href="/static/jtn-icon.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@200;300;400;500;700;900&display=swap" rel="stylesheet">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  :root{
    --red:rgb(188,27,41); --red-strong:rgb(166,22,35); --red-tint:rgba(188,27,41,.05);
    --gray:rgb(102,100,100); --ink:#2c2b2b;
    --line:rgba(102,100,100,.22); --line-soft:rgba(102,100,100,.12);
    --muted:rgba(102,100,100,.66); --bg:#fff; --panel:#fafafa;
  }
  html{ -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility; }
  body{ font-family:'Noto Sans SC',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    background:var(--bg); color:var(--gray); min-height:100vh; font-weight:400;
    display:flex; flex-direction:column; }
  body::before{ content:""; position:fixed; top:0; left:0; right:0; height:3px; background:var(--red); z-index:60; }

  .nav{ background:#fff; border-bottom:1px solid var(--line);
    display:flex; align-items:center; padding:0 40px; height:74px; gap:30px; }
  .nav-logo{ height:34px; width:auto; display:block; }
  .nav-sep{ flex:1; }
  .nav a{ color:var(--gray); text-decoration:none; font-size:13px; font-weight:500; letter-spacing:.04em; transition:color .2s; }
  .nav a:hover{ color:var(--ink); }
  .nav a.active{ color:var(--red); }
  .nav a.active::before{ content:""; display:inline-block; width:7px; height:7px; background:var(--red); margin-right:9px; vertical-align:1px; }

  .container{ flex:1; max-width:1200px; width:100%; margin:0 auto; padding:48px 40px; }
  .hero{ position:relative; margin-bottom:32px; }
  .overline{ display:flex; align-items:center; gap:11px; margin-bottom:16px; }
  .overline .sq{ width:9px; height:9px; background:var(--red); flex:none; }
  .overline em{ font-style:normal; font-size:12px; letter-spacing:.30em; color:var(--red); font-weight:700; text-transform:uppercase; }
  h1{ font-size:32px; font-weight:300; color:var(--ink); letter-spacing:.01em; line-height:1.2; margin-bottom:12px; }
  h1 b{ font-weight:700; }
  .subtitle{ color:var(--muted); font-size:14px; line-height:1.8; }

  .toolbar{ display:flex; gap:14px; align-items:center; margin-bottom:22px; flex-wrap:wrap; }
  .toolbar label{ font-size:13px; color:var(--gray); }
  select, input[type=text]{ background:#fff; border:1px solid var(--line); color:var(--gray);
    padding:8px 12px; font-size:13px; outline:none; font-family:inherit; }
  select:focus, input[type=text]:focus{ border-color:var(--red); }
  #customLangWrap{ display:none; }
  #customLang{ width:140px; }

  .btn{ padding:9px 22px; border:1px solid transparent; cursor:pointer; font-size:13px;
    font-weight:500; font-family:inherit; letter-spacing:.03em; transition:all .18s; }
  .btn-primary{ background:var(--red); color:#fff; }
  .btn-primary:hover{ background:var(--red-strong); }
  .btn-primary:disabled{ background:var(--line); color:var(--muted); cursor:not-allowed; }
  .btn-secondary{ background:#fff; color:var(--gray); border-color:var(--line); }
  .btn-secondary:hover{ border-color:var(--gray); color:var(--ink); }

  .input-tabs{ display:flex; margin-bottom:22px; border:1px solid var(--line); width:fit-content; }
  .tab{ padding:8px 22px; border:none; border-right:1px solid var(--line); font-size:12px;
    cursor:pointer; background:#fff; color:var(--muted); letter-spacing:.04em; transition:all .18s; }
  .tab:last-child{ border-right:none; }
  .tab.active{ background:var(--red); color:#fff; }

  /* ── 粘贴模式 ── */
  .editor-row{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  .editor-pane{ display:flex; flex-direction:column; }
  .pane-header{ display:flex; justify-content:space-between; align-items:center;
    margin-bottom:10px; min-height:34px; }
  .pane-title{ font-size:12px; color:var(--ink); font-weight:700; text-transform:uppercase;
    letter-spacing:.12em; display:flex; align-items:center; gap:8px; }
  .pane-title::before{ content:""; width:7px; height:7px; background:var(--red); }
  .pane-actions{ display:flex; gap:6px; align-items:center; }
  textarea{ width:100%; min-height:520px; background:#fff; color:var(--ink);
    border:1px solid var(--line); padding:18px; font-size:13px;
    font-family:'SF Mono','Fira Code',ui-monospace,monospace; resize:vertical; line-height:1.8; outline:none; }
  textarea:focus{ border-color:var(--red); }
  .btn-sm{ padding:6px 13px; font-size:12px; border:1px solid var(--line); cursor:pointer;
    background:#fff; color:var(--gray); font-family:inherit; transition:all .18s; }
  .btn-sm:hover{ border-color:var(--gray); color:var(--ink); }
  .btn-sm.copied{ background:var(--red); color:#fff; border-color:var(--red); }
  .progress-row{ display:flex; align-items:center; gap:12px; margin-top:14px; }
  .progress-bar-bg{ flex:1; height:3px; background:var(--line-soft); }
  .progress-bar{ height:100%; background:var(--red); transition:width .4s; width:0%; }
  .progress-bar.done{ background:var(--red); }
  .progress-label{ font-size:12px; color:var(--muted); white-space:nowrap; min-width:120px; }

  /* ── 文件模式 ── */
  .drop-zone{ border:1.5px dashed var(--line); background:#fff; padding:48px 20px;
    text-align:center; cursor:pointer; transition:all .2s; margin-bottom:18px; }
  .drop-zone:hover, .drop-zone.drag-over{ border-color:var(--red); background:var(--red-tint); }
  .drop-zone input{ display:none; }
  .drop-zone p{ font-size:15px; color:var(--ink); font-weight:500; }
  .drop-zone small{ font-size:12px; color:var(--muted); margin-top:8px; display:block; }

  .file-queue{ background:#fff; border:1px solid var(--line); padding:14px 18px; margin-bottom:18px; }
  .queue-header{ font-size:12px; color:var(--muted); margin-bottom:10px; letter-spacing:.06em; }
  .queue-item{ display:flex; align-items:center; gap:10px; padding:7px 0; border-bottom:1px solid var(--line-soft); }
  .queue-item:last-child{ border-bottom:none; }
  .queue-name{ flex:1; font-size:13px; color:var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .queue-size{ font-size:11px; color:var(--muted); white-space:nowrap; }
  .btn-remove{ background:none; border:none; color:var(--muted); cursor:pointer; font-size:18px; line-height:1; padding:0 4px; transition:color .2s; }
  .btn-remove:hover{ color:var(--red); }

  /* ── 文件翻译卡片 ── */
  .file-card{ background:#fff; border:1px solid var(--line); border-left:3px solid var(--red); padding:18px 22px; margin-bottom:14px; }
  .file-card-header{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; gap:12px; }
  .file-name{ font-size:14px; font-weight:700; color:var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:68%; }
  .file-status{ font-size:12px; color:var(--red); font-weight:500; }
  .file-status.done{ color:var(--gray); }
  .card-progress-bg{ width:100%; height:3px; background:var(--line-soft); margin-bottom:12px; }
  .card-progress{ height:100%; background:var(--red); transition:width .3s; width:0%; }
  .card-progress.done{ background:var(--red); }
  .result-row{ display:flex; gap:8px; }
  .btn-copy{ background:var(--red); color:#fff; padding:8px 16px; font-size:12px;
    border:1px solid var(--red); cursor:pointer; font-weight:500; font-family:inherit; transition:all .18s; }
  .btn-copy:hover{ background:var(--red-strong); }
  .btn-copy.copied{ background:var(--gray); border-color:var(--gray); }
  .btn-dl{ background:#fff; color:var(--gray); padding:8px 16px; font-size:12px;
    border:1px solid var(--line); cursor:pointer; font-weight:500; font-family:inherit; transition:all .18s; }
  .btn-dl:hover{ border-color:var(--gray); color:var(--ink); }

  .spinner{ display:inline-block; width:11px; height:11px; border:2px solid var(--red);
    border-top-color:transparent; border-radius:50%; animation:spin .8s linear infinite;
    vertical-align:middle; margin-right:7px; }
  @keyframes spin{ to{ transform:rotate(360deg); } }

  .foot{ border-top:1px solid var(--line); padding:24px 40px; display:flex;
    align-items:center; justify-content:space-between; gap:16px; }
  .foot img{ height:22px; width:auto; }
  .foot span{ font-size:11px; color:var(--muted); letter-spacing:.05em; }
</style>
</head>
<body>

<nav class="nav">
  <img class="nav-logo" src="/static/jtn-logo.png" alt="JT&N 金诚同达">
  <span class="nav-sep"></span>
  <a href="/">文档解析</a>
  <a href="/translate" class="active">Markdown 翻译</a>
  <a href="/office">Office 文档翻译</a>
</nav>

<div class="container">
  <div class="hero">
    <div class="overline"><span class="sq"></span><em>Markdown Translation · 文档翻译</em></div>
    <h1>Markdown <b>智能翻译</b></h1>
    <p class="subtitle">粘贴文本或上传多个 Markdown 文件，GLM-5.3-Flash 流式翻译，保留完整格式。</p>
  </div>

  <div class="toolbar">
    <label>目标语言</label>
    <select id="langSelect" onchange="onLangChange()">
      <option value="中文">中文</option>
      <option value="English">English</option>
      <option value="日本語">日本語</option>
      <option value="custom">自定义...</option>
    </select>
    <span id="customLangWrap">
      <input type="text" id="customLang" placeholder="输入语言名称">
    </span>
    <button class="btn btn-primary" id="translateBtn" onclick="startTranslate()">开始翻译</button>
    <button class="btn" id="cancelTransBtn" style="display:none" onclick="cancelCurrentTransJob()">取消翻译</button>
    <button class="btn btn-secondary" onclick="clearAll()">清空</button>
  </div>

  <div class="input-tabs">
    <div class="tab active" id="tab-file" onclick="switchTab('file')">上传文件</div>
    <div class="tab" id="tab-paste" onclick="switchTab('paste')">粘贴文本</div>
  </div>

  <!-- 粘贴模式 -->
  <div id="paste-panel" style="display:none">
    <div class="editor-row">
      <div class="editor-pane">
        <div class="pane-header">
          <span class="pane-title">原文</span>
          <span id="inputStats" style="font-size:11px;color:#444"></span>
        </div>
        <textarea id="inputArea" placeholder="在此粘贴 Markdown 内容..." oninput="updateStats()"></textarea>
      </div>
      <div class="editor-pane">
        <div class="pane-header">
          <span class="pane-title">译文</span>
          <div class="pane-actions">
            <button class="btn-sm" onclick="copyOutput()">复制</button>
            <button class="btn-sm" onclick="downloadOutput()">下载 .md</button>
          </div>
        </div>
        <textarea id="outputArea" readonly placeholder="译文将在此实时显示..."></textarea>
      </div>
    </div>
    <div class="progress-row" id="pasteProgressRow" style="display:none">
      <div class="progress-bar-bg"><div class="progress-bar" id="pasteBar"></div></div>
      <span class="progress-label" id="pasteLabel"></span>
    </div>
  </div>

  <!-- 文件上传模式 -->
  <div id="file-panel">
    <div class="drop-zone" id="dropZone">
      <p>将 .md / .txt 文件拖到这里，或点击选择</p>
      <small>支持同时选择多个文件，选好后点"开始翻译"。Word 文档请移步「Office 文档翻译」</small>
      <input type="file" id="mdFileInput" accept=".md,.markdown,.txt" multiple>
    </div>
    <div class="file-queue" id="fileQueue" style="display:none">
      <div class="queue-header" id="queueHeader"></div>
      <div id="fileQueueList"></div>
    </div>
    <div id="fileCards"></div>
  </div>

</div>

<footer class="foot">
  <img src="/static/jtn-logo.png" alt="JT&N 金诚同达">
  <span>金诚同达律师事务所　·　Doxify 文档智能工具</span>
</footer>

<script src="/static/job-client.js"></script>
<script>
let currentTab = 'file';
let selectedFiles = [];
let isTranslating = false;
// 文件模式状态
let fileStates = {};
// 粘贴模式状态
let pasteChunkBufs = {};
let pasteTotalChunks = 1;
let pasteDoneChunks = 0;
let pasteNameSuffix = '';  // 粘贴模式译文文件名语种标识（如 ESP2CN）

function switchTab(tab) {
  currentTab = tab;
  document.getElementById('tab-paste').classList.toggle('active', tab === 'paste');
  document.getElementById('tab-file').classList.toggle('active', tab === 'file');
  document.getElementById('paste-panel').style.display = tab === 'paste' ? '' : 'none';
  document.getElementById('file-panel').style.display = tab === 'file' ? '' : 'none';
}

function onLangChange() {
  const val = document.getElementById('langSelect').value;
  document.getElementById('customLangWrap').style.display = val === 'custom' ? 'inline' : 'none';
  if (val === 'custom') document.getElementById('customLang').focus();
}

function getTargetLang() {
  const sel = document.getElementById('langSelect').value;
  if (sel === 'custom') return document.getElementById('customLang').value.trim() || '中文';
  return sel;
}

function updateStats() {
  const txt = document.getElementById('inputArea').value;
  document.getElementById('inputStats').textContent = txt.length > 0 ? `${txt.length} 字符` : '';
}

function clearAll() {
  if (currentTab === 'paste') {
    document.getElementById('inputArea').value = '';
    document.getElementById('outputArea').value = '';
    document.getElementById('pasteProgressRow').style.display = 'none';
    updateStats();
  } else {
    selectedFiles = [];
    renderFileQueue();
    document.getElementById('fileCards').innerHTML = '';
  }
}

function copyOutput() {
  navigator.clipboard.writeText(document.getElementById('outputArea').value);
  const btn = event.target;
  btn.textContent = '已复制!'; btn.classList.add('copied');
  setTimeout(() => { btn.textContent = '复制'; btn.classList.remove('copied'); }, 2000);
}

function downloadOutput() {
  const text = document.getElementById('outputArea').value;
  if (!text) return;
  const blob = new Blob([text], { type: 'text/markdown' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'translated' + (pasteNameSuffix ? ('_' + pasteNameSuffix) : '') + '.md';
  a.click();
}

// ── 文件拖放 ──
const dropZone = document.getElementById('dropZone');
const mdFileInput = document.getElementById('mdFileInput');
dropZone.addEventListener('click', () => mdFileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  addFiles([...e.dataTransfer.files]);
});
mdFileInput.addEventListener('change', () => {
  addFiles([...mdFileInput.files]);
  mdFileInput.value = '';
});

function addFiles(list) {
  for (const f of list) {
    if (/\\.(md|markdown|txt)$/i.test(f.name) &&
        !selectedFiles.find(sf => sf.name === f.name && sf.size === f.size)) {
      selectedFiles.push(f);
    }
  }
  renderFileQueue();
}

function removeFile(idx) {
  selectedFiles.splice(idx, 1);
  renderFileQueue();
}

function renderFileQueue() {
  const queue = document.getElementById('fileQueue');
  const list  = document.getElementById('fileQueueList');
  const hdr   = document.getElementById('queueHeader');
  if (selectedFiles.length === 0) { queue.style.display = 'none'; return; }
  queue.style.display = '';
  hdr.textContent = `已选择 ${selectedFiles.length} 个文件`;
  list.innerHTML = selectedFiles.map((f, i) => `
    <div class="queue-item">
      <span class="queue-name" title="${f.name}">${f.name}</span>
      <span class="queue-size">${(f.size/1024).toFixed(1)} KB</span>
      <button class="btn-remove" onclick="removeFile(${i})">×</button>
    </div>`).join('');
}

const TRANS_JOB_KEY = 'doxify.job.translate';
let currentTransJobId = null;
let transStartTime = Date.now();
// 终态处理器（onEnd/onGone/onGiveUp）是否已经还原过界面——startTranslate 的
// finally 靠这个标志判断要不要再碰按钮，见下方 resetTransUi 与 startTranslate。
let transUiRestored = false;

function cancelCurrentTransJob() {
  if (currentTransJobId) JobClient.cancel(currentTransJobId);
}

function attachTransJob(jobId, isFileMode) {
  currentTransJobId = jobId;
  document.getElementById('cancelTransBtn').style.display = '';
  // 重连路径（DOMContentLoaded）不经过 startTranslate，按钮的运行态必须由这里
  // 兜底设置，否则刷新后按钮显示空闲、看不出后台其实还有作业在跑。
  const btn = document.getElementById('translateBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>翻译中...';

  // 重连时（isFileMode === null）不知道原始是文件模式还是粘贴模式，只能等
  // init 事件到达后从作业本身判断：粘贴任务的唯一条目 file_id 固定是
  // text_0（服务端 pre_items 的构造方式决定，见 start_translate_job），文件
  // 任务是 file_0 / file_1...。硬编码成文件模式会把粘贴任务的输出错误地画
  // 进文件卡片、outputArea 永远空白——这正是本轮要修的问题。
  // startTranslate 的原调用已经知道真实模式，直接传布尔值，这里的判断不会
  // 覆盖它（下面的 if 只在仍是 null 时才生效）。
  let resolvedMode = isFileMode;
  function resolveMode(evt) {
    if (resolvedMode === null && evt.type === 'init') {
      resolvedMode = !(evt.files.length === 1 && evt.files[0].file_id.startsWith('text_'));
    }
    return resolvedMode;
  }

  // onEnd / onGone / onGiveUp 三条路径都要把界面还原成空闲态，抽成一个函数只写
  // 一处——按钮态曾经因为“重连路径不经过 startTranslate”漏掉过一次，onGiveUp
  // 是同一类漏洞，不能再让第三个分支单独维护一份还原逻辑。
  // message 参数让 onGiveUp 能在按钮本身留下失败原因——按钮是 resetTransUi
  // 唯一会摸到、且两种模式下都可见的元素，不必再按模式分别找地方提示。
  // transUiRestored 标记「终态已经处理过」：startTranslate 的 finally 会在
  // await 结束后紧跟着运行，如果不设这个标记，它会无条件把按钮文案覆盖回
  // “开始翻译”，onGiveUp 刚写上的失败提示还没让用户看清就被抹掉——同时
  // currentTransJobId 和 isTranslating 双双清空，再点一次就会在第一个作业
  // 仍可能存活的情况下起出第二个，白烧一倍网关配额。
  function resetTransUi(message) {
    transUiRestored = true;
    currentTransJobId = null;
    document.getElementById('cancelTransBtn').style.display = 'none';
    btn.disabled = false;
    btn.textContent = message || '开始翻译';
  }

  // job_end 不带 file_id，handleEvent 按 file_id 派发，所以取消/失败的收尾只能在
  // 这里做。不做的话在途卡片会永远停在「3/12 块」的旋转态，与成功毫无区别。
  function markTransUnfinished(status, error) {
    const label = status === 'cancelled' ? '已取消' : '失败';
    if (resolvedMode === false) {
      document.getElementById('pasteLabel').textContent =
        label + (error ? '：' + error : '');
      return;
    }
    for (const fid of Object.keys(fileStates)) {
      const st = fileStates[fid];
      if (st.finalMd) continue;              // 已完成的不动
      const sta = document.getElementById('status-' + fid);
      if (sta) { sta.textContent = label; sta.className = 'file-status error'; }
      const bar = document.getElementById('bar-' + fid);
      if (bar) bar.style.background = '#e74c3c';
    }
  }

  return JobClient.attach(TRANS_JOB_KEY, jobId, {
    onMeta: (m) => { transStartTime = m.created_at * 1000; },
    onEvent: (evt, replaying) => {
      const mode = resolveMode(evt);
      handleEvent(evt, mode, transStartTime, replaying);
    },
    onReplayDone: () => {
      // 重连成功、backlog 补发完毕：把按钮文案从「重连中 (n)...」换回正常的
      // 运行态提示，否则恢复之后还会一直挂着旧的重连文案。
      btn.innerHTML = '<span class="spinner"></span>翻译中...';
      // 粘贴模式的 chunk_token 在重放期间被抑制了逐条 DOM 写入（否则长文档
      // 重放是 O(n²)，会冻住标签页），这里补一次性渲染。resolvedMode 到这里
      // 一定已经确定——_replay.count > 0 时 init 必然已在补发的 backlog 里
      // 处理过；count === 0 则 pasteChunkBufs 本就是空的，渲染空字符串无害。
      if (resolvedMode === false) renderPastePreview();
    },
    onReconnecting: (n) => {
      // 整个退避重连的 75~90 秒里，不写这个用户就什么反馈都看不到，直到
      // 最后 onGiveUp 弹出失败提示——中间这段空白本身就是体验问题。
      btn.innerHTML = `<span class="spinner"></span>连接中断，重连中 (${n})...`;
    },
    onEnd: (status, error) => {
      resetTransUi();
      if (status && status !== 'done') markTransUnfinished(status, error);
    },
    onGone: () => resetTransUi(),
    onGiveUp: () => {
      // 不调用 JobClient.forget：作业大概率还在服务端跑，留着 localStorage 里的
      // id 才能让用户刷新页面时重新接上，而不是永久失联。提示文案必须说清楚
      // 「作业可能还在跑、刷新是接回不是重开」，否则用户很容易再点一次按钮，
      // 在第一个作业没被取消的情况下起出第二个。
      resetTransUi('重连失败，作业可能仍在运行，刷新页面可接回');
    },
  });
}

function createCard(fileId, filename) {
  fileStates[fileId] = { filename, chunkBufs: {}, totalChunks: 1, doneChunks: 0, finalMd: '' };
  const card = document.createElement('div');
  card.className = 'file-card';
  card.id = 'card-' + fileId;
  card.innerHTML = `
    <div class="file-card-header">
      <span class="file-name" title="${filename}">${filename}</span>
      <span class="file-status" id="status-${fileId}"><span class="spinner"></span>翻译中...</span>
    </div>
    <div class="card-progress-bg"><div class="card-progress" id="bar-${fileId}"></div></div>
    <div class="result-row" id="btns-${fileId}" style="display:none">
      <button class="btn-copy" onclick="copyCard('${fileId}')">复制</button>
      <button class="btn-dl" onclick="downloadCard('${fileId}')">下载 .md</button>
    </div>`;
  document.getElementById('fileCards').appendChild(card);
}

async function startTranslate() {
  // isTranslating 只在走 startTranslate 时才置位；重连路径（DOMContentLoaded →
  // attachTransJob）不经过它，所以必须同时看 currentTransJobId，否则刷新后按钮
  // 看起来空闲，再点一次就会起一个重复且不被跟踪的作业。
  if (isTranslating || currentTransJobId) return;
  const targetLang = getTargetLang();
  if (!targetLang) { alert('请输入目标语言'); return; }

  const formData = new FormData();
  formData.append('target_lang', targetLang);
  const isFileMode = currentTab === 'file';

  if (isFileMode) {
    if (selectedFiles.length === 0) { alert('请先选择要翻译的文件'); return; }
    for (const f of selectedFiles) formData.append('files', f);
    document.getElementById('fileCards').innerHTML = '';
    fileStates = {};
    document.getElementById('fileQueue').style.display = 'none';
  } else {
    const input = document.getElementById('inputArea').value.trim();
    if (!input) { alert('请先输入或粘贴 Markdown 内容'); return; }
    formData.append('text', input);
    document.getElementById('outputArea').value = '';
    pasteChunkBufs = {}; pasteDoneChunks = 0; pasteTotalChunks = 1;
    document.getElementById('pasteProgressRow').style.display = 'flex';
    document.getElementById('pasteBar').style.width = '0%';
    document.getElementById('pasteBar').classList.remove('done');
    document.getElementById('pasteBar').style.background = '';   // 清掉上次失败的红色
    document.getElementById('pasteLabel').textContent = '准备中...';
  }

  isTranslating = true;
  transUiRestored = false;   // 本次 attach 还没有终态处理器还原过界面
  const btn = document.getElementById('translateBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>翻译中...';

  try {
    const resp = await fetch('/jobs/translate', { method: 'POST', body: formData });
    const body = await resp.json();
    if (!body.job_id) {
      alert(body.error || '启动失败');
      return;
    }
    JobClient.save(TRANS_JOB_KEY, body.job_id);
    await attachTransJob(body.job_id, isFileMode);
  } catch(e) {
    if (!isFileMode) document.getElementById('pasteLabel').textContent = '请求失败: ' + e.message;
    console.error(e);
  } finally {
    isTranslating = false;
    // 终态处理器（onEnd/onGone/onGiveUp）若已经还原过界面，就不要再覆盖它——
    // 尤其是 onGiveUp 写在按钮上的失败提示。否则用户等满 90 秒只看到按钮悄悄
    // 变回空闲、毫无解释，还能再点一次起出第二个作业，而第一个可能仍在服务端
    // 跑（give-up 有意不取消它），等于白烧一倍网关配额。
    if (!transUiRestored) {
      btn.disabled = false;
      btn.textContent = '开始翻译';
    }
    if (isFileMode && selectedFiles.length > 0) renderFileQueue();
  }
}

function renderPastePreview() {
  let preview = '';
  for (let i = 1; i <= pasteTotalChunks; i++) {
    if (pasteChunkBufs[i]) preview += (preview ? '\\n\\n' : '') + pasteChunkBufs[i];
  }
  document.getElementById('outputArea').value = preview;
}

function handleEvent(evt, isFileMode, startTime, replaying) {
  if (evt.type === 'init') {
    if (isFileMode) {
      for (const { file_id, filename } of evt.files) createCard(file_id, filename);
    } else {
      pasteTotalChunks = evt.files[0]?.total_chunks || 1;
      document.getElementById('pasteLabel').textContent = `0 / ${pasteTotalChunks} 块`;
    }
  } else if (evt.type === 'file_start') {
    if (isFileMode && fileStates[evt.file_id]) {
      fileStates[evt.file_id].totalChunks = evt.total_chunks;
    }
  } else if (evt.type === 'chunk_token') {
    const { file_id, chunk, token } = evt;
    if (isFileMode) {
      const st = fileStates[file_id]; if (!st) return;
      st.chunkBufs[chunk] = (st.chunkBufs[chunk] || '') + token;
    } else {
      pasteChunkBufs[chunk] = (pasteChunkBufs[chunk] || '') + token;
      // 重放期间跳过逐 token 的 DOM 写入：每次都重建整个 preview 字符串再赋值，
      // 长文档重放是 O(n²)，会冻住标签页。重放结束后 onReplayDone 补一次性渲染。
      if (!replaying) renderPastePreview();
    }
  } else if (evt.type === 'chunk_replace') {
    // 服务端检测到该块残留英文，已生成修正稿；用修正稿覆盖该块缓冲
    const { file_id, chunk, text } = evt;
    if (isFileMode) {
      const st = fileStates[file_id]; if (!st) return;
      st.chunkBufs[chunk] = text;
    } else {
      pasteChunkBufs[chunk] = text;
      if (!replaying) renderPastePreview();
    }
  } else if (evt.type === 'chunk_done') {
    const { file_id, chunk } = evt;
    if (isFileMode) {
      const st = fileStates[file_id]; if (!st) return;
      st.doneChunks++;
      const pct = Math.round(st.doneChunks / st.totalChunks * 100);
      const bar = document.getElementById('bar-' + file_id);
      const sta = document.getElementById('status-' + file_id);
      if (bar) bar.style.width = pct + '%';
      if (sta) sta.innerHTML = `<span class="spinner"></span>${st.doneChunks}/${st.totalChunks} 块`;
    } else {
      pasteDoneChunks++;
      document.getElementById('pasteBar').style.width =
        Math.round(pasteDoneChunks / pasteTotalChunks * 100) + '%';
      document.getElementById('pasteLabel').textContent =
        `${pasteDoneChunks} / ${pasteTotalChunks} 块`;
    }
  } else if (evt.type === 'file_done') {
    if (isFileMode) {
      const st = fileStates[evt.file_id]; if (!st) return;
      st.nameSuffix = evt.name_suffix || '';
      if (evt.download_url) {
        // DOCX：产物在服务端，下载按钮改为取回文件
        st.downloadUrl = evt.download_url;
        st.keptOriginal = evt.kept_original || 0;
        const btn = document.querySelector('#btns-' + evt.file_id + ' .btn-dl');
        if (btn) btn.textContent = '下载 .docx';
      }
      let md = '';
      for (let i = 1; i <= st.totalChunks; i++) md += (md ? '\\n\\n' : '') + (st.chunkBufs[i] || '');
      st.finalMd = md;
      const bar = document.getElementById('bar-' + evt.file_id);
      const sta = document.getElementById('status-' + evt.file_id);
      const bts = document.getElementById('btns-' + evt.file_id);
      if (bar) { bar.style.width = '100%'; bar.classList.add('done'); }
      if (sta) {
        sta.textContent = (st.keptOriginal > 0) ? `完成（${st.keptOriginal} 段未译，保留原文）` : '完成';
        sta.className = 'file-status done';
      }
      if (bts) bts.style.display = 'flex';
    } else if (evt.name_suffix) {
      pasteNameSuffix = evt.name_suffix;
    }
  } else if (evt.type === 'file_error') {
    // 服务端的文件级兜底事件。没有这个分支，单文件翻译崩掉时作业会正确终止，
    // 但卡片停在「n/m 块」旋转且无任何解释。
    const safe = String(evt.error || '未知错误');
    if (isFileMode) {
      const sta = document.getElementById('status-' + evt.file_id);
      if (sta) { sta.textContent = '失败：' + safe; sta.className = 'file-status error'; }
      const bar = document.getElementById('bar-' + evt.file_id);
      if (bar) { bar.style.width = '100%'; bar.style.background = '#e74c3c'; }
    } else {
      document.getElementById('pasteLabel').textContent = '翻译失败：' + safe;
      document.getElementById('pasteBar').style.background = '#e74c3c';
    }
  } else if (evt.type === 'all_done') {
    if (!isFileMode) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      document.getElementById('pasteBar').style.width = '100%';
      document.getElementById('pasteBar').classList.add('done');
      document.getElementById('pasteLabel').textContent = `完成 · 耗时 ${elapsed} 秒`;
    }
  }
}

function copyCard(fileId) {
  navigator.clipboard.writeText(fileStates[fileId]?.finalMd || '');
  const btn = event.target;
  btn.textContent = '已复制!'; btn.classList.add('copied');
  setTimeout(() => { btn.textContent = '复制'; btn.classList.remove('copied'); }, 2000);
}

function downloadCard(fileId) {
  const st = fileStates[fileId];
  if (st?.downloadUrl) { window.location.href = st.downloadUrl; return; }
  if (!st?.finalMd) return;
  const suffix = st.nameSuffix ? ('_' + st.nameSuffix) : '_translated';
  const name = st.filename.replace(/\\.(md|markdown|txt)$/i, '') + suffix + '.md';
  const blob = new Blob([st.finalMd], { type: 'text/markdown' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
}

// 翻译长文档同样耗时，同样会因关标签页/休眠而中断，重连逻辑与解析页一致。
// 重连时不知道原始是文件模式还是粘贴模式，传 null 交给 attachTransJob 在
// init 事件到达后自行判断——硬编码成文件模式会把粘贴任务的输出画错地方。
window.addEventListener('DOMContentLoaded', () => {
  const jobId = JobClient.restore(TRANS_JOB_KEY);
  // 同解析页：兜底记录，避免未来的回归再次变成无人问津的 unhandled rejection。
  if (jobId) attachTransJob(jobId, null).catch(e => console.error('重连失败', e));
});
</script>
</body>
</html>
"""


@app.get("/")
async def index():
    return HTMLResponse(UPLOAD_PAGE_HTML)


@app.get("/translate")
async def translate_page():
    return HTMLResponse(TRANSLATE_PAGE_HTML)


@app.get("/office")
async def office_page():
    font_map = {lang: {"track": track, "fonts": fonts}
                for lang, (track, fonts) in _OFFICE_LANG_FONTS.items()}
    return HTMLResponse(OFFICE_PAGE_HTML_TEMPLATE.replace(
        "__FONT_MAP__", json.dumps(font_map, ensure_ascii=False)))


# ── 翻译文件名语种标识：源语种检测 + 语种代码映射 ──
# langdetect 返回 ISO 639-1（en/es/zh/ja...），按用户偏好映射到文件名代码
# （西语用 ESP、中文用 CN）。未识别回退 XX。
_LANG_CODE_MAP = {
    "en": "EN", "es": "ESP", "zh": "CN", "zh-cn": "CN", "zh-tw": "CN",
    "ja": "JP", "fr": "FR", "de": "DE", "ko": "KO", "ru": "RU",
    "pt": "PT", "it": "IT", "ar": "AR", "nl": "NL", "vi": "VI",
    "th": "TH", "id": "ID", "hi": "HI",
}
# 目标语言显示名（小写）→ 文件名代码
_TARGET_LANG_CODE = {
    "中文": "CN", "简体中文": "CN", "繁體中文": "CN",
    "english": "EN", "英文": "EN", "英语": "EN",
    "日本語": "JP", "日语": "JP", "日文": "JP",
    "español": "ESP", "西班牙语": "ESP", "西语": "ESP",
    "français": "FR", "法语": "FR", "deutsch": "DE", "德语": "DE",
    "한국어": "KO", "韩语": "KO", "русский": "RU", "俄语": "RU",
    "português": "PT", "葡萄牙语": "PT",
    "越南语": "VI", "tiếng việt": "VI", "泰语": "TH", "ไทย": "TH",
    "马来西亚语": "MS", "马来语": "MS", "bahasa melayu": "MS",
}


def _detect_src_lang_code(text: str) -> str:
    """检测源文本语种，返回文件名用代码（EN/ESP/CN...）。失败回退 XX。"""
    sample = (text or "").strip()[:3000]
    if not sample:
        return "XX"
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # 结果可复现
        code = detect(sample).lower()
        return _LANG_CODE_MAP.get(code, code.split("-")[0].upper())
    except Exception:
        return "XX"


def _target_lang_code(name: str) -> str:
    """目标语言显示名 → 文件名用代码。未知回退首两位大写。"""
    n = (name or "").strip()
    return _TARGET_LANG_CODE.get(n.lower(), (n[:2] or "XX").upper())


@app.post("/jobs/translate")
async def start_translate_job(
    text: str = Form(default=""),
    target_lang: str = Form(default="中文"),
    docx_font: str = Form(default="宋体"),
    docx_font_track: str = Form(default=""),
    files: list[UploadFile] = File(default=[]),
):
    """建立翻译作业并立即返回 job_id。理由同解析侧：长文档翻译同样会因断线而作废。"""
    lang = target_lang.strip() or TRANSLATE_TARGET_LANG

    # 预处理：收集所有待翻译内容 (file_id, filename, chunks)
    pre_items: list[tuple[str, str, list[str]]] = []

    docx_items: list[tuple[str, str, bytes]] = []   # (file_id, filename, 原始字节)
    valid_files = [f for f in files if f and f.filename and f.filename.strip()]
    if valid_files:
        for i, f in enumerate(valid_files):
            raw = await f.read()
            if f.filename.lower().endswith(".docx"):
                # DOCX 是 zip 二进制，绝不能走下面的 utf-8 解码——那会把它变成乱码文本
                docx_items.append((uuid.uuid4().hex, f.filename, raw))
                continue
            content = raw.decode("utf-8", errors="replace")
            if content.strip():
                chunks = split_markdown_chunks(content, max_chars=TRANSLATE_CHUNK_CHARS)
                pre_items.append((f"file_{i}", f.filename, chunks))
    elif text.strip():
        chunks = split_markdown_chunks(text, max_chars=TRANSLATE_CHUNK_CHARS)
        pre_items.append(("text_0", "输入文本", chunks))

    if not pre_items and not docx_items:
        return JSONResponse({"success": False, "error": "内容为空"})

    log.info("翻译任务: %d 个 md/文本 + %d 个 docx (并行), 目标语言=%s",
             len(pre_items), len(docx_items), lang)

    async def _translate_file_task(
        file_id: str, filename: str, chunks: list[str], queue: asyncio.Queue
    ) -> None:
        """单个文件的翻译任务：分块并行翻译，受 _api_semaphore 全局限流。"""
        total = len(chunks)
        log.info("开始翻译: %s (%d 块, 分块并行 ≤ %d)",
                 filename, total, MAX_CONCURRENT_REQUESTS)
        await queue.put({"type": "file_start", "file_id": file_id,
                         "filename": filename, "total_chunks": total})

        async def _process_chunk(client: httpx.AsyncClient, idx: int, chunk: str) -> None:
            await queue.put({"type": "chunk_start", "file_id": file_id,
                             "chunk": idx, "total": total})
            log.info("翻译 [%s] 第 %d/%d 块 (%d 字符)", filename, idx, total, len(chunk))

            # translate_chunk_stream 的首段缓冲会压住前 STREAM_HEAD_CHARS 个字符
            # （要整段判定是不是无标签推理），块开头因此有几秒空白。先放一句占位，
            # 第一段真实译文到达时用 chunk_replace 整块换掉——前端对 chunk_replace
            # 的处理就是 chunkBufs[chunk] = text，正好是整块覆盖，不新增事件类型。
            await queue.put({"type": "chunk_token", "file_id": file_id,
                             "chunk": idx, "token": _CHUNK_PLACEHOLDER})

            buf = ""

            async def _stream_once(max_tokens: int) -> None:
                """流一遍。第一段走 chunk_replace 换掉占位符，其余照旧追加。"""
                nonlocal buf
                buf, first = "", True
                async for token in translate_chunk_stream(
                        client, chunk, idx, total, lang, max_tokens):
                    if first:
                        first = False
                        await queue.put({"type": "chunk_replace", "file_id": file_id,
                                         "chunk": idx, "text": token})
                    else:
                        await queue.put({"type": "chunk_token", "file_id": file_id,
                                         "chunk": idx, "token": token})
                    buf += token

            try:
                await _stream_once(TRANSLATE_MAX_TOKENS)
                if not buf.strip():
                    # 零产出多半是思考吃光了 max_tokens（GLM 档案配错时必现），
                    # 也可能是网关异常被归一成空流。翻倍额度自动重发一次。
                    log.warning("[%s] 第 %d 块译文为空，max_tokens 翻倍至 %d 重发",
                                filename, idx, TRANSLATE_MAX_TOKENS * 2)
                    await _stream_once(TRANSLATE_MAX_TOKENS * 2)
            except Exception as e:
                log.exception("[%s] 第 %d 块翻译异常: %s", filename, idx, e)
                # 占位符还在前端的块缓冲里，不换掉它就会原样混进最终译文与下载文件
                await queue.put({"type": "chunk_replace", "file_id": file_id, "chunk": idx,
                                 "text": buf or f"[第 {idx}/{total} 块翻译异常：{e}]"})
                await queue.put({"type": "chunk_done", "file_id": file_id, "chunk": idx})
                return

            if not buf.strip():
                log.error("[%s] 第 %d 块译文为空（输入 %d 字符），翻倍重发后 LLM 仍未返回内容",
                          filename, idx, len(chunk))
                notice = f"[第 {idx}/{total} 块译文为空，LLM 未返回内容，请重试]"
                buf = notice
                # 占位符还挂在前端缓冲里，必须换掉而不是追加
                await queue.put({"type": "chunk_replace", "file_id": file_id,
                                 "chunk": idx, "text": notice})

            # 残留英文检测 + 一次性修正。整段包在 try 里做纵深防御：这里原本裸奔在
            # 上面那个 try 之外，_detect_residual_english 一抛异常整个块任务就死，
            # 而 gather 的同辈任务不会被取消、file_done 也永不到达。译文已经在客户端
            # 手里了，检测/修正失败绝不该升级成整个文件失败。
            try:
                residuals = _detect_residual_english(buf)
                if residuals:
                    log.info("[%s] 第 %d 块检测到残留英文 %s，启动修正",
                             filename, idx, residuals[:8])
                    fixed = await _fix_residual_english(client, buf, residuals, lang)
                    if fixed and fixed != buf:
                        await queue.put({"type": "chunk_replace", "file_id": file_id,
                                         "chunk": idx, "text": fixed})
                        log.info("[%s] 第 %d 块修正完成（%d → %d 字符）",
                                 filename, idx, len(buf), len(fixed))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("[%s] 第 %d 块残留英文检测/修正失败，保留原译: %s",
                            filename, idx, e)

            await queue.put({"type": "chunk_done", "file_id": file_id, "chunk": idx})

        # trust_env=False 理由同 parse_pdf_streaming：绕过系统代理直连 LLM 网关
        async with httpx.AsyncClient(timeout=httpx.Timeout(LLM_TIMEOUT, connect=10.0),
                                     trust_env=False) as client:
            # 同时把所有 chunk 任务投递出去；并发上限由 translate_chunk_stream / _fix_residual_english
            # 内部的 _api_semaphore 自然控制（全局 MAX_CONCURRENT_REQUESTS）。
            await asyncio.gather(
                *(_process_chunk(client, idx, chunk)
                  for idx, chunk in enumerate(chunks, 1)),
                return_exceptions=False,
            )

        src_code = _detect_src_lang_code("\n".join(chunks))
        tgt_code = _target_lang_code(lang)
        name_suffix = f"{src_code}2{tgt_code}"
        await queue.put({"type": "file_done", "file_id": file_id, "filename": filename,
                         "name_suffix": name_suffix})
        log.info("文件翻译完成: %s（%s）", filename, name_suffix)

    async def _run_translate_file_task(
        file_id: str, filename: str, chunks: list[str], queue: asyncio.Queue
    ) -> None:
        """包装单个翻译任务：任何未捕获异常都转成一条 file_error 事件。形状同 _run_parse_task。

        没有这层兜底，子任务一死就再也不投递 file_done，runner 的收集循环
        （`while done_count + error_count < len(tasks)`）永久阻塞，作业永远停在
        running；_evict_old_jobs 又永不淘汰运行中作业，于是它连同完整事件日志驻留到
        进程结束。前端此时**不会**重连——SSE 生成器停在 await q.get()、连接保持打开，
        客户端看到的是一条永不结束的静默流。
        """
        try:
            await _translate_file_task(file_id, filename, chunks, queue)
        except asyncio.CancelledError:
            # 取消不是失败：绝不能被下面的 except 归一成 file_error 占位
            raise
        except Exception as e:
            log.exception("[%s] 翻译失败: %s", file_id, e)
            await queue.put({"type": "file_error", "file_id": file_id,
                             "filename": filename,
                             "error": str(e) or e.__class__.__name__})

    async def _translate_docx_task(
        file_id: str, filename: str, raw: bytes, queue: asyncio.Queue
    ) -> None:
        """单个 DOCX 的翻译任务：引擎批次进度映射为 chunk_done 事件。"""
        def _on_plan(stats):
            # put_nowait：队列无界，且此处在事件循环内，不会丢
            queue.put_nowait({"type": "file_start", "file_id": file_id,
                              "filename": filename,
                              "total_chunks": max(1, stats["batches"]),
                              "docx": True})
            log.info("[docx] %s: %d 分段 / %d 需译 / %d 批 / %d 字符",
                     filename, stats["segments"], stats["translated"],
                     stats["batches"], stats["chars"])

        def _on_progress(done, total):
            queue.put_nowait({"type": "chunk_done", "file_id": file_id, "chunk": done})

        async with httpx.AsyncClient(timeout=httpx.Timeout(LLM_TIMEOUT, connect=10.0),
                                     trust_env=False) as client:
            async def _batch(texts: list[str]) -> list[str]:
                return await _translate_docx_batch_llm(client, texts, lang)

            # 所选字体写入哪条轨由目标语言决定（内置语言查表；自定义语言由前端
            # 经 /office/font_suggest 拿到轨道后随表单传入）。「不调整」关闭字体补齐。
            _fonts = {"eastAsia": "", "latin": "", "cs": ""}
            if docx_font and docx_font != "不调整":
                _fonts[_office_font_track(lang, docx_font_track)] = docx_font
            out, stats = await translate_docx_bytes(
                raw, _batch, progress_cb=_on_progress, on_plan=_on_plan,
                target_lang=lang, cjk_font=_fonts["eastAsia"],
                latin_font=_fonts["latin"], cs_font=_fonts["cs"])

        src_code = _detect_src_lang_code(stats.get("sample", ""))
        name_suffix = f"{src_code}2{_target_lang_code(lang)}"
        out_dir = OUTPUT_DIR / file_id
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"\.docx$", "", filename, flags=re.I)
        (out_dir / f"{stem}_{name_suffix}.docx").write_bytes(out)
        kept = stats.get("kept_original", 0)
        await queue.put({"type": "file_done", "file_id": file_id,
                         "filename": filename, "name_suffix": name_suffix,
                         "docx": True, "kept_original": kept,
                         "download_url": f"/download_docx/{file_id}"})
        if kept:
            log.warning("DOCX 翻译完成但有 %d 个分段未译成、保留原文: %s", kept, filename)
        else:
            log.info("DOCX 翻译完成: %s（%s，%d 批）", filename, name_suffix, stats["batches"])

    async def _run_docx_file_task(
        file_id: str, filename: str, raw: bytes, queue: asyncio.Queue
    ) -> None:
        """兜底形状同 _run_translate_file_task：任何未捕获异常 → 一条 file_error。"""
        try:
            await _translate_docx_task(file_id, filename, raw, queue)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("[%s] DOCX 翻译失败: %s", file_id, e)
            await queue.put({"type": "file_error", "file_id": file_id,
                             "filename": filename,
                             "error": str(e) or e.__class__.__name__})

    async def _runner(job: Job):
        queue: asyncio.Queue = asyncio.Queue()
        tasks = [
            asyncio.create_task(_run_translate_file_task(fid, fname, cks, queue))
            for fid, fname, cks in pre_items
        ] + [
            asyncio.create_task(_run_docx_file_task(fid, fname, raw, queue))
            for fid, fname, raw in docx_items
        ]
        try:
            _publish(job, {"type": "init", "files": [
                {"file_id": fid, "filename": fname, "total_chunks": len(cks)}
                for fid, fname, cks in pre_items
            ] + [
                {"file_id": fid, "filename": fname, "total_chunks": 1, "docx": True}
                for fid, fname, _raw in docx_items
            ]})
            done_count = error_count = 0
            while done_count + error_count < len(tasks):
                evt = await queue.get()
                t = evt["type"]
                if t == "file_done":
                    done_count += 1
                elif t == "file_error":
                    error_count += 1
                _publish(job, evt)
            await asyncio.gather(*tasks)
            _publish(job, {"type": "all_done"})
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    job = _start_job("translate", _runner)
    return {"job_id": job.id}


async def _run_parse_task(task_fn, data, filename, file_id, queue) -> None:
    """包装单个解析任务：任何未捕获异常都转成一条 file_error 事件。

    VLM 路径 parse_pdf_streaming 不自带兜底，pdf_to_images 遇损坏 PDF 抛异常时
    task 会直接死亡、既不发 file_done 也不发 file_error，使 _generate 的事件
    循环永久挂起。
    """
    try:
        await task_fn(data, filename, file_id, queue)
    except Exception as e:
        log.exception("[%s] 解析失败: %s", file_id, e)
        await queue.put({"type": "file_error", "file_id": file_id,
                         "filename": filename, "error": str(e) or e.__class__.__name__})


@app.post("/jobs/parse")
async def start_parse_job(
    files: list[UploadFile] = File(...),
    strip_watermark: str = Form(default="1"),
    page_markers: str = Form(default="1"),
    footnote_fix: str = Form(default="1"),
):
    """建立解析作业并立即返回 job_id。

    此前这里直接返回 SSE 流，任务绑在请求生命周期上，客户端一断就成孤儿。
    现在任务交给注册表后台运行，前端另行订阅 /jobs/{id}/events。
    """
    valid = [f for f in files if f.filename and f.filename.lower().endswith(".pdf")]
    if not valid:
        return JSONResponse({"success": False, "error": "请上传 PDF 文件"})

    file_data = []
    for f in valid:
        data = await f.read()
        file_data.append((uuid.uuid4().hex, f.filename, data))

    strip_wm = strip_watermark == "1"
    pm = page_markers == "1"
    fn_fix = footnote_fix == "1"
    log.info("PDF 解析请求: %d 个文件, strip_watermark=%s, page_markers=%s, footnote_fix=%s",
             len(file_data), strip_wm, pm, fn_fix)

    def task_fn(d, fn, fid, q):
        return parse_pdf_streaming(d, fn, fid, q, strip_wm, pm, fn_fix)

    async def _runner(job: Job):
        queue: asyncio.Queue = asyncio.Queue()
        tasks = [
            asyncio.create_task(_run_parse_task(task_fn, data, filename, file_id, queue))
            for file_id, filename, data in file_data
        ]
        try:
            _publish(job, {"type": "init", "files": [
                {"file_id": fid, "filename": fname} for fid, fname, _ in file_data
            ]})
            done_count = error_count = 0
            while done_count + error_count < len(tasks):
                evt = await queue.get()
                t = evt.get("type")
                if t == "file_done":
                    done_count += 1
                elif t == "file_error":
                    error_count += 1
                _publish(job, evt)
            await asyncio.gather(*tasks)
        finally:
            # 作业被取消时子任务不会自动停，必须显式取消，否则它们会继续烧 API 配额
            for t in tasks:
                if not t.done():
                    t.cancel()

    job = _start_job("parse", _runner)
    return {"job_id": job.id}


_office_font_cache: dict[str, dict] = {}


def _office_parse_font_suggestion(content: str) -> dict:
    """解析 LLM 的字体建议：{"fonts": [...], "track": ...}。轨道非法归一为 latin。"""
    start, stop = content.find("{"), content.rfind("}")
    if start == -1 or stop <= start:
        raise ValueError("响应里没有 JSON 对象")
    blob = content[start:stop + 1]
    try:
        parsed = json.loads(blob)
    except ValueError:
        import json_repair
        parsed = json_repair.loads(blob)
    fonts = [str(f).strip() for f in (parsed.get("fonts") or []) if str(f).strip()]
    if not fonts:
        raise ValueError("fonts 为空")
    track = str(parsed.get("track", "")).strip()
    if track not in ("eastAsia", "latin", "cs"):
        track = "latin"
    return {"fonts": fonts[:3], "track": track}


async def _office_font_suggest_llm(lang: str) -> dict:
    """问私有化 LLM：该语言的 Word 文档最常用什么字体、归哪条 rFonts 轨。"""
    sys_prompt = (
        "你是排版专家。用户给出一种目标语言，回答该语言的 Word 文档最常用的 1-2 种"
        "字体（优先 Windows/Office 自带、正式文书常用的），以及该文字系统在 OOXML "
        "rFonts 里归哪条轨：中日韩文字 eastAsia，泰文/阿拉伯文/希伯来文等复杂文种 cs，"
        "其余（拉丁/西里尔/希腊等）latin。"
        '只输出 JSON：{"fonts": ["字体1", "字体2"], "track": "eastAsia|latin|cs"}'
    )
    body = {
        "model": ACTUAL_MODEL_NAME,
        "messages": [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": lang}],
        # 小额度调用：思考型模型可能几十 token 就把 300 吃光，所以留了翻倍重试。
        "stream": False, "temperature": 0.1, "max_tokens": 300,
        **llm_extra_body(),          # 思考开关由档案决定
    }
    headers = llm_headers()
    client = _get_http_client()

    content = await _llm_post_with_budget_retry(
        client, body, headers, timeout=60, what="[office] 字体建议")
    return _office_parse_font_suggestion(content)


@app.post("/office/font_suggest")
async def office_font_suggest(lang: str = Form(default="")):
    """自定义目标语言的字体建议。内置语言查表即回；其余问 LLM 并缓存；
    失败回退拉丁轨 + Times New Roman——给出可用选项永远好过报错。"""
    lang = lang.strip()
    if not lang:
        return {"fonts": ["Times New Roman"], "track": "latin"}
    if lang in _OFFICE_LANG_FONTS:
        track, fonts = _OFFICE_LANG_FONTS[lang]
        return {"fonts": fonts, "track": track}
    if lang in _office_font_cache:
        return _office_font_cache[lang]
    try:
        got = await _office_font_suggest_llm(lang)
        _office_font_cache[lang] = got
        return got
    except Exception as e:
        log.warning("[office] 字体建议失败（%s），回退拉丁轨: %s", lang, e)
        return {"fonts": ["Times New Roman"], "track": "latin"}


@app.get("/download_docx/{file_id}")
async def download_docx(file_id: str):
    """下载翻译后的 DOCX。file_id 是任务分配的 uuid hex，严格校验防路径注入。"""
    if not re.fullmatch(r"[0-9a-f]{32}", file_id):
        return JSONResponse({"error": "无效的文件 ID"}, status_code=404)
    d = OUTPUT_DIR / file_id
    cands = sorted(d.glob("*.docx")) if d.is_dir() else []
    if not cands:
        return JSONResponse({"error": "文件不存在或已清理"}, status_code=404)
    return FileResponse(
        cands[0], filename=cands[0].name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


@app.get("/download_zip/{file_id}")
async def download_zip(file_id: str):
    """下载 output/<file_id>/ 目录打包成的 ZIP（含 .md 与 images/）。"""
    # file_id 仅允许 uuid hex / 安全字符，防止路径穿越
    if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
        return JSONResponse({"error": "invalid file_id"}, status_code=400)

    src = OUTPUT_DIR / file_id
    # 二次校验：解析后必须仍在 OUTPUT_DIR 下
    try:
        src_resolved = src.resolve()
        if OUTPUT_DIR.resolve() not in src_resolved.parents and src_resolved != OUTPUT_DIR.resolve():
            return JSONResponse({"error": "invalid path"}, status_code=400)
    except OSError:
        return JSONResponse({"error": "invalid path"}, status_code=400)
    if not src.exists() or not src.is_dir():
        return JSONResponse({"error": "not found"}, status_code=404)

    buf = io.BytesIO()
    file_count = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src.rglob("*")):
            if not p.is_file():
                continue
            # 跳过遗留临时目录
            rel = p.relative_to(src)
            if rel.parts and rel.parts[0].startswith("_"):
                continue
            zf.write(p, rel)
            file_count += 1
    if file_count == 0:
        return JSONResponse({"error": "empty"}, status_code=404)
    buf.seek(0)

    md_files = sorted(src.glob("*.md"))
    base_name = md_files[0].stem if md_files else file_id
    zip_name = f"{base_name}.zip"
    encoded = quote(zip_name)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "target_api": TARGET_API_URL or "not configured"}


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Doxify 启动")
    log.info("  监听: http://127.0.0.1:%d", GATEWAY_PORT)
    log.info("  目标 LLM: %s", TARGET_API_URL or "(未配置)")
    log.info("  VLM 模型: %s", ACTUAL_MODEL_NAME)
    log.info("  PDF DPI: %d", PDF_DPI)
    log.info("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=GATEWAY_PORT)
