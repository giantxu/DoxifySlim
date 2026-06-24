"""
Doxify — PDF → Markdown 解析 + Markdown 翻译工具

功能：
1. PDF 解析：四种模式（VLM 远程 / MinerU 文字版 / MinerU 扫描版 / PaddleOCR-VL），
   带逐页进度，输出 Markdown + 图片 ZIP 打包下载
2. Markdown 翻译：分块并行流式翻译，含残留英文自动检测与修正

启动方式：python app.py
"""

import os
import base64
import asyncio
import io
import logging
import json
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path
from urllib.parse import quote

import fitz  # PyMuPDF
import httpx
import uvicorn
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
TARGET_API_URL = os.getenv("TARGET_API_URL", "")
TARGET_API_KEY = os.getenv("TARGET_API_KEY", "")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "4000"))
ACTUAL_MODEL_NAME = os.getenv("ACTUAL_MODEL_NAME", "kimi26")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))
PAGE_TIMEOUT = int(os.getenv("PAGE_TIMEOUT", "120"))
PDF_DPI = int(os.getenv("PDF_DPI", "200"))
# 单文件内并发 Worker 数
CONCURRENCY = int(os.getenv("CONCURRENCY", "5"))
# 启用并发的最小页数
CONCURRENCY_THRESHOLD = int(os.getenv("CONCURRENCY_THRESHOLD", "10"))
# 全局最大同时发出的 API 请求数（多文件×多Worker）
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "8"))
# 翻译：每块最大字符数
TRANSLATE_CHUNK_CHARS = int(os.getenv("TRANSLATE_CHUNK_CHARS", "3000"))
# 翻译：默认目标语言
TRANSLATE_TARGET_LANG = os.getenv("TRANSLATE_TARGET_LANG", "中文")
# MinerU 本地处理超时（秒）
MINERU_TIMEOUT = int(os.getenv("MINERU_TIMEOUT", "1800"))
# MinerU 性能调优：MinerU 3.0.x 在 Apple Silicon 上无法识别 MPS 显存，会回落到
# batch_ratio=1（最慢档）。下列变量在子进程启动时注入，让其按统一内存机器调度。
# 详见 mineru/utils/model_utils.py: get_vram()  +  pipeline_analyze.py / hybrid_analyze.py
MINERU_VIRTUAL_VRAM_SIZE = os.getenv("MINERU_VIRTUAL_VRAM_SIZE", "32")  # GB，→ batch_ratio=16
MINERU_HYBRID_BATCH_RATIO = os.getenv("MINERU_HYBRID_BATCH_RATIO", "16")  # hybrid 后端直接指定
MINERU_PDF_RENDER_THREADS = os.getenv("MINERU_PDF_RENDER_THREADS", "8")  # PDF→图片线程
MINERU_DEVICE_MODE = os.getenv("MINERU_DEVICE_MODE", "mps")  # M-series 用 mps；Intel/无 GPU 改 cpu
# 本地 OCR 并发数（PaddleOCR 逐页并发上限）
LOCAL_OCR_CONCURRENCY = int(os.getenv("LOCAL_OCR_CONCURRENCY", "4"))

# PaddleOCR-VL 块间并发数：多个 PDF 块同时处理（共享单例，实测线程安全）。
# MLX 路径下单 GPU 是瓶颈、块内已并发，块间并发主要重叠版面检测(CPU)+填满 server 队列，
# 实测约 1.2× 提速；调大收益递减。CPU 回退路径下建议设为 1（CPU 已被单块吃满）。
PADDLE_VL_CONCURRENCY = int(os.getenv("PADDLE_VL_CONCURRENCY", "3"))

# ── PaddleOCR-VL MLX 加速（Apple Silicon）──
# 把最慢的 VLM 识别环节卸载到独立的 mlx_vlm.server 进程（MLX/GPU），实测约 20× 快于 CPU。
# Doxify 自身不加载 MLX 模型，只通过 vl_rec_backend="mlx-vlm-server" 把识别请求转发给该 server，
# 因此激进依赖（mlx-vlm>=0.3.11 等）全部隔离在独立 venv 里，不污染 mineru 环境。
# server 不可达时本模块自动回退到 device="cpu"。server 由 start.sh 自动拉起，见 .env。
PADDLE_MLX_ENABLED = os.getenv("PADDLE_MLX_ENABLED", "1") == "1"
_PADDLE_MLX_PORT = os.getenv("PADDLE_MLX_SERVER_PORT", "8111")
PADDLE_MLX_SERVER_URL = os.getenv("PADDLE_MLX_SERVER_URL", f"http://127.0.0.1:{_PADDLE_MLX_PORT}/")
PADDLE_MLX_MODEL_NAME = os.getenv("PADDLE_MLX_MODEL_NAME", "PaddlePaddle/PaddleOCR-VL-1.6")

# mlx-vlm-server 后端不支持 min_pixels/max_pixels（这两个参数只对进程内 paddle 后端有效，
# 用于限制送入 VLM 的图片分辨率）。MLX 路径下 paddlex 会对每个识别块各警告一次，纯噪音、
# 不影响结果，这里按消息精确静音。
import warnings as _warnings
_warnings.filterwarnings("ignore", message=r".*does not support `min_pixels`.*")
_warnings.filterwarnings("ignore", message=r".*does not support `max_pixels`.*")

# 解析输出根目录：每个 file_id 一个子目录，存 .md + images/，供 ZIP 下载
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gateway.log")

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
# PaddleOCR-VL 单例实例（按后端缓存："mlx" / "cpu" 各一个）
_paddle_ocr_vl_instances: dict[str, object] = {}
_paddle_ocr_vl_lock = threading.Lock()

@app.on_event("startup")
async def _init_semaphore():
    global _api_semaphore
    _api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    log.info("并发配置: workers=%d, threshold=%d, max_requests=%d",
             CONCURRENCY, CONCURRENCY_THRESHOLD, MAX_CONCURRENT_REQUESTS)

# ---------------------------------------------------------------------------
# 核心：PDF → 图片 → VLM → Markdown
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Markdown 水印剥离（针对 CBP / EAPA 风格的 Barcode 头与 Filed By 脚水印）
# ---------------------------------------------------------------------------

# 头部：行首 "Barcode:" + 标准 DOC 案号结构（<digits>-<digits> <letter>-<digits>-<digits>）。
# 这个锚点覆盖 Investigation/Admin Review/NSR/Sunset Review 等各种案件阶段。
_WATERMARK_HEADER_RE = re.compile(
    r"(?im)^\s*Barcode:\s*\d+-\d+\s+[A-Z]-\d+-\d+.*?$"
)
# 脚部：行首 "Filed By:" + 同一行内同时含 "Filed Date:" 与 "Submission Status:"
_WATERMARK_FOOTER_RE = re.compile(
    r"(?im)^\s*Filed\s*By:\s*\S+.*?Filed\s*Date:.*?Submission\s*Status:.*?$"
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
    """将 PDF 每页转为 PNG 图片字节列表。"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


async def vlm_recognize_page(
    client: httpx.AsyncClient,
    image_bytes: bytes,
    page_num: int,
    total_pages: int,
) -> str:
    """调用 Kimi 2.6 VLM 识别单页图片，返回 Markdown 文本。"""
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
                "8. 如果有无法识别的文字，用 [?] 标记"
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

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TARGET_API_KEY}",
    }

    body = {
        "model": ACTUAL_MODEL_NAME,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.1,
        # 关闭思考模式：Kimi K2.6 默认开启 thinking，会把 max_tokens 大量耗在 reasoning_content
        # 上导致 content 被截断或为空。OCR 任务无需推理。
        "chat_template_kwargs": {"thinking": False},  # vLLM / SGLang 部署
        "thinking": {"type": "disabled"},             # Kimi 官方 API
        "enable_thinking": False,                     # Qwen 协议兼容（参数名沿用 Qwen 约定）
    }

    max_retries = 3
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
                    log.error("VLM 第 %d 页返回 HTTP %d (尝试 %d/%d): %s",
                              page_num, resp.status_code, attempt, max_retries, resp.text[:300])
                    if attempt < max_retries:
                        await asyncio.sleep(2 * attempt)
                        continue
                    return f"[第 {page_num} 页识别失败：HTTP {resp.status_code}]"

                result = resp.json()
                content = result["choices"][0]["message"]["content"]

                # 去除可能的 think 标签（Kimi K2.6 / Qwen 思考模式都会用 <think>...</think> 包裹）
                if "<think>" in content:
                    think_end = content.find("</think>")
                    if think_end != -1:
                        content = content[think_end + len("</think>"):].strip()

                return content

            except httpx.TimeoutException:
                log.warning("VLM 第 %d 页超时 (尝试 %d/%d, 超时 %.0fs)",
                            page_num, attempt, max_retries, timeout_sec)
                if attempt < max_retries:
                    await asyncio.sleep(2 * attempt)
                    continue
                return f"[第 {page_num} 页识别超时]"
            except Exception as e:
                log.warning("VLM 第 %d 页异常 (尝试 %d/%d): %s",
                            page_num, attempt, max_retries, e)
                if attempt < max_retries:
                    await asyncio.sleep(2 * attempt)
                    continue
                log.exception("VLM 第 %d 页最终失败", page_num)
                return f"[第 {page_num} 页识别异常：{e}]"


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
    images: list[bytes],
    page_indices: list[int],  # 1-based 页码
    total: int,
    results: dict[int, str],
    progress_queue: asyncio.Queue,
) -> None:
    """顺序处理一组页面，每页完成后将结果写入 results 并通知 queue。"""
    for page_num in page_indices:
        img = images[page_num - 1]
        text = await vlm_recognize_page(client, img, page_num, total)
        results[page_num] = text
        await progress_queue.put({"type": "page_done", "page": page_num, "text": text})


async def parse_pdf_streaming(
    pdf_bytes: bytes,
    filename: str,
    file_id: str,
    queue: asyncio.Queue,
    strip_watermark: bool = True,
    page_markers: bool = True,
) -> None:
    """
    解析单个 PDF，将进度事件放入 queue。
    事件格式均携带 file_id 以区分多文件场景。
    """
    log.info("[%s] 开始解析: %s (%.1f MB)", file_id, filename, len(pdf_bytes) / 1024 / 1024)

    images = pdf_to_images(pdf_bytes, dpi=PDF_DPI)
    total = len(images)
    log.info("[%s] 共 %d 页 (DPI=%d)", file_id, total, PDF_DPI)

    await queue.put({"type": "file_start", "file_id": file_id,
                     "filename": filename, "total": total})

    results: dict[int, str] = {}
    progress_queue: asyncio.Queue = asyncio.Queue()

    async with httpx.AsyncClient() as client:
        if total > CONCURRENCY_THRESHOLD:
            # 并发模式：均匀分组，组间并发
            groups = _split_into_groups(total, CONCURRENCY)
            log.info("[%s] 并发模式: %d 组 %s",
                     file_id, len(groups), [len(g) for g in groups])
            workers = [
                _process_group(client, images, g, total, results, progress_queue)
                for g in groups
            ]
            # 启动所有 worker，同时从 progress_queue 收集结果转发到主 queue
            worker_tasks = [asyncio.create_task(w) for w in workers]
            completed = 0
            while completed < total:
                evt = await progress_queue.get()
                completed += 1
                await queue.put({**evt, "file_id": file_id,
                                 "done": completed, "total": total})
            await asyncio.gather(*worker_tasks)
        else:
            # 顺序模式：逐页处理
            log.info("[%s] 顺序模式", file_id)
            for i, img in enumerate(images, 1):
                text = await vlm_recognize_page(client, img, i, total)
                results[i] = text
                await queue.put({"type": "page_done", "file_id": file_id,
                                 "page": i, "text": text, "done": i, "total": total})

    # 按页码顺序合并（page_markers 控制是否插入分页标识）
    if page_markers:
        full_md = "\n\n---\n\n".join(
            f"<!-- 第 {i} 页 -->\n\n{results[i]}"
            for i in range(1, total + 1)
        )
    else:
        full_md = "\n\n".join(results[i] for i in range(1, total + 1))
    full_md = _strip_watermarks(full_md, strip_watermark, file_id)
    log.info("[%s] 解析完成: %d 页, %d 字符", file_id, total, len(full_md))
    await queue.put({"type": "file_done", "file_id": file_id,
                     "filename": filename, "pages": total,
                     "chars": len(full_md), "markdown": full_md})


# ---------------------------------------------------------------------------
# 本地处理：MinerU（文字版 / 扫描版）
# ---------------------------------------------------------------------------

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


def _consolidate_md_and_images(
    source_dir: Path, result_dir: Path, prefix: str = "",
    separator: str = "\n\n---\n\n",
) -> tuple[str, int]:
    """递归扫描 source_dir，把所有图片汇总到 result_dir/images/，把所有 .md 内的
    图片引用重写为 images/<name>，再拼接所有 .md 返回。

    prefix: 给图片文件名加前缀以避免不同来源的同名冲突（如 PaddleOCR-VL 多 chunk）。
    返回 (合并后的 markdown, 实际复制的图片数)。
    """
    images_dir = result_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有图片，建立 basename -> new_basename 映射
    img_paths: list[Path] = []
    for ext in _IMG_EXTS:
        img_paths.extend(source_dir.rglob(f"*.{ext}"))
        img_paths.extend(source_dir.rglob(f"*.{ext.upper()}"))

    # 去重（rglob 大小写两次可能命中同一文件，APFS 默认大小写不敏感）
    seen_paths: set[Path] = set()
    unique_imgs: list[Path] = []
    for p in img_paths:
        rp = p.resolve()
        if rp not in seen_paths:
            seen_paths.add(rp)
            unique_imgs.append(p)

    name_map: dict[str, str] = {}
    copied = 0
    for src in unique_imgs:
        target_name = f"{prefix}{src.name}" if prefix else src.name
        dst = images_dir / target_name
        if dst.exists():
            # 同名同大小视作同文件，跳过；否则加数字后缀
            if dst.stat().st_size == src.stat().st_size:
                name_map[src.name] = dst.name
                continue
            stem, ext = os.path.splitext(target_name)
            n = 1
            while (images_dir / f"{stem}_{n}{ext}").exists():
                n += 1
            dst = images_dir / f"{stem}_{n}{ext}"
        shutil.copy2(src, dst)
        copied += 1
        name_map[src.name] = dst.name

    # 收集并合并所有 .md
    md_files = sorted(p for p in source_dir.rglob("*.md") if "content_list" not in p.name)

    def _rewrite(match: re.Match) -> str:
        alt, path = match.group(1), match.group(2)
        if path.startswith(("http://", "https://", "data:")):
            return match.group(0)
        base = os.path.basename(path)
        new_base = name_map.get(base, base)
        return f"![{alt}](images/{new_base})"

    def _rewrite_html_img(match: re.Match) -> str:
        prefix, quote, path = match.group(1), match.group(2), match.group(3)
        if path.startswith(("http://", "https://", "data:")):
            return match.group(0)
        base = os.path.basename(path)
        new_base = name_map.get(base, base)
        return f"{prefix}{quote}images/{new_base}{quote}"

    parts: list[str] = []
    for md in md_files:
        try:
            text = md.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = md.read_text(encoding="utf-8", errors="replace")
        text = _IMG_REF_RE.sub(_rewrite, text)
        text = _HTML_IMG_SRC_RE.sub(_rewrite_html_img, text)
        parts.append(text)

    return (separator.join(parts), copied)


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
    md = re.sub(r"\$\s*\^\{(.*?)\}\s*\$", r"<sup>\1</sup>", md)
    md = re.sub(r"\$\s*_\{(.*?)\}\s*\$", r"<sub>\1</sub>", md)
    return md


# 剥掉 loguru 风格的 "YYYY-MM-DD HH:MM:SS.SSS | LEVEL | module:fn:line - " 前缀，
# 让前端展示的进度行更紧凑。匹配失败时返回原始行。
_MINERU_LOG_PREFIX_RE = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}[\.,]?\d*\s*\|\s*\w+\s*\|\s*[\w\.\-:]+\s*-\s*"
)


def _clean_mineru_line(line: str) -> str:
    return _MINERU_LOG_PREFIX_RE.sub("", line).strip() or line.strip()


def _run_mineru_sync(pdf_bytes: bytes, filename: str, method: str, file_id: str,
                     strip_watermark: bool = True, progress_cb=None,
                     page_markers: bool = True) -> tuple[str, bool]:
    """调用 mineru CLI 解析 PDF，落盘到 output/<file_id>/，返回 (markdown 文本, 是否含图片)。

    持久化目的：把 MinerU 抽取的 images/ 保留下来，供前端 ZIP 下载。
    progress_cb: 可选的回调，每行 mineru 输出调用一次（已剥前缀）。会在 reader 线程里执行，
                 调用方负责把它路由回 asyncio 事件循环（见 parse_pdf_mineru）。
    """
    result_dir = OUTPUT_DIR / file_id
    result_dir.mkdir(parents=True, exist_ok=True)
    # MinerU 输出落地区，待合并后清理；合并后只留 result_dir/<stem>.md + result_dir/images/
    work_dir = result_dir / "_raw"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    safe = filename.replace(" ", "_")
    pdf_path = work_dir / safe
    pdf_path.write_bytes(pdf_bytes)
    # 注入性能调优环境变量（外部已设置的不覆盖）
    env = os.environ.copy()
    env.setdefault("MINERU_VIRTUAL_VRAM_SIZE", MINERU_VIRTUAL_VRAM_SIZE)
    env.setdefault("MINERU_HYBRID_BATCH_RATIO", MINERU_HYBRID_BATCH_RATIO)
    env.setdefault("MINERU_PDF_RENDER_THREADS", MINERU_PDF_RENDER_THREADS)
    env.setdefault("MINERU_DEVICE_MODE", MINERU_DEVICE_MODE)
    # 关闭 tqdm 的 ANSI/颜色，避免输出里混控制字符
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("NO_COLOR", "1")
    # 强制 pipeline 后端：避开 hybrid-auto-engine 在 Apple Silicon 上的 mlx-engine 大文档崩溃问题。
    # pipeline 只用传统 layout + OCR + 公式检测，内存稳定；hybrid-auto-engine 会拉 Qwen2-VL，
    # 跑超过几百页时 worker 进程会无痕崩溃（参见 git history 中关于 547 页德语 PDF 的诊断）。
    log.info("MinerU 启动: %s -b pipeline --method %s -> %s | vram=%s batch_ratio=%s render_threads=%s device=%s timeout=%ds",
             safe, method, result_dir,
             env["MINERU_VIRTUAL_VRAM_SIZE"], env["MINERU_HYBRID_BATCH_RATIO"],
             env["MINERU_PDF_RENDER_THREADS"], env["MINERU_DEVICE_MODE"], MINERU_TIMEOUT)

    # 滚动保留最近若干行用于报错诊断
    output_tail: list[str] = []
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            ["mineru", "-p", str(pdf_path), "-o", str(work_dir),
             "-b", "pipeline", "--method", method],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        def _pump_output() -> None:
            try:
                assert proc is not None and proc.stdout is not None
                for raw in proc.stdout:
                    line = raw.rstrip()
                    if not line:
                        continue
                    log.info("[mineru:%s] %s", file_id[:8], line)
                    output_tail.append(line)
                    if len(output_tail) > 200:
                        del output_tail[:100]
                    if progress_cb is not None:
                        try:
                            progress_cb(_clean_mineru_line(line))
                        except Exception:
                            pass
            except Exception:
                log.exception("MinerU 输出读取异常")

        reader = threading.Thread(target=_pump_output, daemon=True)
        reader.start()

        try:
            rc = proc.wait(timeout=MINERU_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            reader.join(timeout=5)
            tail = "\n".join(output_tail[-30:])
            raise RuntimeError(
                f"MinerU 超时（{MINERU_TIMEOUT}s），已被终止。最后输出：\n{tail}"
            )

        reader.join(timeout=5)
        if rc != 0:
            tail = "\n".join(output_tail[-30:])
            raise RuntimeError(f"MinerU 退出码 {rc}。最后输出：\n{tail[:1500]}")

        _sep = "\n\n---\n\n" if page_markers else "\n\n"
        combined_md, img_count = _consolidate_md_and_images(work_dir, result_dir, separator=_sep)
        if not combined_md.strip():
            combined_md = "[MinerU 未生成 Markdown 文件]"
        combined_md = _strip_watermarks(combined_md, strip_watermark, file_id)

        # 用原始文件名写最终 .md（ZIP 下载时呈现给用户）
        final_md = result_dir / f"{_safe_doc_stem(filename)}.md"
        final_md.write_text(combined_md, encoding="utf-8")
        log.info("[MinerU] %s: %d 张图片, %d 字符 -> %s",
                 filename, img_count, len(combined_md), final_md)
        return combined_md, img_count > 0
    finally:
        # 防御：如果出现异常但子进程还活着，确保被回收
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        shutil.rmtree(work_dir, ignore_errors=True)


async def parse_pdf_mineru(
    pdf_bytes: bytes,
    filename: str,
    file_id: str,
    queue: asyncio.Queue,
    method: str = "txt",
    strip_watermark: bool = True,
    page_markers: bool = True,
) -> None:
    """调用本地 MinerU CLI 解析 PDF。total=0 表示不定进度（整体黑盒）。
    通过 progress_cb 实时把 mineru 的每行输出转发给前端 SSE。"""
    loop = asyncio.get_event_loop()
    await queue.put({"type": "file_start", "file_id": file_id,
                     "filename": filename, "total": 0, "mode": f"mineru_{method}"})

    def _on_progress(message: str) -> None:
        # 在 reader 线程中调用；用 call_soon_threadsafe 把事件投递回事件循环
        msg = message[:200] if message else ""
        if not msg:
            return
        try:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "file_progress", "file_id": file_id,
                 "filename": filename, "message": msg},
            )
        except RuntimeError:
            # 事件循环已关闭：忽略
            pass

    try:
        md, has_images = await loop.run_in_executor(
            None, _run_mineru_sync, pdf_bytes, filename, method, file_id,
            strip_watermark, _on_progress, page_markers,
        )
        log.info("[%s] MinerU 完成: %d 字符, has_images=%s", file_id, len(md), has_images)
        await queue.put({"type": "file_done", "file_id": file_id,
                         "filename": filename, "pages": 0,
                         "chars": len(md), "markdown": md,
                         "has_images": has_images})
    except Exception as e:
        log.exception("MinerU 处理失败: %s", filename)
        await queue.put({"type": "file_error", "file_id": file_id,
                         "filename": filename, "error": str(e)})


# ---------------------------------------------------------------------------
# 本地处理：PaddleOCR-VL（VLM 文档解析，含表格/公式/图表）
# ---------------------------------------------------------------------------

# 每块页数。注意：PaddleOCR-VL 默认 max_num_input_imgs=100，单次 predict() 超过 100 页的
# 部分会被静默丢弃（见百度 AI Studio API 文档）。这里按 3 页切块，天然规避该限制——
# 不要把此值改到 >100，否则尾页会丢失。
PADDLEOCR_VL_CHUNK_PAGES = 3


def _mlx_server_alive(url: str | None = None) -> bool:
    """探测 mlx_vlm.server 是否在线：直接对 host:port 建 TCP 连接。

    用裸 socket 而非 HTTP，是为了免疫系统/环境代理——urllib 可能把对 127.0.0.1 的请求
    也走代理并返回一个 HTTP 错误页，从而把“没人监听”误判成“在线”。TCP 直连只看端口是否可达。
    """
    import socket
    from urllib.parse import urlparse
    p = urlparse(url or PADDLE_MLX_SERVER_URL)
    host = p.hostname or "127.0.0.1"
    port = p.port or 80
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except Exception:
        return False


def get_paddle_ocr_vl(use_mlx: bool = False):
    """获取 PaddleOCR-VL 单例（懒初始化 + 双重检查锁，按后端分别缓存）。

    use_mlx=True 时走 mlx-vlm-server 后端（Apple Silicon GPU，实测约 20× 快于 CPU）；
    否则用纯 CPU。调用方应先用 _mlx_server_alive() 判断 server 是否可达再决定 use_mlx。
    """
    key = "mlx" if use_mlx else "cpu"
    inst = _paddle_ocr_vl_instances.get(key)
    if inst is None:
        with _paddle_ocr_vl_lock:
            inst = _paddle_ocr_vl_instances.get(key)
            if inst is None:
                from paddleocr import PaddleOCRVL
                if use_mlx:
                    log.info("初始化 PaddleOCR-VL 实例 (mlx-vlm-server: %s, model=%s)",
                             PADDLE_MLX_SERVER_URL, PADDLE_MLX_MODEL_NAME)
                    inst = PaddleOCRVL(
                        vl_rec_backend="mlx-vlm-server",
                        vl_rec_server_url=PADDLE_MLX_SERVER_URL,
                        vl_rec_api_model_name=PADDLE_MLX_MODEL_NAME,
                    )
                else:
                    log.info("初始化 PaddleOCR-VL 实例 (device=cpu)")
                    inst = PaddleOCRVL(device="cpu")
                _paddle_ocr_vl_instances[key] = inst
    return inst


def _split_pdf_chunk(pdf_bytes: bytes, start_page: int, end_page: int) -> bytes:
    """从 PDF 中提取 [start_page, end_page)（0 索引）生成新的 PDF bytes。"""
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    dst = fitz.open()
    dst.insert_pdf(src, from_page=start_page, to_page=end_page - 1)
    return dst.tobytes()


def _run_paddleocr_vl_chunk_sync(chunk_bytes: bytes, stem: str, chunk_idx: int,
                                   start_page: int, end_page: int, file_id: str,
                                   use_mlx: bool = False) -> str:
    """用 PaddleOCR-VL 解析单个 PDF 块（同步阻塞，在 executor 中运行）。

    输出落盘到 output/<file_id>/_chunks/chunk_<idx>/output/，等所有块完成后由调用方
    统一合并到 output/<file_id>/。
    """
    import time

    pipeline = get_paddle_ocr_vl(use_mlx=use_mlx)
    t0 = time.time()
    chunk_root = OUTPUT_DIR / file_id / "_chunks" / f"chunk_{chunk_idx:03d}"
    if chunk_root.exists():
        shutil.rmtree(chunk_root, ignore_errors=True)
    chunk_root.mkdir(parents=True, exist_ok=True)

    pdf_path = chunk_root / f"{stem}_c{chunk_idx}.pdf"
    out_dir = chunk_root / "output"
    out_dir.mkdir(exist_ok=True)
    pdf_path.write_bytes(chunk_bytes)

    pages_res = list(pipeline.predict(str(pdf_path)))
    if not pages_res:
        log.warning("[PaddleOCR-VL] 块 %d（第 %d-%d 页）返回空结果",
                    chunk_idx, start_page + 1, end_page)
        return ""

    restructured = pipeline.restructure_pages(
        pages_res,
        merge_tables=True,
        relevel_titles=True,
        concatenate_pages=True,
    )
    for res in restructured:
        res.save_to_markdown(save_path=str(out_dir))

    md_files = sorted(out_dir.rglob("*.md"))
    elapsed = time.time() - t0
    log.info("[PaddleOCR-VL] 块 %d（第 %d-%d 页）完成，耗时 %.1fs，%d 个 .md",
             chunk_idx, start_page + 1, end_page, elapsed, len(md_files))
    if not md_files:
        return ""
    # 返回的是块的原始 md（含相对图片引用），仅用于前端实时进度展示；
    # file_done 时调用方会用合并后的 md 覆盖之。
    return _normalize_vlm_latex("\n\n".join(p.read_text(encoding="utf-8") for p in md_files))


async def parse_pdf_paddleocr(
    pdf_bytes: bytes,
    filename: str,
    file_id: str,
    queue: asyncio.Queue,
    lang: str = "ch",  # 保留参数以兼容 task_map，VL 模式不使用
    strip_watermark: bool = True,
    use_mlx: bool = True,  # 请求是否希望用 MLX 加速；server 不可达时自动回退 CPU
    page_markers: bool = True,
) -> None:
    """用 PaddleOCR-VL 解析 PDF，按块处理，有实时逐块进度。"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    doc.close()

    chunk_size = PADDLEOCR_VL_CHUNK_PAGES
    num_chunks = (total_pages + chunk_size - 1) // chunk_size
    stem = Path(filename).stem or "doc"
    loop = asyncio.get_event_loop()

    # 决定实际后端：请求要 MLX 且全局开关开 且 server 可达，才用 MLX，否则回退 CPU。
    mlx_active = bool(use_mlx) and PADDLE_MLX_ENABLED and _mlx_server_alive()
    if use_mlx and PADDLE_MLX_ENABLED and not mlx_active:
        log.warning("[%s] PaddleOCR-VL: 请求 MLX 但 server (%s) 不可达，回退 CPU",
                    file_id, PADDLE_MLX_SERVER_URL)
    backend = "mlx" if mlx_active else "cpu"

    log.info("[%s] PaddleOCR-VL 启动: %s, %d 页, %d 块（每块 %d 页），后端=%s",
             file_id, filename, total_pages, num_chunks, chunk_size, backend)
    await queue.put({"type": "file_start", "file_id": file_id,
                     "filename": filename, "total": num_chunks, "mode": "paddleocr",
                     "backend": backend})

    result_dir = OUTPUT_DIR / file_id
    result_dir.mkdir(parents=True, exist_ok=True)

    # 块间并发：MLX 路径用 PADDLE_VL_CONCURRENCY；CPU 回退降为 1（单块已吃满 CPU）。
    concurrency = max(1, PADDLE_VL_CONCURRENCY if mlx_active else 1)
    concurrency = min(concurrency, num_chunks)
    sem = asyncio.Semaphore(concurrency)
    chunk_markdowns: list[str] = [""] * num_chunks
    done_count = 0

    async def _process_chunk(chunk_idx: int) -> None:
        nonlocal done_count
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_pages)
        chunk_bytes = _split_pdf_chunk(pdf_bytes, start, end)
        async with sem:
            log.info("[%s] PaddleOCR-VL 开始块 %d/%d（第 %d-%d 页）",
                     file_id, chunk_idx + 1, num_chunks, start + 1, end)
            chunk_md = await loop.run_in_executor(
                None, _run_paddleocr_vl_chunk_sync, chunk_bytes, stem,
                chunk_idx + 1, start, end, file_id, mlx_active,
            )
        # 完成顺序可能乱序，但按 chunk_idx 回填保证最终顺序；前端按 page 键存储亦不受影响。
        chunk_markdowns[chunk_idx] = chunk_md
        done_count += 1
        await queue.put({"type": "page_done", "file_id": file_id,
                         "page": chunk_idx + 1, "text": chunk_md,
                         "done": done_count, "total": num_chunks})

    try:
        log.info("[%s] PaddleOCR-VL 块间并发度=%d", file_id, concurrency)
        await asyncio.gather(*(_process_chunk(i) for i in range(num_chunks)))

        # 合并所有 chunk 的 md + 图片到 result_dir/，重写图片引用
        chunks_root = result_dir / "_chunks"
        _sep = "\n\n---\n\n" if page_markers else "\n\n"
        try:
            full_md, img_count = await loop.run_in_executor(
                None, _consolidate_md_and_images, chunks_root, result_dir, "", _sep
            )
        finally:
            shutil.rmtree(chunks_root, ignore_errors=True)
        if not full_md.strip():
            full_md = _sep.join(m for m in chunk_markdowns if m.strip())
        full_md = _normalize_vlm_latex(full_md)
        full_md = _strip_watermarks(full_md, strip_watermark, file_id)

        final_md = result_dir / f"{_safe_doc_stem(filename)}.md"
        final_md.write_text(full_md, encoding="utf-8")
        log.info("[%s] PaddleOCR-VL 完成: %d 块, %d 字符, %d 张图片",
                 file_id, num_chunks, len(full_md), img_count)
        await queue.put({"type": "file_done", "file_id": file_id,
                         "filename": filename, "pages": num_chunks,
                         "chars": len(full_md), "markdown": full_md,
                         "has_images": img_count > 0})
    except Exception as e:
        log.exception("[%s] PaddleOCR-VL 失败: %s", file_id, filename)
        await queue.put({"type": "file_error", "file_id": file_id,
                         "filename": filename, "error": str(e)})


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
  /* 卡内子选项（如 PaddleOCR-VL 的 MLX 加速开关）*/
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
</nav>
<div class="main">
  <div class="hero">
    <div class="dotmatrix"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
    <div class="overline"><span class="sq"></span><em>Document Parsing · 文档解析</em></div>
    <h1>PDF <b>智能解析</b></h1>
    <p class="subtitle">选择解析模式，拖入 PDF，实时输出结构化 Markdown，支持多文件并行处理。</p>
  </div>

  <div class="section-label">解析模式</div>
  <!-- 模式选择 -->
  <div class="mode-grid" id="modeGrid">
    <label class="mode-card selected" onclick="selectMode('vlm')">
      <input type="radio" name="mode" value="vlm" checked>
      <div class="mode-icon">🌐</div>
      <div class="mode-title">Kimi 2.6 VLM</div>
      <span class="mode-badge">远程 API</span>
      <div class="mode-desc">手写 / 复杂版式 / 图文混排最佳<br>逐页实时进度</div>
    </label>
    <label class="mode-card" onclick="selectMode('mineru_txt')">
      <input type="radio" name="mode" value="mineru_txt">
      <div class="mode-icon">📄</div>
      <div class="mode-title">MinerU 文字版</div>
      <span class="mode-badge">本地 · 数字 PDF</span>
      <div class="mode-desc">版式结构完美还原<br>无需 API · 整体处理</div>
    </label>
    <label class="mode-card" onclick="selectMode('mineru_ocr')">
      <input type="radio" name="mode" value="mineru_ocr">
      <div class="mode-icon">🖥️</div>
      <div class="mode-title">MinerU 扫描版</div>
      <span class="mode-badge">本地 · 扫描 PDF</span>
      <div class="mode-desc">版式保留好 · ch_lite 模型<br>整体处理 · 适合印刷体</div>
    </label>
    <label class="mode-card" onclick="selectMode('paddleocr')">
      <input type="radio" name="mode" value="paddleocr">
      <div class="mode-icon">🧠</div>
      <div class="mode-title">PaddleOCR-VL</div>
      <span class="mode-badge">本地 · VLM 文档解析</span>
      <div class="mode-desc">0.9B VLM · 表格/公式/图表<br>109 语言 · 结构化 Markdown</div>
      <span class="mode-opt">
        <input type="checkbox" id="paddleMlx" checked>
        <span onclick="var c=document.getElementById('paddleMlx'); c.checked=!c.checked;">MLX 加速 · Apple Silicon 约 20×</span>
      </span>
    </label>
  </div>

  <div class="section-label">处理选项</div>
  <!-- 水印过滤选项（对全部四种解析模式均生效） -->
  <div class="lang-row">
    <label>
      <input type="checkbox" id="stripWatermark" checked>
      去除页眉/页脚水印（Barcode / Filed By 行）· 适用于所有模式
    </label>
  </div>

  <div class="lang-row">
    <label>
      <input type="checkbox" id="pageMarkers" checked>
      在 Markdown 中插入分页标识（页码注释 / 分隔线）· 适用于所有模式
    </label>
  </div>

  <div class="section-label">上传文档</div>
  <div class="drop-zone" id="dropZone">
    <p>将 PDF 文件拖到这里，或点击选择文件</p>
    <small>支持同时上传多个 PDF，自动并行处理</small>
    <input type="file" id="fileInput" accept=".pdf" multiple>
  </div>

  <div id="fileCards"></div>
</div>

<footer class="foot">
  <img src="/static/jtn-logo.png" alt="JT&N 金诚同达">
  <span>金诚同达律师事务所　·　Doxify 文档智能工具</span>
</footer>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileCards = document.getElementById('fileCards');
const fileState = {};

let selectedMode = 'vlm';

function selectMode(mode) {
  selectedMode = mode;
  document.querySelectorAll('.mode-card').forEach(c => c.classList.remove('selected'));
  document.querySelectorAll('.mode-card input[type=radio]').forEach(r => {
    if (r.value === mode) { r.checked = true; r.closest('.mode-card').classList.add('selected'); }
  });
}

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

async function uploadFiles(files) {
  const formData = new FormData();
  for (const f of files) formData.append('files', f);
  formData.append('mode', selectedMode);
  const strip = document.getElementById('stripWatermark');
  formData.append('strip_watermark', (strip && strip.checked) ? '1' : '0');
  const mlx = document.getElementById('paddleMlx');
  formData.append('paddle_mlx', (mlx && mlx.checked) ? '1' : '0');
  const pm = document.getElementById('pageMarkers');
  formData.append('page_markers', (pm && pm.checked) ? '1' : '0');

  const tmpCards = [];
  for (const f of files) {
    const tmpId = 'tmp_' + Math.random().toString(36).slice(2);
    createCard(tmpId, f.name);
    tmpCards.push(document.getElementById('card-' + tmpId));
  }

  const startTime = Date.now();
  try {
    const resp = await fetch('/parse_pdf_stream', { method: 'POST', body: formData });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6);
        if (raw === '[DONE]') continue;
        try { handleEvent(JSON.parse(raw), tmpCards, startTime); } catch(e) {}
      }
    }
  } catch(e) {
    console.error('上传失败', e);
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
      // 不定进度：MinerU 整体处理
      st.indeterminate = true;
      document.getElementById('bar-' + fid).classList.add('indeterminate');
      setStatus(fid, '<span class="spinner"></span>本地处理中（请稍候）...', false);
    } else {
      setStatus(fid, `<span class="spinner"></span>识别中 (0/${evt.total})`, false);
    }
  } else if (evt.type === 'page_done') {
    const { page, text, done, total } = evt;
    st.pageTexts[page] = text;
    const pct = Math.round((done / total) * 100);
    document.getElementById('bar-' + fid).style.width = pct + '%';
    setStatus(fid, `<span class="spinner"></span>识别中 (${done}/${total})`, false);
  } else if (evt.type === 'file_progress') {
    // MinerU 实时进度：把每行输出展示为状态文字
    const safe = String(evt.message || '').replace(/[<>&"]/g, c => ({
      '<':'&lt;', '>':'&gt;', '&':'&amp;', '"':'&quot;'
    }[c]));
    setStatus(fid, '<span class="spinner"></span>' + safe, false);
  } else if (evt.type === 'file_done') {
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

    // 优先使用服务端发来的 markdown（MinerU 模式），否则从 pageTexts 重建
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
    // 含图片时显示 ZIP 下载按钮（MinerU / PaddleOCR-VL 才会有）
    if (evt.has_images) {
      const zb = document.getElementById('zip-' + fid);
      if (zb) zb.style.display = '';
    }
  } else if (evt.type === 'file_error') {
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
</script>
</body>
</html>
"""


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
# 残留英文检测 & 二次修正（针对 Kimi 偶尔保留英文形容词的行为）
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
        "enable_thinking": False,                     # Qwen 协议兼容
        "chat_template_kwargs": {"thinking": False},  # vLLM / SGLang
        "thinking": {"type": "disabled"},             # Kimi 官方 API
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TARGET_API_KEY}",
    }
    async with _api_semaphore:
        resp = await client.post(TARGET_API_URL, json=body, headers=headers,
                                  timeout=LLM_TIMEOUT)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    if "<think>" in content:
        end = content.find("</think>")
        if end != -1:
            content = content[end + len("</think>"):].strip()
    return content.strip()


async def translate_chunk_stream(
    client: httpx.AsyncClient,
    chunk: str,
    chunk_idx: int,
    total_chunks: int,
    target_lang: str,
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
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TARGET_API_KEY}",
    }
    # max_tokens：输入最多 3000 字，翻译后一般不超过 2x，6000 token 足够；
    # 不设上限时模型可能进入循环生成，导致单块耗时 10+ 分钟。
    body = {
        "model": ACTUAL_MODEL_NAME,
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
        "max_tokens": 6000,
        # 关闭思考模式：覆盖 Qwen / Kimi K2.6 / 官方 API 三种约定
        "enable_thinking": False,                     # Qwen 协议兼容（参数名沿用 Qwen 约定）
        "chat_template_kwargs": {"thinking": False},  # vLLM / SGLang
        "thinking": {"type": "disabled"},             # Kimi 官方 API
    }

    try:
        async with _api_semaphore:
            req = client.build_request("POST", TARGET_API_URL, json=body, headers=headers)
            resp = await client.send(req, stream=True)

        buffer = ""
        in_think = False   # 过滤 <think>...</think> 块（Kimi K2.6 / Qwen 思维链输出）
        think_buf = ""
        # 思考泄漏诊断计数器：>0 即说明服务端未完全遵守 thinking=disabled
        think_chars_stripped = 0   # 被剥掉的 <think>...</think>（含标签自身）字符数
        reasoning_chars = 0        # delta.reasoning_content 字段累计长度（Kimi 官方 API 的另一种思考字段）

        def _log_thinking_leak() -> None:
            # 阈值：标签自身 15 字符 + 一点容差；reasoning_content 任何长度都报
            if think_chars_stripped > 20 or reasoning_chars > 0:
                log.warning(
                    "第 %d/%d 块检测到思考泄漏: <think> 段剥离 %d 字符, "
                    "reasoning_content 累计 %d 字符。服务端可能未完全遵守 "
                    "thinking=disabled，请检查 Kimi 部署。",
                    chunk_idx, total_chunks, think_chars_stripped, reasoning_chars,
                )

        async for raw in resp.aiter_bytes():
            buffer += raw.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    _log_thinking_leak()
                    return
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0].get("delta", {})
                    reasoning = delta.get("reasoning_content") or ""
                    if reasoning:
                        reasoning_chars += len(reasoning)
                    token = delta.get("content", "")
                    if not token:
                        continue
                    # 过滤 <think>...</think>，同时统计被剥字符数
                    think_buf += token
                    while True:
                        if in_think:
                            end = think_buf.find("</think>")
                            if end == -1:
                                think_chars_stripped += len(think_buf)
                                think_buf = ""  # 思考内容持续中，全丢弃
                                break
                            think_chars_stripped += end + len("</think>")
                            think_buf = think_buf[end + len("</think>"):]
                            in_think = False
                        else:
                            start = think_buf.find("<think>")
                            if start == -1:
                                yield think_buf
                                think_buf = ""
                                break
                            if start > 0:
                                yield think_buf[:start]
                            think_chars_stripped += len("<think>")
                            think_buf = think_buf[start + len("<think>"):]
                            in_think = True
                except Exception:
                    pass
        await resp.aclose()
        _log_thinking_leak()

    except httpx.TimeoutException:
        yield f"\n[第 {chunk_idx}/{total_chunks} 块翻译超时]"
    except Exception as e:
        log.exception("翻译块 %d 异常", chunk_idx)
        yield f"\n[第 {chunk_idx}/{total_chunks} 块翻译失败: {e}]"


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------

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
</nav>

<div class="container">
  <div class="hero">
    <div class="overline"><span class="sq"></span><em>Markdown Translation · 文档翻译</em></div>
    <h1>Markdown <b>智能翻译</b></h1>
    <p class="subtitle">粘贴文本或上传多个 Markdown 文件，Kimi 2.6 流式翻译，保留完整格式。</p>
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
      <small>支持同时选择多个文件，选好后点"开始翻译"</small>
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
  if (isTranslating) return;
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
    document.getElementById('pasteLabel').textContent = '准备中...';
  }

  isTranslating = true;
  const btn = document.getElementById('translateBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>翻译中...';
  const startTime = Date.now();

  try {
    const resp = await fetch('/translate_stream', { method: 'POST', body: formData });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6);
        if (raw === '[DONE]') continue;
        try { handleEvent(JSON.parse(raw), isFileMode, startTime); } catch(e) {}
      }
    }
  } catch(e) {
    if (!isFileMode) document.getElementById('pasteLabel').textContent = '请求失败: ' + e.message;
    console.error(e);
  } finally {
    isTranslating = false;
    btn.disabled = false;
    btn.textContent = '开始翻译';
    if (isFileMode && selectedFiles.length > 0) renderFileQueue();
  }
}

function handleEvent(evt, isFileMode, startTime) {
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
      let preview = '';
      for (let i = 1; i <= pasteTotalChunks; i++) {
        if (pasteChunkBufs[i]) preview += (preview ? '\\n\\n' : '') + pasteChunkBufs[i];
      }
      document.getElementById('outputArea').value = preview;
    }
  } else if (evt.type === 'chunk_replace') {
    // 服务端检测到该块残留英文，已生成修正稿；用修正稿覆盖该块缓冲
    const { file_id, chunk, text } = evt;
    if (isFileMode) {
      const st = fileStates[file_id]; if (!st) return;
      st.chunkBufs[chunk] = text;
    } else {
      pasteChunkBufs[chunk] = text;
      let preview = '';
      for (let i = 1; i <= pasteTotalChunks; i++) {
        if (pasteChunkBufs[i]) preview += (preview ? '\\n\\n' : '') + pasteChunkBufs[i];
      }
      document.getElementById('outputArea').value = preview;
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
      let md = '';
      for (let i = 1; i <= st.totalChunks; i++) md += (md ? '\\n\\n' : '') + (st.chunkBufs[i] || '');
      st.finalMd = md;
      const bar = document.getElementById('bar-' + evt.file_id);
      const sta = document.getElementById('status-' + evt.file_id);
      const bts = document.getElementById('btns-' + evt.file_id);
      if (bar) { bar.style.width = '100%'; bar.classList.add('done'); }
      if (sta) { sta.textContent = '完成'; sta.className = 'file-status done'; }
      if (bts) bts.style.display = 'flex';
    } else if (evt.name_suffix) {
      pasteNameSuffix = evt.name_suffix;
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
  if (!st?.finalMd) return;
  const suffix = st.nameSuffix ? ('_' + st.nameSuffix) : '_translated';
  const name = st.filename.replace(/\\.(md|markdown|txt)$/i, '') + suffix + '.md';
  const blob = new Blob([st.finalMd], { type: 'text/markdown' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
}
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


@app.post("/translate_stream")
async def translate_stream(
    text: str = Form(default=""),
    target_lang: str = Form(default="中文"),
    files: list[UploadFile] = File(default=[]),
):
    """SSE 流式翻译 Markdown，支持多文件或单段文本。"""
    lang = target_lang.strip() or TRANSLATE_TARGET_LANG

    # 预处理：收集所有待翻译内容 (file_id, filename, chunks)
    pre_items: list[tuple[str, str, list[str]]] = []

    valid_files = [f for f in files if f and f.filename and f.filename.strip()]
    if valid_files:
        for i, f in enumerate(valid_files):
            raw = await f.read()
            content = raw.decode("utf-8", errors="replace")
            if content.strip():
                chunks = split_markdown_chunks(content, max_chars=TRANSLATE_CHUNK_CHARS)
                pre_items.append((f"file_{i}", f.filename, chunks))
    elif text.strip():
        chunks = split_markdown_chunks(text, max_chars=TRANSLATE_CHUNK_CHARS)
        pre_items.append(("text_0", "输入文本", chunks))

    if not pre_items:
        return JSONResponse({"success": False, "error": "内容为空"})

    log.info("翻译任务: %d 个文件 (并行), 目标语言=%s", len(pre_items), lang)

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
            buf = ""
            try:
                async for token in translate_chunk_stream(client, chunk, idx, total, lang):
                    buf += token
                    await queue.put({"type": "chunk_token", "file_id": file_id,
                                     "chunk": idx, "token": token})
            except Exception as e:
                log.exception("[%s] 第 %d 块翻译异常: %s", filename, idx, e)
                await queue.put({"type": "chunk_done", "file_id": file_id, "chunk": idx})
                return

            # 残留英文检测 + 一次性修正
            residuals = _detect_residual_english(buf)
            if residuals:
                log.info("[%s] 第 %d 块检测到残留英文 %s，启动修正",
                         filename, idx, residuals[:8])
                try:
                    fixed = await _fix_residual_english(client, buf, residuals, lang)
                    if fixed and fixed != buf:
                        await queue.put({"type": "chunk_replace", "file_id": file_id,
                                         "chunk": idx, "text": fixed})
                        log.info("[%s] 第 %d 块修正完成（%d → %d 字符）",
                                 filename, idx, len(buf), len(fixed))
                except Exception as e:
                    log.warning("[%s] 第 %d 块修正失败，保留原译: %s",
                                filename, idx, e)

            await queue.put({"type": "chunk_done", "file_id": file_id, "chunk": idx})

        async with httpx.AsyncClient(timeout=httpx.Timeout(LLM_TIMEOUT, connect=10.0)) as client:
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

    async def _generate():
        init_evt = {
            "type": "init",
            "files": [
                {"file_id": fid, "filename": fname, "total_chunks": len(cks)}
                for fid, fname, cks in pre_items
            ],
        }
        yield f"data: {json.dumps(init_evt, ensure_ascii=False)}\n\n"

        queue: asyncio.Queue = asyncio.Queue()
        # 并行启动所有文件的翻译任务
        tasks = [
            asyncio.create_task(_translate_file_task(fid, fname, cks, queue))
            for fid, fname, cks in pre_items
        ]

        # 收集事件直到所有文件完成
        done_count = 0
        while done_count < len(tasks):
            evt = await queue.get()
            if evt["type"] == "file_done":
                done_count += 1
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

        await asyncio.gather(*tasks)
        yield f"data: {json.dumps({'type': 'all_done'}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.post("/parse_pdf_stream")
async def parse_pdf_stream(
    files: list[UploadFile] = File(...),
    strip_watermark: str = Form(default="1"),
    page_markers: str = Form(default="1"),
):
    """
    SSE 流式解析，支持同时上传多个 PDF。
    解析后端：Kimi 2.6 VLM（逐页实时进度）。
    多文件并行处理，前端按 file_id 区分。
    """
    # 过滤非 PDF
    valid = [f for f in files if f.filename and f.filename.lower().endswith(".pdf")]
    if not valid:
        return JSONResponse({"success": False, "error": "请上传 PDF 文件"})

    # 预读所有文件内容（UploadFile 不能在 async generator 外部访问）
    file_data = []
    for f in valid:
        data = await f.read()
        file_data.append((uuid.uuid4().hex, f.filename, data))

    strip_wm = strip_watermark == "1"
    pm = page_markers == "1"
    log.info("PDF 解析请求: %d 个文件, strip_watermark=%s, page_markers=%s",
             len(file_data), strip_wm, pm)

    async def _generate():
        queue: asyncio.Queue = asyncio.Queue()
        tasks = [
            asyncio.create_task(
                parse_pdf_streaming(data, filename, file_id, queue, strip_wm, pm)
            )
            for file_id, filename, data in file_data
        ]

        # 发送初始化事件：告知前端文件列表
        init_evt = {"type": "init", "files": [
            {"file_id": fid, "filename": fname}
            for fid, fname, _ in file_data
        ]}
        yield f"data: {json.dumps(init_evt, ensure_ascii=False)}\n\n"

        # 收集事件直到所有任务完成
        done_count = 0
        total_files = len(tasks)
        error_count = 0

        while done_count + error_count < total_files:
            evt = await queue.get()
            t = evt.get("type")
            if t == "file_done":
                done_count += 1
            elif t == "file_error":
                error_count += 1
            # VLM 和 PaddleOCR 模式：page_done 已携带文本，file_done 的 markdown 可省略
            # MinerU 模式：无 page_done，必须发送 file_done.markdown
            # 策略：始终发送 markdown（前端优先用 evt.markdown，否则重建）
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

        await asyncio.gather(*tasks)
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


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
