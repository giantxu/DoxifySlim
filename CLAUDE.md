# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

DoxifySlim 是 Doxify 的精简版（**已移除 MinerU / PaddleOCR-VL**），仅保留 Kimi 2.6 VLM 解析路径和 Markdown 翻译。  
单文件 FastAPI 应用 (`app.py`)，pip-venv 安装，无本地模型下载。

## 启动方式

```bash
# 激活虚拟环境后直接启动
source .venv/bin/activate
python app.py
```

或使用 macOS 脚本（自动激活 venv）：

```bash
bash start.sh
```

默认监听 `http://127.0.0.1:4000`（由 `.env` 的 `GATEWAY_PORT` 控制）。

## 安装

```bash
bash install.sh   # macOS（多镜像自动回退：阿里云→清华→中科大）
install.bat       # Windows
```

脚本创建 `.venv`，安装 `requirements.txt` 中的 7 个依赖（无模型下载），首次运行时从 `.env.example` 复制 `.env`。

## 环境配置

所有配置读自 `.env`（从 `.env.example` 复制后填写）：

| 变量 | 说明 |
|---|---|
| `TARGET_API_URL` | OpenAI 兼容的 chat/completions 端点 |
| `TARGET_API_KEY` | API 密钥 |
| `ACTUAL_MODEL_NAME` | 模型名（VLM 解析与翻译共用） |
| `GATEWAY_PORT` | 服务端口，默认 4000 |
| `LLM_TIMEOUT` | LLM 请求总超时（秒），默认 300 |
| `PAGE_TIMEOUT` | 单页 VLM 超时（秒），默认 120 |
| `PDF_DPI` | PDF→图片分辨率，默认 200 |
| `CONCURRENCY` | 单文件内并发 Worker 数，默认 5 |
| `CONCURRENCY_THRESHOLD` | 启用并发的最小页数，默认 10 |
| `MAX_CONCURRENT_REQUESTS` | 全局最大同时 API 请求数，默认 8 |
| `TRANSLATE_CHUNK_CHARS` | 翻译每块最大字符数，默认 3000 |
| `TRANSLATE_TARGET_LANG` | 默认翻译目标语言，默认"中文" |

## 架构

单文件：`app.py`。两个功能区：

### 1. PDF 解析 — `/`（Web UI）+ `POST /parse_pdf_stream`（SSE 端点）

**调用链**：  
`POST /parse_pdf_stream` → `parse_pdf_streaming(data, filename, file_id, queue, strip_wm, pm)` → 多个 `vlm_recognize_page` 并发任务

**关键函数**：
- `pdf_to_images(pdf_bytes, dpi)` — PyMuPDF 将 PDF 每页转为 PNG 字节列表
- `vlm_recognize_page(client, image_bytes, page_num, total_pages)` — 单页 VLM 识别，最多 3 次重试，超时递增 50%；发送三个关闭思考模式的标志（`enable_thinking`/`chat_template_kwargs`/`thinking`）
- `_split_into_groups(total, n_groups)` — 将页码均匀分成 n 组，供并发 Worker 使用
- `parse_pdf_streaming(data, filename, file_id, queue, strip_watermark, page_markers)` — 主编排函数；页数 ≥ `CONCURRENCY_THRESHOLD` 时启用并发（`asyncio.gather`），否则顺序处理；`strip_watermark=True` 时对每页结果调用 `_strip_watermarks`；`page_markers=True` 时插入分页分隔符
- `_strip_watermarks(md, enabled, file_id)` — 用正则剥离 EAPA 风格 Barcode 头部水印（`_WATERMARK_HEADER_RE`）和 Filed By 脚部水印（`_WATERMARK_FOOTER_RE`）
- `_api_semaphore` — `asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)`，限制全局同时发出的 API 请求数

**表单字段**：`files`（多文件）、`strip_watermark`（bool）、`page_markers`（bool）

**SSE 事件**：`init` → `file_start`（`total` = 总页数）→ `page_done` × N → `file_done`（含 `markdown` 字段）→（`file_error`）

### 2. Markdown 翻译 — `/translate`（Web UI）+ `POST /translate_stream`（SSE 端点）

**关键函数**：
- `split_text_into_chunks(text, max_chars)` — 在段落边界切块，块大小不超过 `TRANSLATE_CHUNK_CHARS`
- `translate_chunk_stream(client, chunk, target_lang, chunk_id, file_id)` — 单块流式翻译；过滤 `<think>...</think>` 内容；统计 `reasoning_content` 长度，非零时记 WARNING
- `_detect_residual_english(text)` — 用 `langdetect` 检测块中残留英文（排除代码、URL、白名单缩写及 `中文（English）` 注释模式）
- `_fix_residual_english(client, chunk, target_lang)` — 一次性非流式修正调用，触发 `chunk_replace` 事件告知前端替换缓冲区
- `translate_file_streaming(...)` — 单文件编排：`split_text_into_chunks` → `asyncio.gather` 并发翻译各块（受 `_api_semaphore` 约束）→ 按块序拼接

**SSE 事件**：`init` → `file_start` → (`chunk_start` → `chunk_token`... → `chunk_done` [→ `chunk_replace` 如有修正]) × N → `file_done` → `all_done`

### 3. 其他端点

- `GET /health` — 返回 `{"status": "ok", "model": "...", "port": ...}`，可用于存活检测
- `GET /` — 解析页 HTML（内联在 `app.py` 的 `_INDEX_HTML` 字符串常量中）
- `GET /translate` — 翻译页 HTML（内联在 `_TRANSLATE_HTML`）

## 关闭思考模式

所有 LLM 调用（`vlm_recognize_page`、`translate_chunk_stream`、`_fix_residual_english`）均发送以下三个字段以兼容不同部署协议：

```python
"enable_thinking": False,                     # Qwen 协议
"chat_template_kwargs": {"thinking": False},  # vLLM / SGLang
"thinking": {"type": "disabled"},             # Kimi 官方 API
```

## 依赖

`requirements.txt` 共 7 个包（无本地模型，无 GPU 依赖）：

```
fastapi>=0.115.0
uvicorn>=0.30.0
httpx>=0.27.0
python-dotenv>=1.0.0
PyMuPDF>=1.24.0
langdetect>=1.0.9
python-multipart>=0.0.9
```

## 测试

```bash
source .venv/bin/activate
python -m pytest tests/test_strip_watermarks.py -q   # 应输出 12 passed
```

测试覆盖 `_strip_watermarks` 函数的各种水印模式。

## 日志

- 控制台实时输出 + `gateway.log`（与 `app.py` 同目录）
- Logger 名称：`gateway`（不传播至根 logger）
- `_strip_watermarks` 命中时记 INFO 含 `file_id`
- 翻译块的 `reasoning_content` 非零时记 WARNING

## 注意事项

- 无认证机制，设计为本地 `127.0.0.1` 使用
- 输出文件（解析结果）写入 `output/<file_id>/`，已加入 `.gitignore`
- 品牌标识 "JT&N 金诚同达" 保留于前端 HTML，不得删除
