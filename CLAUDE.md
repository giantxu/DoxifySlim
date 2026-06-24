# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running Doxify

```bash
# Preferred: activates conda env and starts the service
bash start.sh

# Direct (if mineru conda env is already active)
python app.py
```

Listens on `http://127.0.0.1:4000`. The conda environment is named `mineru`.

## Environment Configuration

All config lives in `.env` (copy from `.env.example`). Key variables:

| Variable | Purpose |
|---|---|
| `TARGET_API_URL` | OpenAI-compatible LLM endpoint |
| `TARGET_API_KEY` | API key for the LLM |
| `ACTUAL_MODEL_NAME` | Model name used for VLM and translation (e.g. `kimi26`) |
| `GATEWAY_PORT` | Default 4000 |
| `PDF_DPI` | Resolution for PDF→image conversion (default 200) |
| `CONCURRENCY` | VLM parallel workers per file (default 5) |
| `MAX_CONCURRENT_REQUESTS` | Global cap across all files+workers (default 8) |
| `MINERU_TIMEOUT` | Timeout for local MinerU subprocess (default 1800s) |
| `MINERU_VIRTUAL_VRAM_SIZE` | Reported VRAM in GB; drives MinerU's batch_ratio (default 32 → batch_ratio=16). MPS has no native VRAM detection in MinerU 3.0.x, so this is required on Apple Silicon. |
| `MINERU_HYBRID_BATCH_RATIO` | Direct batch_ratio for hybrid backend (default 16) |
| `MINERU_PDF_RENDER_THREADS` | Threads for PDF→image rendering inside MinerU (default 8) |
| `MINERU_DEVICE_MODE` | Force device for MinerU subprocess (default `mps` on M-series) |
| `LOCAL_OCR_CONCURRENCY` | PaddleOCR parallel pages (default 4, M5 Max) |
| `PADDLE_MLX_ENABLED` | Use MLX-accelerated PaddleOCR-VL backend (default 1; falls back to CPU if server down) |
| `PADDLE_MLX_VENV` | Isolated venv that runs `mlx_vlm.server` (default `~/paddleocr_vl_venv`) |
| `PADDLE_MLX_SERVER_PORT` | Port for the MLX server (default 8111) |
| `PADDLE_MLX_MODEL_NAME` | VLM model served by MLX (default `PaddlePaddle/PaddleOCR-VL-1.6`) |
| `PADDLE_VL_CONCURRENCY` | PaddleOCR-VL chunks processed concurrently (default 3; auto-forced to 1 on the CPU fallback path) |

## Architecture

Single file: `app.py`. FastAPI app with two functional areas:

### 1. PDF Parsing — `/` (web UI) + `POST /parse_pdf_stream` (SSE endpoint)

Four modes, selected in the web UI:

- **vlm** — PDF→images via PyMuPDF → Kimi 2.6 VLM API per page → Markdown. Pages are split into `CONCURRENCY` groups processed in parallel, bounded by `_api_semaphore`.
- **mineru_txt** — calls `mineru -p <pdf> -o <out_dir> --method txt` in a subprocess via `run_in_executor`. For digital PDFs with embedded text.
- **mineru_ocr** — same as above but `--method ocr`. For scanned PDFs; uses MinerU's internal `ch_lite` OCR model.
- **paddleocr** — PaddleOCR-VL chunked document parsing, 3 pages per chunk (`PADDLEOCR_VL_CHUNK_PAGES`). Chunks run concurrently via `asyncio.gather` + a `Semaphore(PADDLE_VL_CONCURRENCY)` (forced to 1 on the CPU fallback path); results are back-filled by chunk index so order is preserved. Supports tables/formulas/layout. MLX-accelerated by default (see PaddleOCR-VL Instance Management).

Outputs persist to `output/<file_id>/` and are zipped for download via `GET /download_zip/{file_id}`. The `<file_id>` is a uuid hex generated per request.

SSE event protocol (all events carry `file_id`):
- `init` — file list, sent before processing starts
- `file_start` — `total > 0` means per-page progress; `total == 0` means indeterminate (MinerU modes)
- `page_done` — per-page result (vlm, paddleocr only)
- `file_done` — includes `markdown` field; `has_images: true` triggers the ZIP-download button on the UI
- `file_error` — MinerU subprocess failure

### 2. Markdown Translation — `/translate` (web UI) + `POST /translate_stream` (SSE endpoint)

Two input modes (paste text / upload multiple `.md` files). Files run in parallel via `asyncio.create_task` + shared `asyncio.Queue`. **Chunks within a single file also run in parallel** via `asyncio.gather`, bounded by `MAX_CONCURRENT_REQUESTS` through `_api_semaphore`.

After each chunk streams, `_detect_residual_english(buf)` scans for English words that escaped translation (excluding code, URLs, whitelisted acronyms/proper nouns, and the `中文（English）` annotation pattern). If any residuals are found, `_fix_residual_english(...)` does a one-shot non-streaming correction call and a `chunk_replace` event tells the frontend to swap the chunk's buffer.

SSE event protocol: `init` → `file_start` → (`chunk_start` → `chunk_token`... → `chunk_done` [→ `chunk_replace` if fix applied]) × N → `file_done` → `all_done`.

## Thinking-mode disable

All outbound LLM calls (VLM `_recognize_image`, translation `translate_chunk_stream`, fix `_fix_residual_english`) send three flags to defeat reasoning mode across deployments:

```python
"enable_thinking": False,                     # Qwen protocol
"chat_template_kwargs": {"thinking": False},  # vLLM / SGLang
"thinking": {"type": "disabled"},             # Kimi official API
```

`translate_chunk_stream` additionally counts `<think>...</think>` content stripped on the fly and `reasoning_content` field length; if either is non-trivial it logs a WARNING per chunk so the user can confirm whether the deployment honored the flags.

## PaddleOCR-VL Instance Management

`get_paddle_ocr_vl(use_mlx)` returns a process-wide singleton, cached per backend in `_paddle_ocr_vl_instances` (`"mlx"` / `"cpu"`) with double-checked locking (`_paddle_ocr_vl_lock`).

**MLX acceleration (default on, Apple Silicon):** when `use_mlx=True` the instance is built with `vl_rec_backend="mlx-vlm-server"` pointing at a standalone `mlx_vlm.server` (`PADDLE_MLX_SERVER_URL`, model `PADDLE_MLX_MODEL_NAME`). This offloads the slow VLM recognition step to MLX/GPU — benchmarked ~20× faster than CPU on M5 Max (~45s → ~2.2s per page). Doxify itself does NOT load the MLX model; the separate server process does, so the bleeding-edge deps (`mlx-vlm>=0.3.11`, which pulls `transformers>=5` / `huggingface-hub>=1`) live in an isolated venv (`PADDLE_MLX_VENV`, default `~/paddleocr_vl_venv`) and never touch the `mineru` conda env. **Do not install `mlx-vlm>=0.3.11` into the `mineru` env** — it violates MinerU's `transformers<5.0.0` / `mlx-vlm<0.4` pins.

`start.sh` auto-launches the server (with a 30s health check) and tears it down on exit. `parse_pdf_paddleocr` calls `_mlx_server_alive()` per request and **falls back to `device="cpu"` automatically** when the server is unreachable or `PADDLE_MLX_ENABLED=0`. A per-request UI checkbox (`paddle_mlx` form field, default on) and the global `PADDLE_MLX_ENABLED` env both gate it. The `file_start` SSE event carries `backend: "mlx"|"cpu"` so the frontend knows which path ran.

The installed `paddleocr 3.4.0` in the `mineru` env already exposes the `vl_rec_backend` params, so the client needs no upgrade even though the served model is 1.6 (verified: 3.4.0 client + 1.6 server produces correct output).

## MinerU Subprocess

`_run_mineru_sync()` runs in `run_in_executor` (blocking). It writes the PDF to `output/<file_id>/_raw/`, runs the `mineru` CLI, then `_consolidate_md_and_images` walks the output tree, copies all images to `output/<file_id>/images/`, concatenates `.md` files, and rewrites image references to point at the consolidated `images/` folder. The `_raw/` workspace is cleaned up on success.

The subprocess inherits the parent env plus `MINERU_VIRTUAL_VRAM_SIZE` / `MINERU_HYBRID_BATCH_RATIO` / `MINERU_PDF_RENDER_THREADS` / `MINERU_DEVICE_MODE` (via `env.setdefault`). These are required on Apple Silicon — MinerU 3.0.x's `get_vram()` has no MPS branch and otherwise reports 1 GB, forcing `batch_ratio=1` (slowest tier).

## Dependencies

Install into the `mineru` conda environment:
```bash
conda activate mineru
pip install -r requirements.txt          # fastapi, uvicorn, httpx, python-dotenv, PyMuPDF
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install paddleocr
# MinerU is already installed in the mineru env
```
