# Doxify

A local-first FastAPI tool with two features: **PDF → Markdown parsing** and **Markdown translation**. Drag, drop, watch tokens stream in.

```
PDF → [VLM | MinerU | PaddleOCR-VL] → Markdown (+ images)
       │
       └── Markdown → chunked parallel translation → translated Markdown
```

---

## Features

### PDF Parsing (four modes)

| Mode | Best for | Engine |
|---|---|---|
| **Kimi 2.6 VLM** | Handwriting, complex layouts, mixed text+image | Remote VLM API, page-by-page |
| **MinerU (text)** | Digital PDFs with embedded text | Local `mineru` CLI subprocess |
| **MinerU (scan)** | Scanned printed PDFs | Local `mineru` CLI subprocess |
| **PaddleOCR-VL** | High-accuracy document parsing, tables/formulas | Local VLM inference, chunked |

- Drag-and-drop web UI at `http://127.0.0.1:4000`
- Real-time per-page progress (VLM / PaddleOCR) or spinner (MinerU)
- Multiple files processed in parallel via SSE streaming
- Extracted figures preserved alongside Markdown; downloadable as ZIP

### Markdown Translation

- Paste text or upload multiple `.md` files
- All files translated in parallel; **chunks within each file also run in parallel** (bounded by global concurrency cap)
- Streaming token-by-token output
- Preserves Markdown formatting, code blocks, tables, and links
- Automatic residual-English detection with one-shot self-correction
- Respects Chinese-term-with-English-annotation pattern: `数据空间（dataspace）` stays intact

---

## Requirements

- macOS (tested on Apple Silicon M-series)
- [Miniforge](https://github.com/conda-forge/miniforge) / conda
- Python 3.10 (in conda env `mineru`)
- [MinerU](https://github.com/opendatalab/MinerU) installed in the `mineru` conda env
- An OpenAI-compatible LLM API endpoint (e.g. self-hosted Kimi)

---

## Installation

```bash
# 1. Create conda environment
conda create -n mineru python=3.10 -y
conda activate mineru

# 2. Install MinerU (includes its own OCR stack)
pip install -U "mineru[all]"

# 3. Install Doxify dependencies
pip install -r requirements.txt

# 4. Install PaddleOCR-VL (optional, for PaddleOCR mode)
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install paddleocr

# 5. Configure environment
cp .env.example .env
# Edit .env and fill in your LLM API URL and key
```

---

## Configuration

Copy `.env.example` to `.env` and set:

| Variable | Default | Description |
|---|---|---|
| `TARGET_API_URL` | — | OpenAI-compatible chat completions endpoint |
| `TARGET_API_KEY` | — | API key for the target LLM |
| `ACTUAL_MODEL_NAME` | `kimi26` | Model name used for VLM and translation |
| `GATEWAY_PORT` | `4000` | Service listening port |
| `PDF_DPI` | `200` | PDF → image resolution (higher = slower but clearer) |
| `CONCURRENCY` | `5` | VLM parallel workers per file |
| `MAX_CONCURRENT_REQUESTS` | `8` | Global cap on simultaneous LLM API requests |
| `MINERU_TIMEOUT` | `600` | Timeout for MinerU subprocess (seconds) |
| `LOCAL_OCR_CONCURRENCY` | `4` | PaddleOCR parallel pages (4 recommended for M-series) |
| `TRANSLATE_CHUNK_CHARS` | `3000` | Max characters per translation chunk |
| `TRANSLATE_TARGET_LANG` | `中文` | Default translation target language |

---

## Usage

```bash
bash start.sh
```

| Access point | URL |
|---|---|
| PDF parsing web UI | `http://127.0.0.1:4000` |
| Markdown translation web UI | `http://127.0.0.1:4000/translate` |

---

## Architecture

Single-file FastAPI app (`app.py`). Two functional areas:

**PDF parsing** (`POST /parse_pdf_stream`, SSE)
Parallel tasks per file → shared `asyncio.Queue` → single SSE connection.
Event sequence: `init` → `file_start` → `page_done` × N → `file_done`
Output persisted under `output/<file_id>/` for ZIP download via `GET /download_zip/{file_id}`.

**Markdown translation** (`POST /translate_stream`, SSE)
All files launched as concurrent tasks; chunks within each file fan out in parallel, bounded by `MAX_CONCURRENT_REQUESTS`.
Event sequence: `init` → `file_start` → (`chunk_start` → `chunk_token`... → `chunk_done` [→ `chunk_replace` if residual English fixed]) × N → `file_done` → `all_done`

---

## Known Limitations

- **PaddleOCR server models unavailable on macOS ARM** — Bus error 10 on CPU mode; mobile models are used instead (slightly lower accuracy)
- **MinerU has no per-page progress** — processed as a black-box CLI; only a spinner is shown
- **MinerU internal OCR is fixed to `ch_lite`** — hardcoded in MinerU's CPU path; cannot be overridden
- **No authentication** — designed for local `127.0.0.1` use only

---

## File Structure

```
.
├── app.py              # All business logic
├── start.sh            # One-command startup script
├── requirements.txt    # pip dependencies (excludes PaddlePaddle/PaddleOCR)
├── .env.example        # Configuration template
└── output/             # Per-file parsed results (gitignored)
```
