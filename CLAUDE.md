# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

DoxifySlim 是 [Doxify](https://github.com/giantxu/Doxify) 的精简版：**只保留远程多模态 LLM 解析路径**（当前网关背后是 GLM-5.3-Flash），去掉了 MinerU / PaddleOCR-VL / MLX 等本地模型。pip-venv 安装，**必须兼容 Windows**（用户主要在 Windows 上部署）。

两者共享绝大部分代码。Doxify 有新迭代时，同步方式是：以 Doxify 当前 `app.py` 为源，按 AST 删除引擎专属函数、按正则删除引擎常量与 UI 模式选择区、删孤儿函数，然后在干净 venv 里跑全套测试 + 真机冒烟。不要试图逐个 cherry-pick 提交。

## 启动

```bash
bash install.sh  /  install.bat     # 一次性：建 .venv、装依赖、生成 .env
bash start.sh    /  start.bat       # 启动，默认 http://127.0.0.1:4000
```

`start.bat` / `start.sh` 都设置 `PYTHONUTF8=1`——Windows 默认 GBK，中文日志和文件名会乱码。

## Windows 兼容性纪律

- 所有文本读写显式 `encoding="utf-8"`（`read_text` / `write_text` / `open` / `FileHandler`）。审计命令：`grep -nE "read_text\(\)|write_text\(|open\(" app.py | grep -v encoding=`，必须为空
- 路径一律走 `pathlib.Path` / `os.path.join`，不用字符串拼 `/`
- 不用 `os.setsid` / `fcntl` / `signal.SIGKILL` / `/tmp` 等 Unix 专属调用
- 依赖必须全部有 Windows 预编译轮子（当前 10 个包均满足：fastapi、uvicorn、httpx、python-dotenv、PyMuPDF、langdetect、python-multipart、lxml、json-repair、python-docx）
- 测试不得依赖 shell 特性；`seed_page_cache` 的 CLI 测试用 `sys.executable` 拉子进程

## 环境配置

见 `.env.example`，每项有注释。关键的：

| 变量 | 说明 |
|---|---|
| `TARGET_API_URL` / `TARGET_API_KEY` / `ACTUAL_MODEL_NAME` | 网关三件套 |
| `LLM_PROFILE` | 模型档案（`glm53flash` 当前默认 / `kimi26`），决定关闭思考模式用哪套参数。见「模型档案与思考模式」 |
| `PDF_DPI` | 页缓存键的一部分，改了缓存全失效 |
| `TRANSLATE_CONCURRENCY` | 翻译流并发（嵌套在 `MAX_CONCURRENT_REQUESTS` 内，覆盖整个流生命周期） |
| `JOB_HISTORY_MAX` / `SUBSCRIBER_QUEUE_MAX` | 作业注册表内存边界 |
| `DOXIFY_LOG_FILE` | 日志路径覆写。`tests/conftest.py` 设它，否则测试会污染生产 `gateway.log` |

## 架构

单文件 `app.py`（约 5400 行）+ `llm_common.py`（模型档案、响应清洗）+ `static/job-client.js`（前端重连）。三个页面：`/` 解析、`/translate` Markdown 翻译、`/office` Office 翻译。

### 作业注册表（parse + translate 共用）

长任务不绑在请求上：`POST /jobs/parse|translate` 立即返回 `job_id`，任务在进程级 `_jobs` 里跑；`GET /jobs/{id}/events?cursor=N` 重放 `events[N:]` 再流式续接；`POST /jobs/{id}/cancel`。

不变量：
- `_subscribe` 在加入订阅者和快照事件之间**不得 `await`**，否则丢事件（有测试钉住）
- `job_end` 是事件日志里的终态事件，`_run_job` 从 `finally` 发出，取消路径也发。没有它前端会永远重连
- 每个子任务必须有文件级 catch-all（`_run_parse_task` / `_run_translate_file_task`），把异常转成 `file_error`；否则 runner 的收集循环永久阻塞、作业卡在 `running` 且永不淘汰（内存泄漏）
- `CancelledError` 必须显式 re-raise 在通用 `except` 之前，不得被洗成 `file_error`

### PDF 解析（`parse_pdf_streaming`）

- `_PageRenderer` 惰性逐页渲染（`fitz.Document` 非线程安全，每文件一个单线程 executor）
- 每页结果落 `output/<file_id>/_pages/pNNNN.md`；页缓存 `output/_page_cache/<sha256(pdf)+dpi+model>/`。**失败页不缓存**（`_PAGE_FAILURE_RE`），缓存存的是**未注入图片**的文本（保留 `[[FIGURE]]` 占位，`_inject_figures` 每次都跑，图片提取改进能作用于旧缓存）
- 图表抽取 `_extract_and_save_figures_sync`：位图原样、矢量簇裁剪成 PNG；`_is_ruled_table` 丢掉纯横平竖直的簇（那是表格框线，VLM 已转成 Markdown 表）
- 每页 body 包在 try/except 里：worker 一死 `progress_queue` 断供，收集循环永久挂起
- `_heartbeat_loop` 每 `HEARTBEAT_SEC` 记一行各文件进度，`+0` 即卡住

### Markdown 后处理（解析结果统一执行，顺序有讲究）

`_strip_watermarks` → `_normalize_footnotes` → `_bullet_chars_to_markdown` → `_unquote_list_blocks` → `_unwrap_hard_linebreaks` → `_merge_broken_paragraphs`（仅 `page_markers` 关时）

- **水印剥离必须最先**：页脚行不带句末标点，合并器会把下一页正文粘上去，粘连后再剥就带走正文
- **脚注归一化必须先于段落合并**：合并器要认得 `[^N]:` 才不会把下一页正文粘到脚注上
- 裸数字脚注只信「连排 + 正文角标佐证 ≥ 50%」：连排长度分不开脚注和编号段落（一份 300 段的法律文书连排 208），引用关系分得开
- 脚注替换跳过链接/图片路径、行内代码、URL（`_FN_PROTECT_RE`）：`images/p003_1.jpeg` 里的 `003` 曾被当成脚注标记
- `_html_tables_to_markdown` 只转不含合并单元格的表；转换后前后必须补空行（GFM 要求表格自成块）；接缝清理只作用于表格两侧，**不得**全文清行尾空格（会抹掉硬换行）

### Markdown 翻译（`translate_chunk_stream`）

- `_translate_semaphore` 覆盖**整个流**而非仅发起请求（`stream=True` 时 `send()` 拿到头就返回）
- 首段缓冲 `STREAM_HEAD_CHARS`：攒够再过 `strip_thinking`，无标签推理前缀不会当译文流出；源块以 `1. **…**` 开头时关闭启发式（真标题会被误剥）
- `finish_reason == "length"` 且译文为空 → 以 2× `max_tokens` 重跑一次
- 残留英文检测 + 一次性修正，整段包在 chunk 的 try 里

### DOCX 翻译（`translate_docx_bytes`）

- **lxml 原位改 `w:t` / `w:delText` 文本，XML 树一个节点不增不删**——格式、修订、批注、书签、域、图片天然保留。不要改走 python-docx 重建的路，必丢东西
- 分段：修订语境切换（normal/ins/del）必断；`w:tab`/`w:br` 必断；嵌套段落恰好收集一次；段内加粗/斜体/下划线用 `**·**`/`*·*`/`<u>·</u>` 标记送 LLM，解析失败退化为整段填首 run
- ≤3 字符或无字母的分段不送翻译；空译文/形状不符/语言跑偏 → 保留原文并计入 `kept_original`
- 字体三轨 `_docx_apply_fonts`：`eastAsia`（中日韩）/ `ascii`+`hAnsi`（拉丁）/ `w:cs`（泰文等复杂文种）。只动译过的 run，styles.xml 不碰。OOXML 顺序：`rPr` 是 run 首子元素，`rFonts` 排 `rStyle` 后
- 语言→字体表 `_OFFICE_LANG_FONTS` 前后端共用；自定义语言走 `POST /office/font_suggest`（LLM + 进程缓存 + 拉丁轨兜底）

## 模型档案与思考模式（`llm_common.py`）

网关可能在不改 URL/模型名的情况下换底层模型（2026-09-04 Kimi → GLM 实发）。`PROFILES` 按档案给出关闭思考的 `chat_template_kwargs`；`llm_extra_body()` 展开进每个 payload。五个调用点全部经 `extract_content()` 清洗：剥 `<think>`、剥无标签中英推理前缀（`prefix_heuristic`）；OCR 与残留英文修正传 `prefix_heuristic=False`（转写/译文可能真的以编号粗体标题开头）。`finish_reason == "length"` 一律翻倍 `max_tokens` 与超时重试一次。

`app.py` 里不得再直接读 `["message"]["content"]`；`tests/test_llm_profiles.py` 扫源码保证这一点。

## 测试

```bash
python -m pytest tests/ -q     # 约 415 个，3 秒，不需网络
```

新功能一律先写失败测试再实现。提交前闸门：测试全绿 + `gateway.log` 零增量 + 全仓无客户标识（案号、当事人名、内网 IP、真实邮箱）。

## 注意事项

- 无认证，设计为本机使用；`host="0.0.0.0"` 可局域网共享但不得暴露公网
- 品牌标识 "JT&N 金诚同达" 保留于前端，不得删除
- `output/`、`.env`、`*.log` 已 gitignore
