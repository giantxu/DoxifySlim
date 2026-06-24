# DoxifySlim

基于 **Kimi 2.6 VLM** 的精简版 PDF→Markdown 解析工具 + Markdown 翻译工具（JT&N 金诚同达内部使用）。

```
PDF → Kimi 2.6 VLM（逐页识别）→ Markdown
                                   │
                                   └── Markdown → 分块并行翻译 → 译文 Markdown
```

## 界面预览

**PDF 解析页**（`/`）

![PDF 解析页](docs/images/parse.png)

**Markdown 翻译页**（`/translate`）

![Markdown 翻译页](docs/images/translate.png)

---

## 功能

### ① PDF 解析

- **引擎**：Kimi 2.6 VLM（远程 API），将每页 PDF 转为图片后逐页调用 VLM 识别
- **实时进度**：SSE 流式推送，浏览器实时显示每页完成情况
- **两个处理选项**（勾选即生效）：
  - **去除页眉/页脚水印**：自动剥离 EAPA 风格的 Barcode 头、Filed By 脚水印行
  - **插入分页标识**：每页 Markdown 之间插入 `--- [第 N 页] ---` 分隔符
- **结果获取**：页面下方提供"复制 Markdown"和"下载 .md"两个按钮，无 ZIP 打包
- **多文件并行**：同时拖入多个 PDF，各文件独立并行处理

### ② Markdown 翻译

- 粘贴文本或上传多个 `.md` 文件
- 分块并行翻译，流式 token-by-token 输出
- 自动检测残留英文并一次性修正（`_detect_residual_english` + `_fix_residual_english`）
- 保留 Markdown 格式、代码块、表格、链接
- 支持选择目标语言（默认"中文"）

---

## 系统要求

| 要求 | 说明 |
|---|---|
| Python | **3.10 或更高**（3.11 / 3.12 均可） |
| 操作系统 | macOS 或 Windows |
| 网络 | 需可访问 Kimi API（或其他 OpenAI 兼容端点） |
| 模型下载 | **无需下载任何本地模型** |

---

## 安装

### macOS

```bash
bash install.sh
```

### Windows

双击 `install.bat`，或在命令提示符中运行：

```cmd
install.bat
```

脚本会自动：

1. 检测 Python 3.10+（不满足时提示华为云下载地址）
2. 创建虚拟环境 `.venv`
3. 从国内镜像安装依赖（**阿里云 → 清华 → 中科大** 自动回退，无需手动配置）
4. 初始化 `.env`（首次运行时从 `.env.example` 复制）

> 如果所有镜像均失败，可手动安装：
> ```bash
> source .venv/bin/activate   # macOS
> # .venv\Scripts\activate.bat  # Windows
> pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
> ```

---

## 配置

编辑项目根目录的 `.env` 文件（安装后自动生成）：

```dotenv
TARGET_API_URL=https://your-api-endpoint.com/v1/chat/completions
TARGET_API_KEY=your_api_key_here
ACTUAL_MODEL_NAME=kimi26
```

**必填项**：`TARGET_API_URL`、`TARGET_API_KEY`、`ACTUAL_MODEL_NAME`

**可选项**（已有合理默认值，一般无需修改）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GATEWAY_PORT` | `4000` | 服务监听端口 |
| `LLM_TIMEOUT` | `300` | LLM 请求总超时（秒） |
| `PAGE_TIMEOUT` | `120` | 单页 VLM 识别超时（秒） |
| `PDF_DPI` | `200` | PDF→图片分辨率（越高越清晰但越慢） |
| `CONCURRENCY` | `5` | 单文件内并发 Worker 数 |
| `CONCURRENCY_THRESHOLD` | `10` | 启用并发的最小页数 |
| `MAX_CONCURRENT_REQUESTS` | `8` | 全局最大同时 API 请求数 |
| `TRANSLATE_CHUNK_CHARS` | `3000` | 每块最大翻译字符数 |
| `TRANSLATE_TARGET_LANG` | `中文` | 默认翻译目标语言 |

---

## 启动

### macOS

```bash
bash start.sh
```

### Windows

双击 `start.bat`，或在命令提示符中运行 `start.bat`。

### 直接启动（任意平台，已激活 venv）

```bash
source .venv/bin/activate   # macOS
python app.py
```

浏览器访问：

| 功能 | 地址 |
|---|---|
| PDF 解析 | `http://127.0.0.1:4000` |
| Markdown 翻译 | `http://127.0.0.1:4000/translate` |
| 健康检查 | `http://127.0.0.1:4000/health` |

---

## 使用说明

### PDF 解析页（`/`）

1. 先按需勾选处理选项（拖入文件后会立即开始处理，所以请先设置）：
   - ☑ **去除页眉/页脚水印**：剔除 EAPA 文件中的 Barcode / Filed By 水印行
   - ☑ **插入分页标识**：每页之间插入分隔符，便于对照原件
2. 将一个或多个 PDF 拖入上传区（或点击选择）——选好后**自动开始解析**，无需额外按钮
3. 等待进度条逐页更新（多文件并行处理）
4. 完成后点击"**复制 Markdown**"或"**下载 .md**"获取结果

### Markdown 翻译页（`/translate`）

1. 在文本框粘贴 Markdown 内容，或上传 `.md` 文件（支持多选）
2. 选择目标语言（默认"中文"）
3. 点击"开始翻译"，流式实时显示译文
4. 翻译完成后复制或下载结果

---

## 常见问题

**Q：Python 未安装或版本过低？**  
A：请从华为云镜像下载 Python 3.12 安装包：  
`https://mirrors.huaweicloud.com/python/`

**Q：pip 安装慢或报错？**  
A：`install.sh` / `install.bat` 已内置阿里云/清华/中科大多镜像自动回退。若仍失败，手动指定：  
```bash
source .venv/bin/activate
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

**Q：端口 4000 已被占用？**  
A：编辑 `.env`，将 `GATEWAY_PORT` 改为其他端口（如 `4001`），重启服务即生效。

**Q：API 报错（401 / 403 / 连接超时）？**  
A：检查 `.env` 中的 `TARGET_API_URL`、`TARGET_API_KEY`、`ACTUAL_MODEL_NAME` 是否正确，以及网络是否可访问该端点。

**Q：解析结果有页眉/页脚杂项行？**  
A：在解析页勾选"去除页眉/页脚水印"选项后重新解析。

---

## 文件结构

```
DoxifySlim/
├── app.py              # 全部业务逻辑（FastAPI 单文件应用）
├── requirements.txt    # 7 个 pip 依赖（无需安装模型）
├── .env.example        # 配置模板（install 脚本自动复制为 .env）
├── install.sh          # macOS 安装脚本
├── install.bat         # Windows 安装脚本
├── start.sh            # macOS 启动脚本
├── start.bat           # Windows 启动脚本
├── static/             # 静态资产（品牌 logo 等）
└── tests/              # 单元测试
```

---

## 技术说明

- 单文件 FastAPI 应用，`python app.py` 启动，无需额外构建步骤
- PDF→图片：PyMuPDF（`fitz`），DPI 可配
- VLM 识别：每页独立 HTTP 请求，最多 3 次自动重试，超时递增 50%
- 翻译分块：按 `TRANSLATE_CHUNK_CHARS` 在自然段落边界切块，各块并行，结果合并后输出
- 全局信号量 `_api_semaphore` 限制同时发出的 API 请求数，防止触发 API 速率限制
- 日志写入 `gateway.log`（同目录）

---

## 许可证

本项目以 [MIT License](LICENSE) 开源。前端保留的 "JT&N 金诚同达" 品牌标识为商标，不在 MIT 授权范围内。

---

*JT&N 金诚同达*
