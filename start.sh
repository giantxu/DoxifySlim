#!/bin/bash
# DoxifySlim 启动脚本（macOS / Linux）。用法: bash start.sh
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "[X] 未找到 .venv，请先运行 bash install.sh"; exit 1
fi
if [ ! -f ".env" ]; then
  echo "[X] 未找到 .env，请先复制 .env.example 为 .env 并填写"; exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUTF8=1

PORT=$(grep -E '^GATEWAY_PORT=' .env 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '[:space:]')
PORT=${PORT:-4000}

echo "[*] 启动 DoxifySlim"
echo "    PDF 解析:        http://127.0.0.1:${PORT}"
echo "    Markdown 翻译:   http://127.0.0.1:${PORT}/translate"
echo "    Office 文档翻译: http://127.0.0.1:${PORT}/office"
echo "    按 Ctrl+C 停止"
python app.py
