#!/usr/bin/env bash
# DoxifySlim 启动脚本 (macOS)
# 用法: bash start.sh   （前置：先运行 bash install.sh）
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "❌ 未找到 .venv，请先运行: bash install.sh"
  exit 1
fi

if [ ! -f .env ]; then
  echo "❌ 未找到 .env，请先复制并填写: cp .env.example .env"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

PORT="$(grep -E '^GATEWAY_PORT=' .env 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '[:space:]')"
PORT="${PORT:-4000}"

# 延迟打开浏览器（等服务起来）
( sleep 2; open "http://127.0.0.1:${PORT}" >/dev/null 2>&1 || true ) &

echo "🚀 启动 DoxifySlim"
echo "   PDF 解析: http://127.0.0.1:${PORT}"
echo "   MD 翻译:  http://127.0.0.1:${PORT}/translate"
echo "   按 Ctrl+C 停止"
python app.py
