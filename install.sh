#!/usr/bin/env bash
# DoxifySlim 安装脚本 (macOS) —— 使用国内镜像（多镜像自动回退）
set -e
cd "$(dirname "$0")"

# 国内镜像，按顺序尝试直到成功（Tsinghua 偶发 403，故含多个备选）
MIRRORS=(
  "https://mirrors.aliyun.com/pypi/simple"
  "https://pypi.tuna.tsinghua.edu.cn/simple"
  "https://pypi.mirrors.ustc.edu.cn/simple"
)

# 1. 检测 Python 3.10+
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)' 2>/dev/null; then
  echo "❌ 需要 Python 3.10 或更高版本。"
  echo "   未检测到可用 Python，请先安装："
  echo "   • 华为云镜像: https://mirrors.huaweicloud.com/python/"
  echo "   • 或 Homebrew: brew install python@3.12"
  exit 1
fi
echo "✅ 使用 $("$PY" --version)"

# 2. 创建虚拟环境
if [ ! -d .venv ]; then
  echo "📦 创建虚拟环境 .venv ..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. 安装依赖（逐个尝试镜像，成功即止）
pip_install() {
  local ok=1
  for m in "${MIRRORS[@]}"; do
    echo "⬇️  尝试镜像: $m"
    if python -m pip install --upgrade pip -i "$m" \
       && python -m pip install -r requirements.txt -i "$m"; then
      ok=0; echo "✅ 依赖安装完成（镜像: $m）"; break
    else
      echo "⚠️  该镜像失败，尝试下一个..."
    fi
  done
  return $ok
}
if ! pip_install; then
  echo "❌ 所有国内镜像均失败，请检查网络后重试，或手动："
  echo "   source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# 4. 初始化 .env
if [ ! -f .env ]; then
  cp .env.example .env
  echo "📝 已生成 .env，请填写 TARGET_API_URL / TARGET_API_KEY / ACTUAL_MODEL_NAME"
fi

echo ""
echo "🎉 安装完成！"
echo "   1) 编辑 .env 填入 API 信息"
echo "   2) 运行: bash start.sh"
