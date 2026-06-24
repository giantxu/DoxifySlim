#!/bin/bash
# ============================================================
# Doxify 启动脚本
# 用法: bash start.sh
# 前置条件: conda 环境 "mineru" 已创建并安装依赖
# 可选: PaddleOCR-VL MLX 加速需独立 venv（见 .env 的 PADDLE_MLX_*）
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Doxify${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查 .env 文件
if [ ! -f .env ]; then
    echo -e "${RED}错误: .env 文件不存在${NC}"
    echo -e "请先复制并编辑配置文件："
    echo -e "  ${YELLOW}cp .env.example .env${NC}"
    echo -e "  然后编辑 .env 填入 API 地址和密钥"
    exit 1
fi

# 从 .env 读取单个配置项（去掉首尾引号），带默认值
get_env() {
    local val
    val="$(grep -E "^$1=" .env 2>/dev/null | tail -1 | cut -d= -f2- | sed 's/^[\"'\'']//;s/[\"'\'']$//')"
    [ -n "$val" ] && echo "$val" || echo "$2"
}

# ── 可选：拉起 PaddleOCR-VL MLX 加速 server，并在脚本退出时清理 ──
MLX_PID=""
cleanup() {
    if [ -n "$MLX_PID" ] && kill -0 "$MLX_PID" 2>/dev/null; then
        echo -e "\n${YELLOW}停止 MLX server (PID $MLX_PID)...${NC}"
        kill "$MLX_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

PADDLE_MLX_ENABLED="$(get_env PADDLE_MLX_ENABLED 1)"
PADDLE_MLX_VENV="$(get_env PADDLE_MLX_VENV "$HOME/paddleocr_vl_venv")"
PADDLE_MLX_SERVER_PORT="$(get_env PADDLE_MLX_SERVER_PORT 8111)"
# 展开开头的 ~
PADDLE_MLX_VENV="${PADDLE_MLX_VENV/#\~/$HOME}"

if [ "$PADDLE_MLX_ENABLED" = "1" ]; then
    echo -e "\n${YELLOW}[1/3] 启动 PaddleOCR-VL MLX server...${NC}"
    if lsof -nP -iTCP:"$PADDLE_MLX_SERVER_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        echo -e "${GREEN}  ✓ 端口 $PADDLE_MLX_SERVER_PORT 已有 server 在跑，复用${NC}"
    elif [ -x "$PADDLE_MLX_VENV/bin/python" ]; then
        "$PADDLE_MLX_VENV/bin/python" -m mlx_vlm.server \
            --host 127.0.0.1 --port "$PADDLE_MLX_SERVER_PORT" \
            > "$SCRIPT_DIR/mlx_server.log" 2>&1 &
        MLX_PID=$!
        # 健康检查：最多等 30s（模型在首个识别请求时才懒加载，这里只确认 HTTP 起来）
        ok=0
        for _ in $(seq 1 30); do
            if curl -s -m 2 -o /dev/null "http://127.0.0.1:$PADDLE_MLX_SERVER_PORT/" 2>/dev/null; then
                ok=1; break
            fi
            sleep 1
        done
        if [ "$ok" = "1" ]; then
            echo -e "${GREEN}  ✓ MLX server 就绪 (PID $MLX_PID, 端口 $PADDLE_MLX_SERVER_PORT)${NC}"
        else
            echo -e "${YELLOW}  ⚠ MLX server 未在 30s 内就绪，PaddleOCR-VL 将回退 CPU（日志: mlx_server.log）${NC}"
        fi
    else
        echo -e "${YELLOW}  ⚠ 未找到 MLX venv: $PADDLE_MLX_VENV${NC}"
        echo -e "${YELLOW}    PaddleOCR-VL 将回退到 CPU。如需加速，请按 .env 注释创建该 venv。${NC}"
    fi
else
    echo -e "\n${YELLOW}[1/3] MLX 加速已禁用 (PADDLE_MLX_ENABLED=0)，PaddleOCR-VL 走 CPU${NC}"
fi

# 激活 conda 环境
echo -e "\n${YELLOW}[2/3] 激活 conda 环境...${NC}"
eval "$(conda shell.bash hook)"
conda activate mineru
echo -e "${GREEN}  ✓ conda 环境 'mineru' 已激活${NC}"

# 启动服务
echo -e "\n${YELLOW}[3/3] 启动 Doxify...${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  PDF 解析:   http://127.0.0.1:4000${NC}"
echo -e "${GREEN}  MD 翻译:    http://127.0.0.1:4000/translate${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n按 Ctrl+C 停止服务\n"

python app.py
