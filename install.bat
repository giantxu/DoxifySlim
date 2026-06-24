@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM 1. 检测 Python 3.10+
where python >nul 2>nul
if errorlevel 1 (
  echo [X] 未检测到 Python，请先安装 Python 3.10+：
  echo     华为云镜像: https://mirrors.huaweicloud.com/python/
  echo     安装时请勾选 "Add python.exe to PATH"
  pause
  exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info[:2]>=(3,10) else 1)"
if errorlevel 1 (
  echo [X] Python 版本过低，需要 3.10 或更高。
  pause
  exit /b 1
)
for /f "delims=" %%v in ('python --version') do echo [OK] %%v

REM 2. 创建虚拟环境
if not exist ".venv" (
  echo [*] 创建虚拟环境 .venv ...
  python -m venv .venv
)
call .venv\Scripts\activate.bat

REM 3. 安装依赖（多镜像自动回退；Tsinghua 偶发 403）
set MIRRORS=https://mirrors.aliyun.com/pypi/simple https://pypi.tuna.tsinghua.edu.cn/simple https://pypi.mirrors.ustc.edu.cn/simple
set "INSTALLED="
for %%m in (%MIRRORS%) do (
  if not defined INSTALLED (
    echo [*] 尝试镜像: %%m
    python -m pip install --upgrade pip -i %%m && python -m pip install -r requirements.txt -i %%m && set "INSTALLED=1"
    if not defined INSTALLED echo [!] 该镜像失败，尝试下一个...
  )
)
if not defined INSTALLED (
  echo [X] 所有国内镜像均失败，请检查网络后重试。
  pause
  exit /b 1
)
echo [OK] 依赖安装完成

REM 4. 初始化 .env
if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo [*] 已生成 .env，请填写 TARGET_API_URL / TARGET_API_KEY / ACTUAL_MODEL_NAME
)

echo.
echo [DONE] 安装完成！
echo    1) 用记事本编辑 .env 填入 API 信息
echo    2) 双击 start.bat 启动
pause
