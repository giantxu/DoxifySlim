@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM 1. 检测 Python 3.10+（优先 py 启动器，回退 python；可用者记入 PYCMD）
set "PYCMD="
py -3 -c "import sys; sys.exit(0 if sys.version_info[:2]>=(3,10) else 1)" >nul 2>nul && set "PYCMD=py -3"
if not defined PYCMD (python -c "import sys; sys.exit(0 if sys.version_info[:2]>=(3,10) else 1)" >nul 2>nul && set "PYCMD=python")
if not defined PYCMD (
  echo [X] 未找到 Python 3.10 或更高版本。当前检测到：
  echo     - python --version :
  python --version 2>nul || echo         ^(不可用：未安装，或 python 指向微软商店占位程序^)
  echo     - py -3 --version  :
  py -3 --version 2>nul || echo         ^(不可用：未安装 py 启动器^)
  echo.
  echo     请安装 Python 3.10+（华为云镜像）: https://mirrors.huaweicloud.com/python/
  echo     安装时务必勾选 "Add python.exe to PATH"，装好后重开 PowerShell 再运行本脚本。
  pause
  exit /b 1
)
for /f "delims=" %%v in ('%PYCMD% --version') do echo [OK] 使用 %%v

REM 2. 创建虚拟环境
if not exist ".venv" (
  echo [*] 创建虚拟环境 .venv ...
  %PYCMD% -m venv .venv
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
