@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if not exist ".venv" (
  echo [X] 未找到 .venv，请先运行 install.bat
  pause
  exit /b 1
)
if not exist ".env" (
  echo [X] 未找到 .env，请先复制 .env.example 为 .env 并填写
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat

REM 强制 Python 以 UTF-8 处理控制台与文件（Windows 默认 GBK，中文日志/文件名会乱码或报错）
set PYTHONUTF8=1

set "PORT=4000"
for /f "tokens=2 delims==" %%p in ('findstr /b "GATEWAY_PORT=" .env 2^>nul') do set "PORT=%%p"

start "" "http://127.0.0.1:%PORT%"
echo [*] 启动 DoxifySlim
echo     PDF 解析:        http://127.0.0.1:%PORT%
echo     Markdown 翻译:   http://127.0.0.1:%PORT%/translate
echo     Office 文档翻译: http://127.0.0.1:%PORT%/office
echo     按 Ctrl+C 停止
python app.py
