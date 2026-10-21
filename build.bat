@echo off
chcp 65001 >nul
REM ============================================================
REM  AI Engineer Mentor - 一键打包脚本
REM  产物: dist\AIEngineerMentor\AIEngineerMentor.exe
REM ============================================================

echo.
echo ============================================================
echo   AI Engineer Mentor - 开始打包
echo ============================================================
echo.

REM 1) 激活 conda 环境
call conda activate myai
if errorlevel 1 (
    echo [错误] 未找到 conda 环境 myai，请先运行 setup.sh 安装
    pause
    exit /b 1
)

REM 2) 确保 PyInstaller 已安装
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [信息] 安装 PyInstaller ...
    pip install pyinstaller --quiet
)

REM 3) 清理旧产物
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist

REM 4) 打包
echo [信息] PyInstaller 正在打包，请耐心等待 2~5 分钟 ...
pyinstaller build.spec --noconfirm

if not exist "dist\AIEngineerMentor\AIEngineerMentor.exe" (
    echo.
    echo [错误] 打包失败，请查看上面的日志
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   打包完成！
echo   产物: %CD%\dist\AIEngineerMentor\AIEngineerMentor.exe
echo.
echo   你可以把整个 dist\AIEngineerMentor 文件夹压缩后
echo   发给任何人；首次运行会自动打开浏览器。
echo ============================================================
echo.

REM 自动打开产物目录
start "" "dist\AIEngineerMentor"

pause
