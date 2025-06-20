@echo off
chcp 65001 >nul
echo ========================================
echo 📚 文献引用验证系统启动器
echo ========================================
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装或未添加到PATH
    echo 请先安装Python 3.8或更高版本
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python已安装
echo.

:: 检查是否在正确的目录
if not exist "app.py" (
    echo ❌ 找不到app.py文件
    echo 请确保在正确的项目目录中运行此脚本
    pause
    exit /b 1
)

echo ✅ 项目文件存在
echo.

:: 检查是否已安装依赖
echo 🔍 检查依赖包...
python -c "import flask, torch, sentence_transformers, chromadb" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  依赖包未完全安装
    echo 正在安装依赖包...
    echo.
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ 依赖安装失败
        echo 请手动运行: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo ✅ 依赖安装完成
) else (
    echo ✅ 依赖包已安装
)
echo.

:: 启动应用
echo 🚀 启动文献引用验证系统...
echo ========================================
echo 📱 浏览器将自动打开Web界面
echo 🌐 如果没有自动打开，请访问: http://localhost:5000
echo ⏹️  按 Ctrl+C 停止服务器
echo ========================================
echo.

python run_web_app.py

echo.
echo 👋 系统已停止
pause