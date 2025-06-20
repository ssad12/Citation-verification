#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文献引用验证系统 Web 应用启动脚本

这个脚本会检查必要的依赖和文件，然后启动 Web 应用。
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import WebConfig, ModelConfig, create_directories, validate_config
except ImportError:
    print("❌ 无法导入配置文件，请确保 config.py 文件存在")
    sys.exit(1)

def check_dependencies():
    """检查必要的依赖是否已安装"""
    required_packages = [
        'flask', 'torch', 'sentence_transformers', 'chromadb', 
        'fitz', 'thefuzz', 'arxiv', 'openai', 'flask_cors'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == 'fitz':
                try:
                    __import__('PyMuPDF')
                except ImportError:
                    missing_packages.append('PyMuPDF')
            else:
                missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装依赖:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_model_files():
    """检查本地模型文件是否存在"""
    model_path = ModelConfig.EMBEDDING_MODEL_PATH
    
    if not os.path.exists(model_path):
        if ModelConfig.AUTO_DOWNLOAD_MODEL:
            print(f"⚠️  本地模型文件不存在: {model_path}")
            print("   系统将在首次运行时自动下载模型")
            print("   这可能需要几分钟时间，请耐心等待")
            return True
        else:
            print(f"❌ 本地模型文件不存在: {model_path}")
            print("   请下载模型文件或在config.py中设置AUTO_DOWNLOAD_MODEL=True")
            return False
    
    print("✅ 本地模型文件已存在")
    return True

def check_required_folders():
    """检查并创建必要的文件夹"""
    try:
        create_directories()
        print("✅ 所有必要文件夹已准备就绪")
        return True
    except Exception as e:
        print(f"❌ 创建文件夹失败: {e}")
        return False

def open_browser():
    """延迟打开浏览器"""
    time.sleep(3)  # 等待服务器启动
    url = f'http://localhost:{WebConfig.PORT}'
    try:
        webbrowser.open(url)
        print(f"🌐 浏览器已打开: {url}")
    except Exception as e:
        print(f"⚠️  无法自动打开浏览器: {e}")
        print(f"请手动访问: {url}")

def main():
    """主函数"""
    print("🚀 启动文献引用验证系统...")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请安装缺少的包后重试")
        return
    
    # 验证配置
    if not validate_config():
        print("\n❌ 配置验证失败，请检查config.py文件")
        return
    
    # 检查模型文件
    if not check_model_files():
        print("\n❌ 模型文件检查失败")
        return
    
    # 检查并创建必要文件夹
    if not check_required_folders():
        print("\n❌ 文件夹创建失败")
        return
    
    print("\n" + "=" * 50)
    print("🌐 启动 Web 服务器...")
    print(f"📱 浏览器将自动打开，如果没有请手动访问: http://localhost:{WebConfig.PORT}")
    print("⏹️  按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    # 延迟打开浏览器
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # 启动 Flask 应用
    try:
        from app import app
        app.run(debug=WebConfig.DEBUG, host=WebConfig.HOST, port=WebConfig.PORT)
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print(f"请检查是否有其他程序占用了 {WebConfig.PORT} 端口")

if __name__ == "__main__":
    main()