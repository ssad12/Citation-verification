#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文献引用验证系统安装脚本

这个脚本会帮助您快速设置和安装系统所需的所有依赖。
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """打印安装脚本标题"""
    print("="*60)
    print("📚 文献引用验证系统 - 安装脚本")
    print("="*60)
    print(f"Python 版本: {sys.version}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"当前目录: {os.getcwd()}")
    print("="*60)

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要Python 3.8或更高版本")
        print(f"   当前版本: {sys.version}")
        return False
    
    print(f"✅ Python版本符合要求: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_pip():
    """检查pip是否可用"""
    print("📦 检查pip...")
    
    try:
        import pip
        print("✅ pip已安装")
        return True
    except ImportError:
        print("❌ pip未安装，请先安装pip")
        return False

def install_requirements():
    """安装requirements.txt中的依赖"""
    print("📥 安装Python依赖包...")
    
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"❌ 找不到{requirements_file}文件")
        return False
    
    try:
        # 升级pip
        print("   升级pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # 安装依赖
        print("   安装依赖包...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        
        print("✅ 依赖包安装完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装依赖包失败: {e}")
        print("   请尝试手动运行: pip install -r requirements.txt")
        return False

def create_directories():
    """创建必要的目录"""
    print("📁 创建必要目录...")
    
    directories = [
        "uploads",
        "chroma_db", 
        "papers",
        "local_models",
        "downloaded_arxiv_papers"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"   ✅ {directory}/")
        except Exception as e:
            print(f"   ❌ 创建{directory}失败: {e}")
            return False
    
    print("✅ 目录创建完成")
    return True

def download_model():
    """下载嵌入模型"""
    print("🤖 检查嵌入模型...")
    
    model_path = "./local_models/all-MiniLM-L6-v2"
    
    if os.path.exists(model_path):
        print("✅ 嵌入模型已存在")
        return True
    
    print("   模型不存在，将在首次运行时自动下载")
    print("   如果您想现在下载，请运行以下Python代码:")
    print("   ")
    print("   from sentence_transformers import SentenceTransformer")
    print("   model = SentenceTransformer('all-MiniLM-L6-v2')")
    print("   model.save('./local_models/all-MiniLM-L6-v2')")
    print("   ")
    
    return True

def check_config():
    """检查配置文件"""
    print("⚙️  检查配置文件...")
    
    config_file = "config.py"
    if not os.path.exists(config_file):
        print(f"❌ 找不到{config_file}文件")
        return False
    
    try:
        # 尝试导入配置
        sys.path.insert(0, os.getcwd())
        from config import validate_config
        
        if validate_config():
            print("✅ 配置文件验证通过")
            return True
        else:
            print("⚠️  配置文件验证失败，请检查config.py")
            return False
            
    except Exception as e:
        print(f"❌ 配置文件导入失败: {e}")
        return False

def create_sample_files():
    """创建示例文件"""
    print("📄 创建示例文件...")
    
    # 创建示例论文目录
    sample_papers_dir = "papers/samples"
    os.makedirs(sample_papers_dir, exist_ok=True)
    
    # 创建示例README
    sample_readme = os.path.join(sample_papers_dir, "README.txt")
    if not os.path.exists(sample_readme):
        with open(sample_readme, 'w', encoding='utf-8') as f:
            f.write("这里可以放置您的论文PDF文件\n")
            f.write("支持的格式: .pdf, .txt\n")
            f.write("\n")
            f.write("使用方法:\n")
            f.write("1. 将论文文件复制到此目录\n")
            f.write("2. 在Web界面中点击'添加论文到知识库'\n")
            f.write("3. 选择论文目录进行批量添加\n")
    
    print("✅ 示例文件创建完成")
    return True

def run_tests():
    """运行基本测试"""
    print("🧪 运行基本测试...")
    
    try:
        # 测试导入主要模块
        print("   测试模块导入...")
        
        import flask
        import torch
        import sentence_transformers
        import chromadb
        import fitz  # PyMuPDF
        import openai
        
        print("   ✅ 核心模块导入成功")
        
        # 测试配置文件
        from config import WebConfig, ModelConfig
        print("   ✅ 配置文件导入成功")
        
        print("✅ 基本测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def print_next_steps():
    """打印后续步骤"""
    print("\n" + "="*60)
    print("🎉 安装完成！")
    print("="*60)
    print("\n📋 后续步骤:")
    print("\n1. 配置API密钥 (可选):")
    print("   - 编辑 config.py 文件")
    print("   - 设置您的 DeepSeek API 密钥")
    print("\n2. 启动系统:")
    print("   python run_web_app.py")
    print("\n3. 访问Web界面:")
    print("   http://localhost:5000")
    print("\n4. 添加论文:")
    print("   - 将PDF文件放入 papers/ 目录")
    print("   - 在Web界面中添加到知识库")
    print("\n5. 开始使用:")
    print("   - 上传文章进行引用验证")
    print("   - 查询知识库")
    print("\n📚 更多信息请查看 README_WEB.md")
    print("="*60)

def main():
    """主安装流程"""
    print_header()
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查pip
    if not check_pip():
        sys.exit(1)
    
    # 安装依赖
    if not install_requirements():
        print("\n⚠️  依赖安装失败，但您可以尝试手动安装")
        print("   pip install -r requirements.txt")
    
    # 创建目录
    if not create_directories():
        print("\n⚠️  目录创建失败，请检查权限")
    
    # 检查模型
    download_model()
    
    # 检查配置
    if not check_config():
        print("\n⚠️  配置检查失败，请检查config.py文件")
    
    # 创建示例文件
    create_sample_files()
    
    # 运行测试
    if not run_tests():
        print("\n⚠️  基本测试失败，可能存在依赖问题")
    
    # 打印后续步骤
    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 安装被用户中断")
    except Exception as e:
        print(f"\n\n❌ 安装过程中出现错误: {e}")
        print("请检查错误信息并重试")