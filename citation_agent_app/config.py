# -*- coding: utf-8 -*-
"""
文献引用验证系统配置文件

在这里修改系统的各种配置参数
"""

import os

# =============================================================================
# Web 应用配置
# =============================================================================

class WebConfig:
    """Web 应用相关配置"""
    
    # 服务器配置
    HOST = '0.0.0.0'  # 监听地址，0.0.0.0 表示监听所有网络接口
    PORT = 5000       # 端口号
    DEBUG = False     # 调试模式，生产环境请设为 False
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 最大文件大小 (50MB)
    UPLOAD_FOLDER = 'uploads'               # 临时上传文件夹
    ALLOWED_EXTENSIONS = {'pdf', 'txt'}     # 允许的文件扩展名
    
    # 安全配置
    SECRET_KEY = 'your-secret-key-here'  # Flask 密钥，生产环境请修改

# =============================================================================
# 模型和数据库配置
# =============================================================================

class ModelConfig:
    """模型和数据库相关配置"""
    
    # 向量数据库配置
    CHROMA_DB_PATH = './chroma_db'  # ChromaDB 数据库路径
    
    # 嵌入模型配置
    EMBEDDING_MODEL_PATH = './local_models/all-MiniLM-L6-v2'  # 本地嵌入模型路径
    EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'                 # 模型名称
    
    # 如果本地模型不存在，是否自动下载
    AUTO_DOWNLOAD_MODEL = True
    
    # 模型设备配置 ('cpu', 'cuda', 'auto')
    DEVICE = 'auto'  # auto 会自动检测是否有 GPU

# =============================================================================
# LLM API 配置
# =============================================================================

class LLMConfig:
    """大语言模型 API 配置"""
    
    # DeepSeek API 配置
    BASE_URL = ''
    API_KEY = ''  # 请替换为您的 API 密钥
    MODEL_NAME = ''
    
    # API 调用参数
    MAX_TOKENS = 200
    TEMPERATURE = 0.1
    
    # 重试配置
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # 秒

# =============================================================================
# 引用验证配置
# =============================================================================

class VerificationConfig:
    """引用验证相关配置"""
    
    # 默认阈值
    DEFAULT_SIM_THRESH = 0.05           # 语义相似度阈值
    DEFAULT_KEYWORD_OVERLAP_THRESH = 0.3 # 关键词重合度阈值
    
    # 上下文提取配置
    CONTEXT_WINDOW_SIZE = 200  # 引用上下文窗口大小（字符数）
    MIN_CONTEXT_LENGTH = 30    # 最小上下文长度
    
    # 引用格式正则表达式
    CITATION_PATTERNS = [
        r'\[\d+\]',                                  # [1], [12]
        r'\[[A-Za-z]+(?:\s*et al\.)?\s*,\s*\d{4}\]', # [Smith, 2020], [Smith et al., 2020]
        r'\[[A-Za-z]+[0-9]{4}\]',                    # [Smith2020]
        r'\((?:[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*\s*(?:and|&)\s*)?[A-Z][a-z]+(?:\s*et al\.)?,\s*\d{4}[a-z]?\)', # (Smith, 2020)
        r'【[^】]{2,40}\s*(?:20\d{2}|19\d{2})】'      # 【王2020】
    ]

# =============================================================================
# 知识库查询配置
# =============================================================================

class QueryConfig:
    """知识库查询相关配置"""
    
    # 默认查询参数
    DEFAULT_N_RESULTS = 5      # 默认返回结果数量
    DEFAULT_SIM_THRESH = 0.6   # 默认相似度阈值
    MAX_N_RESULTS = 20         # 最大返回结果数量
    
    # 结果显示配置
    MAX_CONTENT_LENGTH = 500   # 显示的内容片段最大长度

# =============================================================================
# 论文处理配置
# =============================================================================

class PaperConfig:
    """论文处理相关配置"""
    
    # 论文存储路径
    PAPERS_DIR = './papers'  # 论文文件存储目录
    
    # PDF 处理配置
    MAX_PAGES_PER_PAPER = 100  # 每篇论文最大页数限制
    
    # 文本提取配置
    MIN_TEXT_LENGTH = 100      # 最小文本长度（字符数）
    
    # arXiv 下载配置
    ARXIV_DOWNLOAD_DIR = './downloaded_arxiv_papers'
    MAX_ARXIV_DOWNLOADS = 10   # 单次最大下载数量

# =============================================================================
# 日志配置
# =============================================================================

class LogConfig:
    """日志相关配置"""
    
    # 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    LOG_LEVEL = 'INFO'
    
    # 日志文件配置
    LOG_FILE = 'citation_agent.log'
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5
    
    # 日志格式
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# =============================================================================
# 性能配置
# =============================================================================

class PerformanceConfig:
    """性能相关配置"""
    
    # 批处理配置
    BATCH_SIZE = 10           # 批处理大小
    MAX_WORKERS = 4           # 最大工作线程数
    
    # 缓存配置
    ENABLE_CACHE = True       # 是否启用缓存
    CACHE_SIZE = 1000         # 缓存大小
    CACHE_TTL = 3600          # 缓存过期时间（秒）
    
    # 内存管理
    MAX_MEMORY_USAGE = 0.8    # 最大内存使用率

# =============================================================================
# 辅助函数
# =============================================================================

def get_config(config_name):
    """获取指定的配置类"""
    config_map = {
        'web': WebConfig,
        'model': ModelConfig,
        'llm': LLMConfig,
        'verification': VerificationConfig,
        'query': QueryConfig,
        'paper': PaperConfig,
        'log': LogConfig,
        'performance': PerformanceConfig
    }
    return config_map.get(config_name.lower())

def create_directories():
    """创建必要的目录"""
    directories = [
        WebConfig.UPLOAD_FOLDER,
        ModelConfig.CHROMA_DB_PATH,
        PaperConfig.PAPERS_DIR,
        PaperConfig.ARXIV_DOWNLOAD_DIR,
        os.path.dirname(ModelConfig.EMBEDDING_MODEL_PATH)
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"创建目录: {directory}")

def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 检查必要的目录
    if not os.path.exists(ModelConfig.EMBEDDING_MODEL_PATH) and not ModelConfig.AUTO_DOWNLOAD_MODEL:
        errors.append(f"嵌入模型路径不存在: {ModelConfig.EMBEDDING_MODEL_PATH}")
    
    # 检查 API 密钥
    if not LLMConfig.API_KEY or LLMConfig.API_KEY == 'your-api-key-here':
        errors.append("请设置有效的 LLM API 密钥")
    
    # 检查端口号
    if not (1 <= WebConfig.PORT <= 65535):
        errors.append(f"无效的端口号: {WebConfig.PORT}")
    
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

if __name__ == "__main__":
    # 测试配置
    print("配置测试:")
    print(f"Web 服务器: {WebConfig.HOST}:{WebConfig.PORT}")
    print(f"嵌入模型: {ModelConfig.EMBEDDING_MODEL_PATH}")
    print(f"向量数据库: {ModelConfig.CHROMA_DB_PATH}")
    print(f"LLM API: {LLMConfig.BASE_URL}")
    
    # 创建目录
    create_directories()
    
    # 验证配置
    if validate_config():
        print("✅ 配置验证通过")
    else:
        print("❌ 配置验证失败")