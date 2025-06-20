# 文献引用验证系统 - Web 前端应用

## 📖 项目简介

这是一个基于 Flask 的 Web 前端应用，为文献引用验证与知识库查询系统提供了直观易用的图形界面。用户可以通过浏览器轻松完成论文上传、引用验证、知识库查询等操作。

## ✨ 主要功能

### 🔧 系统管理
- **系统初始化**: 一键初始化嵌入模型和向量数据库
- **状态监控**: 实时显示系统运行状态
- **知识库更新**: 手动重建 RAG 索引

### 📚 论文管理
- **批量上传**: 支持多个 PDF 文件同时上传
- **拖拽上传**: 直观的拖拽文件上传界面
- **进度显示**: 实时显示上传和处理进度

### 🔍 引用验证
- **智能检测**: 自动识别文中的各种引用格式
- **准确性验证**: 基于语义相似度和关键词匹配验证引用准确性
- **详细报告**: 提供完整的验证结果和相似度分数
- **参数调节**: 可调节相似度阈值和关键词重合度阈值

### 🔎 知识库查询
- **语义搜索**: 基于向量相似度的智能搜索
- **结果排序**: 按相似度排序显示搜索结果
- **内容预览**: 显示匹配的文献片段

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 8GB+ 内存 (推荐)
- 2GB+ 可用磁盘空间

### 方法一：自动安装（推荐）

1. **运行安装脚本**
   ```bash
   python install.py
   ```
   
2. **启动系统**
   ```bash
   python run_web_app.py
   ```
   
   或者在Windows上双击 `start.bat` 文件

### 方法二：手动安装

1. **克隆或下载项目**
   ```bash
   cd citation_agent
   ```

2. **安装依赖包**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置系统**
   - 编辑 `config.py` 文件
   - 设置API密钥和其他参数

4. **启动 Web 应用**
   ```bash
   python run_web_app.py
   ```
   
   或者直接运行 Flask 应用:
   ```bash
   python app.py
   ```

5. **访问应用**
   
   在浏览器中打开: http://localhost:5000
   
   浏览器会自动打开，如果没有请手动访问上述地址。

## 📱 使用指南

### 1. 系统初始化

首次使用时，需要点击右上角的 "初始化系统" 按钮。系统会:
- 加载本地嵌入模型
- 初始化向量数据库
- 准备 RAG 检索系统

### 2. 添加论文到知识库

1. 在 "添加论文" 卡片中点击上传区域或拖拽 PDF 文件
2. 选择一个或多个 PDF 文件
3. 点击 "添加到知识库" 按钮
4. 系统会自动提取论文内容并添加到向量数据库

### 3. 验证文章引用

1. 在 "验证引用" 卡片中上传待验证的文章 (PDF 或 TXT)
2. 调整验证参数:
   - **语义相似度阈值**: 控制语义匹配的严格程度 (0-1)
   - **关键词重合度阈值**: 控制关键词匹配的严格程度 (0-1)
3. 点击 "开始验证" 按钮
4. 查看详细的验证结果报告

### 4. 查询知识库

1. 在 "查询知识库" 卡片中输入查询内容
2. 设置返回结果数量 (1-20)
3. 调整相似度阈值
4. 点击 "查询" 按钮
5. 浏览搜索结果和相关文献片段

### 5. 更新知识库

当添加新论文后，建议点击 "更新知识库" 来重建索引，提高查询性能。

## 🎨 界面特性

### 响应式设计
- 支持桌面和移动设备
- 自适应屏幕尺寸
- 现代化的卡片式布局

### 交互体验
- 实时状态反馈
- 进度条显示
- 拖拽文件上传
- 动画过渡效果

### 结果展示
- 分类显示验证结果 (成功/警告/错误)
- 详细的相似度分数
- 匹配的论文信息
- 可滚动的结果列表

## ⚙️ 配置选项

### 配置文件

系统使用 `config.py` 文件进行配置，包含以下主要配置类：

#### WebConfig - Web应用配置
```python
class WebConfig:
    HOST = '0.0.0.0'          # 服务器地址
    PORT = 5000               # 端口号
    DEBUG = False             # 调试模式
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 最大文件大小(50MB)
    SECRET_KEY = 'your-secret-key-here'    # Flask密钥
```

#### ModelConfig - 模型配置
```python
class ModelConfig:
    CHROMA_DB_PATH = './chroma_db'  # 向量数据库路径
    EMBEDDING_MODEL_PATH = './local_models/all-MiniLM-L6-v2'  # 嵌入模型路径
    AUTO_DOWNLOAD_MODEL = True      # 自动下载模型
    DEVICE = 'auto'                 # 设备选择(cpu/cuda/auto)
```

#### LLMConfig - 大语言模型API配置
```python
class LLMConfig:
    BASE_URL = 'https://api-inference.modelscope.cn/v1/'
    API_KEY = 'your-api-key-here'   # 请替换为您的API密钥
    MODEL_NAME = 'deepseek-ai/DeepSeek-R1-0528'
    MAX_TOKENS = 200
    TEMPERATURE = 0.1
```

#### VerificationConfig - 引用验证配置
```python
class VerificationConfig:
    DEFAULT_SIM_THRESH = 0.05           # 语义相似度阈值
    DEFAULT_KEYWORD_OVERLAP_THRESH = 0.3 # 关键词重合度阈值
    CONTEXT_WINDOW_SIZE = 200           # 上下文窗口大小
```

### 重要配置项说明

1. **API密钥设置**
   - 编辑 `config.py` 中的 `LLMConfig.API_KEY`
   - 设置您的DeepSeek API密钥

2. **模型路径**
   - 默认模型会下载到 `./local_models/all-MiniLM-L6-v2`
   - 可以修改 `EMBEDDING_MODEL_PATH` 指定其他路径

3. **数据库路径**
   - ChromaDB数据库默认存储在 `./chroma_db`
   - 可以修改 `CHROMA_DB_PATH` 指定其他位置

4. **性能调优**
   - 修改 `DEVICE` 设置使用GPU加速
   - 调整 `BATCH_SIZE` 和 `MAX_WORKERS` 优化性能

### 环境变量支持

系统也支持通过环境变量覆盖配置：

- `DEEPSEEK_API_KEY`: DeepSeek API密钥
- `CHROMA_DB_PATH`: ChromaDB数据库路径
- `UPLOAD_FOLDER`: 文件上传目录
- `WEB_PORT`: Web服务器端口

### 配置验证

运行以下命令验证配置：
```bash
python config.py
```

## 🔧 API 接口

### 系统管理
- `POST /api/init` - 初始化系统
- `GET /api/status` - 获取系统状态
- `POST /api/update_knowledge_base` - 更新知识库

### 文件处理
- `POST /api/upload_papers` - 上传论文
- `POST /api/verify_citations` - 验证引用
- `POST /api/query_knowledge_base` - 查询知识库

## 🐛 故障排除

### 常见问题

1. **系统初始化失败**
   - 检查本地模型文件是否存在
   - 确保有足够的内存和磁盘空间
   - 查看控制台错误信息

2. **文件上传失败**
   - 检查文件大小是否超过 50MB 限制
   - 确保文件格式为 PDF 或 TXT
   - 检查文件是否损坏

3. **引用验证结果不准确**
   - 调整相似度阈值参数
   - 确保知识库中有相关论文
   - 检查文章中的引用格式

4. **查询无结果**
   - 降低相似度阈值
   - 尝试不同的查询关键词
   - 确保知识库中有相关内容

### 日志查看

应用运行时会在控制台输出详细日志，包括:
- 请求处理信息
- 错误堆栈跟踪
- 系统状态变化

## 📊 性能优化

### 建议配置

- **内存**: 8GB+ (处理大量论文时)
- **存储**: SSD 硬盘 (提高向量检索速度)
- **网络**: 稳定的网络连接 (用于 API 调用)

### 优化建议

1. **批量处理**: 一次上传多个论文文件
2. **定期更新**: 添加新论文后及时更新知识库
3. **参数调优**: 根据实际需求调整相似度阈值
4. **缓存清理**: 定期清理临时文件和缓存

## 🔒 安全注意事项

1. **文件安全**: 上传的文件会临时存储并自动清理
2. **API 密钥**: 确保 API 密钥安全，不要泄露
3. **网络安全**: 生产环境建议使用 HTTPS
4. **访问控制**: 可以添加用户认证和权限管理

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目！

### 开发环境设置

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 📞 支持

如有问题或建议，请:
1. 查看本文档的故障排除部分
2. 提交 GitHub Issue
3. 联系项目维护者

---

**享受使用文献引用验证系统！** 🎉