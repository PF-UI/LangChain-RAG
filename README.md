# 基于Langchain的简易RAG系统

## 项目概述
本项目是一个基于Langchain框架构建的检索增强生成（RAG）系统，能够处理PDF文档并构建本地向量数据库，结合外部知识源（百度百科）为用户提供智能问答服务。系统通过Streamlit提供友好的Web界面，支持多工具协同调用，优先使用本地文档信息，必要时补充外部权威解释。

## 核心功能
- PDF文档批量处理与结构化解析（支持章节层级识别、表格识别）
- 本地向量数据库构建与高效检索（基于Chroma）
- 多工具协同问答（本地向量库检索 + 百度百科查询）
- 对话历史记录与展示
- 工具调用流程可视化

## 环境要求
- Python 3.8+
- 相关依赖库（见下方安装说明）

## 安装步骤

1. 克隆或下载项目到本地
```bash
git clone <项目仓库地址>  # 若使用Git
cd 基于Langchain的简易RAG系统
```

2. 安装依赖包
```bash
pip install langchain langchain-community langchain-chroma langchain-huggingface langchain-openai
pip install aiohttp beautifulsoup4 streamlit chromadb python-multipart
```

3. 模型与配置准备
- 下载嵌入模型（如`bge-large-zh-v1.5`），并在`PDF.py`中修改模型路径：
  ```python
  model_name = r"你的模型路径"  # 在init_embedding_model函数中
  ```
- （可选）百度百科查询需要配置Cookie：在`BaiduMCP.py`中填写`BDUSS` cookie值
- 配置LLM服务：在`test.py`和`main.py`中修改`base_url`和`api_key`以匹配本地LLM服务

## 目录结构说明
```
基于Langchain的简易RAG系统/
├── BaiduMCP.py         # 百度百科查询工具实现
├── chroma_db/          # 向量数据库存储目录
├── PDF/                # 存放待处理的PDF文档
├── PDF.py              # PDF处理与向量数据库构建逻辑
├── test.py             # 系统测试脚本
└── main.py             # Streamlit主界面与核心逻辑
```

## 使用方法

1. 准备PDF文档
   将需要处理的PDF文件放入`PDF`目录下

2. 构建向量数据库
   运行`PDF.py`处理文档并生成向量数据库：
   ```bash
   python PDF.py
   ```
   处理完成后，向量数据将存储在`chroma_db/ai_teaching_standard`目录

3. 启动问答系统
   运行主程序启动Streamlit界面：
   ```bash
   streamlit run main.py
   ```
   系统将自动在浏览器中打开（默认地址：http://localhost:8501）

4. 开始问答
   在输入框中填写问题并提交，系统将自动调用相关工具并返回结果，右侧将显示工具调用顺序

## 示例问题
- "人工智能技术应用专业代码是什么？"
- "课程考核方式是什么？"
- "人工智能是什么？"

## 注意事项
- 首次运行`PDF.py`时，若`PDF`目录为空，将无法生成向量数据库
- 确保本地LLM服务已启动并能正常访问（默认地址：http://localhost:3000/v1）
- 百度百科查询功能可能受网络状况和Cookie有效性影响
- 大文件处理可能需要较长时间，请耐心等待

## 工具调用逻辑
1. 所有问题均先调用"VectorDB检索"查询本地文档
2. 非名词类问题：若本地有结果则直接回答，否则调用百度百科
3. 名词类问题：无论本地是否有结果，均需调用百度百科补充解释
4. 回答将明确区分本地文档信息与百度百科信息
