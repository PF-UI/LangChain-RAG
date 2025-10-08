import re
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#------------------------------------------------------------------
# 初始化文本分割器
SEPARATORS = [
    r"\n\n(\d+)\s+[^\n]+(?=\n)",  # 一级章节
    r"\n(\d+\.\d+)\s+[^\n]+(?=\n)",  # 二级章节
    r"\n(\d+\.\d+\.\d+)\s+[^\n]+(?=\n)",  # 三级章节
    "\n\n", "\n", " ", ""  # 段落分隔符
]

structured_splitter = RecursiveCharacterTextSplitter(
    separators=SEPARATORS,
    chunk_size=2000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=True
)


# -------------------------- 工具函数定义 --------------------------
def init_embedding_model():
    """初始化嵌入模型"""
    # 替换为你的嵌入模型路径
    model_name = r"D:\ai\download\bge-large-zh-v1.5"
    return HuggingFaceEmbeddings(model_name=model_name)


def add_structured_metadata(split_docs):
    """为分割后的文档添加结构化元数据"""
    processed_docs = []
    current_level1 = ""
    current_level2 = ""
    current_level3 = ""

    # 正则表达式：匹配各级章节标题
    level1_pattern = r"^(\d+)\s+([^\n]+)"  # 匹配“1 概述”
    level2_pattern = r"^(\d+\.\d+)\s+([^\n]+)"  # 匹配“8.1 课程设置”
    level3_pattern = r"^(\d+\.\d+\.\d+)\s+([^\n]+)"  # 匹配“8.1.1 公共基础课程”
    table_pattern = r"\|.+\|.+\|"  # 匹配表格（以“|”分隔的行）

    for doc in split_docs:
        content = doc.page_content
        metadata = doc.metadata.copy()

        # 匹配一级章节，更新当前层级
        level1_match = re.search(level1_pattern, content, re.MULTILINE)
        if level1_match:
            current_level1 = f"{level1_match.group(1)} {level1_match.group(2)}"
            current_level2 = ""  # 重置二级/三级层级
            current_level3 = ""

        # 匹配二级章节，更新当前层级
        level2_match = re.search(level2_pattern, content, re.MULTILINE)
        if level2_match:
            current_level2 = f"{level2_match.group(1)} {level2_match.group(2)}"
            current_level3 = ""  # 重置三级层级

        # 匹配三级章节，更新当前层级
        level3_match = re.search(level3_pattern, content, re.MULTILINE)
        if level3_match:
            current_level3 = f"{level3_match.group(1)} {level3_match.group(2)}"

        # 判断是否为表格
        is_table = bool(re.search(table_pattern, content, re.MULTILINE))

        # 添加元数据
        metadata.update({
            "level1_chapter": current_level1,
            "level2_chapter": current_level2,
            "level3_chapter": current_level3,
            "content_type": "table" if is_table else "text"
        })

        processed_doc = Document(page_content=content, metadata=metadata)
        processed_docs.append(processed_doc)

    return processed_docs


def process_single_pdf(pdf_path, splitter):
    """处理单个PDF文件，返回分割并添加元数据后的文档片段"""
    try:
        # 加载PDF
        loader = PyPDFLoader(file_path=pdf_path)
        pdf_docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in pdf_docs])

        # 创建总文档对象
        doc = Document(
            page_content=full_text,
            metadata={"source": pdf_path, "doc_type": "总文档"}
        )

        # 分割文档
        split_docs = splitter.split_documents([doc])

        # 添加元数据
        final_docs = add_structured_metadata(split_docs)
        return final_docs
    except Exception as e:
        print(f"处理PDF {pdf_path} 时出错: {str(e)}")
        return []


def process_all_pdfs(pdf_dir="./PDF", splitter=None):
    """处理目录下所有PDF文件，返回合并后的文档片段列表"""
    if not splitter:
        raise ValueError("请提供文本分割器实例")

    # 检查目录是否存在
    if not os.path.exists(pdf_dir):
        print(f"错误: 目录 {pdf_dir} 不存在")
        return []

    # 获取所有PDF文件
    pdf_files = [
        f for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(pdf_dir, f))
    ]

    if not pdf_files:
        print(f"警告: 目录 {pdf_dir} 下未找到PDF文件")
        return []

    # 批量处理PDF
    all_docs = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"正在处理: {pdf_path}")
        docs = process_single_pdf(pdf_path, splitter)
        all_docs.extend(docs)
        print(f"  完成，生成 {len(docs)} 个片段")

    print(f"\n所有PDF处理完成，共生成 {len(all_docs)} 个片段")
    return all_docs


def search_vector_db(query, k=3):
    """
    向量数据库查询函数

    参数:
        query: 查询字符串
        vector_db: 向量数据库实例
        k: 返回结果数量，默认3

    返回:
        检索结果列表
    """
    persist_directory = "./chroma_db/ai_teaching_standard"
    vector_db = Chroma(
        collection_name="ai_tech_application_course",
        embedding_function=init_embedding_model(),
        persist_directory=persist_directory
    )
    if not query or not vector_db:
        print("查询参数或向量数据库实例不能为空")
        return []

    print(f"\n=== 检索查询: {query} ===")
    results = vector_db.similarity_search(query=query, k=k)

    for i, res in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"  来源文件: {res.metadata.get('source', '未知')}")
        print(f"  章节层级: {res.metadata.get('level1_chapter', '无')} > {res.metadata.get('level2_chapter', '无')}")
        print(f"  内容类型: {res.metadata.get('content_type', '未知')}")
        print(f"  文本预览: {res.page_content.strip()[:]}...")  # 显示前300字符

    return results


# -------------------------- 主流程 --------------------------
if __name__ == "__main__":


    # 处理所有PDF并获取文档片段
    all_final_docs = process_all_pdfs(splitter=structured_splitter)

    if not all_final_docs:
        print("没有可写入向量库的文档片段，程序退出")
        exit()

    # 初始化向量数据库
    persist_directory = "./chroma_db/ai_teaching_standard"
    vector_db = Chroma(
        collection_name="ai_tech_application_course",
        embedding_function=init_embedding_model(),
        persist_directory=persist_directory
    )

    # 写入向量库
    print(f"\n正在将 {len(all_final_docs)} 个片段写入向量库...")
    vector_db.add_documents(documents=all_final_docs)
    print(f"向量库已保存至: {persist_directory}")

    # 示例查询
    sample_queries = [
        "人工智能技术应用专业代码",
  ]

    # 执行示例查询
    for query in sample_queries:
        search_vector_db(query, vector_db, k=2)