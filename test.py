from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMChain
from typing import List, Optional
import asyncio
from PDF import search_vector_db
from BaiduMCP import query_introduction






# 初始化LLM
llm = ChatOpenAI(
    base_url="http://localhost:3000/v1",
    api_key="sk-V5Nou8za8Jaqz7RZ5a501569043846D98e2aA0AaEaCeB6Bb",
    model_name="qwen3-max",
    streaming=True,
    temperature=0.2
)

# 定义工具列表（向量检索工具 + 百度百科工具）
tools = [
    Tool(
        name="VectorDB检索",
        func=search_vector_db,
        description="用于从本地向量库中查询已存储的文档内容，优先使用该工具回答与文档相关的问题"
    ),
    Tool(
        name="百度百科查询",
        func=lambda q: asyncio.run(query_introduction(q)),  # 适配同步调用
        description="当问题涉及名词解释或向量库中没有相关内容时，用于查询百度百科获取外部信息"
    )
]

# 初始化Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,  # 调试模式，可查看工具调用过程
    handle_parsing_errors=True
)

# 系统提示词（指导Agent行为）
system_prompt = """
你是一个智能问答助手，遵循以下流程回答问题：
    1. 你必须先使用"VectorDB检索"工具查询本地文档内容
    2. 如果检索结果包含有效信息，基于此进行简洁准确的回答
    3. 如果检索结果显示"未找到相关内容"，使用"百度百科查询"工具获取信息并回答
    4. 如果问题涉及名词解释,请先使用"VectorDB检索"工具查询本地文档内容，然后结合"百度百科查询"工具回答问题
    5. 回答需严格基于工具返回的结果，不编造内容
"""


# 流式处理函数
async def stream_agent_response(query: str, chat_history: List = []):
    """流式获取Agent响应并输出"""
    # 构建消息列表
    messages = [SystemMessage(content=system_prompt)]
    # 添加历史对话
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    # 添加当前查询
    messages.append(HumanMessage(content=query))

    # 流式处理响应
    full_response = ""
    async for chunk in agent.astream({"input": query, "chat_history": chat_history}):
        if "output" in chunk:
            content = chunk["output"]
            full_response += content
            print(content, end="", flush=True)
    return full_response


# 示例运行
if __name__ == "__main__":
    # 示例查询（可替换为实际问题）
    sample_queries = [
        "人工智能是什么？",
    ]

    chat_history = []  # 维护对话历史
    for query in sample_queries:
        print(f"\n用户问：{query}")
        print("回答：", end="")
        response = asyncio.run(stream_agent_response(query, chat_history))
        chat_history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ])