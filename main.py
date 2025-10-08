import streamlit as st
import asyncio
import re  # 新增：用于清除ANSI转义码和提取工具名称
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from typing import List
from io import StringIO
from contextlib import redirect_stdout

# 导入您的自定义模块（确保路径正确）
from PDF import search_vector_db
from BaiduMCP import query_introduction

# 页面基础配置
st.set_page_config(
    page_title="智能问答助手",
    page_icon="🤖",
    layout="wide"
)


# ---------------------- 1. 初始化核心组件（缓存优化） ----------------------
@st.cache_resource
def init_llm():
    return ChatOpenAI(
        base_url="http://localhost:3000/v1",
        api_key="sk-V5Nou8za8Jaqz7RZ5a501569043846D98e2aA0AaEaCeB6Bb",
        model_name="qwen3-max",
        streaming=True,
        temperature=0.2
    )


# @st.cache_resource
# def init_agent(llm):
#     tools = [
#         Tool(
#             name="VectorDB检索",
#             func=search_vector_db,
#             description="用于从本地向量库中查询已存储的文档内容，优先使用该工具回答与文档相关的问题"
#         ),
#         Tool(
#             name="百度百科查询",
#             func=lambda q: asyncio.run(query_introduction(q)),  # 适配同步调用
#             description="当问题涉及名词解释或向量库中没有相关内容时，用于查询百度百科获取外部信息"
#         )
#     ]
#
#     return initialize_agent(
#         tools=tools,
#         llm=llm,
#         agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
#         verbose=True,  # 需保留verbose=True以捕获调用记录
#         handle_parsing_errors=True
#     )


# ---------------------- 2. 核心工具：清除乱码+提取工具名称 ----------------------
def process_tool_calls(raw_output: str) -> List[str]:
    """
    1. 清除ANSI转义码（解决乱码）
    2. 提取工具名称（只保留action字段的值）
    3. 返回工具调用顺序列表
    """
    # 步骤1：清除ANSI转义码（匹配类似[1m、[32;1m等格式）
    ansi_pattern = r'\x1B\[[0-9;]*[mK]'
    clean_output = re.sub(ansi_pattern, '', raw_output)

    # 步骤2：提取工具名称（匹配 "action": "工具名" 格式）
    tool_pattern = r'"action": "([^"]+)"'
    tool_names = re.findall(tool_pattern, clean_output)

    # 步骤3：去重并保留顺序（避免重复调用记录）
    unique_tools = []
    for tool in tool_names:
        if tool not in unique_tools and tool != "Final Answer":  # 排除最终答案标记
            unique_tools.append(tool)

    return unique_tools


async def process_query(query: str, chat_history: List, agent):
    # 构建对话历史（调整为分级处理逻辑）
    messages = [SystemMessage(content="""
    你是一个智能问答助手，需根据问题类型灵活调用工具，严格遵守以下流程：

    一、通用流程（所有问题必须执行的第一步）：
    1. 无论任何问题，第一步必须调用"VectorDB检索"工具查询本地文档内容，禁止直接回答或跳过此步骤。

    二、分情况处理：
    1. 若问题不涉及特定名词（如事实查询、流程咨询等，例："课程考核方式是什么？"）：
       - 若"VectorDB检索"返回有效内容（非"未找到相关内容"），仅基于该结果回答，不调用百度百科。
       - 若"VectorDB检索"返回"未找到相关内容"，调用"百度百科查询"补充回答。

    2. 若问题涉及特定名词（如技术术语、学科名称、专有名词等，例："人工智能是什么？"、"计算机科学包含哪些内容？"）：
       - 第一步：调用"VectorDB检索"获取本地文档中该名词的相关内容。
       - 第二步：必须继续调用"百度百科查询"获取外部权威解释（无论VectorDB是否有结果）。
       - 最终答案需融合两者结果，优先保留本地文档内容，补充百度百科的扩展信息。

    三、关键定义：
    - "特定名词"指：学科名称（如"数据科学"）、技术术语（如"机器学习"）、专有概念（如"区块链"）等具有明确指向性的专业词汇。
    - 禁止颠倒工具调用顺序，必须先"VectorDB检索"，后"百度百科查询"（若需第二步）。

    四、输出要求：
    回答需明确区分本地文档信息和百度百科信息（可标注来源），不编造内容，不混淆信息来源。
    """)]
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=query))
    buffer = StringIO()
    full_response = ""
    with redirect_stdout(buffer):
        async for chunk in agent.astream({"input": query, "chat_history": chat_history}):
            if "output" in chunk:
                full_response += chunk["output"]

    raw_tool_calls = buffer.getvalue()
    simplified_tools = process_tool_calls(raw_tool_calls)

    return full_response, simplified_tools


@st.cache_resource
def init_agent(llm):
    tools = [
        Tool(
            name="VectorDB检索",
            func=search_vector_db,
            description="所有问题的第一步必须调用此工具！用于查询本地向量库中的文档内容，优先获取内部信息"
        ),
        Tool(
            name="百度百科查询",
            func=lambda q: asyncio.run(query_introduction(q)),
            description="两种情况需调用：1. 非特定名词问题且VectorDB无结果；2. 特定名词问题（无论VectorDB是否有结果），用于补充外部权威解释"
        )
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )


# ---------------------- 4. Streamlit界面渲染 ----------------------
def main():
    st.title("智能问答助手 🤖")
    st.write("输入问题后，系统将自动调用工具查询信息并返回结果")

    # 初始化会话状态（保存历史对话、LLM、Agent）
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm" not in st.session_state:
        st.session_state.llm = init_llm()
    if "agent" not in st.session_state:
        st.session_state.agent = init_agent(st.session_state.llm)

    # 问题输入区域
    with st.form(key="query_form", clear_on_submit=True):
        user_query = st.text_input(
            "请输入您的问题：",
            placeholder="例如：人工智能技术应用专业 基本修业年限是多久？",
            label_visibility="visible"
        )
        submit_btn = st.form_submit_button("获取答案", type="primary")

    # 处理查询请求
    if submit_btn and user_query.strip():
        # 显示加载状态
        with st.spinner("正在调用工具查询信息..."):
            response, tool_calls = asyncio.run(
                process_query(user_query, st.session_state.chat_history, st.session_state.agent)
            )

        # 更新对话历史
        st.session_state.chat_history.extend([
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": response}
        ])

        # ---------------------- 分栏展示：最终答案 + 工具调用顺序 ----------------------
        col1, col2 = st.columns(2, gap="large")

        # 左侧：最终答案
        with col1:
            st.subheader("📝 最终答案", divider="blue")
            st.info(response, icon="ℹ️")

        # 右侧：简化的工具调用顺序（无乱码、只显名称）
        with col2:
            st.subheader("🔧 工具调用顺序", divider="green")
            if tool_calls:
                # 按顺序列出工具（带序号）
                for idx, tool in enumerate(tool_calls, 1):
                    st.success(f"{idx}. {tool}", icon="✅")
            else:
                st.warning("未调用任何外部工具（直接基于内置逻辑回答）", icon="⚠️")

    # 可选：展示对话历史（折叠面板）
    with st.expander("📜 对话历史", expanded=False):
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                role = "用户" if msg["role"] == "user" else "助手"
                st.write(f"**{role}**{role}**：{msg['content']}")
                st.divider()
        else:
            st.write("暂无对话历史")


if __name__ == "__main__":
    main()