from urllib.parse import quote_plus
import aiohttp
from bs4 import BeautifulSoup
from aiohttp import ClientError, ClientTimeout
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务器
mcp = FastMCP("WeatherServer")


user_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0'
}
user_cookies = {
    'BDUSS': '请填写cookie值'
}


async def get_baike_description(query: str, headers=user_headers, cookies=user_cookies) -> str:
    """
    异步从百度百科获取指定词条的简介（修复await使用错误）
    """
    try:
        # 1. 关键词URL编码（处理中文/特殊字符）
        encoded_query = quote_plus(query, encoding="utf-8")
        baike_url = f"https://baike.baidu.com/item/{encoded_query}"

        # 2. 异步发送GET请求（核心：ClientSession是异步会话）
        timeout = ClientTimeout(total=10)  # 总超时10秒，避免阻塞
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                    url=baike_url,
                    headers=headers,
                    cookies=cookies
            ) as response:
                # 关键修正：raise_for_status()是同步方法，无需await！
                response.raise_for_status()  # 直接调用，非200状态码会抛异常
                # 3. 异步获取HTML文本（response.text()是协程，必须await）
                html_text = await response.text()

        # 4. 解析HTML，提取<meta name="description">标签
        soup = BeautifulSoup(html_text, 'html.parser')
        desc_meta = soup.find('meta', {'name': 'description'})

        # 5. 验证并返回简介
        if desc_meta and desc_meta.get('content'):
            return desc_meta.get('content').strip()
        else:
            return f"解析失败：未找到词条「{query}」的<meta name='description'>标签"

    # 捕获aiohttp请求异常（网络错误、超时、HTTP状态码错误等）
    except ClientError as e:
        return f"请求失败：{str(e)}"
    # 捕获其他未知异常（如HTML解析错误）
    except Exception as e:
        return f"未知错误：{str(e)}"




@mcp.tool()
async def query_introduction(query: str) -> str:
    """
    输入要查询的百度百科名词，返回百度百科查询结果。
    :param query: 百度百科查询名词
    :return: 格式化后的百度百科信息
    """
    data = await get_baike_description(query)
    return str(data)


# 使用示例
if __name__ == "__main__":
    # 以标准I/O方式运行MCP服务器
    mcp.run(transport='stdio')