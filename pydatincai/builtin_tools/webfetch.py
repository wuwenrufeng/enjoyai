from dotenv import load_dotenv
from pydantic_ai import Agent, WebFetchTool

load_dotenv()

agent = Agent(
    "google-gla:gemini-2.5-pro",
    instructions="你是一个网页内容分析助手",
    builtin_tools=[WebFetchTool(allowed_domains=['ai.pydantic.dev'])],
)

# result = agent.run_sync('请抓取 https://ai.pydantic.dev/ 首页，告诉我 PydanticAI 的主要特性')
result = agent.run_sync('请抓取 https://www.baidu.com 的网页内容并告诉我页面上有什么')
print(result.output)