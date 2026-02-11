from dotenv import load_dotenv
from pydantic_ai import Agent, ImageGenerationTool

load_dotenv()

agent = Agent(
    "google-gla:gemini-3-pro-image-preview",
    instructions="你是一个图片生成助手",
    builtin_tools=[ImageGenerationTool()],
)

result = agent.run_sync("生成一张游戏数据分析报告的封面图，风格简洁专业")
print(result.output)
