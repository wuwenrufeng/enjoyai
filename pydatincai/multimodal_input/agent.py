from dotenv import load_dotenv
from pydantic_ai import Agent, BinaryContent

load_dotenv()

agent = Agent(
    "google-gla:gemini-2.5-pro",
    instructions="你是一个游戏数据分析洞察助手",
)

with open("test_data.csv", "rb") as f:
    result = agent.run_sync(
        ["分析这个csv", BinaryContent(data=f.read(), media_type="text/csv")]
    )
    print(result.output)
