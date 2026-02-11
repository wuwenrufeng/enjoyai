from dotenv import load_dotenv
from pydantic_ai import Agent, BinaryContent

load_dotenv()

agent = Agent(
    "google-gla:gemini-2.5-pro",
    instructions="你是一个报表分析助手,能从用户上传的图片的数据进行数据分析",
)

with open("report.png", "rb") as f:
    result = agent.run_sync(
        [
            "请核查这张报表截图中的数据，分析数据趋势",
            BinaryContent(data=f.read(), media_type="image/png"),
        ]
    )
    print(result.output)
