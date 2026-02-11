from typing import Union

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent

load_dotenv()


class QueryResult(BaseModel):
    sql: str
    explanation: str


class ExplanationResult(BaseModel):
    topic: str
    explanation: str


agent = Agent(
    "google-gla:gemini-2.5-pro",
    instructions="你是一个SQL生成和SQL查询解释助手",
    output_type=Union[QueryResult, ExplanationResult],
    output_retries=3,  # 最多重试3次
)

# result = agent.run_sync("查询所有的用户")
# print(type(result.output))
# print(result.output.sql)
# print(result.output.explanation)

# result = agent.run_sync("什么是 JOIN")
# print(type(result.output))
# print(result.output.topic)
# print(result.output.explanation)


def save_query_result(result: QueryResult) -> QueryResult:
    """保存 SQL 查询结果"""
    print(f"[保存] SQL : {result.sql}")
    return result


agent = Agent(
    "google-gla:gemini-2.5-pro",
    instructions="你是一个SQL生成和SQL查询解释助手",
    output_type=save_query_result,
    output_retries=3,  # 最多重试3次
)
result = agent.run_sync("查询所有的用户")
print(result)
