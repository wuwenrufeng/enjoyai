import asyncio
import os
from dataclasses import dataclass
from typing import List

import clickhouse_connect
from clickhouse_connect.driver.asyncclient import AsyncClient
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import RunContext, agent

load_dotenv()


@dataclass
class Config:
    ck_client: AsyncClient


class Column(BaseModel):
    name: str
    type: str
    comment: str


agent = agent.Agent(
    "openai:kimi-k2-0711-preview",
    system_prompt="你是一个ClickHouse 表结构查询助手",
    deps_type=Config,
)


@agent.tool
async def list_database(ctx: RunContext[Config]) -> List[str]:
    result = await ctx.deps.ck_client.query("SHOW DATABASES")
    databases = [row[0] for row in result.result_rows]
    return databases


@agent.tool
async def list_tables(ctx: RunContext[Config], database) -> List[str]:
    result = await ctx.deps.ck_client.query(f"SHOW TABLES IN {database}")
    tables = [row[0] for row in result.result_rows]
    return tables


@agent.tool
async def describe_table(ctx: RunContext[Config], database, table: str) -> List[Column]:
    result = await ctx.deps.ck_client.query(f'DESCRIBE TABLE "{database}"."{table}"')
    columns = []
    for row in result.result_rows():
        column = Column(name=row["name"], type=row["type"], comment=row["comment"])
        columns.append(column)
    return columns


async def main():
    ck_client = await clickhouse_connect.get_async_client(
        host=os.getenv("CLICKHOUSE_HOST")
    )
    config = Config(ck_client=ck_client)
    history = []
    while True:
        user_input = input("User: ")
        if user_input == "q":
            break
        result = await agent.run(user_input, deps=config, message_history=history)
        history.extend(result.new_messages())
        print("Agent: ", result.output)


async def test_query():
    client = await clickhouse_connect.get_async_client(dsn=os.getenv("CLICKHOUSE_DSN"))
    result = await client.query("SHOW DATABASES")
    print("类型:", type(result.result_rows()))
    print("第一行:", result.result_rows()[0] if result.result_rows() else None)
    print(
        "第一行类型:", type(result.result_rows()[0]) if result.result_rows() else None
    )


if __name__ == "__main__":
    # asyncio.run(main())
    asyncio.run(test_query())
