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


db_agent = agent.Agent(
    "openai:kimi-k2-0711-preview",
    system_prompt="你是一个ClickHouse 表结构查询助手",
    deps_type=Config,
)


@db_agent.tool
async def list_database(ctx: RunContext[Config]) -> List[str]:
    result = await ctx.deps.ck_client.query("SHOW DATABASES")
    databases = [row[0] for row in result.result_rows]
    return databases


@db_agent.tool
async def list_tables(ctx: RunContext[Config], database) -> List[str]:
    result = await ctx.deps.ck_client.query(f"SHOW TABLES IN {database}")
    tables = [row[0] for row in result.result_rows]
    return tables


@db_agent.tool
async def describe_table(ctx: RunContext[Config], database, table: str) -> List[Column]:
    result = await ctx.deps.ck_client.query(f'DESCRIBE TABLE "{database}"."{table}"')
    columns = []
    for row in result.result_rows:
        item = dict(zip(result.column_names, row))
        column = Column(
            name=item.get("name", ""),
            type=item.get("type", ""),
            comment=item.get("comment", ""),
        )
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
        result = await db_agent.run(user_input, deps=config, message_history=history)
        history.extend(result.new_messages())
        print("Agent: ", result.output)


if __name__ == "__main__":
    asyncio.run(main())
