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


class Table(BaseModel):
    database: str
    table: str


db_agent = agent.Agent(
    "openai:kimi-k2-0711-preview",
    instructions="你是一个ClickHouse数据库查询助手",
    deps_type=Config,
)


@db_agent.tool
async def list_database(ctx: RunContext[Config]) -> List[str]:
    """List all databases on your ClickHouse cluster."""
    result = await ctx.deps.ck_client.query("SHOW DATABASES")
    databases = [row[0] for row in result.result_rows]
    return databases


@db_agent.tool
async def list_tables(ctx: RunContext[Config], database: str) -> List[str]:
    """List tables in a database"""
    result = await ctx.deps.ck_client.query(
        "SELECT name FROM system.tables WHERE database = %(db)s",
        parameters={"db": database},
    )
    tables = [row[0] for row in result.result_rows]
    return tables


@db_agent.tool
async def describe_table(
    ctx: RunContext[Config], database: str, table: str
) -> List[Column]:
    """List columns info in a database.table"""
    result = await ctx.deps.ck_client.query(
        "SELECT name, type, comment FROM system.columns WHERE database = %(db)s AND table = %(tb)s",
        parameters={"db": database, "tb": table},
    )
    columns = []
    for row in result.result_rows:
        column = Column(
            name=row[0],
            type=row[1],
            comment=row[2],
        )
        columns.append(column)
    return columns


@db_agent.tool
async def find_tables_with_column(
    ctx: RunContext[Config], column_name: str
) -> List[Table]:
    """查询包含输入的列名的表信息"""
    result = await ctx.deps.ck_client.query(
        "SELECT database,table FROM system.columns WHERE name = %(name)s",
        parameters={"name": column_name},
    )
    tables = [Table(database=r[0], table=r[1]) for r in result.result_rows]
    return tables


async def main():
    ck_client = await clickhouse_connect.get_async_client(
        dsn=os.getenv("CLICKHOUSE_DSN")
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
