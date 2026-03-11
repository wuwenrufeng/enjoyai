import asyncio
import json
import os
from dataclasses import dataclass
from io import BytesIO
from typing import List

import clickhouse_connect
import matplotlib.pyplot as plt
import pandas as pd
from clickhouse_connect.driver.asyncclient import AsyncClient
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRetry,
    RunContext,
    ToolReturn,
    agent,
)

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
    # "openai:kimi-k2-0711-preview",
    "google-gla:gemini-2.5-pro",
    instructions="你是一个ClickHouse数据库查询助手",
    deps_type=Config,
)


@db_agent.tool(timeout=10)
async def list_database(ctx: RunContext[Config]) -> List[str]:
    """List all databases on your ClickHouse cluster."""
    result = await ctx.deps.ck_client.query("SHOW DATABASES")
    databases = [row[0] for row in result.result_rows]
    return databases


@db_agent.tool(timeout=10)
async def list_tables(ctx: RunContext[Config], database: str) -> List[str]:
    """List tables in a database"""
    result = await ctx.deps.ck_client.query(
        "SELECT name FROM system.tables WHERE database = %(db)s",
        parameters={"db": database},
    )
    tables = [row[0] for row in result.result_rows]
    return tables


@db_agent.tool(timeout=10)
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


@db_agent.tool(timeout=10)
async def find_tables_with_column(
    ctx: RunContext[Config], column_name: str
) -> List[Table]:
    """查询包含输入的列名的表信息"""
    if not column_name.strip():
        raise ModelRetry(message="列名不能为空值")
    result = await ctx.deps.ck_client.query(
        "SELECT database,table FROM system.columns WHERE name = %(name)s",
        parameters={"name": column_name},
    )
    tables = [Table(database=r[0], table=r[1]) for r in result.result_rows]
    return tables


@db_agent.tool_plain
def get_company_logo() -> ImageUrl:
    return ImageUrl(url="https://iili.io/3Hs4FMg.png")


@db_agent.tool(timeout=60)
async def generate_trend_chart(
    ctx: RunContext[Config],
    database: str,
    table: str,
    metric_column: str,
    time_column: str = "ds",
) -> ToolReturn:
    """生成趋势图表"""
    sql = f'SELECT {metric_column} metric, {time_column} ds FROM "{database}"."{table}" LIMIT 1000'
    df = await ctx.deps.ck_client.query_df(sql)
    # 数据清洗
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").dropna(subset=["metric"])

    # 画图（不显示）
    plt.figure(figsize=(12, 6))
    plt.plot(df["ds"], df["metric"], color="#2E86AB", linewidth=2)
    plt.title(f"{metric_column} Trend")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存到内存字节流
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()  # 释放内存，防止告警

    img_bytes = buf.getvalue()
    buf.close()
    return ToolReturn(
        return_value="生成数据趋势图成功",
        content=[
            f"请解读图片中 {metric_column} 按 时间列ds 的趋势",
            BinaryContent(data=img_bytes, media_type="image/png"),
        ],
        metadata={"sql": sql},
    )


HISTORY_FILE = "chat_history.json"


def save_history(history: List[ModelMessage]):
    data = ModelMessagesTypeAdapter.dump_json(history, indent=2)
    with open(HISTORY_FILE, "wb") as f:
        f.write(data)


def load_history() -> List[ModelMessage]:
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE) as f:
        data = json.load(f)
        hist = ModelMessagesTypeAdapter.validate_python(data)
        return hist


async def main():
    ck_client = await clickhouse_connect.get_async_client(
        dsn=os.getenv("CLICKHOUSE_DSN")
    )
    config = Config(ck_client=ck_client)
    history = load_history()
    while True:
        user_input = input("User: ")
        if user_input == "q":
            break
        result = await db_agent.run(user_input, deps=config, message_history=history)
        history.extend(result.new_messages())
        print("Agent: ", result.output)

    save_history(history)


if __name__ == "__main__":
    asyncio.run(main())
