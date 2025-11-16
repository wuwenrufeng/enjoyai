"""06-tools-工具列表"""

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

client = OpenAI()

server_params = StdioServerParameters(
    command="uv",
    args=[
        "run",
        "server",
        "simple-tools-v1-FastMCP",
        "stdio",
    ],
    env={"UV_INDEX": os.environ.get("UV_INDEX", "")},
)


def call_llm_with_tools(messages, tools: list):
    response = client.chat.completions.create(
        model="deepseek-chat", messages=messages, tools=tools, tool_choice="auto"
    )
    return response.choices[0].message


async def run():
    async with stdio_client(server_params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            response = await session.list_tools()
            tools = response.tools
            print(
                "欢迎使用工具调用系统!\n 可用工具列表已加载。\n请输入您的需求(输入exit结束):"
            )

            tool_list = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in tools
            ]

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.请根据用户输入选择合适的工具并构造函数入参,或直接回复用户.",
                }
            ]
            while True:
                user_input = input("> ")
                if user_input == "exit":
                    break
                messages.append({"role": "user", "content": user_input})
                message = call_llm_with_tools(messages, tool_list)
                messages.append(message.model_dump())
                if not message.tool_calls:
                    print(message.content)
                    continue
                for tool_call in message.tool_calls:
                    args = json.loads(tool_call.function.arguments)  # type: ignore
                    result = await session.call_tool(tool_call.function.name, args)  # type: ignore
                    messages.append(
                        {
                            "role": "tool",
                            "content": str(result),
                            "tool_call_id": tool_call.id,
                        }
                    )
                message = call_llm_with_tools(messages, tool_list)
                print(message.content)
                messages.append(message.model_dump())


def main():
    """Entry point for the client script."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
