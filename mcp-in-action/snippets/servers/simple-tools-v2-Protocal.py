"""06-tools-工具列表"""

import anyio

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("tools-server")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all available tools."""
    return [
        types.Tool(
            name="calculator",
            description="执行基本的数学运算（加减乘除",
            inputSchema={
                "type": "object",
                "properties": {
                    "opt": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["opt", "a", "b"],
            },
        ),
        types.Tool(
            name="text_analyzer",
            description="分析文本，统计字符数和单词数",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """处理工具调用请求"""
    if name == "calculator":
        opt = arguments["opt"]
        a = arguments["a"]
        b = arguments["b"]

        if opt == "add":
            result = a + b
        elif opt == "subtract":
            result = a - b
        elif opt == "multiply":
            result = a * b
        elif opt == "divide":
            result = a / b
        else:
            raise ValueError(f"Invalid operation: {opt}")

        return [types.TextContent(type="text", text=f"计算结果：{result}")]
    elif name == "text_analyzer":
        text = arguments["text"]
        char_count = len(text)
        word_count = len(text.split())

        return [
            types.TextContent(
                type="text", text=f"字符数：{char_count}\n,单词数:{word_count}"
            ),
        ]
    return []


async def run_stdio_async(app) -> None:
    async with stdio_server() as (reader, writer):
        await app.run(reader, writer, app.create_initialization_options())


def main():
    anyio.run(run_stdio_async, app)


if __name__ == "__main__":
    main()
