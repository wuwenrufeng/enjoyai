"""
cd to the `examples/snippets/clients` directory and run:
    uv run mcp_sampling_client
"""

import asyncio
import base64
import pathlib
import re
import os
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext

client = OpenAI()

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="uv",  # Using uv to run the server
    args=[
        "run",
        "server",
        "mcp_sampling",
        "stdio",
    ],  # We're already in snippets dir
    env={"UV_INDEX": os.environ.get("UV_INDEX", "")},
)


# Optional: create a sampling callback
async def handle_sampling_message(
    context: RequestContext[ClientSession, None],
    params: types.CreateMessageRequestParams,
) -> types.CreateMessageResult:
    print(f"Sampling request: {params.systemPrompt}")
    messages = []
    if params.systemPrompt:
        messages.append({"role": "system", "content": params.systemPrompt})
    for msg in params.messages:
        if msg.content.type == "text":
            messages.append({"role": msg.role, "content": msg.content.text})
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=params.temperature,
        max_tokens=params.maxTokens,
    )
    # print(f"Sampling response: {response.choices[0].message.content}")
    code = (
        response.choices[0].message.content
        if response.choices[0].message.content
        else ""
    )

    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(
            type="text",
            text=code,
        ),
        model="deepseek-chat",
        stopReason="endTurn",
    )


def save_b64_image(b64_text: str, save_path: str | pathlib.Path) -> None:
    """
    将 base64 图片保存到本地
    :param b64_text:  可能带 data URI 头的 base64 字符串
    :param save_path: 目标文件路径，推荐含后缀（.jpg/.png…）
    """
    # 1. 去掉 data URI 头（如果有）
    data_url_pattern = re.compile(r"^data:image/\w+;base64,", re.IGNORECASE)
    base64_data = data_url_pattern.sub("", b64_text).strip()

    # 2. 解码
    try:
        img_bytes = base64.b64decode(base64_data, validate=True)
    except Exception as e:
        raise ValueError("非法 base64 内容") from e

    # 3. 写入文件
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(img_bytes)
    print(f"已保存 -> {save_path.absolute()}")


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write, sampling_callback=handle_sampling_message
        ) as session:
            # Initialize the connection
            await session.initialize()
            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools.tools]}")

            # Call a tool (add tool from fastmcp_quickstart)
            result = await session.call_tool(
                "analyze_csv_and_plot",
                arguments={
                    "csv_path": "/Users/wuwen/workspace/python/github/enjoyai/mcp-in-action/snippets/clients/cereal.csv"
                },
            )
            result_unstructured = result.content[0]
            if isinstance(result_unstructured, types.TextContent):
                print(f"Tool result: {result_unstructured.text}")
            if isinstance(result_unstructured, types.ImageContent):
                save_b64_image(
                    result_unstructured.data,
                    "result_plot.png",
                )
                print("Plot image saved as result_plot.png")


def main():
    """Entry point for the client script."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
