"""
07-prompts-提示词模板
"""

from mcp.server import Server
import mcp.types as types
import anyio
from mcp.server.stdio import stdio_server


# 定义可用的提示模板
PROMPTS = {
    "code-review": types.Prompt(
        name="code-review",
        description="审查给定的代码片段",
        arguments=[
            types.PromptArgument(
                name="code",
                description="需要审查的代码片段",
                required=True,
            ),
            types.PromptArgument(
                name="language",
                description="代码的编程语言（例如：Python, JavaScript）",
                required=True,
            ),
            types.PromptArgument(
                name="focus",
                description="审查的重点（可选: performace, security, readability）",
                required=False,
            ),
        ],
    ),
    "explain-code": types.Prompt(
        name="explain-code",
        description="解释给定代码的功能和工作原理",
        arguments=[
            types.PromptArgument(
                name="code",
                description="需要解释的代码片段",
                required=True,
            ),
            types.PromptArgument(
                name="language",
                description="代码的编程语言（例如：Python, JavaScript）",
                required=True,
            ),
        ],
    ),
}

# 初始化服务器
app = Server("code-assistant")


@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return list(PROMPTS.values())


@app.get_prompt()
async def get_prompt(
    name: str, arguments: dict[str, str] | None = None
) -> types.GetPromptResult:
    """根据名称获取提示内容"""
    prompt = PROMPTS.get(name)
    if not prompt:
        raise ValueError(f"Prompt '{name}' not found.")
    if name == "code-review":
        code = arguments.get("code", "") if arguments else ""
        language = arguments.get("language", "") if arguments else ""
        focus = arguments.get("focus", "general") if arguments else "general"
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="assistant",
                    content=types.TextContent(
                        type="text",
                        text=f"你是一个代码审查助手。请审查以下{language}代码，重点关注{focus}方面.",
                    ),
                ),
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text", text=f"请审查以下代码:\n{code}"
                    ),
                ),
            ]
        )
    elif name == "explain-code":
        code = arguments.get("code", "") if arguments else ""
        language = arguments.get("language", "") if arguments else ""
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="assistant",
                    content=types.TextContent(
                        type="text",
                        text=f"你是一个专业的代码解释助手。请解释以下{language}代码的功能和工作原理.",
                    ),
                ),
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text", text=f"请解释以下代码:\n{code}"
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported prompt name: {name}")


async def run_stdio_async(app) -> None:
    async with stdio_server() as (reader, writer):
        await app.run(reader, writer, app.create_initialization_options())


def main():
    anyio.run(run_stdio_async, app)


if __name__ == "__main__":
    main()
