"""
07-prompts-提示词模板
"""

import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from openai import OpenAI
import os
import mcp.types as types

server_params = StdioServerParameters(
    command="uv",
    args=[
        "run",
        "server",
        "prompts_v2_protocal",
        "stdio",
    ],
    env={"UV_INDEX": os.environ.get("UV_INDEX", "")},
)


class CodeAssistantClient:
    def __init__(self):
        self.session = None
        self.transport = None
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE_URL"),
        )
        self.prompts = None

    async def connect(self):
        self.transport = stdio_client(server_params)
        self.read, self.write = await self.transport.__aenter__()
        self.session = await ClientSession(self.read, self.write).__aenter__()
        await self.session.initialize()

        self.prompts = await self.session.list_prompts()

        if not isinstance(self.prompts, dict):
            self.prompts = dict(self.prompts)

        prompt_list = self.prompts.get("prompts", [])
        print("可用的prompts:")
        for prompt in prompt_list:
            print(f"- {prompt.name}: {prompt.description}")

    async def use_prompt(self, prompt_name: str, arguments: dict[str, str]):
        prompt_result = await self.session.get_prompt(prompt_name, arguments)  # type: ignore

        messages = []
        for msg in prompt_result.messages:
            if isinstance(msg.content, types.TextContent):
                messages.append({"role": msg.role, "content": msg.content.text})
        print(f"\n 使用prompt '{prompt_name}' 生成的消息: {messages}")

        response = self.client.chat.completions.create(
            model="deepseek-chat", messages=messages
        )

        return response.choices[0].message.content

    async def close(self):
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self.transport:
            await self.transport.__aexit__(None, None, None)


async def run():
    print(">>> 开始初始化代码助手客户端...")
    client = CodeAssistantClient()
    try:
        await client.connect()
        # code
        sample_code = """
    def fibonacci(n):
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        elif n == 2:
            return [0, 1]
        seq = [0, 1]
        for i in range(2, n):
            seq.append(seq[-1] + seq[-2])
        return seq
                """
        while True:
            print("\n 请选择操作:")
            print("1. 代码审查")
            print("2. 代码解释")
            print("3. 退出")

            choice = input("> ")

            if choice == "3":
                break

            if choice == "1":
                print("\n请选择审查重点:")
                print("1. 性能")
                print("2. 安全性")
                print("3. 可读性")
                print("4. 综合")

                focus_choice = input("> ")
                focus_map = {
                    "1": "performance",
                    "2": "security",
                    "3": "readability",
                    "4": "general",
                }
                focus = focus_map.get(focus_choice, "general")
                print("\n 正在进行代码审查...")
                response = await client.use_prompt(
                    "code-review",
                    {"code": sample_code, "language": "python", "focus": focus},
                )
                print("\n 代码审查结果:", response)
            elif choice == "2":
                print("\n 正在进行代码解释...")
                response = await client.use_prompt(
                    "explain-code", {"code": sample_code, "language": "python"}
                )
                print("\n 代码解释结果:", response)
    except Exception as e:
        print(f"连接服务器时出错: {e}")
        return
    finally:
        await client.close()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
