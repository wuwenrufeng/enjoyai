"""06-tools-工具列表"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("tools-server")


@mcp.tool()
async def calculator(opt: str, a: float, b: float) -> str:
    """
    执行基本的数学运算
    Args:
        opt (str): 运算符，支持加减乘除
        a (float): 数字1
        b (float): 数字2
    Returns:
        str: 运算结果
    """
    if opt == "add":
        return f"计算结果:{a + b}"
    elif opt == "sub":
        return f"计算结果:{a - b}"
    elif opt == "mul":
        return f"计算结果:{a * b}"
    elif opt == "div":
        if b == 0:
            return "错误，除数不能为0"
        return f"计算结果:{a / b}"
    else:
        return f"不支持的运算符:{opt}"


@mcp.tool()
async def text_analyzer(text: str) -> str:
    """分析文本：统计字符数和单词数
    Args:
        text (str): 文本内容
    Returns:
        str: 统计结果
    """
    char_count = len(text)
    word_count = len(text.split())
    return f"字符数:{char_count}, 单词数:{word_count}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
