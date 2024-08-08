
import ast


# 原始 Python 代码
source_code = """
def greet(name):
    print(f"Hello, {name}!")
greet("World")
"""
parsed_code = ast.parse(source_code)
exec(compile(parsed_code, filename="<string>", mode="exec"))
