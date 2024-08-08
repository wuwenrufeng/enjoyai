import ast
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI()

def display_message(messages):
    """
    打印发送给GPT或GPT回复的消息
    """
    for msg in messages:
        role = msg['role']
        content = msg['content']
        print(f'\n{role}\n{content}')

# 定义被测试的函数
example_function = '''
    def caesar_cipher(message, offset):
        result = ""
        for char in message:
            if char.isalpha():
                ascii_offset = 65 if char.isupper() else 97
                shifted_ascii = (ord(char) - ascii_offset + offset) % 26
                new_char = chr(shifted_ascii + ascii_offset)
                result += new_char
            else:
                result += char
        return result
'''

# 1. 生成函数的解释说明
explain_system_message = {
    'role': 'system',
    'content': '你是一位世界级的Python开发者,对意料之外的bug和边缘情况有着敏锐的洞察力。你总是非常仔细和准确地解释代码。你用markdown格式的项目列表来组织解释。'
}
explain_user_message = {
    'role':'user',
    'content': f'''请解释下面这个Python函数。仔细审查函数的每个元素在精确地做什么,作者的意图可能是什么。用markdown格式的项目列表组织你的解释。

    ```python
    {example_function}
    ```''',
}
explain_messages = [explain_system_message, explain_user_message]
display_message(explain_messages)

explanation_respond = client.chat.completions.create(
    model=os.getenv('DEFAULT_MODEL'),
    messages=explain_messages,
    temperature=0.4,
)
explanation = explanation_respond.choices[0].message.content
explain_assistant_message = {'role': 'assistant', 'content': explanation}
display_message([explain_assistant_message])

# 2. 生成编写单元测试的计划
plan_user_message = {
    'role': 'user',
    'content': f"""一个好的单元测试套件应该致力于:
                    - 测试函数在各种可能的输入下的行为
                    - 测试作者可能没有预料到的边缘情况 
                    - 利用 `pytest` 的特性使测试更容易编写和维护
                    - 易于阅读和理解,代码干净,名称描述清晰
                    - 具有确定性,测试总是以相同的方式通过或失败

                    为了帮助对上述函数进行单元测试,请列出函数应该能够处理的不同场景(每个场景下再用子项目列出几个例子)。 """,
}
plan_messages = [
    explain_system_message,
    explain_user_message,
    explain_assistant_message,
    plan_user_message
]
display_message(plan_messages)

plan_respond = client.chat.completions.create(
    model=os.getenv('DEFAULT_MODEL'),
    messages=plan_messages,
    temperature=0.4,
)
plan = plan_respond.choices[0].message.content
plan_assistant_message = {'role': 'assistant', 'content': plan}
display_message([plan_assistant_message])

# 3. 生成单元测试代码
execute_system_message = {
    'role': 'system',
    'content': '你是一位世界级的Python开发者,对意料之外的bug和边缘情况有着敏锐的洞察力。你编写谨慎、准确的单元测试。当要求只用代码回复时,你把所有代码写在一个代码块里。'
}
execute_user_message = {
    'role': 'user',
    'content': f""""使用Python和 `pytest` 包,按照上面的案例为函数编写一套单元测试。包含有助于解释每一行的注释。只需回复代码,格式如下:

                    ```python  
                    # 导入
                    import pytest  # 用于单元测试
                    {{根据需要插入其他导入}}

                    # 要测试的函数
                    {example_function}

                    # 单元测试 
                    # 下面,每个测试用例都用一个元组表示,传递给 @pytest.mark.parametrize 装饰器
                    {{在此插入单元测试代码}}
                    ```""",
}
execute_messages = [
    execute_system_message,
    explain_user_message,
    explain_assistant_message,
    plan_user_message,
    plan_assistant_message,
    execute_user_message,
]
display_message(execute_messages)
execute_respond = client.chat.completions.create(
    model=os.getenv('DEFAULT_MODEL'),
    messages=execute_messages,
    temperature=0.4,
)
execution = execute_respond.choices[0].message.content
execution_message = {'role': 'assistant', 'content': execution}
display_message([execution_message])

# 4. 检查生成的代码是否有语法错误
code = execution.split('```python')[1].split('```')[0].strip()
try:
    ast.parse(code)
    print(f'\n生成代码有效\n')
except SyntaxError as e:
    print(f'\n生成代码无效, 错误信息: {e}\n')
    raise ValueError('生成代码无效')

# 打印生成的单元测试代码
print(f'\n生成的单元测试代码:\n{code}\n')
