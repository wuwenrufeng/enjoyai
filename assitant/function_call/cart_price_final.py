import os
import json
from openai import AzureOpenAI
from openai.types.beta.threads.run import Run
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI()

assistant_id="asst_uHP16yah8A4camMoQch84p21"

# 创建一个新的Thread
# 并初始化messages
thread = client.beta.threads.create(
    messages=[{"role": "user", "content": "你好，我购买了一本书和一个电子产品,请帮我计算一下订单总价。"}]
)
print(thread)

# 向thread中添加用户的消息
# message = client.beta.threads.messages.create(
#     thread_id=thread.id, 
#     role="user", 
#     content="你好，我购买了一本书和一个电子产品,请帮我计算一下订单总价。")
# print(message)

# 调用assistant_id执行thread
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id="asst_uHP16yah8A4camMoQch84p21",
    extra_headers={
        'Accept': 'application/json; charset=utf-8'
    }
)
# print(run)

# 读取function元数据信息
def get_function_details(run: Run):
  output = run.required_action.submit_tool_outputs.tool_calls[0]
  function_name = output.function.name
  arguments = output.function.arguments
  function_id = output.id
  return function_name, arguments, function_id

function_name, arguments, function_id = get_function_details(run)
print(function_name, arguments, function_id)

# 定义计算订单总价函数
def calculate_order_total(items):
    item_prices = {
        "书籍": 10,
        "文具": 5,
        "电子产品": 100
    }
    total_price = 0
    for item in items:
        price_per_item = item_prices.get(item['item_type'], 0)
        total_price += price_per_item * item['quantity']
    return total_price

# 调用计算订单总价函数
arguments_dict = json.loads(arguments.encode('iso-8859-1').decode('utf-8'))
order_total = globals()[function_name](**arguments_dict)
# 打印结果以验证
print(f"订单总价: {order_total}")

# 提交结果
run = client.beta.threads.runs.submit_tool_outputs_and_poll(
    run_id=run.id,
    thread_id=thread.id,
    tool_outputs=[
        {
            "tool_call_id": function_id,
            "output": str(order_total),
        }
    ]

)
print(run)

# 获取Assistant在Thread中的回应
messages = client.beta.threads.messages.list(thread_id=thread.id)
print(f'全部消息: {messages}')

# 输出响应
print('下面打印最终的Assistant回应:')
for m in messages.data:
    if m.role == 'assistant':
        print(f'{m.content}')