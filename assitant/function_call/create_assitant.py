import os
import json
from openai import AzureOpenAI
from openai.types.beta.threads.run import Run
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI()

def create_assistant():
    assistant = client.beta.assistants.create(
        instructions="您是一个订单助手。请使用提供的函数来计算订单总价并回答问题。",
        model="gpt-4o",
        tools=[{
            "type": "function",
            "function": {
                "name": "calculate_order_total",
                "description": "根据多个商品类型和数量计算订单总价",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "item_type": {
                                        "type": "string",
                                        "description": "商品类型,例如:书籍,文具,电子产品"
                                    },
                                    "quantity": {
                                        "type": "integer",
                                        "description": "商品数量"
                                    }
                                },
                                "required": [
                                    "item_type",
                                    "quantity"
                                ]
                            }
                        }
                    },
                    "required": [
                        "items"
                    ]
                }
            }
        }]
    )

    print(assistant)
    # Assistant(id='asst_uHP16yah8A4camMoQch84p21', created_at=1720443660, description=None, instructions='您是一个订单助手。请使用提供的函数来计算订单总价并回答问题。', metadata={}, model='gpt-4o', name=None, object='assistant', tools=[FunctionTool(function=FunctionDefinition(name='calculate_order_total', description='根据多个商品类型和数量计算订单总价', parameters={'type': 'object', 'properties': {'items': {'type': 'array', 'items': {'type': 'object', 'properties': {'item_type': {'type': 'string', 'description': '商品类型,例如:书籍,文具,电子产品'}, 'quantity': {'type': 'integer', 'description': '商品数量'}}, 'required': ['item_type', 'quantity']}}}, 'required': ['items']}), type='function')], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0)

# 创建一个新的Thread
thread = client.beta.threads.create()
print(thread)

# 向thread中添加用户的消息
message = client.beta.threads.messages.create(
    thread_id=thread.id, 
    role="user", 
    content="你好，我购买了一本书和一个电子产品,请帮我计算一下订单总价。")
print(message)

# 调用assistant_id执行thread
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id="asst_uHP16yah8A4camMoQch84p21"
)
# print(run)

# 读取function元数据信息
def get_function_details(run: Run):
  output = run.required_action.submit_tool_outputs.tool_calls[0]
  function_name = output.function.name
  arguments = output.function.arguments
  arguments = json.loads(arguments)
  function_id = output.id
  return function_name, arguments, function_id

function_name, arguments, function_id = get_function_details(run)
print(function_name, arguments, function_id)
