import time
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI()

assistant_id="asst_uHP16yah8A4camMoQch84p21"
# 检索已创建的assistant
assistant = client.beta.assistants.retrieve(assistant_id)
print(assistant)

# 创建一个thread
thread = client.beta.threads.create(
    messages=[{"role": "user", "content": "你好，请问你能做什么?"}]
)
print(thread)

# 运行Assistant来处理thread
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant_id,
)
print(f'Run: {run}')

# 定义一个轮询run状态的函数
def poll_run_status(thread_id, run_id: str, interval=3):
    while True:
        run = client.beta.threads.runs.retrieve(
            run_id=run_id,
            thread_id=thread_id,
        )
        print(f'Run的状态: {run.status}')
        print(f'Run的轮询信息: \n{run}\n')
        if run.status in ['require_action', 'completed', 'cancelled']:
            return run
        if run.status == 'in_progress':
            client.beta.threads.runs.cancel(run_id=run_id, thread_id=thread_id)
        time.sleep(interval)

# 轮询run状态
run = poll_run_status(thread.id, run.id)
print(f'Run的最终信息: \n{run}\n')

# 获取Assistant在Thread中的respond
messages = client.beta.threads.messages.list(thread.id)
print(f'Thread中的消息: \n{messages}\n')

# 输出Assistant的最终respond
print(f'下面打印最终的Assistant respond')
for message in messages:
    if message.role == 'assistant':
        print(f'{message.content}\n')
