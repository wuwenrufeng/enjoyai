import os
import time
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI()

def create_assistant(instructions, file_id):
    assistant = client.beta.assistants.create(
        instructions=instructions,
        model=os.getenv('DEFAULT_MODEL'),
        tools=[{"type": "code_interpreter"}],
        tool_resources={
            "code_interpreter": {
                "file_ids": [file_id]
            }
        }
    )
    return assistant 

def create_thread(user_message, file_id):
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": user_message,
                "attachments": [
                    {
                        "file_id": file_id,
                        "tools": [{"type": "code_interpreter"}]
                    }
                ]
            }
        ]
    )
    return thread

def run_assistant(thread_id, assistant_id):
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    return run

def poll_run_status(thread_id, run_id: str, interval=10):
    while True:
        run = client.beta.threads.runs.retrieve(
            run_id=run_id,
            thread_id=thread_id,
        )
        print(f'Run的状态: {run.status}')
        print(f'Run的轮询信息: \n{run}\n')
        if run.status in ['require_action', 'completed', 'cancelled']:
            return run
        time.sleep(interval)

def get_assistant_reply(thread_id):
    respond = client.beta.threads.messages.list(thread_id=thread_id)
    msg = respond.data[-1]
    msg_content = msg.content[0].text
    annotations = msg.content[0].annotations
    citations = []
    for index, annotation in enumerate(annotations):
        msg_content.value = msg_content.value.replace(annotation.text, f'[{index}]')
        if (file_citation := getattr(annotation, 'file_citation', None)):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f'[{index}] {file_citation.quote} from {cited_file.filename}')
        elif (file_path := getattr(annotation, 'file_path', None)):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f'[{index}] Click <here> to download {cited_file.filename}')
            download_file(cited_file.filename, file_path.file_id)
        msg_content.value += '\n' + '\n'.join(citations)
        return msg_content.value

def download_file(filename, file_id):
    file_content = client.files.content(file_id)
    with open(filename, 'wb') as file:
        file.write(file_content.read())

def main():
    instructions = """
    Please use the flower_sales.csv data to complete the following analysis tasks:
    1. Group the data by region and calculate the total revenue for each region. Visualize the results using a bar chart.
    2. Group the data by date and calculate the daily total revenue. Create a line chart to show the revenue trend over time.
    3. Calculate the sales proportion of each flower type and display the results in a pie chart.
    4. Find the top 5 most profitable flowers based on the total profit.
    5. Using the historical sales data, forecast the total revenue for the next 7 days using a Random Forest Regressor model.
    """
    file_path = './03_Asst_Code_Intepreter/flower_sales.csv'
    with open(file_path, 'rb') as file:
        file_obj = client.files.create(file=file, purpose='assistants')
        file_id = file_obj.id

    assistant = create_assistant(instructions, file_id)

    user_msg = 'Please perform the data analysis tasks as instructed.'
    thread = create_thread(user_msg, file_id)
    run = run_assistant(thread.id, assistant.id)
    print(f'Run的初始信息: {run.status}')

    # 轮询Run直到完成或需要操作
    run = poll_run_status(thread.id, run.id)

    msg = client.beta.threads.messages.list(thread_id=thread.id)
    print(f'全部的msg:{msg}')

    print('下面打印最终的Assistant回复:')
    for msg in msg.data:
        if msg.role == 'assistant':
            print(f'Assistant回复: {msg.content}\n')

main()