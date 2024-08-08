import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI()


def create_assistant(instructions):
    assistant = client.beta.assistants.create(
        instructions=instructions,
        model=os.getenv('DEFAULT_MODEL'),
        tools=[{"type": "file_search"}],
    )
    return assistant 

def create_vector_store(name, file_paths):
    try:
        # 创建一个新的Vector Store
        vector_store = client.beta.vector_stores.create(name=name)
        # 准备要上传到openai的文件
        file_streams = [open(path, 'rb') for path in file_paths]
        # 使用SDK的上传和轮询辅助方法来上传文件,将它们添加到Vector Store中,
        # 并轮询文件批次的状态直到完成
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=file_streams
        )
        # 打印批次的状态和文件计数,查看此操作的结果
        print(f'文件批次状态: {file_batch.status}')
        print(f'文件计数: {file_batch.file_counts}')
        return vector_store, file_batch
    except Exception as e:
        print(f'创建Vector Store时发生错误: {e}')
        raise e

def update_assistant_vector_store(assistant_id, vector_store_id):
    try:
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store_id]
                }
            }
        )
        return assistant
    except Exception as e:
        print(f'更新Assistant时发生错误: {e}')
        raise e

def create_thread(user_message, file_id):
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": user_message,
                "attachments": [
                    {
                        "file_id": file_id,
                        "tools": [{"type": "file_search"}]
                    }
                ]
            }
        ],
    )
    print(f'Thread的tool_resources: {thread.tool_resources}')
    return thread

def run_assistant(thread_id, assistant_id, instructions):
    try:
        # 使用create_and_poll SDK辅助方法创建run并轮询状态直到完成 
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id, assistant_id=assistant_id,
            instructions=instructions 
        )

        # 获取run生成的消息
        messages = list(client.beta.threads.messages.list(thread_id=thread_id, run_id=run.id))
        
        # 提取消息的文本内容
        message_content = messages[0].content[0].text
        annotations = message_content.annotations
        citations = []

        # 处理文件引用,将原文中的引用替换为[index]的形式
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = client.files.retrieve(file_citation.file_id) 
                citations.append(f"[{index}] {cited_file.filename}")

        print(message_content.value)
        print("\n".join(citations))
        
    except Exception as e:
        print(f"运行Assistant失败: {e}")
        raise e

def main():
    instructions = "你是一位销售数据分析助手。请利用提供的销售数据,尽可能准确完整地回答用户的问题。"
    # 创建启用了file_search工具的Assistant
    assistant = create_assistant(instructions)
    print(f'Assistant的ID: {assistant.id}')
    # 创建Vector store并上传销售数据文件
    file_paths = ['./04_File_Search/flower_sales.docx']
    vector_store, file_batch = create_vector_store('Sales Data', file_paths)
    # 将新的Vector store关联到Assistant
    assistant = update_assistant_vector_store(assistant.id, vector_store.id)
    user_message = "请分析一下各种花卉的销售情况,哪个品种卖得最好,哪个卖得最差?对于销量不佳的品种,有什么推广建议吗?"

    # 获取Vector store中的文件列表
    files = list(client.beta.vector_stores.files.list(vector_store.id))
    file_id = files[0].id
    thread = create_thread(user_message, file_id)
    run_instructions = "以花店店长的身份回答问题。" 
    run_assistant(thread.id, assistant.id, run_instructions)

# vectors = client.beta.vector_stores.list()
# for vector in vectors:
#     client.beta.vector_stores.delete(vector.id)

main()
