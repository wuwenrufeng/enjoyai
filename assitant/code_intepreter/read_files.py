import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI()


def download_file(filename, file_id):
    file_content = client.files.content(file_id)
    with open(filename, 'wb') as file:
        file.write(file_content.read())

def get_assistant_messaages(thread_id, assistant_id):
    respond = client.beta.threads.messages.list(thread_id=thread_id)
    messages = [message for message in respond.data if message.role == 'assistant' and message.assistant_id == assistant_id]
    return messages

def process_message(message):
    for content_block in message.content:
        if content_block.type == 'image_file':
            file_id = content_block.image_file.file_id
            filename = f'image_{file_id}.png'
            download_file(filename, file_id)
            print(f'Downloaded image: {filename}')
        elif content_block.type == 'text':
            text = content_block.text.value
            annotations = content_block.text.annotations
            for annotation in annotations:
                if annotation.type == 'file_path':
                    file_id = annotation.file_path.file_id
                    filename = annotation.text.split('/')[-1]
                    download_file(filename, file_id)
                    print(f'Downloaded file: {filename}')

def main():
    assistant_id = "asst_e5C28fk3pILbZVdke81IhgUj"
    thread_id = "thread_wO08Eroi4g6pM2W38Ulevr7u"

    messages = get_assistant_messaages(thread_id, assistant_id)
    for message in messages:
        process_message(message)

main()