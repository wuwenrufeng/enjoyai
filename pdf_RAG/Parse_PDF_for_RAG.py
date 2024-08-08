import os
import base64
import re
from io import BytesIO
from openai import AzureOpenAI
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
from rich import print
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai


###
# 1.从 PDF 文件中提取文本内容。
# 2.将 PDF 文件转换为一系列图像（即 PPT 图片）。
# 3.遍历每一张 PPT 图片（除了第一张，因为它通常是简介），使用GPT-4o模型分析图片内容，并将分析结果存储在图片内容列表中。
# 4.将提取的文本内容按照 \f 分隔符分割成多个文本页面（去除第一页）。
# 5.遍历每一个文本页面，尝试找到与之匹配的图片描述（通过比较标题），并将文本内容和匹配的图片描述组合在一起，形成一个完整的内容片段，存储在新列表中。
# 6.对于未被匹配到的图片描述，将它们单独添加到新列表中。
# 7.对组合后的内容进行清理，去除多余的空格、换行符和页码等。
# 8.为每个内容片段生成嵌入向量
# 9.检索搜索相关内容
# 10.根据检索结果生成回答
###

load_dotenv()

client = AzureOpenAI()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

file_path = "./09_PDF_RAG/GPT图解.pdf"

# 定义系统提示,用于指导GPT-4V分析图像内容
system_for_image_prompt = '''
你将获得一张PDF页面或幻灯片的图片。你的目标是以技术术语描述你所看到的内容,就像你在进行演示一样。

如果有图表,请描述图表并解释其含义。
例如:如果有一个描述流程的图表,可以说"流程从X开始,然后有Y和Z..."

如果有表格,请从逻辑上描述表格中的内容。
例如:如果有一个列出项目和价格的表格,可以说"价格如下:A为X,B为Y..."  

不要包含涉及内容格式的术语。
不要提及内容类型,而是专注于内容本身。
例如:如果图片中有图表和文本,请同时描述两者,而不要提及一个是图表,另一个是文本。
只需描述你在图表中看到的内容以及从文本中理解到的内容。

你应该保持简洁,但请记住,你的听众看不到图片,所以要详尽地描述内容。

排除与内容无关的元素:
不要提及页码或图片上元素的位置。

------

如果有明确的标题,请按以下格式输出:

{标题}

{内容描述}  

如果没有明确的标题,只需返回内容描述即可。

'''
# 定义系统提示,用于指导GPT-4回复输入查询
system_for_answer_prompt = '''
    你将获得一个输入提示和一些作为上下文的内容,可以用来回复提示。
    
    你需要做两件事:
    
    1. 首先,你要在内部评估提供的内容是否与回答输入提示相关。
    
    2a. 如果内容相关,直接使用这些内容进行回答。如果内容相关,使用内容中的元素来回复输入提示。
    
    2b. 如果内容不相关,使用你自己的知识回答,如果你的知识不足以回答,就说你不知道如何回应。 
    
    保持回答简洁,具体回复输入提示,不要提及上下文内容中提供的额外信息。
'''

def extract_text_from_doc(path):
    """
    从pdf文档中提取文本内容
    """
    text = extract_text(path)
    return text

def covert_doc_to_images(path):
    """
    将pdf文档转换为图片
    """
    images = convert_from_path(path)
    return images

def get_img_uri(img):
    """
    将图像转换为base64编码的数据URI格式
    """
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_image}"
    return data_uri

def analyze_image(img_url):
    """
    使用GPT-4o分析图像内容
    """
    response = client.chat.completions.create(
        model=os.getenv("DEFAULT_MODEL"),
        temperature=0,
        messages=[
            {
                'role': 'system',
                'content': system_for_image_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
        top_p=0.1
    )

    return response.choices[0].message.content

def analyze_doc_image(img):
    """
    分析PDF文档中的图像
    """
    img_uri = get_img_uri(img)
    content = analyze_image(img_uri)
    return content

def get_pages_description(images):
    """
    获取每张图片的描述
    """
    pages_descriptions = []
    for img in images[1:]:
        res = analyze_doc_image(img)
        pages_descriptions.append(res) 
    return  pages_descriptions

def combine_content(text, pages_descriptions):
    """
    组合文本和图像分析结果
    """
    combined_content = []
    # 去除第一张幻灯片
    text_pages = text.split('\f')[1:]
    description_indexes = []
    for text_content in text_pages:
        slide_content = f'{text_content}\n'
        # 尝试找到匹配的幻灯片描述
        slide_title = text_content.split('\n')[0]
        for i, page_content in enumerate(pages_descriptions):
            description_title = page_content.split('\n')[0]
            if slide_title.lower() == description_title.lower():
                slide_content += page_content.replace(description_title, '')
                # 记录已添加的描述的索引
                description_indexes.append(i)
        combined_content.append(slide_content) 
    # 添加未匹配到的描述
    for i, page_content in enumerate(pages_descriptions):
        if i not in description_indexes:
            combined_content.append(page_content)
    return combined_content

def clean_content(content):
    """
    清理组合内容
    """
    cleaned_content = []
    for c in content:
        if c.strip() == '':
            continue
        text = c.replace(' \n', '').replace('\n\n', '\n').replace('\n\n\n', '\n').strip()
        text = re.sub(r"(?<=\n)\d{1,2}", "", text)
        text = re.sub(r"\b(?:the|this)\s*slide\s*\w+\b", "", text, flags=re.IGNORECASE)
        cleaned_content.append(text)
    return cleaned_content

# def get_embeddings(text):
#     """
#     获取文本嵌入向量
#     """
#     embeddings = client.embeddings.create(
#         model='text-embedding-3-small',
#         input=text,
#         encoding_format='float'
#     )
#     return embeddings.data[0].embedding

def get_embeddings(text):
    """
    获取文本嵌入向量
    """
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document",
        title="Embedding of single string",
    )

    return result['embedding']

def search_content(df, input_text, top_k):
    """
    搜索与输入文本最相关的内容
    """
    embedded_value = get_embeddings(input_text)
    df['similarity'] = df.embeddings.apply(lambda x: cosine_similarity(np.array(x).reshape(1, -1), np.array(embedded_value).reshape(1, -1)).item())
    res = df.sort_values('similarity', ascending=False).head(top_k)
    return res

def get_similarity(row):
    """
    获取给定行的相似度得分
    """
    similarity_score = row['similarity']
    if isinstance(similarity_score, np.ndarray):
        similarity_score = similarity_score[0][0]
    return similarity_score

def generate_output(system_prompt, input_prompt, similar_content, threshold=0.5):
    """
    根据输入的提示和相似的内容生成回答
    """
    content = similar_content.iloc[0]['content']
    # 如果相似度高于阈值,添加更多匹配的内容
    if len(similar_content) > 1:
        for i, row in similar_content.iterrows():
            similarity_score = get_similarity(row)
            if similarity_score > threshold:
                content += f"\n\n{row['content']}"
    prompt = f'输入提示:\n{input_prompt}\n------\n内容:\n{content}'
    completion = client.chat.completions.create(
        model=os.getenv("DEFAULT_MODEL"),
        temperature=0,
        messages=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    return completion.choices[0].message.content
# text = extract_text_from_doc(file_path)
# images = covert_doc_to_images(file_path)
# pages_description = get_pages_description(images)
# combined_content = combine_content(text, pages_description)
# cleaned_content = clean_content(combined_content)
cleaned_content = ["新加坡科技研究局 资深研究员\\n极客时间专栏《零基础实战机器学习》作者\\n畅销书《零基础学机器学习》作者\\n畅销书《数据分析咖哥十话》作者\\n前埃森哲新加坡公司资深顾问\\n- 极客时间专栏《零基础实战机器学习》作者\\n- 畅销书《零基础学机器学习》作者\\n- 畅销书《数据分析咖哥十话》作者\\n- 前埃森哲新加坡公司资深顾问\\n图中展示了三本书的封面，分别是《零基础学机器学习》、《数据分析咖哥十话》和《GPT图解》。", "在一个遥远的未来世界，科技的力量无处不在。在这个世界里，聪明的人工智能助\\n手们为人类带来了无尽的便利。这个世界里，有两位闻名遐迩的人工智能研究员：\\n咖哥和小冰。他们是一对师徒，携手在人工智能领域研究进步。为了让更多人了解\\n他们的故事，咖哥决定将他们的故事写成一本书。\\n故事从11月底那场ChatGPT发布会开始。那天，全世界都在关注这场发布会。在那个激\\n动人心的时刻，咖哥感受到了人工智能的力量和希望。然而，他也清楚地看到了人\\n工智能所带来的挑战和风险。\\n就在这时，一个陨石从天而降，砸在了发布会现场，瞬间引起了一片混乱。人们四\\n处奔逃，尖叫声此起彼伏。而陨石中，竟然有一个神秘的宝盒。咖哥好奇地走上前\\n去，小心翼翼地打开了宝盒。\\n宝盒里，竟然是一封古老的羊皮卷。咖哥展开羊皮卷，发现上面写着一段预言：只\\n有通过历练和传承，才能将人工智能的力量驾驭得更加完美。咖哥意识到，他的使\\n命就是要传授给小冰和其它NLP小白ChatGPT的奥义！……", "AI技术的发端时期\\nThresholded\\nLogic Unit\\nPerceptron\\nAdaline\\n神经\\n网络\\n寒冬\\nXOR\\nPeoblem\\nAI技术的发展时期\\n神经\\n网络\\n寒冬\\n深度学习时代\\nChatGPT时代\\nMultilayerBackprop\\nCNNs\\nLSTMs\\nSVMs\\nAlex Net\\nTransformer\\nChatGPT\\nGPT4\\n43\\n57\\n60\\n69\\n82\\n86\\n89\\n97\\n95\\n12\\n18\\n22\\n23\\n40\\n50\\n60\\n70\\n80\\n90\\n00\\n10\\n20\\nS.McCulloch - W.Pitts\\nR.Rosenblatt\\nB.Widrow - M.Hoff\\nM.Minsky - S.Papert\\nP.Werbos\\nD.Rumelhart - G.Hinton - R.Williams\\nY.Lecun  J.Schmidhuber\\nC.Cortes - V.Vapnik\\nG.Hinton - A.Krizhevsky - I.Sutskever\\nAshish Vaswani  (Google 团队)\\nIlya Sutskever - Sam Altman( Open AI团队)", "信息/语言\\n你俩定\\n点啥菜？\\n别问我\\n特色菜谱\\n宫保鸡丁\\n芹菜炒肉\\n小鸡炖蘑菇\\n酸菜炖粉条\\n锅包肉\\n…\\n人类社会的核心就是——语言交流。语言不仅是信息的载体，也是人类情\\n感和思想的直接表达方式。它构成了人类社会交往的基础，是文化和知识\\n传承的关键，也承载着我们的情感。ChatGPT作为一个能够理解并使用人\\n类语言的模型，打破了人与机器之间交流的障碍，让人们感受到了前所未\\n有的亲密和便利。具有深远的意义，它不仅改变了人类与机器的互动方\\n式，还可能对教育、娱乐、甚至整个社会的发展产生深刻影响。\\n人类社会的核心就是——语言交流。语言不仅是信息的载体，也是人类情感和思想的直接表达方式。它构成了人类社会交往的基础，是文化和知识传承的关键，也承载着我们的情感。ChatGPT作为一个能够理解并使用人类语言的模型，打破了人与机器之间交流的障碍，让人们感受到了前所未有的亲密和便利。具有深远的意义，它不仅改变了人类与机器的互动方式，还可能对教育、娱乐、甚至整个社会的发展产生深刻影响。\\n左侧展示了三种古代文字的图像，分别是象形文字、甲骨文和另一种古文字。右侧是一幅古代人物的绘画，人物之间的对话框显示了他们在讨论点菜的问题，菜单上列出了几道特色菜：宫保鸡丁、芹菜炒肉、小鸡炖蘑菇、酸菜炖粉条和锅包肉。", "语言模型本质上是什么？\\n信息\\n编码\\n发送人\\n信息\\n信道\\n解码\\n信息\\n接收人\\n简化的通信模型\\n是信息编码和解码的通道\\n无论是原始人含混的声音，还是现代的文本，如果无法解码，那么信息就无法传递\\n语言模型本质上是一种复杂的算法，用于对人类语言进行编码和解码。在编码过程中，它将文本或语音转换为机器可\\n以理解的数据形式；在解码过程中，它再将这些数据转换回人类可以理解的语言形式。通过这种方式，语言模型能够\\n捕捉和模拟人类语言的复杂结构和含义，实现有效的沟通和理解。简而言之，语言模型是信息交流的桥梁，连接了人\\n类的语言世界和计算机的数字世界。\\n信息从发送人开始，通过编码过程转化为机器可理解的形式，然后通过信道传输。接收人通过解码过程将信息转化为人类可理解的语言形式。这个过程被称为简化的通信模型。\\n语言模型本质上是一种复杂的算法，用于对人类语言进行编码和解码。在编码过程中，它将文本或语音转换为机器可以理解的数据形式；在解码过程中，它再将这些数据转换回人类可以理解的语言形式。通过这种方式，语言模型能够捕捉和模拟人类语言的复杂结构和含义，实现有效的沟通和理解。简而言之，语言模型是信息交流的桥梁，连接了人类的语言世界和计算机的数字世界。", "吃了吗，亲？\\n吃了吗，亲？\\n吃了吗，亲？\\n对话\\nNLP\\n000\\n110\\n100\\nNLP的目标是缩小人类语言和计算机代码之间的差距，使机器能够更自\\n然、更有效地与人类交流，提高信息处理的效率和智能化程度。", "语言模型的进化\\n50s\\n70s\\n70s\\n90s\\n现在\\n阿兰·图灵（Alan Turing）在论\\n文中描述了会“思考型”机\\n器：能够与人类自然的对话\\n基于规则的语言处理方法是\\n由语言学家开发的，用于确\\n定计算机如何处理语言。\\n随着统计学和数据驱动的\\n自然语言处理方法逐渐成\\n为主流。\\n起源于\\n图灵测试\\n基于规则\\n基于统计\\n深度学习时代到来，开始出\\n现基于深度神经网络的预训\\n练语言模型，ChatGPT等聊天\\n机器人。\\n深度学习\\n大数据驱动\\n50年代：\\n阿兰·图灵（Alan Turing）在论文中描述了会“思考型”机器：能够与人类自然的对话。这个阶段的语言模型起源于图灵测试。\\n70年代：\\n基于规则的语言处理方法是由语言学家开发的，用于确定计算机如何处理语言。这个阶段的语言模型基于规则。\\n90年代：\\n随着统计学和数据驱动的自然语言处理方法逐渐成为主流。这个阶段的语言模型基于统计。\\n现在：\\n深度学习时代到来，开始出现基于深度神经网络的预训练语言模型，ChatGPT等聊天机器人。这个阶段的语言模型深度学习大数据驱动。", "前言", "在一个遥远的未来世界，科技的力量无处不在。在这个世界里，聪明的人工智能助手们为人类带来了无尽的便利。这个世界里，有两位闻名遐迩的人工智能研究员：咖哥和小冰。他们是一对师徒，携手在人工智能领域研究进步。为了让更多人了解他们的故事，咖哥决定将他们的故事写成一本书。\\n故事从11月底那场ChatGPT发布会开始。那天，全世界都在关注这场发布会。在那个激动人心的时刻，咖哥感受到了人工智能的力量和希望。然而，他也清楚地看到了人工智能所带来的挑战和风险。\\n就在这时，一个陌生的人在示威场，砸在了发布会现场，瞬间引起了一片混乱。人们四处奔逃，尖叫声此起彼伏。而与此同时，发现有一个神秘的宝盒。咖哥好奇地走上前去，小心翼翼地打开了宝盒。\\n宝盒里，竟然是一封古老的羊皮卷。咖哥展开羊皮卷，发现上面写着一段预言：只有通过历史和传承，才能将人工智能的重要性得到加倍体现。咖哥意识到，他的使命就是要传授给小冰和其它N", "AI技术的发展历程\\n. **AI技术的发端时期**\\n   - **1943年**: Thresholded Logic Unit\\n   - **1957年**: Perceptron\\n   - **1960年**: Adaline\\n   - **1969年**: XOR Problem\\n. **AI技术的发展时期**\\n   - **1982年**: Multilayer Backpropagation\\n   - **1986年**: CNNs (Convolutional Neural Networks)\\n   - **1989年**: LSTMs (Long Short-Term Memory)\\n   - **1995年**: SVMs (Support Vector Machines)\\n. **深度学习时期**\\n   - **2012年**: AlexNet\\n   - **2018年**: Transformer\\n. **ChatGPT时代**\\n   - **2022年**: ChatGPT\\n   - **2023年**: GPT-4\\n**图示说明**:\\n- 时间轴从1940年到2020年，展示了AI技术的主要发展节点。\\n- 每个节点下方展示了相关的研究人员（面部模糊处理）。\\n- 底部的图示展示了各个技术的示意图，例如神经网络的结构、支持向量机的分类图等。", "NLP的目标\\n第一行展示了两个人物之间没有对话的情况，旁边有一个变色龙的图标，表示无法交流。\\n第二行展示了两个人物之间的对话，表示人类之间的交流。\\n第三行展示了一个人类和一个计算机之间的交流，旁边有二进制代码，表示通过自然语言处理（NLP）实现人机交流。\\n底部的文字解释了NLP的目标：缩小人类语言和计算机代码之间的差距，使机器能够更自然、更有效地与人类交流，提高信息处理的效率和智能化程度。"]
# 将清理后的内容转换为DataFrame  
df = pd.DataFrame(cleaned_content, columns=['content'])
# 为每个内容片段生成嵌入向量
df['embeddings'] = df['content'].apply(lambda x: get_embeddings(x))

# 定义与内容相关的示例用户查询
example_inputs = [
    '语言模型的本质是什么?', 
    'NLP技术的目标是什么?',
    '语言模型经历了怎样的发展历程?',
    '作者解释语言时候，怎样用了古人进行对话?',
    '请介绍一下GPT图解一书作者?',
]
for ex in example_inputs:
    print(f"[deep_pink4][bold]查询:[/bold] {ex}[/deep_pink4]\n\n")
    matching_content = search_content(df, ex, 3)
    print(f"[grey37][b]匹配内容:[/b][/grey37]\n")
    for i, row in matching_content.iterrows():
        print(f"[grey37][i]相似度: {get_similarity(row):.2f}[/i][/grey37]")
        content = str(row['content'])
        print(f"[grey37]{content[:100]}{'...' if len(content) > 100 else ''}[/[grey37]]\n\n")
    reply = generate_output(system_for_answer_prompt, ex, matching_content)
    print(f"[turquoise4][b]回复:[/b][/turquoise4]\n\n[spring_green4]{reply}[/spring_green4]\n\n--------------\n\n")
