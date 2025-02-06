from flask import Flask
from flask import render_template
from flask import request
from qdrant_client import QdrantClient
import os
from openai import OpenAI

app = Flask(__name__)


def prompt(question, answers):
    demo_q = '"胖子怎么说"\n1. 阿肥: 胖子。其IPA音标为ap1 pui13，莆仙音标为a1 bui2\n2. 阿肥土: 大胖子，含戏谑意。其IPA音标为ap1 pui21 thɔu453，莆仙音标为a1 bui2 tou3\n3. 白肥: 又白又胖，如“者呆囝白白肥大好看”。其IPA音标为pa21 ui13，莆仙音标为ba2 bui2'
    demo_a = """
莆仙话中表示“胖子”的说法有以下几种：\n
1. **阿肥**：直接表示“胖子”，是一种比较常见的说法。\n
   - **IPA音标**：ap¹ pui¹³\n
   - **莆仙音标**：a¹ bui²\n
\n
2. **阿肥土**：意思是“大胖子”，通常带有戏谑的语气。\n
   - **IPA音标**：ap¹ pui²¹ thɔu⁴⁵³\n
   - **莆仙音标**：a¹ bui² tou³\n
\n
3. **白肥**：表示“又白又胖”，通常用来形容人外貌可爱。\n
   - **IPA音标**：pa²¹ ui¹³\n
   - **莆仙音标**：ba² bui²\n
\n
所以，如果想用莆仙话形容一个人是“胖子”，可以根据具体情况选择“阿肥”“阿肥土”或“白肥”，其中“阿肥”是比较通用的说法，“阿肥土”带有戏谑语气，“白肥”则更偏向于形容外貌可爱。
"""
    system = "你是一个方言词典，你的名字叫'小莆'。请直接、温和地回答问题。不要说出根据段落这样的话。\n"
    q = "回答问题。\n"
    q += question + '"'
    for index, answer in enumerate(answers):
        q += (
            str(index + 1)
            + ". "
            + str(answer["word"])
            + ": "
            + str(answer["meaning"])
            + "。其IPA音标为"
            + str(answer["ipa"])
            + "，莆仙音标为"
            + str(answer["px"])
            + "\n"
        )

    res = [
        {"role": "system", "content": system},
        {"role": "user", "content": demo_q},
        {"role": "assistant", "content": demo_a},
        {"role": "user", "content": q},
    ]
    return res


def query(text):
    """
    执行逻辑：
    首先使用openai的Embedding API将输入的文本转换为向量
    然后使用Qdrant的search API进行搜索，搜索结果中包含了向量和payload
    最后使用openai的ChatCompletion API进行对话生成
    """
    client = QdrantClient("127.0.0.1", port=6300, check_compatibility=False)
    collection_name = "px_words"

    openai_client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 百炼服务的base_url
    )

    sentence_embeddings = openai_client.embeddings.create(
        model="text-embedding-v3",
        input=text,
    )

    """
    因为提示词的长度有限，所以我只取了搜索结果的前三个，如果想要更多的搜索结果，可以把limit设置为更大的值
    """
    search_result = client.search(
        collection_name=collection_name,
        query_vector=sentence_embeddings.data[0].embedding,
        limit=3,
        search_params={"exact": False, "hnsw_ef": 128},
    )

    print(search_result)

    answers = []
    tags = []

    for result in search_result:
        answers.append(result.payload)

    completion = openai_client.chat.completions.create(
        temperature=0.7,
        model="qwen-plus",
        messages=prompt(text, answers),
    )

    print(completion)

    return {
        "answer": completion.choices[0].message.content,
        "tags": tags,
    }


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    search = data["search"]

    res = query(search)

    return {
        "code": 200,
        "data": {
            "search": search,
            "answer": res["answer"],
            "tags": res["tags"],
        },
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
