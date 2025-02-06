import os
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from openai import OpenAI


class PxWordsProcessor:
    def __init__(
        self, excel_file, qdrant_client, openai_client, collection_name="px_words"
    ):
        self.excel_file = excel_file
        self.qdrant_client = qdrant_client
        self.openai_client = openai_client
        self.collection_name = collection_name

    def process_excel(self):
        """
        读取并处理 Excel 文件，将数据整合到 DataFrame 中
        """
        df = pd.read_excel(self.excel_file, header=None)
        df["meaning"] = df.apply(self.combine_columns, axis=1)
        df = df.drop(columns=[1, 2])
        df.rename(
            columns={0: "word", 3: "pronunciation_ipa", 4: "pronunciation_px"},
            inplace=True,
        )
        df["meaning"] = df.apply(self.replace_meaning, axis=1)
        return df

    def combine_columns(self, row):
        """
        合并列1和列2为 "meaning" 列
        """
        if pd.isna(row[2]):
            return row[1]
        return row[1] + row[2]

    def replace_meaning(self, row):
        """
        替换 'meaning' 列中的 "～" 为 "word" 列的值
        """
        return (
            row["word"]
            if pd.isna(row["meaning"])
            else row["meaning"].replace("～", row["word"])
        )

    def to_embedding(self, word, meaning):
        """
        调用 OpenAI API 获取 word 和 meaning 的嵌入向量
        """
        input_text = f"{word} {meaning}"
        embedding_response = self.openai_client.embeddings.create(
            model="text-embedding-v3", input=input_text
        )
        return embedding_response.data[0].embedding

    def create_qdrant_collection(self):
        """
        创建 Qdrant 数据库集合，若已存在则删除
        """
        if self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.delete_collection(self.collection_name)

        return self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    def upsert_to_qdrant(self, df):
        """
        将 DataFrame 中的数据插入到 Qdrant
        """
        for index, row in df.iterrows():
            embedding = self.to_embedding(row["word"], row["meaning"])
            point = PointStruct(
                id=index,
                vector=embedding,
                payload={
                    "word": row["word"],
                    "meaning": row["meaning"],
                    "ipa": row["pronunciation_ipa"],
                    "px": row["pronunciation_px"],
                },
            )
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[point],
            )

            # 只插入前 10 条数据
            if index == 10:
                break


def main():
    # 初始化 Qdrant 和 OpenAI 客户端
    qdrant_client = QdrantClient("127.0.0.1", port=6333, check_compatibility=False)
    openai_client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    file_path = os.path.join(os.getcwd(), "../data/莆仙词汇表.xlsx")

    processor = PxWordsProcessor(file_path, qdrant_client, openai_client)

    processor.create_qdrant_collection()

    df = processor.process_excel()
    processor.upsert_to_qdrant(df)


if __name__ == "__main__":
    main()
