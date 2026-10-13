"""检索:查询多模态向量库"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import chromadb
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("BAILIAN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

CHROMA_PATH = Path(__file__).parent.parent / "chroma_db_multimodal"
chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))

def get_embedding(text:str)->list[float]:
    response = client.embeddings.create(
        model="text-embedding-v3",
        input=text,
    )
    return response.data[0].embedding

def retrieve(query:str,top_k:int=5,only_type:str|None=None)->list[dict]:
    """检索;only_type可限定'text' 或'image',None=不限"""
    col = chroma.get_collection("multimodal_docs")

    #metadata 过滤:先按结构化条件筛选,再计算向量相似度
    where = {"type":only_type} if only_type else None

    results = col.query(
        query_embeddings=[get_embedding(query)],
        n_results=top_k,
        include=["documents","metadatas","distances"],
        where=where,
    )

    out = []
    
    for i in range(len(results['documents'][0])):
        meta = results["metadatas"][0][i]
        out.append({
            "content":results["documents"][0][i],
            "type":meta["type"],
            "page":meta["page"],
            "source":meta["source"],
            "image_path":meta.get("image_path"),
            "score":1-results["distances"][0][i],
        })

    return out

if __name__ == "__main__":
    for q in ["蓝铜肽精华水的卖点","学分要求","学位授予条件"]:
        print(f"\n===查询:{q}===")
        for r in retrieve(q,top_k=3):
            tag = "[图]" if r["type"] == "image" else "[文]"
            print(f"{tag}第{r['page']}页 score={r['score']:.3f} | {r['content'][:50]}...")


