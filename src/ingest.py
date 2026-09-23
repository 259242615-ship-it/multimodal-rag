"""多模态入库:文本块+图片描述->向量库"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
from pathlib import Path
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("BAILIAN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

CHROMA_PATH = Path(__file__).parent.parent/"chroma_db_multimodal"
chroma = chromadb.PersistentClient(path=(CHROMA_PATH))

def get_embedding(text:str)->list[float]:
    """文本转向量"""
    response = client.embeddings.create(
        model="text-embedding-v3",
        input=text,
    )
    return response.data[0].embedding

def chunk_text(text:str,chunk_size:int=500,overlap:int=50)->list[str]:

    start = 0
    chunks = []
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

def build_multimodal_index(items:list[dict],captions:dict,source_name:str):
    """文本块+图片描述 统一进库"""
    #重建库
    try:
        chroma.delete_collection("multimodal_docs")
    except Exception:
        pass
    col = chroma.create_collection("multimodal_docs",metadata={"hnsw:space":"cosine"})

    n_text,n_img = 0,0

    for item in items:
        if item["type"] == "text":
            for ci,chunk in enumerate(chunk_text(item["content"])):
                col.add(
                    embeddings=[get_embedding(chunk)],
                    ids=[f"t_{item['page']}_{ci}"],
                    documents=[chunk],
                    metadatas=[{
                        "type":"text",
                        "page":item["page"],
                        "source":source_name
                    }]
                )

                n_text += 1
        else:
            #图片:用caption向量化入库
            caption = captions.get(item["path"],"")
            if not caption:
                continue
            col.add(
                embeddings=[get_embedding(caption)],
                ids=[f"i_{item['page']}_{Path(item['path']).stem}"],
                documents=[caption],
                metadatas=[{
                    "type":"image",
                    "page":item["page"],
                    "source":source_name,
                    "image_path":item["path"]
                }]
            )

            n_img += 1

    print(f"[OK]入库:文本{n_text}块,图片{n_img}张")
    return col


if __name__ == "__main__":
    from parse_pdf import parse_pdf

    root = Path(__file__).parent.parent
    pdf_path = list((root/"data"/"pdfs").glob("*.pdf"))[0]

    #1.PDF->文字+图片
    items = parse_pdf(str(pdf_path),str(root/"data"/"images"))
    #2.读图片描述
    captions = json.loads((root/"data"/"captions.json").read_text(encoding="utf-8"))
    #3.入库
    build_multimodal_index(items,captions,pdf_path.name)

