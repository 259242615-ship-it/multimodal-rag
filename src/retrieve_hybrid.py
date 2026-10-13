"""混合检索:BM25+向量+RRF融合"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import jieba
import chromadb
from pathlib import Path
from openai import OpenAI
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from retrieve import get_embedding,chroma

# sentence_transformers(含PyTorch)只在精排时才需要,改为延迟导入:
# 未安装时 use_rerank=False 的混合检索依然可用,不至于 import 就崩
_reranker = None
COLLECTION = "multimodal_docs"



def tokenize(text:str)->list[str]:
    """中文分词"""
    return list(jieba.cut(text))

def build_bm25_index():
   """从向量数据库中取出所有片段,建立BM25索引"""
   col = chroma.get_collection(COLLECTION)
   data = col.get(include=["documents","metadatas"])

   ids = data["ids"]
   docs = data["documents"]
   metas = data["metadatas"]

   bm25 = BM25Okapi([tokenize(d) for d in docs])
   return bm25,ids,docs,metas

def bm25_search(bm25,ids,query:str,top_k:int=10)->list[tuple[str,float]]:
    """关键词检索->[(chunk_id,分数)]"""
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(ids,scores),key=lambda x:x[1],reverse=True)
    return ranked[:top_k]

def vector_search(query:str,top_k:int=10)->list[tuple[str,float]]:
    """向量检索->[(chunk_id,分数)]"""
    col = chroma.get_collection(COLLECTION)
    r = col.query(
        query_embeddings=[get_embedding(query)],
        n_results=top_k,
        include=["documents","metadatas","distances"],
    )
    ids = r["ids"][0]
    scores = [1-d for d in r["distances"][0]]
    return list(zip(ids,scores))

def rrf_fusion(list_a:list[tuple[str,float]],
               list_b:list[tuple[str,float]],
               k:int=60)->list[tuple[str,float]]:
    """RRF融合两个榜单"""
    scores = {}

    for rank,(chunk_id,_) in enumerate(list_a):
        scores[chunk_id] = scores.get(chunk_id,0) + 1/(k + rank + 1)

    for rank,(chunk_id,_) in enumerate(list_b):
        scores[chunk_id] = scores.get(chunk_id,0) + 1/(k + rank + 1)

    return sorted(scores.items(),key=lambda x:x[1],reverse=True)

def get_reranker():
    """懒加载:第一次调用才导入并载入模型(模型约1.1GB,走 hf-mirror 镜像)"""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("正在加载reranker模型...")
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker

def rerank(query:str,candidates:list[dict],top_k:int=3)->list[dict]:
    """Cross-Encoder精排:"""
    if not candidates:
        return []
    model = get_reranker()
    pairs = [(query,c['content']) for c in candidates]
    scores = model.predict(pairs)

    for c,s in zip(candidates,scores):
        c["score"] = float(s)

    ranked = sorted(candidates,key=lambda x:x['score'],reverse=True)
    return ranked[:top_k]

def hybrid_retrieve(query:str,top_k:int=5,candidates:int=10,use_rerank=True)->list[dict]:
    """混合检索:BM25+向量+RRF->统一格式结果"""
    #1.建立索引(每次调用重建,文档少无所谓,大规模应该缓存)
    bm25,ids,docs,metas = build_bm25_index()
    if not ids:
        print("[!] 向量库为空,混合检索直接返回空")
        return []

    #2.两个榜单出牌
    a = bm25_search(bm25,ids,query,top_k=candidates)
    b = vector_search(query,top_k=candidates)

    #3.RRF融合
    fused = rrf_fusion(a,b)
    #4.id->内容映射,组装和retrieve一致的格式
    id2idx = {cid:i for i,cid in enumerate(ids)}
    out = []
    for chunk_id,score in fused[:candidates]:
        meta = metas[id2idx[chunk_id]]
        out.append({
            "content":docs[id2idx[chunk_id]],
            "type":meta["type"],
            "page":meta["page"],
            "source":meta["source"],
            "image_path":meta.get("image_path"),
            "score":score,
        })

    if use_rerank:
        return rerank(query,out,top_k=top_k)

    return out[:top_k]

if __name__ == "__main__":
    queries = [
        "学分要求",
        "蓝铜肽精华水的卖点",
        "学位授予条件",
    ]

    for q in queries:
        print(f"\n{'='*55}")
        print(f"查询:{q}")

        print("\n-----粗排-----")
        for r in hybrid_retrieve(q,use_rerank=False):
            tag = "[图]" if r['type'] == "image" else "[文]"
            print(f"当前类型为{tag}第{r['page']}页 rrf得分:{r['score']:.3f} | 内容:{r['content'][:35]}...")

        print("\n-----精排-----")
        for r in hybrid_retrieve(q,use_rerank=True):
            tag = "[图]" if r['type'] == "image" else "[文]"
            print(f"当前类型为{tag}第{r['page']}页 精排分数:{r['score']:.3f} | 内容:{r['content'][:35]}...")












