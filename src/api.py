"""FastAPI后端:多模态问答API"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from fastapi import FastAPI,HTTPException,File,UploadFile
from uuid import uuid4
from fastapi.responses import FileResponse
from pydantic import BaseModel
from retrieve_hybrid import hybrid_retrieve
from retrieve import retrieve,get_embedding,chroma
from ingest import chunk_text
from generate import generate_answer

IMAGES_DIR = Path(__file__).parent.parent/"data"/"images"

app = FastAPI(title="多模态文档问答API")

#====请求/响应模型====
class AskRequest(BaseModel):
    question:str
    top_k:int = 3
    use_hybrid:bool = False
    use_rerank:bool = False

class SourceItem(BaseModel):
    index:int
    type:str
    page:int
    score:float
    content:str
    image_url:str|None=None

class AskResponse(BaseModel):
    answer:str
    sources:list[SourceItem]
    retrieval_mode:str = "vector"

#====接口====

#====上传入库====
MAX_SIZE = 1_000_000

@app.post("/upload")
def upload(file:UploadFile=File(...)):
    """上传txt/md->切块->向量化->增量入库"""
    #1.校验后缀(放乱传/路径穿越)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt",".md"}:
        raise HTTPException(status_code=400,detail="只支持.txt/.md")

    #2.读取+解码(中文txt常见GBK编码)
    if file.size is not None and file.size > MAX_SIZE:
        raise HTTPException(status_code=413,detail="文本太大,上限1MB")
    data = file.file.read(MAX_SIZE + 1)
    if len(data) > MAX_SIZE:
        raise HTTPException(status_code=413,detail="文本太大,上限1MB")
    try:
        text = data.decode("utf-8")
    except UnicodeError:
        text = data.decode("gbk",errors="ignore")

    #3.切块
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400,detail="文件是空的")

    #4.增量入库
    try:
        col = _collection()
        tag = uuid4().hex[:8]
        for i,chunk in enumerate(chunks):
            col.add(
                embeddings=[get_embedding(chunk)],
                ids=[f"u_{tag}_{i}"],
                documents=[chunk],
                metadatas=[{"type":"text","page":0,"source":file.filename}],
            )
        return {"filename":file.filename,"chunks":len(chunks),"total":col.count()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"入库失败:{type(e).__name__}:{e}")
def _collection():
    """取向量库collection;不存在时给可读的503,而不是裸500"""
    try:
        return chroma.get_collection("multimodal_docs")
    except Exception as e:
        raise HTTPException(status_code=503,detail=f"向量库未就绪,请先运行ingest:{e}")


@app.post("/ask",response_model=AskResponse)
def ask(req:AskRequest):
    """问答:检索+生成"""
    try:
        if req.use_hybrid:
            chunks = hybrid_retrieve(req.question,top_k=req.top_k,use_rerank=req.use_rerank)
            mode = "hybrid+rerank" if req.use_rerank else "hybrid"
        else:
            chunks = retrieve(req.question,top_k=req.top_k)
            mode = "vector"

        answer = generate_answer(req.question,chunks)
        sources = []
        for i,c in enumerate(chunks,start=1):
            image_url = None
            if c["image_path"]:
                name = Path(c["image_path"]).name
                if (IMAGES_DIR/name).exists():
                    image_url = f"/images/{name}"
            sources.append(SourceItem(
                index=i,
                type=c["type"],
                page=c["page"],
                score=c["score"],
                content=c["content"][:300],
                image_url=image_url
            ))
        return AskResponse(answer=answer,sources=sources,retrieval_mode=mode)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"问答失败:{type(e).__name__}:{e}")


@app.get("/search")
def search(q:str,top_k:int=5,only_type:str|None=None):
    """检索调试接口(检索仪表盘用) 只检索 不调生成模型"""
    return {"query":q,"results":retrieve(q,top_k=top_k,only_type=only_type)}


@app.get("/images/{name}")
def get_image(name:str):
    """静态图片服务:把本地图片作为HTTP资源暴露"""
    p = (IMAGES_DIR/name).resolve()
    if not str(p).startswith(str(IMAGES_DIR.resolve())) or not p.is_file():
        raise HTTPException(status_code=404,detail="图片不存在")
    return FileResponse(p)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)



















