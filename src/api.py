"""FastAPI后端:多模态问答API"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from retrieve import retrieve
from generate import generate_answer

IMAGES_DIR = Path(__file__).parent.parent/"data"/"images"

app = FastAPI(title="多模态文档问答API")

#====请求/响应模型====
class AskRequest(BaseModel):
    question:str
    top_k:int = 3

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

#====接口====
@app.post("/ask",response_model=AskResponse)
def ask(req:AskRequest):
    """问答:检索+生成"""
    chunks = retrieve(req.question,top_k=req.top_k)
    answer = generate_answer(req.question,chunks)

    sources = []
    for i,c in enumerate(chunks,start=1):
        sources.append(SourceItem(
            index=i,
            type=c["type"],
            page=c["page"],
            score=c["score"],
            content=c["content"][:300],
            image_url=f"/images/{Path(c["image_path"]).name}" if c["image_path"] else None
        ))
    return AskResponse(answer=answer,sources=sources)


@app.get("/search")
def search(q:str,top_k:int=5,only_type:str|None=None):
    """检索调试接口(检索仪表盘用) 只检索 不调生成模型"""
    return {"query":q,"results":retrieve(q,top_k=top_k,only_type=only_type)}


@app.get("/images/{name}")
def get_image(name:str):
    """静态图片服务:把本地图片作为HTTP资源暴露"""
    p = IMAGES_DIR/name
    if not p.exists():
        raise HTTPException(status_code=404,detail="图片不存在")
    return FileResponse(p)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)



















