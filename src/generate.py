"""生成:文本块+原始图片->视觉模型生成答案"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import base64
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 图片目录:以本文件位置为锚,跨环境可移植
IMAGES_DIR = Path(__file__).parent.parent / "data" / "images"

client = OpenAI(
    api_key=os.getenv("BAILIAN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

SYSTEM_PROMPT = """你是文档知识库助手。根据提供的文本片段和图片回答问题。
要求:
1. 引用材料信息时,在句末标注来源编号,如[1]
2. 图片中的表格/图表数据以图片实际内容为准
3. 材料中找不到答案时明确说明,不要编造"""

def resolve_image(path_str: str) -> Path:
    """解析图片真实路径
    向量库里可能存的是入库机器的本地绝对路径(如 D:\\...\\page5_img0.png),
    部署到容器后该路径不存在 → 退回用文件名在当前环境的 data/images 里找
    """
    p = Path(path_str)
    if p.exists():
        return p
    return IMAGES_DIR / p.name


def encode_image(img_path:str)->str:
    """本地图片转base64 data URL"""
    path = resolve_image(img_path)
    with open(path,"rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def generate_answer(query:str,chunks:list[dict])->str:
    """把检索结果(文本+原图)组成多模态请求"""
    parts = []

    for i,c in enumerate(chunks,start=1):
        if c["type"] == "text":
            parts.append({
                "type":"text",
                "text":f"[{i}] (第{c['page']}页) {c['content']}"
            })
        else:
            #图片:传原始图片,让视觉模型直接看
            parts.append({
                "type":"text",
                "text":f"[{i}](第{c['page']}页 图片)"
            })

            parts.append({
                "type":"image_url",
                "image_url":{"url":encode_image(c["image_path"])}
            })

    parts.append({"type":"text","text":f"\n用户问题:{query}"})

    response = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":parts}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    from retrieve import retrieve

    q = "蓝铜肽精华水的卖点是什么"
    chunks = retrieve(q,top_k=3)
    print(f"检索到:{[c['type'] for c in chunks]}\n")
    print(generate_answer(q,chunks))