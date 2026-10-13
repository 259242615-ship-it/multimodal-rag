"""图片->qwen-v1生成描述(检索用caption)"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import base64
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("BAILIAN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

#为检索优化的caption prompt
CAPTION_PROMPT = """请详细描述这张图片,内容将用于建立文档检索索引。要求:
1. 如果表格:逐行提取所有字段名和数据,用文字完整表述
2. 如果流程图:说明每个步骤及其先后关系
3. 如果示意图/架构图:说明各组成部分和图中标注文字
4. 提取图中所有可读文字和关键数字
只客观描述图片内容,不要添加评论或推测"""

def encode_image(img_path:str)->str:
    """本地图片转base64 data URL"""
    with open(img_path,"rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def describe_all(images_dir:Path,out_file:Path)->dict:
    """批量描述:带缓存,已经处理的跳过(省钱+端点续跑)"""
    cache = {}
    if out_file.exists():
        cache = json.loads(out_file.read_text(encoding="utf-8"))
        print(f"[i]已经有缓存{len(cache)}张")

    imgs = sorted(images_dir.glob("*.png"))
    for i,img in enumerate(imgs,start=1):
        key = str(img)
        if key in cache:
            print(f"[skip]{img.name}已经缓存")
            continue
        try:
            cache[key] = describe_image(str(img))
            #每次存一次:中断也不丢进度
            out_file.write_text(
                json.dumps(cache,ensure_ascii=False,indent=2),
                encoding="utf-8",
            )
            print(f"[{i}/{len(imgs)}]{img.name}完成")
        except Exception as e:
            print(f"[!]{img.name} 失败:{e}")

    return cache



def describe_image(img_path:str)->str:
    """调用视觉模型描述单张图片"""
    response = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=[
            {
                "role":"user",
                "content":[
                    #多模态消息:content不再是字符串,而是列表
                    {"type":"image_url",
                     "image_url":{"url":encode_image(img_path)}},
                    {"type":"text",
                     "text":CAPTION_PROMPT},
                ]
            }
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    cache = describe_all(root/'data'/'images',root/'data'/'captions.json')
    print(f"[OK]共{len(cache)}张图片描述->data/captions.json")

