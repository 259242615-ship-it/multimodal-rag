"""PDF解析:抽取文本块+图片->多模态管线第一环"""
import sys
sys.stdout.reconfigure(encoding="utf-8") #防止emoji报错

import pymupdf
from pathlib import Path

def parse_pdf(pdf_path:str,images_dir:str)->list[dict]:
    """解析PDF,返回统一元素列表(文本块+图片引用)"""
    doc = pymupdf.open(pdf_path)
    out_dir = Path(images_dir)
    out_dir.mkdir(parents=True,exist_ok=True)

    items = []

    for page_num,page in enumerate(doc):
        #抽文字
        text = page.get_text().strip()
        if text:
            items.append({
                "type":"text",
                "page":page_num + 1,
                "content":text,
            })

        for img_index,img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = pymupdf.Pixmap(doc,xref)

            #过滤装饰性小图(logo/图标/分割线)
            if pix.width < 100 or pix.height < 100:
                pix = None
                continue

            #CMYK等印刷色彩空间转RGB
            if pix.n - pix.alpha > 3:
                pix = pymupdf.Pixmap(pymupdf.csRGB,pix)

            img_path = out_dir / f"page{page_num+1}_img{img_index}.png"
            pix.save(img_path)
            pix = None

            items.append({
                "type":"image",
                "page":page_num + 1,
                "path":str(img_path),
            })

    doc.close()
    return items

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    pdfs = list((root/"data"/"pdfs").glob("*.pdf"))

    if not pdfs:
        print(f"[x]没找到pdf,请放一份到{root/'data'/'pdfs'}")
    else:
        items = parse_pdf(str(pdfs[0]),str(root/'data'/'images'))
        texts = [i for i in items if i["type"]=="text"]
        imgs = [i for i in items if i["type"]=="image"]
        print(f"[ok] 解析 {pdfs[0].name}:文本块{len(texts)}个,图片{len(imgs)}张")
        for i in imgs[:5]:
            print(f"第{i['page']}页->{Path(i['path']).name}")


