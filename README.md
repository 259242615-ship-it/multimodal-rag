---
domain:
  - multi-modal
tags:
  - RAG
  - Multimodal
  - Document-QA
  - FastAPI
  - Streamlit
datasets:
  evaluation:
models:
license: Apache License 2.0
---

# 多模态文档问答系统 (Multimodal RAG)

支持**含图表 PDF** 的智能问答系统。除了文本内容，还能理解文档中的**表格、示意图、流程图**，并给出带引用溯源的答案。

> 🚀 **在线体验**：https://www.modelscope.cn/studios/aalllllong/multimodal-rag

---

## ✨ 核心设计：两阶段多模态架构

传统 RAG 只能处理纯文本，而真实文档里大量信息藏在**图片和表格**中。本项目采用两阶段设计：

```
【检索阶段】 图片 ──视觉模型──→ 文字描述(caption) ──embedding──→ 向量库
                 (快、便宜、适合语义匹配)

【生成阶段】 命中图片 → 回传【原始图片】给视觉模型 → 生成答案
                 (无损、精确,保留表格中的具体数字)
```

**为什么这样做？**
- 只用 caption 生成：模型看到的是「表格包含学分、外语等级等字段」这种**转述**，具体数字容易丢失
- 回传原图：模型直接读表格中的精确数据，答案准确性显著提升
- 检索用 caption 而非原图：图片 embedding 需要额外模型且中文效果弱，文本 embedding 更成熟可靠

---

## 🏗️ 系统架构

```
┌─────────────┐   HTTP    ┌──────────────┐        ┌──────────────┐
│  Streamlit  │ ────────→ │   FastAPI    │ ─────→ │  ChromaDB    │
│   前端(7860)│           │   后端(8000) │        │  向量库      │
│  · 问答界面 │ ←──────── │  · /ask      │        └──────────────┘
│  · 检索仪表盘│           │  · /search   │               ↑
└─────────────┘           │  · /images   │        ┌──────────────┐
                          └──────┬───────┘        │  DashScope   │
                                 └───────────────→│ qwen-vl /    │
                                                  │ embedding-v3 │
                                                  └──────────────┘

【离线管线】PDF ─→ 解析(文本+图片) ─→ 图片生成描述 ─→ 向量化入库
【在线管线】提问 ─→ 检索(caption匹配) ─→ 原图+文本 ─→ 视觉模型 ─→ 答案
```

---

## 🛠️ 技术栈

| 层 | 技术 |
|----|------|
| 文档解析 | PyMuPDF（文本抽取 + 图片导出 + CMYK→RGB 转换 + 小图标过滤） |
| 多模态理解 | Qwen-VL-Plus（图片描述生成 / 多模态答案生成） |
| 向量化 | DashScope text-embedding-v3 |
| 向量库 | ChromaDB（持久化 + cosine 相似度） |
| 后端 | FastAPI + Pydantic |
| 前端 | Streamlit（问答 + 检索仪表盘） |
| 部署 | Docker + ModelScope 创空间 |

---

## 📁 项目结构

```
multimodal-rag/
├── src/
│   ├── parse_pdf.py        # PDF解析:文本块 + 图片导出
│   ├── describe_images.py  # 图片 → 视觉模型描述(带缓存)
│   ├── ingest.py           # 多模态入库(文本+caption→向量库)
│   ├── retrieve.py         # 检索(支持类型过滤)
│   ├── generate.py         # 生成(文本+原图→多模态模型)
│   ├── api.py              # FastAPI 后端
│   └── app.py              # Streamlit 前端
├── data/
│   ├── pdfs/               # 源PDF
│   ├── images/             # 抽取的图片
│   └── captions.json       # 图片描述缓存
├── chroma_db_multimodal/   # 持久化向量库
├── Dockerfile
├── start.sh
└── requirements.txt
```

---

## 🚀 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置密钥(.env)
BAILIAN_API_KEY=your_dashscope_key

# 3. 入库(离线跑一次)
python src/ingest.py

# 4. 启动后端
python src/api.py

# 5. 启动前端(另开终端)
streamlit run src/app.py
```

---

## 🔍 功能特性

- **多模态检索**：文本块与图片描述统一入库，支持按类型过滤
- **引用溯源**：答案标注 `[n]` 来源编号，图片可溯源查看原图
- **检索仪表盘**：实时查看检索结果的分数分布、类型统计、排名明细
- **工程细节**：图片描述缓存（断点续跑/省钱）、metadata 过滤、持久化向量库

---

## ⚠️ 已知限制

- 图片描述质量依赖视觉模型能力，复杂图表可能描述不完整
- 当前单文档场景验证，尚未测试大规模文档集
- 检索层为纯向量检索，尚未接入混合检索（BM25 + RRF 融合）——规划中

---

## 📄 License

Apache License 2.0
