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
  - qwen3-vl-plus
  - text-embedding-v3
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
| 多模态理解 | Qwen3-VL-Plus（`qwen3-vl-plus`，图片描述生成 / 多模态答案生成） |
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

# 6. (可选)混合检索 + 精排 —— 线上未启用,本地可跑
pip install jieba rank-bm25           # BM25 混合检索必需(已含在 requirements.txt)
pip install sentence-transformers     # 精排必需:含 PyTorch,约 2-3GB,首次再下载 1.1GB 模型
python src/retrieve_hybrid.py         # use_rerank=True 即走 Cross-Encoder 精排
```

> 线上（魔搭创空间）默认走纯向量检索：`api.py → retrieve()`，不依赖 `sentence_transformers`，
> 因此 Docker 镜像保持轻量（`requirements.txt` 共 10 个包）；混合检索与精排作为**本地实验链路**独立存在。

---

## 🧪 测试

```bash
uv run pytest -q          # 19 passed, 1 skipped（约 4 秒）
```

三层测试，**全部离线可跑**：不需要 API key、不出网、不写真实向量库。

| 层 | 文件 | 覆盖点 |
|----|------|--------|
| 纯函数 | `tests/test_chunk.py` | 切块丢字、重叠区对齐、空输入 |
| 检索逻辑 | `tests/test_retrieve_hybrid.py` | RRF 只用排名、空库不除零、精排分数真正写回 |
| 接口契约 | `tests/test_api_contract.py` | 引用编号连续、正文截断、路径穿越 404、413/422/503 语义、异常转为结构化 JSON |

依赖（检索 / 生成 / 向量化 / 向量库）全部用 `monkeypatch` 替换，所以跑测试不会花钱。

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
- 检索层的现状是**主动取舍**，不是没做：混合检索（BM25 + RRF 融合）与 Cross-Encoder 精排均已实现于 `src/retrieve_hybrid.py`，但只作本地实验链路，**线上默认纯向量**。原因是成本：`sentence-transformers` 带 PyTorch，镜像体积 +约 2GB、冷启动显著变慢，精排模型（`BAAI/bge-reranker-base`）首次运行还要再下载 1.1GB；而本项目单文档、142 个片段，属于小库小流量场景，收益不足以覆盖这笔启动成本。三档检索质量的实测对照见 `eval/`。

---

## 📄 License

Apache License 2.0
