"""HTTP 契约:TestClient 在进程内打真实 app,但检索/生成/向量化/向量库全换假的"""
import pytest
from fastapi.testclient import TestClient

import api

client = TestClient(api.app, raise_server_exceptions=False)

FAKE_CHUNKS = [
    {"content": "甲" * 400, "type": "text", "page": 1, "source": "t.pdf",
     "image_path": None, "score": 0.9},
    {"content": "乙", "type": "image", "page": 2, "source": "t.pdf",
     "image_path": "D:/nope/not_exist.png", "score": 0.8},
]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """切掉一切外部依赖:检索 / 生成 / 向量化全部换成假的"""
    monkeypatch.setattr(api, "retrieve", lambda q, top_k=5, only_type=None: FAKE_CHUNKS)
    monkeypatch.setattr(api, "hybrid_retrieve",
                        lambda q, top_k=5, use_rerank=True: FAKE_CHUNKS)
    monkeypatch.setattr(api, "generate_answer", lambda q, chunks: "模拟答案[1]")
    monkeypatch.setattr(api, "get_embedding", lambda text: [0.0] * 1024)


class FakeCol:
    def __init__(self):
        self.added = []

    def add(self, **kw):
        self.added.append(kw)

    def count(self):
        return len(self.added)


def _fake_chroma(get_collection):
    return type("C", (), {"get_collection": staticmethod(get_collection)})()


def test_ask_contract():
    r = client.post("/ask", json={"question": "蓝铜肽"})
    assert r.status_code == 200
    d = r.json()
    assert d["retrieval_mode"] == "vector"
    assert [s["index"] for s in d["sources"]] == [1, 2]      # 编号从 1 连号
    assert len(d["sources"][0]["content"]) == 300            # 截断到 300 生效
    assert d["sources"][1]["image_url"] is None              # 图不存在 -> 不给坏链接


def test_ask_hybrid_and_rerank_flag(monkeypatch):
    calls = {}

    def fake_hybrid(q, top_k=5, use_rerank=True):
        calls["use_rerank"] = use_rerank
        return FAKE_CHUNKS

    monkeypatch.setattr(api, "hybrid_retrieve", fake_hybrid)

    r = client.post("/ask", json={"question": "x", "use_hybrid": True, "use_rerank": True})
    assert r.status_code == 200
    assert r.json()["retrieval_mode"] == "hybrid+rerank"
    assert calls["use_rerank"] is True                       # 参数真的透传下去了


def test_ask_requires_question():
    assert client.post("/ask", json={}).status_code == 422   # Pydantic 校验兜住


def test_ask_generation_error_returns_structured_500(monkeypatch):
    """B1 的语义:模型挂了也要是带 detail 的 JSON 500,不是裸栈"""
    def boom(q, chunks):
        raise RuntimeError("模拟模型挂了")

    monkeypatch.setattr(api, "generate_answer", boom)
    r = client.post("/ask", json={"question": "x"})
    assert r.status_code == 500
    assert r.headers["content-type"].startswith("application/json")
    assert "模拟模型挂了" in r.json()["detail"]


def test_search_passes_filters(monkeypatch):
    seen = {}

    def fake(q, top_k=5, only_type=None):
        seen.update(q=q, top_k=top_k, only_type=only_type)
        return []

    monkeypatch.setattr(api, "retrieve", fake)

    r = client.get("/search", params={"q": "学分", "top_k": 2, "only_type": "image"})
    assert r.status_code == 200
    assert seen == {"q": "学分", "top_k": 2, "only_type": "image"}


def test_images_404_and_path_traversal():
    assert client.get("/images/not_exist.png").status_code == 404
    assert client.get("/images/..%2F..%2Fpyproject.toml").status_code == 404  # 不许越界读文件


def test_upload_rejects_suffix_size_and_empty():
    assert client.post("/upload",
                       files={"file": ("a.pdf", b"x", "application/pdf")}).status_code == 400
    big = b"x" * 1_000_001
    assert client.post("/upload",
                       files={"file": ("b.txt", big, "text/plain")}).status_code == 413
    assert client.post("/upload",
                       files={"file": ("c.txt", b"", "text/plain")}).status_code == 400


def test_upload_ok_then_503(monkeypatch):
    col = FakeCol()
    monkeypatch.setattr(api, "chroma", _fake_chroma(lambda name: col))

    r = client.post("/upload", files={"file": ("d.txt", "一句话".encode(), "text/plain")})
    assert r.status_code == 200
    assert r.json()["chunks"] == 1 and col.added

    def boom(name):
        raise RuntimeError("no collection")

    monkeypatch.setattr(api, "chroma", _fake_chroma(boom))
    r = client.post("/upload", files={"file": ("e.txt", b"hi", "text/plain")})
    assert r.status_code == 503 and "detail" in r.json()     # B1 修好的结构化 503
