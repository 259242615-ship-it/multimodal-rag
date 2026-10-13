"""混合检索逻辑:全部 monkeypatch,不出网、不读真实向量库"""
import pytest

import retrieve_hybrid as rh


def test_rrf_fusion_rewards_agreement():
    """两个榜单都排第一的,融合后必须第一(RRF 的核心语义)"""
    a = [("A", 9.9), ("B", 8.8)]
    b = [("A", 0.9), ("C", 0.8)]
    fused = dict(rh.rrf_fusion(a, b, k=60))
    assert fused["A"] == pytest.approx(2 / 61)
    assert fused["B"] == pytest.approx(1 / 62)
    assert max(fused, key=fused.get) == "A"


def test_rrf_ignores_raw_scores():
    """只换原始分数、顺序不变,融合结果必须逐位相同 —— 证明 RRF 只用排名"""
    b = [("B", 5.0), ("A", 4.0)]
    assert rh.rrf_fusion([("A", 100.0), ("B", 1.0)], b) == \
           rh.rrf_fusion([("A", 0.01), ("B", 0.001)], b)


def test_tokenize_splits_chinese():
    """jieba 分词:这里退化成整句,BM25 就永远命中不了"""
    tokens = rh.tokenize("学位授予条件")
    assert len(tokens) > 1


def test_empty_library_returns_empty(monkeypatch):
    """空库不能崩、不能除零(复刻版就是在这里 ZeroDivisionError)"""
    monkeypatch.setattr(rh, "build_bm25_index", lambda: (None, [], [], []))
    assert rh.hybrid_retrieve("年假") == []


def test_rerank_writes_scores_back(monkeypatch):
    """精排分数必须真正写回 score 字段并重排(复刻版写成 == 比较,等于没生效)"""
    class FakeModel:
        def predict(self, pairs):
            return [0.1, 0.9]                  # 故意让第二个更高

    monkeypatch.setattr(rh, "get_reranker", lambda: FakeModel())
    cands = [{"content": "甲", "score": 0.7}, {"content": "乙", "score": 0.2}]
    out = rh.rerank("q", cands, top_k=2)
    assert [c["content"] for c in out] == ["乙", "甲"]
    assert out[0]["score"] == pytest.approx(0.9)


def test_hybrid_retrieve_pipeline(monkeypatch):
    ids = ["id1", "id2", "id3"]
    docs = ["文档一", "文档二", "文档三"]
    metas = [
        {"type": "text", "page": 1, "source": "t.pdf"},
        {"type": "image", "page": 2, "source": "t.pdf", "image_path": "a.png"},
        {"type": "text", "page": 3, "source": "t.pdf"},
    ]
    monkeypatch.setattr(rh, "build_bm25_index", lambda: ("fake", ids, docs, metas))
    monkeypatch.setattr(rh, "bm25_search",
                        lambda bm25, ids, q, top_k=10: [("id1", 5.0), ("id2", 1.0)])
    monkeypatch.setattr(rh, "vector_search",
                        lambda q, top_k=10: [("id3", 0.8), ("id1", 0.7)])

    out = rh.hybrid_retrieve("q", top_k=2, candidates=3, use_rerank=False)
    assert len(out) == 2                                  # top_k 生效
    assert out[0]["content"] == "文档一"                   # id1 两榜都在 -> RRF 第一
    assert set(out[0]) == {"content", "type", "page", "source", "image_path", "score"}
    assert out[0]["score"] > 0.03                         # RRF 量级 ~ 1/61+1/62
