"""入库切块:纯函数,不联网、不读库"""
import pytest

from ingest import chunk_text


def test_empty_text_returns_no_chunk():
    assert chunk_text("") == []


def test_short_text_is_single_chunk():
    text = "很短的一段话"
    assert chunk_text(text, chunk_size=500) == [text]


def test_no_overlap_covers_text_exactly():
    """overlap=0 时块拼回去必须与原文完全相等 —— 切块最容易在这里丢字"""
    text = "0123456789" * 3                    # 30 字
    chunks = chunk_text(text, chunk_size=10, overlap=0)
    assert len(chunks) == 3
    assert "".join(chunks) == text


def test_chunk_count_matches_sliding_window():
    text = "x" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 3                    # 起点 0 / 450 / 900
    assert all(len(c) <= 500 for c in chunks)


def test_overlap_tail_matches_next_head():
    """重叠区必须一致:前一块的尾巴 == 后一块的开头"""
    text = "".join(str(i % 10) for i in range(1000))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev[-20:] == nxt[:20]


@pytest.mark.skip(reason="已知缺陷:overlap >= chunk_size 时 start 不前进会死循环,待加 assert 0 <= overlap < chunk_size")
def test_overlap_bigger_than_size_should_be_rejected():
    """TODO 标记:体检报告 P2 那条"""
