"""检索评测:纯向量 / 混合(BM25+RRF) / 混合+精排 三档对照

语料:与线上服务完全同一份向量库(chroma_db_multimodal),
      其中 14 块来自真实文档 学位授予细则.pdf,其余块是干扰文档。
标注:eval/golden.jsonl,每题给出「正确 chunk id 列表」+ 题型(literal 原词题 / paraphrase 改述题),
      top-k 命中任一即算召回。
指标:HitRate@3 / MRR@3 / Top1 命中率;整体 + 按题型分组(改述题不含文档原词,是真正拉开差距的地方)。
输出:控制台摘要 + eval/report.md 逐题明细。

用法:
    uv run python eval/run_eval.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from retrieve import retrieve, chroma          # noqa: E402
from retrieve_hybrid import hybrid_retrieve    # noqa: E402

COLLECTION = "multimodal_docs"
TOP_K = 3
GOLDEN = ROOT / "eval" / "golden.jsonl"
OUT = ROOT / "eval" / "report.md"


def build_content2id() -> dict:
    """内容 -> chunk id。retrieve / hybrid_retrieve 的返回里都没有 id,只能靠内容反查。"""
    col = chroma.get_collection(COLLECTION)
    d = col.get(include=["documents"])
    m = {}
    for cid, doc in zip(d["ids"], d["documents"]):
        m.setdefault(doc, cid)
    return m


def resolve_ids(chunks: list[dict], content2id: dict) -> list[str]:
    out = []
    for c in chunks:
        cid = content2id.get(c["content"])
        out.append(cid if cid else f'?{c["source"]}:p{c["page"]}')
    return out


CONFIGS = [
    ("纯向量", lambda q: retrieve(q, top_k=TOP_K)),
    ("混合(BM25+RRF)", lambda q: hybrid_retrieve(q, top_k=TOP_K, candidates=10, use_rerank=False)),
    ("混合+精排", lambda q: hybrid_retrieve(q, top_k=TOP_K, candidates=10, use_rerank=True)),
]

TAGS = ["literal", "paraphrase"]


class Acc:
    """一个(配置 × 题型)桶的累计指标"""

    def __init__(self) -> None:
        self.n = self.hit = self.top1 = 0
        self.rr = 0.0

    def add(self, rank: int | None) -> None:
        self.n += 1
        ok = rank is not None and rank <= TOP_K
        self.hit += ok
        if ok:
            self.rr += 1 / rank
            self.top1 += rank == 1

    @property
    def hitrate(self) -> float:
        return self.hit / self.n if self.n else 0.0

    @property
    def mrr(self) -> float:
        return self.rr / self.n if self.n else 0.0

    @property
    def top1rate(self) -> float:
        return self.top1 / self.n if self.n else 0.0


def main() -> None:
    golden = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    content2id = build_content2id()
    n_lit = sum(1 for g in golden if g.get("set", "literal") == "literal")
    n_par = len(golden) - n_lit
    print(f"语料:{len(content2id)} 块(内容去重) / 评测题:{len(golden)} 条"
          f"(原词题 {n_lit} + 改述题 {n_par}) / top_k={TOP_K}")
    print(f"随机基线:HitRate@{TOP_K} ≈ {TOP_K / len(content2id):.1%}\n")

    summary, details = [], []
    for name, fn in CONFIGS:
        t0 = time.time()
        buckets = {"all": Acc(), **{t: Acc() for t in TAGS}}
        rows, misses = [], []
        for item in golden:
            tag = item.get("set", "literal")
            got = resolve_ids(fn(item["q"]), content2id)
            gold = set(item["ids"])
            rank = next((i for i, cid in enumerate(got, 1) if cid in gold), None)
            buckets["all"].add(rank)
            buckets[tag].add(rank)
            if not (rank is not None and rank <= TOP_K):
                misses.append((item["q"], got, item["ids"]))
            rows.append((tag, item["q"], rank, got))
        sec = time.time() - t0
        a = buckets["all"]
        summary.append((name, a, buckets, sec))
        details.append((name, rows, misses))
        print(f"[{name}] HitRate@{TOP_K}={a.hitrate:.1%}  MRR@{TOP_K}={a.mrr:.3f}  "
              f"Top1={a.top1rate:.1%}  用时={sec:.1f}s")
        print(f"   原词题: Top1={buckets['literal'].top1rate:.1%} MRR={buckets['literal'].mrr:.3f}"
              f" | 改述题: Top1={buckets['paraphrase'].top1rate:.1%}"
              f" MRR={buckets['paraphrase'].mrr:.3f}")

    lines = [
        "# 检索质量三档对照",
        "",
        f"- 语料:与线上服务同一份向量库 `chroma_db_multimodal`,{len(content2id)} 块(内容去重);"
        f"其中真实文档 `学位授予细则.pdf` 14 块,其余为干扰文档",
        f"- 标注集:`eval/golden.jsonl` {len(golden)} 条(原词题 {n_lit} + 改述题 {n_par}),"
        f"正确答案为 chunk id,top-{TOP_K} 命中任一即算召回",
        f"- 随机基线:单块正确答案 HitRate@{TOP_K} ≈ {TOP_K / len(content2id):.1%}",
        "- 改述题不含文档原词(如“挂科太多还能拿到学位吗”),用来暴露关键词匹配的短板",
        "",
        f"## 整体(top-{TOP_K})",
        "",
        f"| 检索配置 | HitRate@{TOP_K} | MRR@{TOP_K} | Top1 命中率 | 耗时 |",
        "|---|---|---|---|---|",
    ]
    for name, a, _b, sec in summary:
        lines.append(f"| {name} | {a.hitrate:.1%} | {a.mrr:.3f} | {a.top1rate:.1%} | {sec:.1f}s |")

    lines += ["", "## 按题型分组(Top1 / MRR)", "",
              "| 检索配置 | 原词题 Top1 | 原词题 MRR | 改述题 Top1 | 改述题 MRR |",
              "|---|---|---|---|---|"]
    for name, _a, b, _sec in summary:
        lit, par = b["literal"], b["paraphrase"]
        lines.append(f"| {name} | {lit.top1rate:.1%} | {lit.mrr:.3f} "
                     f"| {par.top1rate:.1%} | {par.mrr:.3f} |")

    for name, rows, misses in details:
        lines += ["", f"## {name} — 逐题明细", "",
                  "| 题型 | 问题 | 正确块排名 | top-3 实际命中 |", "|---|---|---|---|"]
        for tag, q, rank, got in rows:
            if rank != 1:                      # 只列非首位命中的,报告更短更有信息量
                lines.append(f"| {tag} | {q} | {rank if rank else '未命中'} | {', '.join(got)} |")
        if misses:
            lines += ["", f"未命中 {len(misses)} 条:"]
            for q, got, gold in misses:
                lines.append(f"- {q} → 期望 {gold},实际 {got}")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入 {OUT}")


if __name__ == "__main__":
    main()
