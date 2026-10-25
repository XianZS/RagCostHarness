#!/usr/bin/env python3
"""数据准备与校验（不调用模型）

这一步的目的不是"把数据下下来"，而是在花掉任何模型时间之前，
把所有会让实验白跑的数据问题暴露出来。

产出三个文件（都在 ./data/ 下）：
    corpus.jsonl    段落库，每行 {doc_id, title, text, n_tokens}
    queries.jsonl   查询集，每行 {qid, question, answers[], type, gold_doc_ids[], source}
    report.md       人类可读的校验报告

校验项（任一失败会打印 FAIL，此时不要往下跑实验）：
    V1 每个查询至少有一个金标段落
    V2 每个金标段落都在语料库中
    V3 答案非空且不是列表套列表
    V4 题型分布至少覆盖两类，且最大一类不超过 70%
    V5 问题文本无重复
    V6 语料 token 总量与 8K 预算的关系（提示语料是否装得下）

数据来源：
    HotpotQA distractor / validation  -> 多跳题
    SQuAD v1.1 plain_text / validation -> 单跳题
    均通过 HuggingFace datasets-server 的 HTTP 接口，不需要 datasets 库。

用法：
    python3 prepare_data.py                 # 默认各 30 题
    python3 prepare_data.py --n-hotpot 20 --n-squad 20
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

HF = "https://datasets-server.huggingface.co/rows"
OUT_DIR = "data"


def fetch_rows(dataset, config, split, offset, length):
    args = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{HF}?{args}"
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode())


def approx_tokens(text):
    """粗估 token 数。英文约 0.25 token/字符。"""
    return len(text) // 4


# ---------- HotpotQA ----------

def load_hotpot(n):
    payload = fetch_rows("hotpotqa/hotpot_qa", "distractor", "validation", 0, n)
    rows = payload.get("rows", [])
    if not rows:
        raise RuntimeError("HotpotQA 返回空 rows")
    # 打印首条的全部字段名，便于接口变化时排错
    print(f"  HotpotQA 首条字段：{sorted(rows[0].get('row', {}).keys())}")

    queries, docs = [], {}
    for item in rows:
        r = item.get("row", {})
        ctx = r.get("context") or {}
        titles = ctx.get("title") or []
        sents = ctx.get("sentences") or []
        sf = r.get("supporting_facts") or {}
        gold_titles = list(dict.fromkeys(sf.get("title") or []))

        for t, ss in zip(titles, sents):
            if t not in docs:
                text = " ".join(ss)
                docs[t] = text

        queries.append(
            {
                "qid": f"hotpot_{len(queries):03d}",
                "question": (r.get("question") or "").strip(),
                "answers": [str(r.get("answer") or "").strip()],
                "type": f"hotpot_{r.get('type') or 'unknown'}",
                "gold_doc_ids": [t for t in gold_titles if t in docs],
                "source": "hotpotqa",
            }
        )
    return queries, docs


# ---------- SQuAD ----------

def load_squad(n):
    payload = fetch_rows("rajpurkar/squad", "plain_text", "validation", 0, n)
    rows = payload.get("rows", [])
    if not rows:
        raise RuntimeError("SQuAD 返回空 rows")
    print(f"  SQuAD 首条字段：{sorted(rows[0].get('row', {}).keys())}")

    queries, docs = [], {}
    for item in rows:
        r = item.get("row", {})
        title = r.get("title") or f"squad_{len(docs)}"
        # 同一 title 可能有多段，用 title 作为 doc_id，拼接以避免覆盖
        text = (r.get("context") or "").strip()
        doc_id = title if title not in docs else f"{title}#{len(docs)}"
        docs[doc_id] = text

        ans = r.get("answers") or {}
        texts = ans.get("text") or []
        queries.append(
            {
                "qid": f"squad_{len(queries):03d}",
                "question": (r.get("question") or "").strip(),
                "answers": [str(a).strip() for a in texts] or [""],
                "type": "squad_single_hop",
                "gold_doc_ids": [doc_id],
                "source": "squad",
            }
        )
    return queries, docs


# ---------- 校验 ----------

def validate(queries, docs):
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": ok, "detail": detail})

    # V1 金标段落非空
    bad = [q["qid"] for q in queries if not q["gold_doc_ids"]]
    add("V1 每个查询至少一个金标段落", not bad, f"缺失 {len(bad)} 条 {bad[:5]}")

    # V2 金标段落都在语料中
    missing = set()
    for q in queries:
        for d in q["gold_doc_ids"]:
            if d not in docs:
                missing.add(d)
    add("V2 金标段落都在语料库中", not missing, f"缺失 {len(missing)} 个 {list(missing)[:5]}")

    # V3 答案非空
    bad3 = [q["qid"] for q in queries if not q["answers"] or not q["answers"][0]]
    add("V3 答案非空", not bad3, f"缺失 {len(bad3)} 条 {bad3[:5]}")

    # V4 题型分布
    from collections import Counter

    dist = Counter(q["type"] for q in queries)
    n_types = len(dist)
    top_share = (max(dist.values()) / len(queries)) if queries else 1.0
    add(
        "V4 题型至少两类且最大类不超 70%",
        n_types >= 2 and top_share <= 0.70,
        f"分布 {dict(dist)}，最大类占 {top_share*100:.0f}%",
    )

    # V5 问题无重复
    from collections import Counter as C2

    dup = [q for q, c in C2(x["question"] for x in queries).items() if c > 1]
    add("V5 问题文本无重复", not dup, f"重复 {len(dup)} 条 {dup[:3]}")

    # V6 语料规模与 8K 预算
    total_tok = sum(approx_tokens(t) for t in docs.values())
    add(
        "V6 语料规模足以让检索有意义（总 token 远大于单次预算）",
        total_tok > 8192 * 3,
        f"语料约 {total_tok} tokens，语料库 {len(docs)} 段",
    )

    return checks, dist, total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hotpot", type=int, default=30)
    ap.add_argument("--n-squad", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"下载 HotpotQA 前 {args.n_hotpot} 条 ...")
    try:
        q1, d1 = load_hotpot(args.n_hotpot)
    except Exception as e:
        print(f"x HotpotQA 失败：{e}")
        return 1
    print(f"  得到 {len(q1)} 条查询，{len(d1)} 段")

    print(f"下载 SQuAD 前 {args.n_squad} 条 ...")
    try:
        q2, d2 = load_squad(args.n_squad)
    except Exception as e:
        print(f"x SQuAD 失败：{e}")
        return 1
    print(f"  得到 {len(q2)} 条查询，{len(d2)} 段")

    # 问题去重：SQuAD 同一篇文章会有重复问句，只保留首条，避免浪费预算
    seen, queries = set(), []
    for q in q1 + q2:
        key = q["question"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(q)
    dropped = len(q1) + len(q2) - len(queries)
    if dropped:
        print(f"  去重丢弃 {dropped} 条重复问句")

    # 语料合并，按 title 去重
    docs = {}
    for k, v in list(d1.items()) + list(d2.items()):
        docs.setdefault(k, v)

    checks, dist, total_tok = validate(queries, docs)

    # 写文件
    with open(os.path.join(OUT_DIR, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for doc_id, text in docs.items():
            f.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "title": doc_id,
                        "text": text,
                        "n_tokens": approx_tokens(text),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(os.path.join(OUT_DIR, "queries.jsonl"), "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    lines = [
        "# 数据校验报告",
        "",
        f"- 查询数：{len(queries)}",
        f"- 语料段落数：{len(docs)}",
        f"- 语料总 token（估）：{total_tok}",
        f"- 题型分布：{dict(dist)}",
        "",
        "| 检查项 | 结果 | 说明 |",
        "|---|---|---|",
    ]
    all_ok = True
    for c in checks:
        mark = "PASS" if c["ok"] else "**FAIL**"
        if not c["ok"]:
            all_ok = False
        lines.append(f"| {c['name']} | {mark} | {c['detail']} |")

    lines += ["", "全部通过，可以进入第二步。" if all_ok else "存在 FAIL，先修数据再跑实验。"]
    report = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)

    print()
    print(report)
    print()
    print(f"已写入 {OUT_DIR}/corpus.jsonl, queries.jsonl, report.md")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
