#!/usr/bin/env python3
"""补测 $C_{\\text{maint}}$：索引与语料的同步成本

论文 2.6.2 把新鲜度成本列为第三类，并指出它在文献中几乎从未被度量。
本脚本实测它，产出可直接进论文的数字。

测什么
    一次语料更新事件（新增 N 篇 + 修改 M 篇）之后，重建索引的代价：
        扁平索引（BM25）  全量重建，几乎为零
        组织式索引        全量重建 vs 增量重建
    并据此推算运维成本随更新频率的增长。

怎么构造更新事件
    新增：从 SQuAD 再取若干条（与已有语料不重叠）
    修改：对现有若干篇追加一句，使摘要必须重新生成
    两者都不会写回 data/，只在内存中构造，保证主实验语料不变。

产出
    runs/maint.jsonl   每次更新的实测记录
    终端输出           增量成本、全量成本的推算、与构造成本的关系

用法
    python3 exp_maint.py --dry-run
    python3 exp_maint.py
    python3 exp_maint.py --n-new 20 --n-mod 10 --budget 4096
"""

import argparse
import json
import os
import sys
import time

import run_experiment as RX

OUT = os.path.join("runs", "maint.jsonl")
RESULTS = os.path.join("runs", "results.jsonl")


def guard(path):
    if os.path.abspath(path) == os.path.abspath(RESULTS):
        raise SystemExit("拒绝写入 " + path)
    return path


def fetch_new_docs(n):
    """从 SQuAD 取与已有语料不同的段落。失败时返回空列表。"""
    try:
        from prepare_data import fetch_rows
    except Exception as e:
        print("  取不到 prepare_data.fetch_rows: %s" % e)
        return []
    try:
        payload = fetch_rows("rajpurkar/squad", "plain_text", "validation", 30, n)
    except Exception as e:
        print("  下载新段落失败: %s" % e)
        return []
    out = []
    for item in payload.get("rows", []):
        r = item.get("row", {})
        title = r.get("title") or "new"
        text = (r.get("context") or "").strip()
        if text:
            out.append({"doc_id": "new_" + title + "_" + str(len(out)), "text": text})
    return out


def summarize_one(model, text):
    p = (
        "Summarize the passage in about 40 words, keeping entity names.\n\n"
        + text
        + "\n\nSummary:"
    )
    ans, m = RX.gen(model, p, 96, 2048)
    return ans, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:1.5b-instruct")
    ap.add_argument("--n-new", type=int, default=20)
    ap.add_argument("--n-mod", type=int, default=10)
    ap.add_argument("--events", type=int, default=2, help="重复几次更新事件")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    corpus = RX.load_jsonl(os.path.join("data", "corpus.jsonl"))
    if not corpus:
        raise SystemExit("找不到 data/corpus.jsonl，先运行 prepare_data.py")
    corpus_map = {d["doc_id"]: d["text"] for d in corpus}
    print("语料 %d 段" % len(corpus))

    new_docs = fetch_new_docs(args.n_new)
    print("新增文档 %d 篇（来自 SQuAD 第 30 条之后）" % len(new_docs))
    if not new_docs:
        print("x 无法构造更新事件，退出")
        return 1

    # 修改：对现有文档追加一句，摘要必须重新生成
    modified = []
    for d in corpus[: args.n_mod]:
        modified.append({"doc_id": d["doc_id"], "text": d["text"] + " This entry was revised on 2026-10-24."})
    print("修改文档 %d 篇" % len(modified))

    touched = new_docs + modified
    print("每次更新事件涉及 %d 篇，需重新生成摘要 %d 次\n" % (len(touched), len(touched)))

    if args.dry_run:
        # 只测 BM25 重建，不调模型
        t0 = time.perf_counter()
        RX.BM25(corpus_map)
        flat_full = time.perf_counter() - t0
        merged = dict(corpus_map)
        for d in touched:
            merged[d["doc_id"]] = d["text"]
        t0 = time.perf_counter()
        RX.BM25(merged)
        flat_inc = time.perf_counter() - t0
        print("BM25 全量重建 %.3fs，含更新后重建 %.3fs" % (flat_full, flat_inc))
        print("dry-run 通过：扁平索引的运维成本可忽略")
        return 0

    if not RX.check_server():
        print("x 连不上 Ollama。先运行：  ollama serve")
        return 1
    if not RX.has_model(args.model):
        print("x 本机没有模型 " + args.model)
        return 1

    guard(OUT)
    os.makedirs("runs", exist_ok=True)

    # 扁平索引的重建成本（不需模型）
    t0 = time.perf_counter()
    RX.BM25(corpus_map)
    flat_full_ms = (time.perf_counter() - t0) * 1000

    records = []
    for ev in range(1, args.events + 1):
        print("第 %d 次更新事件" % ev)
        tok_p = tok_c = 0
        wall = 0.0
        t0 = time.perf_counter()
        for n, d in enumerate(touched, 1):
            try:
                s, m = summarize_one(args.model, d["text"])
            except Exception as e:
                print("  ! %s 摘要失败: %s" % (d["doc_id"], e))
                continue
            tok_p += m["prompt_tokens"]
            tok_c += m["completion_tokens"]
            wall += m["wall_s"]
            if n % 10 == 0 or n == len(touched):
                print("    摘要 %d/%d" % (n, len(touched)))
        total_wall = time.perf_counter() - t0

        merged = dict(corpus_map)
        for d in touched:
            merged[d["doc_id"]] = d["text"]
        t0 = time.perf_counter()
        RX.BM25(merged)
        flat_inc_ms = (time.perf_counter() - t0) * 1000

        rec = {
            "event": ev,
            "n_new": len(new_docs),
            "n_modified": len(modified),
            "n_touched": len(touched),
            "maint_prompt_tokens": tok_p,
            "maint_completion_tokens": tok_c,
            "maint_wallclock_s": round(total_wall, 3),
            "flat_rebuild_ms": round(flat_inc_ms, 3),
        }
        records.append(rec)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("    增量重建：prompt %d + completion %d tokens，墙钟 %.1fs，BM25 重建 %.2fms\n"
              % (tok_p, tok_c, total_wall, flat_inc_ms))

    # ---------------- 报告 ----------------
    builds = RX.load_jsonl(os.path.join("runs", "build_costs.jsonl"))
    build_tok = {}
    for b in builds:
        t = b.get("build_prompt_tokens", 0) + b.get("build_completion_tokens", 0)
        build_tok[b["index"]] = max(build_tok.get(b["index"], 0), t)

    n_docs = len(corpus)
    per_event = sum(r["maint_prompt_tokens"] + r["maint_completion_tokens"] for r in records) / len(records)
    per_doc = per_event / max(records[0]["n_touched"], 1)
    full_good = build_tok.get("summaries_good", 0)
    full_bad = build_tok.get("summaries_bad", 0)

    print("=" * 84)
    print("$C_{\\text{maint}}$ 实测结果\n")
    print("  一次更新事件（新增 %d 篇 + 修改 %d 篇）" % (records[0]["n_new"], records[0]["n_modified"]))
    print("    组织式索引增量重建：%.0f token，%.1f 秒" % (per_event, records[0]["maint_wallclock_s"]))
    print("    每篇平均：%.0f token" % per_doc)
    print("    扁平索引（BM25）重建：%.2f ms，0 token" % (records[0]["flat_rebuild_ms"]))
    print()
    print("  与构造成本相比")
    print("    组织式索引全量构造：%d token（%d 篇，每篇约 %.0f token）"
          % (full_good, n_docs, full_good / max(n_docs, 1)))
    print("    增量重建是全量构造的 %.1f%%（%d 篇 / %d 篇）"
          % (per_event / full_good * 100 if full_good else 0, records[0]["n_touched"], n_docs))
    print()
    print("  按更新频率推算的年运维成本（以组织式索引计）\n")
    print("    %-12s%14s%18s%16s" % ("更新频率", "年更新次数", "年 C_maint (token)", "相对 C_build"))
    for name, times in (("每日", 365), ("每周", 52), ("每月", 12), ("每季", 4)):
        annual = per_event * times
        print("    %-12s%14d%18.0f%15.1f 倍"
              % (name, times, annual, annual / full_good if full_good else 0))
    print()
    print("  注意：这里的更新事件按 %d 篇计。若语料整体换代，增量即退化为全量。" % records[0]["n_touched"])
    print("  扁平索引的运维成本在任何频率下都可忽略（%.2f ms/次）。" % records[0]["flat_rebuild_ms"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
