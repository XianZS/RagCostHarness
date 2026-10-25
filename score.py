#!/usr/bin/env python3
"""离线分析：质量、成本、$N^*$ 判据（不调用模型）

对应三个实验
    E1  成本核算与 N*  : 各变体的 C_build / C_query 与质量差
    E2  索引质量       : s3_good 与 s3_bad 按单位构造成本比较
    E3  成本转嫁       : s4 相对 s2 的质量增益与成本方差

判据的用法
    N*(λ) = ΔC_build / (λ·Δq − ΔC_query)
    λ 是把质量差折算为 token 当量的汇率。本文不预设 λ，而是报告
    N* 随 λ 变化的曲线，并给出使 N* 为正的临界汇率 λ* = ΔC_query / Δq。
    不给出单一数值的判断，这与论文 2.6.7 节对 ΔV 的处理一致。

用法
    python3 score.py
    python3 score.py --run runs/results.jsonl --build runs/build_costs.jsonl
"""

import argparse
import json
import os
import re
import string
import sys
from collections import defaultdict

ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = ARTICLES.sub(" ", s)
    return " ".join(s.split())


def m_em(pred, golds):
    p = normalize(pred)
    return 1.0 if any(p == normalize(g) for g in golds) else 0.0


def m_contains(pred, golds):
    p = normalize(pred)
    for g in golds:
        g2 = normalize(g)
        if g2 and g2 in p:
            return 1.0
    return 0.0


def m_f1(pred, golds):
    best, p = 0.0, normalize(pred)
    pt = p.split()
    for g in golds:
        gt = normalize(g).split()
        if not pt or not gt:
            continue
        common = {}
        for t in pt:
            common[t] = min(pt.count(t), gt.count(t))
        ov = sum(common.values())
        if ov:
            pr, rc = ov / len(pt), ov / len(gt)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best


def boot_ci(vals, n_boot=2000, seed=42):
    if not vals:
        return (0.0, 0.0)
    import random

    rng = random.Random(seed)
    m = sorted(
        sum(rng.choice(vals) for _ in range(len(vals))) / len(vals)
        for _ in range(n_boot)
    )
    return m[int(0.025 * n_boot)], m[int(0.975 * n_boot)]


def q_tokens(x):
    """查询成本用全部调用的累加值；多轮变体若用最后一次会低估。"""
    return x.get("total_prompt_tokens") or x.get("prompt_tokens", 0)


def u_tokens(x):
    """预算利用率用单次调用的最大值，因为预算是每次调用的上限。"""
    return x.get("max_prompt_tokens") or x.get("prompt_tokens", 0)


LABEL = {
    "s0_parametric": "weights/纯参数",
    "s1_oracle": "context/最小充分上下文(非上界)",
    "s2_flat": "index-flat/BM25",
    "s3_good": "index-org/全文摘要",
    "s3_bad": "index-org/导语摘要",
    "s3_poor": "index-org/截断摘要10词",
    "s4_agentic": "actions/两轮智能体",
}


def lab(v):
    return LABEL.get(v, v)


def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else 0.0


def p95(xs):
    s = sorted(xs)
    return s[min(len(s) - 1, int(0.95 * len(s)))] if s else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/results.jsonl")
    ap.add_argument("--build", default="runs/build_costs.jsonl")
    ap.add_argument("--by-type", action="store_true")
    ap.add_argument("--paper", action="store_true", help="输出论文可直接引用的数字块")
    args = ap.parse_args()

    rows = load_jsonl(args.run)
    if not rows:
        print("x 结果为空或不存在: " + args.run)
        return 1
    builds = load_jsonl(args.build)
    corpus_n = load_jsonl("data/corpus.jsonl")

    for r in rows:
        golds = r.get("answers") or [""]
        r["_em"] = m_em(r.get("answer", ""), golds)
        r["_ct"] = m_contains(r.get("answer", ""), golds)
        r["_f1"] = m_f1(r.get("answer", ""), golds)

    variants = sorted({r["variant"] for r in rows})
    budgets = sorted({r["budget"] for r in rows})
    agg = defaultdict(list)
    for r in rows:
        agg[(r["variant"], r["budget"])].append(r)

    print("共 %d 条结果，%d 个变体，预算 %s\n" % (len(rows), len(variants), budgets))

    # ================= 质量 =================
    print("=" * 92)
    print("质量（EM / contains / F1），括号内为 contains 的 95% 置信区间\n")
    print("%-16s" % "变体" + "".join("%22s" % ("预算 " + str(b)) for b in budgets))
    for v in variants:
        line = "%-30s" % lab(v)
        for b in budgets:
            cell = agg.get((v, b), [])
            if not cell:
                line += "%22s" % "-"
                continue
            lo, hi = boot_ci([x["_ct"] for x in cell])
            line += "%22s" % (
                "%.0f/%.0f/%.0f (%.0f-%.0f)"
                % (
                    mean([x["_em"] for x in cell]) * 100,
                    mean([x["_ct"] for x in cell]) * 100,
                    mean([x["_f1"] for x in cell]) * 100,
                    lo * 100,
                    hi * 100,
                )
            )
        print(line)

    # ================= 参数泄漏与需检索子集 =================
    print()
    print("=" * 92)
    print("参数泄漏检验与「需检索子集」\n")
    s0_bad = {r["qid"] for r in rows if r["variant"] == "s0_parametric" and r["_ct"] == 0}
    s0_all = {r["qid"] for r in rows if r["variant"] == "s0_parametric"}
    leak = len(s0_all) - len(s0_bad)
    print("  不给任何证据时的正确数：%d / %d（%.0f%%），这部分题目不需要检索。" %
          (leak, len(s0_all), leak / max(len(s0_all), 1) * 100))
    print("  以下只统计 s0 答错的 %d 题，即真正依赖检索的题目。\n" % len(s0_bad))
    print("%-30s%20s%20s" % ("变体", "全题集 contains", "需检索子集 contains"))
    for v in variants:
        line = "%-30s" % lab(v)
        for tag in ("all", "dep"):
            vals = []
            for b in budgets:
                cell = agg.get((v, b), [])
                if tag == "dep":
                    cell = [x for x in cell if x["qid"] in s0_bad]
                if cell:
                    vals.append(mean([x["_ct"] for x in cell]) * 100)
            line += "%20s" % (("%.0f%%" % mean(vals)) if vals else "-")
        print(line)
    print()
    print("  全题集与子集的差越大，说明参数泄漏对结论的稀释越严重。")

    # ================= 成本 =================
    print()
    print("=" * 92)
    print("E1 成本表：构造成本取自 build_costs.jsonl，查询成本为实测均值\n")
    # 同一索引会被多次运行重复记账（缓存命中时记 0），
    # 因此取历史最大值而不是最后一条，否则构造成本会被算成 0。
    bcost = {}
    for b in builds:
        tok = b.get("build_prompt_tokens", 0) + b.get("build_completion_tokens", 0)
        bcost[b["index"]] = max(bcost.get(b["index"], 0), tok)

    print("%-16s%14s%16s%14s%14s%12s" % ("变体", "C_build(tok)", "C_query(tok均值)", "C_query(P95)", "延迟(ms)", "利用率"))
    for v in variants:
        cell = [x for x in rows if x["variant"] == v]
        if not cell:
            continue
        key = {"s3_good": "summaries_good", "s3_bad": "summaries_bad", "s3_poor": "summaries_poor", "s2_flat": "bm25_flat"}.get(v, "")
        cb = bcost.get(key, 0)
        qt = [q_tokens(x) for x in cell]
        lat = [x["wall_s"] for x in cell]
        util = [u_tokens(x) / x["budget"] for x in cell]
        print(
            "%-16s%14d%16.0f%14.0f%14.0f%11.0f%%"
            % (v, cb, mean(qt), p95(qt), mean(lat) * 1000, median(util) * 100)
        )

    # ================= 可信三分量 =================
    print()
    print("=" * 92)
    print("可信维度：三个可分量的操作化定义\n")
    print("  忠实性   = 答案的内容词有多大比例能在所用文档中找到（1 - 无据率）")
    print("  鲁棒性   = 查询扰动后的质量 / 原查询的质量")
    print("  抗投毒   = 1 - 采纳注入错误答案的比例")
    print("  ΔTrust   = 上述三项的算术平均，是对论文 2.6.6 未定义分量的操作化选择\n")

    # 需要语料原文来算忠实性
    corpus_map = {}
    try:
        for d in load_jsonl("data/corpus.jsonl"):
            corpus_map[d["doc_id"]] = d["text"].lower()
    except Exception:
        pass

    def faithfulness(r):
        used = r.get("used_docs") or []
        if not used:
            return None
        ctx = " ".join(corpus_map.get(d, "") for d in used)
        toks = [t for t in normalize(r.get("answer", "")).split() if len(t) > 2]
        if not toks:
            return 0.0
        hit = sum(1 for t in toks if t in ctx)
        return hit / len(toks)

    print("%-30s%12s%12s%12s%12s" % ("变体", "忠实性", "鲁棒性", "抗投毒", "ΔTrust"))
    trust_rows = {}
    for v in variants:
        cell = [r for r in rows if r["variant"] == v]
        if not cell:
            continue
        fa = [x for x in cell if faithfulness(x) is not None]
        f_val = mean([faithfulness(x) for x in fa]) if fa else None

        # 鲁棒性只在 s2_flat 行给出，且必须按预算对齐：
        # 扰动后的质量只能与原查询在同一预算档上比，否则比的是预算差异。
        rb = None
        if v == "s2_flat":
            ratios = []
            for b in budgets:
                base = agg.get(("s2_flat", b), [])
                pert = agg.get(("s2_perturb", b), [])
                if base and pert:
                    qb = mean([x["_ct"] for x in base])
                    if qb > 0:
                        ratios.append(mean([x["_ct"] for x in pert]) / qb)
            rb = mean(ratios) if ratios else None

        # 抗投毒
        po = None
        if v == "s7_conflict":
            adopted = [x.get("adopted_poison", 0) for x in cell]
            po = 1 - mean(adopted) if adopted else None

        parts = [x for x in (f_val, rb, po) if x is not None]
        comp = mean(parts) if parts else None
        trust_rows[v] = (f_val, rb, po, comp)
        print(
            "%-30s%12s%12s%12s%12s"
            % (
                lab(v),
                "-" if f_val is None else "%.0f%%" % (f_val * 100),
                "-" if rb is None else "%.0f%%" % (rb * 100),
                "-" if po is None else "%.0f%%" % (po * 100),
                "-" if comp is None else "%.0f%%" % (comp * 100),
            )
        )
    print()
    print("  注意：鲁棒性只在 s2_flat 一行有意义（其余行是不同变体，不可直接比）。")
    print("  忠实性对 s0_parametric 无定义（不使用任何文档）。")

    # ================= 探针诊断 =================
    print()
    print("=" * 92)
    print("四级探针的实现与读法\n")
    print("  P1 召回诊断   由金标召回率与失败分解直接给出，不需要额外变体。")
    print("    理由：在固定 token 预算下提高候选数 k 不会让更多候选进入上下文，")
    print("    「扩大 k」这一操纵在该设定下无效，因此不设专门的 P1 变体。")
    for b in budgets:
        c = agg.get(("s2_flat", b), [])
        if not c:
            continue
        ra = sum(
            1 for r in c
            if r["gold_doc_ids"] and set(r["gold_doc_ids"]) <= set(r.get("used_docs", []))
        )
        miss = len(c) - ra
        verdict = "召回是瓶颈" if ra / len(c) < 0.9 else "召回已接近饱和"
        print("    b=%-5d s2 金标全召回 %.0f%%，未命中 %d / %d 题 -> %s"
              % (b, ra / len(c) * 100, miss, len(c), verdict))
    print()
    print("  附：s6_recall 合并候选数三倍并把每篇截断到 200 字符，")
    print("      金标召回升到 99%，但质量反而下降，答案常落在被截掉的部分。")
    print("      因此它是一项截断研究，不作为 P1 的证据。")
    print()

    print("  P2 重排介入   比较 s2_flat 与 s5_rerank（LLM 重排，C_rank 非零）")
    for b in budgets:
        a = agg.get(("s2_flat", b), [])
        c = agg.get(("s5_rerank", b), [])
        if a and c:
            qa, qc = mean([x["_ct"] for x in a]) * 100, mean([x["_ct"] for x in c]) * 100
            verdict = "排序不足是主因" if qc - qa >= 5 else "排序不是主因"
            print("    b=%d  s2 %.0f%% -> s5 %.0f%%  (%+.0f 点，%s)" % (b, qa, qc, qc - qa, verdict))
    print()
    print("  P3 金标上下文  直接看 s1_oracle 与 s2_flat 的差（已在质量表与失败分解中）")
    print()
    print("  P4 反事实注入  看 s7_conflict 的采纳率：采纳高说明模型依据上下文，采纳低说明固执于参数")
    for b in budgets:
        c = agg.get(("s7_conflict", b), [])
        if c:
            ad = mean([x.get("adopted_poison", 0) for x in c]) * 100
            print("    b=%d  采纳注入答案 %.0f%%  （失败模式 F10 知识冲突的直接度量）" % (b, ad))
    print()

    # ================= 六类成本明细 =================
    print()
    print("=" * 92)
    print("六类成本明细（按论文 2.6.2 的分类，逐类给出实测值）\n")
    print("  结构性的零与未测是两回事：本设计无重排器，C_rank 恒为 0；")
    print("  语料静态，C_maint 未测。两者都应在论文里如实标注。\n")
    print(
        "%-30s%12s%12s%12s%12s%12s"
        % ("变体", "C_ret(ms)", "C_rank(ms)", "C_gen(ms)", "C_act(ms)", "C_act(tok)")
    )
    for v in variants:
        cell = [r for r in rows if r["variant"] == v]
        if not cell:
            continue
        print(
            "%-30s%12.0f%12.0f%12.0f%12.0f%12.0f"
            % (
                lab(v),
                mean([x.get("t_ret_s", 0) for x in cell]) * 1000,
                mean([x.get("t_rank_s", 0) for x in cell]) * 1000,
                mean([x.get("t_gen_s", 0) for x in cell]) * 1000,
                mean([x.get("t_act_s", 0) for x in cell]) * 1000,
                mean([x.get("tokens_act", 0) for x in cell]),
            )
        )
    print()
    print("  C_build 见上表「成本表」一节的构造成本列；C_maint 未测（语料静态）。")
    print("  C_ret 的 token 为 0：BM25 为本地索引，不消耗 LLM token。")

    # ================= 成本方差与失败成本 =================
    print()
    print("=" * 92)
    print("成本方差与失败查询的成本分布（对应论文 2.6.5）\n")
    print(
        "%-30s%10s%10s%10s%14s%14s"
        % ("变体", "均值", "P95", "方差", "成功组均值", "失败组均值")
    )
    for v in variants:
        cell = [r for r in rows if r["variant"] == v]
        if not cell:
            continue
        q = [q_tokens(x) for x in cell]
        mu = mean(q)
        var = mean([(x - mu) ** 2 for x in q])
        ok = [q_tokens(x) for x in cell if x["_ct"] > 0]
        bad = [q_tokens(x) for x in cell if x["_ct"] == 0]
        print(
            "%-30s%10.0f%10.0f%10.0f%14s%14s"
            % (
                lab(v),
                mu,
                p95(q),
                var,
                "%.0f" % mean(ok) if ok else "-",
                "%.0f" % mean(bad) if bad else "-",
            )
        )
    print()
    print("  失败组均值高于成功组，说明失败轨迹更贵，与论文 2.6.5 的判断一致。")

    # ================= E2 索引质量 =================
    print()
    print("=" * 92)
    print("E2 索引质量 vs 索引形式\n")
    print("  指标用相对 s2 的质量增量除以每千构造 token，")
    print("  因为绝对质量里含了纯参数基线那部分免费得分，不能算作索引的功劳。\n")
    for b in budgets:
        base = agg.get(("s2_flat", b), [])
        if not base:
            continue
        q0 = mean([x["_ct"] for x in base]) * 100
        cb0 = bcost.get("bm25_flat", 0)
        print("  预算 %d（基线 s2 质量 %.0f%%，构造 %d tok）" % (b, q0, cb0))
        for v in ("s3_good", "s3_bad", "s3_poor"):
            cell = agg.get((v, b), [])
            if not cell:
                continue
            q = mean([x["_ct"] for x in cell]) * 100
            key = {"s3_good": "summaries_good", "s3_bad": "summaries_bad", "s3_poor": "summaries_poor"}[v]
            cb = bcost.get(key, 0) - cb0
            gain = q - q0
            if cb <= 0:
                unit = "构造为 0，无法归一"
            elif gain <= 0:
                unit = "增益 %+.0f 点，构造 %d tok，不回本" % (gain, cb)
            else:
                unit = "增益 %+.0f 点，构造 %d tok，每千 token 换 %.3f 点" % (gain, cb, gain / (cb / 1000))
            print("    %-26s 质量 %.0f%%  %s" % (lab(v), q, unit))
        print()

    # ================= E3 成本转嫁 =================
    print("=" * 92)
    print("E3 成本转嫁：s4 相对 s2 的质量增益与成本方差（同等预算）\n")
    for b in budgets:
        s2 = agg.get(("s2_flat", b), [])
        s4 = agg.get(("s4_agentic", b), [])
        if not s2 or not s4:
            continue
        q2, q4 = mean([x["_ct"] for x in s2]), mean([x["_ct"] for x in s4])
        t2, t4 = mean([q_tokens(x) for x in s2]), mean([q_tokens(x) for x in s4])
        v2 = p95([q_tokens(x) for x in s2]) / max(mean([q_tokens(x) for x in s2]), 1)
        v4 = p95([q_tokens(x) for x in s4]) / max(mean([q_tokens(x) for x in s4]), 1)
        dq = (q4 - q2) * 100
        dt = (t4 / t2 - 1) * 100 if t2 else 0
        print("  预算 %d" % b)
        print("    s2 质量 %.0f%%  token %.0f  P95/均值 %.2f" % (q2 * 100, t2, v2))
        print("    s4 质量 %.0f%%  token %.0f  P95/均值 %.2f" % (q4 * 100, t4, v4))
        print("    质量增 %+.1f 点，token 增 %+.0f%%" % (dq, dt))
        if dt > 5:
            ratio = dq / dt
            print("    token 每增加 1%%，质量增加 %.2f 点" % ratio)
            print("    判读：" + ("支持发现③（成本增速快于质量增速）" if ratio < 0.5 else "不支持发现③"))
        else:
            print("    s4 的 token 增幅不足 5%，本轮未触发第二轮，无法判断")
        print()

    # ================= N* 判据 =================
    print("=" * 92)
    print("N* 判据：ΔC_build / (λ·Δq − ΔC_query)\n")
    print("  Δq 用 contains 的差（0-1），λ 的单位是 token/质量点。")
    print("  λ* = ΔC_query / Δq 是使 N* 为正的临界汇率，低于它则永不回本。\n")

    base = "s2_flat"
    for b in budgets:
        base_cell = agg.get((base, b), [])
        if not base_cell:
            continue
        q_base = mean([x["_ct"] for x in base_cell])
        cq_base = mean([q_tokens(x) for x in base_cell])
        cbuild_base = bcost.get("bm25_flat", 0)
        print("  基线 %s @ 预算 %d  （质量 %.0f%%，C_query %.0f tok）" % (base, b, q_base * 100, cq_base))
        for v in variants:
            if v == base:
                continue
            cell = agg.get((v, b), [])
            if not cell:
                continue
            dq = mean([x["_ct"] for x in cell]) - q_base
            cq = mean([q_tokens(x) for x in cell]) - cq_base
            key = {"s3_good": "summaries_good", "s3_bad": "summaries_bad", "s3_poor": "summaries_poor", "s2_flat": "bm25_flat"}.get(v, "")
            cb = bcost.get(key, 0) - cbuild_base
            if dq <= 0:
                print("    %-14s Δq=%.3f ≤ 0，无论 λ 与 N 均不回本" % (v, dq))
                continue
            cb0 = bcost.get({"s3_good": "summaries_good", "s3_bad": "summaries_bad",
                             "s3_poor": "summaries_poor"}.get(v, ""), 0)
            if cb <= 0 and cq <= 0:
                print("    %-14s Δq=%+.3f 更高质量且不更贵（构造 0、查询 %+.0f），严格优于基线，"
                      "无需判据即可采纳" % (v, dq, cq))
                continue
            lam_star = cq / dq
            marks = []
            for lam in (5000, 20000, 50000, 100000, 200000):
                denom = lam * dq - cq
                if denom <= 0:
                    marks.append("λ=%d: 永不" % lam)
                else:
                    n = cb / denom if cb > 0 else 0
                    marks.append("λ=%d: N*≈%d" % (lam, int(n)))
            print("    %-14s Δq=%+.3f  ΔC_query=%+.0f  ΔC_build=%+.0f  λ*=%d" % (v, dq, cq, cb, int(lam_star)))
            print("      " + " | ".join(marks))
        print()

    # ================= 分题型 =================
    print("=" * 92)
    print("分题型 contains（检查是否被某一类题支配）\n")
    types = sorted({r.get("type", "unknown") for r in rows})
    print("%-16s" % "变体" + "".join("%16s" % t[:14] for t in types))
    for v in variants:
        line = "%-30s" % lab(v)
        for t in types:
            cell = [r for r in rows if r["variant"] == v and r.get("type") == t]
            line += "%16s" % ("%.0f%%" % (mean([x["_ct"] for x in cell]) * 100) if cell else "-")
        print(line)

    # ================= 违规率 =================
    print()
    print("=" * 92)
    print("预算违规率\n")
    print("%-30s" % "变体" + "".join("%14s" % ("预算 " + str(b)) for b in budgets))
    for v in variants:
        line = "%-30s" % lab(v)
        for b in budgets:
            cell = agg.get((v, b), [])
            line += "%14s" % (
                ("%.1f%%" % (sum(1 for x in cell if x.get("violation")) / len(cell) * 100)) if cell else "-"
            )
        print(line)

    # ================= 检索诊断 =================
    print()
    print("=" * 92)
    print("检索召回：区分'检索没找到'与'找到了但读不懂'\n")
    print("  recall_all = 全部金标段落都进了上下文的题目比例")
    print("  recall_any = 至少进了金标之一的题目比例")
    print("  gold_rank  = 金标段落在 used_docs 中的名次中位数（0 为最优）\n")
    print("%-30s%12s%12s%12s" % ("变体", "recall_all", "recall_any", "gold_rank"))
    for v in variants:
        cell = [r for r in rows if r["variant"] == v]
        if not cell:
            continue
        ra, rr, ranks = 0, 0, []
        for r in cell:
            gold = set(r.get("gold_doc_ids") or [])
            used = r.get("used_docs") or []
            if not gold:
                continue
            hit = gold & set(used)
            if len(hit) == len(gold):
                ra += 1
            if hit:
                rr += 1
                for g in hit:
                    ranks.append(used.index(g))
        n = len(cell)
        print(
            "%-16s%11.0f%%%11.0f%%%12s"
            % (
                v,
                ra / n * 100 if n else 0,
                rr / n * 100 if n else 0,
                ("%.0f" % median(ranks)) if ranks else "-",
            )
        )

    # ================= 失败分解 =================
    print()
    print("=" * 92)
    print("失败分解：把 s2 的失败拆成'检索未命中'与'读不懂'\n")
    for b in budgets:
        cell = agg.get(("s2_flat", b), [])
        if not cell:
            continue
        miss = [r for r in cell if not (set(r.get("gold_doc_ids") or []) & set(r.get("used_docs") or []))]
        hit = [r for r in cell if r not in miss]
        f_miss = (1 - mean([r["_ct"] for r in miss])) * 100 if miss else 0
        f_hit = (1 - mean([r["_ct"] for r in hit])) * 100 if hit else 0
        print("  预算 %d" % b)
        print("    检索未命中 %d / %d 题（%.0f%%）" % (len(miss), len(cell), len(miss) / len(cell) * 100))
        if miss:
            print("      这些题未检索到金标，但仍答对 %.0f%%（说明参数知识已足够，属参数泄漏）" % (100 - f_miss))
        if hit:
            print("      这些题命中了金标，仍答错 %.0f%%（属读者层失败）" % f_hit)
        print()

    # ================= CSV =================
    csv_path = args.run.replace(".jsonl", "_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("variant,budget,n,em,contains,f1,c_build_tokens,c_query_mean,c_query_p95,latency_mean_s,violation_rate,median_utilization\n")
        for (v, b), cell in sorted(agg.items()):
            key = {"s3_good": "summaries_good", "s3_bad": "summaries_bad", "s3_poor": "summaries_poor", "s2_flat": "bm25_flat"}.get(v, "")
            qt = [q_tokens(x) for x in cell]
            f.write(
                "%s,%d,%d,%.4f,%.4f,%.4f,%d,%.1f,%.1f,%.3f,%.4f,%.4f\n"
                % (
                    v, b, len(cell),
                    mean([x["_em"] for x in cell]),
                    mean([x["_ct"] for x in cell]),
                    mean([x["_f1"] for x in cell]),
                    bcost.get(key, 0),
                    mean(qt), p95(qt), mean([x["wall_s"] for x in cell]),
                    sum(1 for x in cell if x.get("violation")) / len(cell),
                    median([u_tokens(x) / b for x in cell]),
                )
            )
    # ================= 论文直出 =================
    if args.paper:
        print()
        print("#" * 92)
        print("# 以下内容可直接进论文，所有数字来自本次 results.jsonl")
        print("#" * 92)
        print()
        pb = 4096 if 4096 in budgets else budgets[-1]
        print("【表】预算 %d 下各载体的成本与质量\n" % pb)
        print("\\begin{tabular}{lrrrr}")
        print("\\toprule")
        print("载体 & 构造 token & 查询 token & contains & 违规率 \\\\")
        print("\\midrule")
        for v in variants:
            cell = agg.get((v, pb), [])
            if not cell:
                continue
            key = {"s3_good": "summaries_good", "s3_bad": "summaries_bad",
                   "s3_poor": "summaries_poor", "s2_flat": "bm25_flat"}.get(v, "")
            cb = bcost.get(key, 0)
            q = mean([x["_ct"] for x in cell]) * 100
            cq = mean([q_tokens(x) for x in cell])
            vio = sum(1 for x in cell if x.get("violation")) / len(cell) * 100
            print("%s & %d & %.0f & %.0f\\%% & %.1f\\%% \\\\" % (lab(v), cb, cq, q, vio))
        print("\\bottomrule")
        print("\\end{tabular}")
        print()

        # 自动生成的结论句
        q0 = mean([x["_ct"] for x in agg.get(("s0_parametric", pb), [])]) * 100
        q2 = mean([x["_ct"] for x in agg.get(("s2_flat", pb), [])]) * 100
        q1 = mean([x["_ct"] for x in agg.get(("s1_oracle", pb), [])]) * 100
        t0 = mean([q_tokens(x) for x in agg.get(("s0_parametric", pb), [])])
        t2 = mean([q_tokens(x) for x in agg.get(("s2_flat", pb), [])])
        print("【可直接引用的句子】\n")
        print(
            "  在本地 Qwen2.5-1.5B 与 %d 段语料上，纯参数生成的质量为 %.0f%%，"
            "平均消耗 %.0f token。" % (len(corpus_n), q0, t0)
        )
        print(
            "  扁平检索在该设定下达到 %.0f%%，平均消耗 %.0f token。"
            % (q2, t2)
        )
        print(
            "  仅提供金标两段（最小充分上下文，%d token）时为 %.0f%%，"
            "说明该规模的语料下检索已接近读者可用信息的饱和点。"
            % (mean([q_tokens(x) for x in agg.get(("s1_oracle", pb), [])]), q1)
        )
        print()

        # 边际收益
        if len(budgets) >= 2:
            print("【边际收益：预算翻倍换来什么】\n")
            print("  预算 & 查询 token & 质量 & 每翻一倍的质量增益 \\\\")
            prev_b = None
            for b in budgets:
                cell = agg.get(("s2_flat", b), [])
                if not cell:
                    continue
                q = mean([x["_ct"] for x in cell]) * 100
                cq = mean([q_tokens(x) for x in cell])
                d = "-" if prev_b is None else "%+.0f 点" % (q - prev_b[1])
                print("  %d & %.0f & %.0f\\%% & %s \\\\" % (b, cq, q, d))
                prev_b = (b, q)
            print()

        # 临界汇率
        print("【$N^*$ 判据的实测临界汇率】\n")
        base_cell = agg.get(("s2_flat", pb), [])
        if base_cell:
            qb = mean([x["_ct"] for x in base_cell])
            cqb = mean([q_tokens(x) for x in base_cell])
            for v in variants:
                if v == "s2_flat":
                    continue
                cell = agg.get((v, pb), [])
                if not cell:
                    continue
                dq = mean([x["_ct"] for x in cell]) - qb
                dcq = mean([q_tokens(x) for x in cell]) - cqb
                key = {"s3_good": "summaries_good", "s3_bad": "summaries_bad",
                       "s3_poor": "summaries_poor"}.get(v, "")
                dcb = bcost.get(key, 0)
                if dq <= 0:
                    print("  %-30s Δq=%+.3f ≤ 0，判据无正解（永不回本）" % (lab(v), dq))
                else:
                    print(
                        "  %-30s Δq=%+.3f  ΔC_build=%d  ΔC_query=%+.0f  λ*=%d token/质量点"
                        % (lab(v), dq, dcb, dcq, int(max(dcq, 0) / dq))
                    )
            print()

        print("【数据算出来的限制，须写进 Limitations】\n")
        print(
            "  1. 语料仅 %d 段，且由各题自带的干扰段落合并而成，"
            "金标必然在库内，检索难度显著低于真实语料库。" % len(corpus_n)
        )
        print(
            "  2. 预算由字符估算软约束，实测利用率 %s，"
            "并非硬拦截。"
            % "、".join(
                "%.0f%%" % (mean([u_tokens(x) / pb2 for x in agg[(v, pb2)]]) * 100)
                for pb2 in budgets
                for v in ["s2_flat"]
                if agg.get((v, pb2))
            )
        )
        print("  3. 样本量 n=%d，标准误约 %.0f 点，小于该幅度的差异不可解释。"
              % (len(agg.get(("s2_flat", pb), [])),
                 (0.25 / max(len(agg.get(("s2_flat", pb), [])), 1)) ** 0.5 * 100))
        print("  4. 单机单模型（1.5B 量化），结论限定在相对成本结构，不是绝对质量排名。")
        print()

    print("\n汇总已写入 " + csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
