#!/usr/bin/env python3
"""补充实验：判定器一致性（发现④、§6.2 L2/L3）与位置效应（§4.1.1）

两个实验都只读 runs/results.jsonl，写入各自的新文件，绝不改动已有结果。
脚本开头有硬性检查：任何输出路径若与 results.jsonl 同名，直接退出。

实验一  judge   判定器一致性
    取已有的模型输出（不重新生成），用两套口径打分：
        字符串匹配 contains（已有）
        判定器 A  「候选答案是否表达了与参考答案相同的事实」
        判定器 B  「参考答案 X，候选 Y，正确填 1，否则 0」
    报告三件事：
        1. A 与 B 的一致率与 Cohen's kappa        -> §6.2 L3 判定器非中立
        2. 三种口径下各变体的得分与排名是否翻转  -> §6.2 L2 内部不可比
        3. 翻转的具体例子                        -> 发现④

实验二  position  位置效应
    固定证据集合（把金标段落强制纳入），只改它在上下文中的位置：
        首部 / 中部 / 尾部
    报告三个位置的质量与召回，用于检验 §4.1.1 所述的位置效应。
    注意：证据集合按预算的 80% 裁剪，保证三种排列都不会被预算截断，
    否则测到的就是截断而不是位置。

用法
    python3 exp_extra.py --dry-run
    python3 exp_extra.py --exp judge
    python3 exp_extra.py --exp position
    python3 exp_extra.py --exp all
"""

import argparse
import json
import os
import re
import statistics
import string
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

import run_experiment as RX

RUNS = "runs"
RESULTS = os.path.join(RUNS, "results.jsonl")
JUDGE_OUT = os.path.join(RUNS, "judge.jsonl")
POS_OUT = os.path.join(RUNS, "position.jsonl")
RETEST_OUT = os.path.join(RUNS, "retest.jsonl")

JUDGE_A = (
    "Does the candidate answer convey the same fact as the reference answer? "
    "Reply with exactly one word, YES or NO.\n\n"
    "Reference: {gold}\nCandidate: {cand}\nAnswer:"
)
JUDGE_B = (
    "Reference answer: {gold}\nCandidate answer: {cand}\n"
    "Is the candidate correct? Reply only 1 for correct, 0 for incorrect.\n"
)
POSITIONS = ("start", "middle", "end")

#  faithfulness 判定：给上下文，问答案是否有依据。
#  这是真实 RAG 评测里判定器的典型用法，比"答案是否等于金标"难得多，
#  更可能暴露判定器的不稳定。
JUDGE_FAITH = (
    "You are grading whether an answer is supported by the evidence.\n\n"
    "Evidence:\n{ctx}\n\nQuestion: {q}\nAnswer: {cand}\n\n"
    "Is every claim in the answer supported by the evidence? "
    "Reply with exactly one word, YES or NO.\nVerdict:"
)


# ---------------- 通用 ----------------

def guard(path):
    """硬性防止覆盖已有结果文件。"""
    if os.path.abspath(path) == os.path.abspath(RESULTS):
        raise SystemExit("拒绝写入 " + path + "：那是主实验的原始结果，不可覆盖。")
    return path


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


ART = re.compile(r"\b(a|an|the)\b")


def normalize(s):
    s = str(s).lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = ART.sub(" ", s)
    return " ".join(s.split())


def contains(pred, golds):
    p = normalize(pred)
    return 1.0 if any(normalize(g) and normalize(g) in p for g in golds) else 0.0


def kappa(a, b):
    """Cohen's kappa，两个判定器的一致性中扣除偶然一致。"""
    n = len(a)
    if n == 0:
        return 0.0, 0.0
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return agree, ((agree - pe) / (1 - pe)) if pe < 1 else 0.0



def gen_temp(model, prompt, num_predict, num_ctx, temp, seed=None):
    """本地实现，带温度。不改 run_experiment.gen，避免影响主实验的可复现性。"""
    data = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temp,
            "top_p": 0.95,
            **({"seed": seed} if seed is not None else {}),
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read().decode())
    return out.get("response", "").strip()


# ---------------- 实验一：判定器 ----------------

def run_judge(args):
    rows = load_jsonl(RESULTS)
    if not rows:
        raise SystemExit("找不到 " + RESULTS + "，先跑主实验")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    picked = [
        r for r in rows
        if r["variant"] in variants and r["budget"] == args.budget
    ]
    if not picked:
        raise SystemExit("在预算 %d 下没有这些变体的记录：%s" % (args.budget, variants))

    guard(JUDGE_OUT)
    os.makedirs(RUNS, exist_ok=True)
    done = {(r["qid"], r["variant"]) for r in load_jsonl(JUDGE_OUT)}
    todo = [r for r in picked if (r["qid"], r["variant"]) not in done]

    print("实验一 判定器一致性")
    print("  变体 %s   预算 %d   共 %d 条，已完成 %d 条，待判 %d 条"
          % (variants, args.budget, len(picked), len(picked) - len(todo), len(todo)))
    print("  预计调用 %d 次（每条两个判定器）\n" % (len(todo) * 2))

    if args.dry_run:
        print("dry-run 通过：不会改动 %s" % RESULTS)
        return 0

    num_ctx = 2048
    t0 = time.perf_counter()
    with open(JUDGE_OUT, "a", encoding="utf-8") as f:
        for n, r in enumerate(todo, 1):
            gold = (r.get("answers") or [""])[0]
            cand = r.get("answer", "")
            rec = {
                "qid": r["qid"],
                "variant": r["variant"],
                "budget": r["budget"],
                "question": r.get("question", ""),
                "gold": gold,
                "candidate": cand,
                "string_match": contains(cand, [gold]),
            }
            for tag, tmpl, npred in (("a", JUDGE_A, 8), ("b", JUDGE_B, 8)):
                try:
                    out, _ = RX.gen(
                        args.model,
                        tmpl.format(gold=gold, cand=cand),
                        npred,
                        num_ctx,
                    )
                except Exception as e:
                    print("  ! %s %s 判定 %s 失败: %s" % (r["qid"], r["variant"], tag, e))
                    out = ""
                if tag == "a":
                    rec["judge_a"] = 1 if re.search(r"\byes\b", out, re.I) else 0
                    rec["raw_a"] = out[:40]
                else:
                    rec["judge_b"] = 1 if re.search(r"\b1\b", out) else 0
                    rec["raw_b"] = out[:40]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 20 == 0 or n == len(todo):
                rate = n / (time.perf_counter() - t0)
                print("  [%d/%d] %.2f 条/s" % (n, len(todo), rate))

    report_judge(args)
    return 0


def report_judge(args):
    recs = load_jsonl(JUDGE_OUT)
    if not recs:
        print("没有判定结果")
        return
    for r in recs:
        r["_s"] = r["string_match"]
        r["_a"] = r.get("judge_a", 0)
        r["_b"] = r.get("judge_b", 0)

    print()
    print("=" * 88)
    print("实验一结果：判定器一致性（§6.2 L3）\n")
    a = [r["_a"] for r in recs]
    b = [r["_b"] for r in recs]
    agree, k = kappa(a, b)
    print("  样本 %d 条" % len(recs))
    print("  判定器 A 通过率 %.0f%%   判定器 B 通过率 %.0f%%"
          % (sum(a) / len(a) * 100, sum(b) / len(b) * 100))
    print("  两者一致率 %.0f%%，Cohen's kappa = %.3f" % (agree * 100, k))
    verdict = ("一致性低，支持 L3「判定器非中立」" if k < 0.6
               else "一致性尚可，L3 的证据不强")
    print("  判读：%s" % verdict)

    print()
    print("=" * 88)
    print("实验一结果：同一批输出的三种口径与排名（§6.2 L2）\n")
    agg = defaultdict(list)
    for r in recs:
        agg[r["variant"]].append(r)
    print("  %-28s%14s%14s%14s" % ("变体", "字符串匹配", "判定器A", "判定器B"))
    score = {}
    for v in sorted(agg):
        cell = agg[v]
        s = sum(x["_s"] for x in cell) / len(cell) * 100
        aa = sum(x["_a"] for x in cell) / len(cell) * 100
        bb = sum(x["_b"] for x in cell) / len(cell) * 100
        score[v] = (s, aa, bb)
        print("  %-28s%13.0f%%%13.0f%%%13.0f%%" % (RX.VARIANT_LABEL.get(v, v), s, aa, bb))

    # 排名是否翻转
    def ranks(idx):
        order = sorted(score, key=lambda v: score[v][idx], reverse=True)
        return {v: i + 1 for i, v in enumerate(order)}

    r0, r1, r2 = ranks(0), ranks(1), ranks(2)
    flips = [
        v for v in score
        if not (r0[v] == r1[v] == r2[v])
    ]
    print()
    if flips:
        print("  排名发生变化的变体：%s" % ", ".join(RX.VARIANT_LABEL.get(v, v) for v in flips))
        for v in flips:
            print("    %-28s 字符串匹配第 %d，判定器A第 %d，判定器B第 %d"
                  % (RX.VARIANT_LABEL.get(v, v), r0[v], r1[v], r2[v]))
        print("  判读：同一批输出换口径后排名改变，支持 L2「同一维度内部不可比」")
    else:
        print("  三种口径下排名一致，L2 的证据不强")



# ---------------- 实验三：重测信度（L3） ----------------

def run_retest(args):
    """同一个判定提示、非零温度、重复多次，测判定器自身的稳定性。

    与 Akarsu 与 Ozdemir 报告的 same-judge-self-k 与 41% 翻转率直接可比。
    这一步才对应论文 §6.2 L3 所说的「判定器非中立」，
    而「两种措辞是否一致」测的是同义反复，不能作数。
    """
    rows = load_jsonl(RESULTS)
    if not rows:
        raise SystemExit("找不到 " + RESULTS + "，先跑主实验")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    picked = [r for r in rows if r["variant"] in variants and r["budget"] == args.budget]
    if not picked:
        raise SystemExit("预算 %d 下没有这些变体的记录" % args.budget)

    corpus_ctx = {d["doc_id"]: d["text"] for d in load_jsonl("data/corpus.jsonl")}
    guard(RETEST_OUT)
    os.makedirs(RUNS, exist_ok=True)
    done = {r["qid"] + "|" + r["variant"] for r in load_jsonl(RETEST_OUT)}
    todo = [r for r in picked if r["qid"] + "|" + r["variant"] not in done]

    print("实验三 重测信度（L3）")
    print("  变体 %s   预算 %d" % (variants, args.budget))
    print("  判定提示 A，temperature=%.2f，每题重复 %d 次" % (args.temp, args.reps))
    print("  共 %d 条，已完成 %d，待跑 %d 条 = %d 次调用\n"
          % (len(picked), len(picked) - len(todo), len(todo), len(todo) * args.reps))

    if args.dry_run:
        print("dry-run 通过：不会改动 %s" % RESULTS)
        return 0

    t0 = time.perf_counter()
    with open(RETEST_OUT, "a", encoding="utf-8") as f:
        for n, r in enumerate(todo, 1):
            gold = (r.get("answers") or [""])[0]
            cand = r.get("answer", "")
            ctx = " ".join(
                (corpus_ctx.get(d) or "")[:400] for d in (r.get("used_docs") or [])[:6]
            )
            tasks = {
                "match": JUDGE_A.format(gold=gold, cand=cand),
                "faith": JUDGE_FAITH.format(ctx=ctx, q=r.get("question", ""), cand=cand),
            }
            rec = {
                "qid": r["qid"], "variant": r["variant"], "budget": r["budget"],
                "gold": gold, "candidate": cand,
                "string_match": contains(cand, [gold]),
            }
            for tag, prompt in tasks.items():
                votes = []
                for k in range(args.reps):
                    # 关键：每次用不同的种子，否则 temperature 形同虚设
                    try:
                        out = gen_temp(args.model, prompt, 8, 2048, args.temp, seed=1000 + k)
                    except Exception as e:
                        print("  ! %s %s %s 失败: %s" % (r["qid"], r["variant"], tag, e))
                        out = ""
                    votes.append(1 if re.search(r"\byes\b", out, re.I) else 0)
                rec[tag + "_votes"] = votes
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 20 == 0 or n == len(todo):
                print("  [%d/%d] %.2f 条/s" % (n, len(todo), n / (time.perf_counter() - t0)))

    report_retest(args)
    return 0


def report_retest(args):
    recs = load_jsonl(RETEST_OUT)
    if not recs:
        print("没有重测结果")
        return
    print()
    print("=" * 88)
    print("实验三结果：判定器重测信度（§6.2 L3）\n")
    n = len(recs)
    print("  样本 %d 条，每题 %d 次重复，temperature=%.2f，每次用不同种子"
          % (n, args.reps, args.temp))
    print("  对照：Akarsu 与 Ozdemir 报告 41%% 的判定在重复测量中翻转\n")
    print("  %-10s%10s%14s%16s" % ("判定任务", "通过率", "翻转条目", "翻转率"))
    for tag, name in (("match", "答案匹配"), ("faith", "忠实性")):
        vs = [r.get(tag + "_votes") for r in recs if r.get(tag + "_votes")]
        if not vs:
            continue
        fl = [v for v in vs if len(set(v)) > 1]
        rate = sum(sum(v) for v in vs) / (len(vs) * args.reps) * 100
        print("  %-10s%9.0f%%%14d%15.0f%%"
              % (name, rate, len(fl), len(fl) / len(vs) * 100))
        if len(fl) / len(vs) >= 0.15:
            print("            判读：该任务下判定器不稳定，支持 L3")
        else:
            print("            判读：该任务下判定器稳定，L3 证据不强")

    print()
    print("  按票数分布（忠实性任务）：")
    from collections import Counter as C
    fv = [r.get("faith_votes") for r in recs if r.get("faith_votes")]
    dist = C(sum(v) for v in fv)
    for k in range(args.reps + 1):
        if dist.get(k):
            print("    %d / %d 票通过：%d 条（%.0f%%）" % (k, args.reps, dist[k], dist[k] / len(fv) * 100))

    print()
    print("  各次重复给出的通过率与排名（检验结论是否随重复变化）\n")
    agg = defaultdict(list)
    for r in recs:
        agg[r["variant"]].append(r)
    print("  %-30s" % "变体" + "".join("%12s" % ("第%d次" % (i + 1)) for i in range(args.reps)))
    for v in sorted(agg):
        cell = agg[v]
        line = "  %-30s" % RX.VARIANT_LABEL.get(v, v)
        for i in range(args.reps):
            vv = [x["match_votes"][i] for x in cell if x.get("match_votes")]
            line += "%11.0f%%" % (sum(vv) / len(vv) * 100 if vv else 0)
        print(line)

    ranks = []
    for i in range(args.reps):
        order = sorted(
            agg,
            key=lambda v: sum(x["match_votes"][i] for x in agg[v] if x.get("match_votes"))
            / max(len(agg[v]), 1),
            reverse=True,
        )
        ranks.append(tuple(order))
    if len(set(ranks)) == 1:
        print("\n  各次重复的排名一致，聚合层面稳定。")
    else:
        print("\n  各次重复的排名不一致，仅凭一次测量得到的排名不可复现。")
        for i, rk in enumerate(ranks, 1):
            print("    第%d次：%s" % (i, " > ".join(RX.VARIANT_LABEL.get(v, v) for v in rk)))


# ---------------- 实验二：位置 ----------------

def fit_docs(doc_ids, corpus_map, budget, must_include):
    """按预算的 80% 挑出可完整放入的文档，并保证 must_include 在其中。"""
    cap = int(budget * RX.CHARS_PER_TOKEN * 0.8)
    chosen, used = [], 0
    for d in doc_ids:
        t = corpus_map.get(d)
        if t is None:
            continue
        cost = len(t) + len(d) + 8
        if used + cost > cap:
            continue
        chosen.append(d)
        used += cost
    for g in must_include:
        if g in corpus_map and g not in chosen:
            # 用金标替换最后一篇，保证三种排列的证据集合完全相同
            if chosen:
                chosen[-1] = g
            else:
                chosen.append(g)
    return chosen


def reorder(docs, gold, where):
    rest = [d for d in docs if d not in gold]
    g = [d for d in docs if d in gold]
    if where == "start":
        return g + rest
    if where == "end":
        return rest + g
    mid = len(rest) // 2
    return rest[:mid] + g + rest[mid:]


def run_position(args):
    rows = load_jsonl(RESULTS)
    if not rows:
        raise SystemExit("找不到 " + RESULTS + "，先跑主实验")
    base = [r for r in rows if r["variant"] == args.base and r["budget"] == args.budget]
    if not base:
        raise SystemExit("预算 %d 下没有 %s 的记录" % (args.budget, args.base))

    corpus_map = {d["doc_id"]: d["text"] for d in load_jsonl("data/corpus.jsonl")}
    if not corpus_map:
        raise SystemExit("找不到 data/corpus.jsonl")

    guard(POS_OUT)
    os.makedirs(RUNS, exist_ok=True)
    done = {(r["qid"], r["position"]) for r in load_jsonl(POS_OUT)}
    plan = []
    for r in base:
        chosen = fit_docs(r.get("used_docs") or [], corpus_map, args.budget, r.get("gold_doc_ids") or [])
        if not chosen:
            continue
        for pos in POSITIONS:
            if (r["qid"], pos) in done:
                continue
            if not any(g in chosen for g in (r.get("gold_doc_ids") or [])):
                continue
            plan.append((r, pos, chosen))

    print("实验二 位置效应")
    print("  基线 %s 预算 %d   %d 题 × %d 个位置，待跑 %d 次"
          % (args.base, args.budget, len(base), len(POSITIONS), len(plan)))
    print("  证据集合按预算的 80%% 裁剪，保证三种排列都不被截断\n")

    if args.dry_run:
        print("dry-run 通过：不会改动 %s" % RESULTS)
        return 0

    num_ctx = args.budget + 512
    t0 = time.perf_counter()
    with open(POS_OUT, "a", encoding="utf-8") as f:
        for n, (r, pos, chosen) in enumerate(plan, 1):
            ordered = reorder(chosen, set(r.get("gold_doc_ids") or []), pos)
            prompt, used, trunc = RX.build_prompt(
                r["question"], [(d, corpus_map[d]) for d in ordered], args.budget
            )
            try:
                ans, m = RX.gen(args.model, prompt, args.num_predict, num_ctx)
            except Exception as e:
                print("  ! %s %s 失败: %s" % (r["qid"], pos, e))
                continue
            rec = {
                "qid": r["qid"],
                "position": pos,
                "budget": args.budget,
                "question": r["question"],
                "answers": r["answers"],
                "answer": ans,
                "used_docs": used,
                "gold_pos_index": next(
                    (i for i, d in enumerate(ordered) if d in set(r.get("gold_doc_ids") or [])), -1
                ),
                "n_docs": len(ordered),
                "truncated": trunc,
                **m,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 20 == 0 or n == len(plan):
                print("  [%d/%d] %.2f 条/s" % (n, len(plan), n / (time.perf_counter() - t0)))

    report_position()
    return 0


def report_position():
    recs = load_jsonl(POS_OUT)
    if not recs:
        print("没有位置实验结果")
        return
    print()
    print("=" * 88)
    print("实验二结果：证据位置对质量的影响（§4.1.1）\n")
    agg = defaultdict(list)
    for r in recs:
        r["_c"] = contains(r.get("answer", ""), r.get("answers") or [""])
        agg[r["position"]].append(r)
    print("  %-10s%8s%12s%14s%14s" % ("位置", "样本", "contains", "金标名次中位", "截断率"))
    for pos in POSITIONS:
        cell = agg.get(pos, [])
        if not cell:
            continue
        idx = [x["gold_pos_index"] for x in cell if x["gold_pos_index"] >= 0]
        print("  %-10s%8d%11.0f%%%14s%13.0f%%"
              % (pos, len(cell),
                 sum(x["_c"] for x in cell) / len(cell) * 100,
                 ("%.0f" % statistics.median(idx)) if idx else "-",
                 sum(1 for x in cell if x["truncated"]) / len(cell) * 100))
    vals = {p: sum(x["_c"] for x in agg[p]) / len(agg[p]) * 100 for p in POSITIONS if agg.get(p)}
    if len(vals) >= 2:
        spread = max(vals.values()) - min(vals.values())
        print()
        print("  最好与最差位置相差 %.0f 个百分点" % spread)
        if spread >= 8:
            print("  判读：位置效应显著，支持 §4.1.1「证据位置显著影响利用率」")
        else:
            print("  判读：位置效应不显著，§4.1.1 的结论在本设定下未复现")
            print("        这与论文引用的 Lost in the Evidence 反证方向一致。")


# ---------------- 入口 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["judge", "position", "retest", "all"])
    ap.add_argument("--model", default="qwen2.5:1.5b-instruct")
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--variants", default="s2_flat,s3_good,s5_rerank")
    ap.add_argument("--base", default="s2_flat")
    ap.add_argument("--num-predict", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    if not RX.check_server():
        print("x 连不上 Ollama。先运行：  ollama serve")
        return 1
    if not RX.has_model(args.model):
        print("x 本机没有模型 " + args.model)
        return 1

    if args.exp in ("judge", "all"):
        run_judge(args)
    if args.exp in ("retest", "all"):
        print()
        run_retest(args)
    if args.exp in ("position", "all"):
        print()
        run_position(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
