#!/usr/bin/env python3
"""地板检验：1.5B 模型在 HotpotQA 上到底能不能答对

为什么先做这个
    如果模型的读者能力本身接近零，那么无论检索策略怎么设计，
    五条质量曲线都会贴在底部，后续实验就没有信息量。
    所以在做任何策略对比之前，先测"给定正确证据，能不能答对"。

本脚本只测读者地板，不测检索
    每条样本把 HotpotQA 的金标支撑段落直接放进上下文，
    因此分数反映的是模型的阅读与抽取能力，与检索器无关。

判读标准（预先声明，避免事后解释）
    读者 EM >= 25%            -> 1.5B 可用，按协议继续
    10% <= 读者 EM < 25%      -> 边缘，建议换 3B
    读者 EM < 10%             -> 太弱，换 3B 或 7B

数据来源
    HuggingFace datasets-server 的 HTTP 接口，不需要安装 datasets 库。
    只取 validation 集的前 N 条。

用法：
    python3 floor_check.py
    python3 floor_check.py --n 30 --model qwen2.5:3b-instruct
    python3 floor_check.py --budgets 2048 8192
"""

import argparse
import json
import re
import string
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

OLLAMA = "http://localhost:11434"
HF = "https://datasets-server.huggingface.co/rows"


# ---------- Ollama ----------

def ollama_get(path, timeout=10):
    with urllib.request.urlopen(OLLAMA + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def check_server():
    try:
        ollama_get("/api/tags")
        return True
    except (urllib.error.URLError, ConnectionError, OSError):
        return False


def has_model(name):
    try:
        tags = ollama_get("/api/tags")
    except Exception:
        return False
    installed = {m.get("name", "") for m in tags.get("models", [])}
    return name in installed or any(
        n.split(":")[0] == name.split(":")[0] for n in installed
    )


def ask(model, prompt, num_predict, num_ctx):
    data = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 1,
                "seed": 42,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        }
    ).encode()
    req = urllib.request.Request(
        OLLAMA + "/api/generate", data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read().decode())
    return out.get("response", "").strip(), time.perf_counter() - t0, out


# ---------- 数据 ----------

def fetch_hotpot(n):
    """从 datasets-server 取 HotpotQA distractor/validation 的前 n 条。"""
    args = urllib.parse.urlencode(
        {
            "dataset": "hotpotqa/hotpot_qa",
            "config": "distractor",
            "split": "validation",
            "offset": 0,
            "length": n,
        }
    )
    url = f"{HF}?{args}"
    print(f"  下载 {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        payload = json.loads(r.read().decode())
    rows = []
    for item in payload.get("rows", []):
        row = item.get("row", {})
        ctx = row.get("context") or {}
        titles = ctx.get("title") or []
        sentences = ctx.get("sentences") or []
        # 金标支撑段落即 supporting_facts 里的 title 对应段落
        sf = row.get("supporting_facts") or {}
        gold_titles = set(sf.get("title") or [])
        paras = []
        for t, sents in zip(titles, sentences):
            if t in gold_titles:
                paras.append(t + ". " + " ".join(sents))
        rows.append(
            {
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "context": "\n\n".join(paras) if paras else "\n\n".join(
                    (t + ". " + " ".join(s)) for t, s in list(zip(titles, sentences))[:2]
                ),
                "n_gold_paras": len(paras),
            }
        )
    return rows


# ---------- 评分 ----------

def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_f1(pred, gold):
    p, g = normalize(pred), normalize(gold)
    em = 1.0 if p == g else 0.0
    pt, gt = p.split(), g.split()
    if not pt or not gt:
        return em, em
    common = {}
    for t in pt:
        common[t] = min(pt.count(t), gt.count(t))
    overlap = sum(common.values())
    if overlap == 0:
        return em, 0.0
    prec = overlap / len(pt)
    rec = overlap / len(gt)
    return em, 2 * prec * rec / (prec + rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:1.5b-instruct")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budgets", nargs="+", type=int, default=[8192])
    ap.add_argument("--num-predict", type=int, default=64)
    args = ap.parse_args()

    if not check_server():
        print("x 连不上 Ollama。先运行：  ollama serve")
        return 1
    if not has_model(args.model):
        print(f"x 本机没有模型 {args.model}")
        print(f"  先运行：  ollama pull {args.model}")
        return 1

    print(f"模型 {args.model}    样本 {args.n} 条    预算 {args.budgets}\n")
    try:
        rows = fetch_hotpot(args.n)
    except Exception as e:
        print(f"x 下载 HotpotQA 失败：{e}")
        print("  可在浏览器打开下面这个地址确认接口是否可用：")
        print(
            "  https://datasets-server.huggingface.co/rows?dataset=hotpotqa/hotpot_qa"
            "&config=distractor&split=validation&offset=0&length=2"
        )
        return 1
    if not rows:
        print("x 没有取到样本")
        return 1
    print(f"取到 {len(rows)} 条，平均金标段落 {sum(r['n_gold_paras'] for r in rows)/len(rows):.1f} 段\n")

    for budget in args.budgets:
        num_ctx = budget + 512
        print(f"=== 预算 {budget}（num_ctx={num_ctx}）===")
        ems, f1s, walls, skipped = [], [], [], 0
        for k, r in enumerate(rows, 1):
            prompt = (
                "Read the evidence below and answer the question with a short phrase "
                "copied from the evidence. Do not explain.\n\n"
                f"Evidence:\n{r['context']}\n\n"
                f"Question: {r['question']}\nShort answer:"
            )
            # 简单裁剪：超出预算按字符粗略截断（约 4 字符/token）
            if len(prompt) > budget * 3.6:
                prompt = prompt[: int(budget * 3.6)] + "\n\nShort answer:"
                skipped += 1
            try:
                pred, wall, _ = ask(args.model, prompt, args.num_predict, num_ctx)
            except Exception as e:
                print(f"  [{k}] 调用失败：{e}")
                continue
            em, f1 = em_f1(pred, r["answer"])
            ems.append(em)
            f1s.append(f1)
            walls.append(wall)
            if k <= 5:
                print(
                    f"  [{k}] pred={pred[:50]!r}  gold={r['answer'][:30]!r}"
                    f"  EM={em:.0f} F1={f1:.2f}"
                )

        if not ems:
            print("  全部失败\n")
            continue

        n = len(ems)
        mean_em = sum(ems) / n
        mean_f1 = sum(f1s) / n
        # bootstrap 95% 置信区间
        import random

        rng = random.Random(42)
        boots = []
        for _ in range(2000):
            sample = [rng.choice(ems) for _ in range(n)]
            boots.append(sum(sample) / n)
        boots.sort()
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]

        print(f"  样本 {n}   截断 {skipped}")
        print(f"  EM  {mean_em*100:5.1f}%   95%CI [{lo*100:.0f}, {hi*100:.0f}]")
        print(f"  F1  {mean_f1*100:5.1f}%")
        print(f"  平均耗时 {sum(walls)/len(walls):.2f}s\n")

        if mean_em >= 0.25:
            verdict = "1.5B 可用，按协议继续"
        elif mean_em >= 0.10:
            verdict = "边缘，建议换 3B"
        else:
            verdict = "太弱，换 3B 或 7B"
        print(f"  判读：{verdict}\n")

    print("提示：若 EM 偏低但 F1 尚可，说明模型答对了内容但格式不对，")
    print("      正式实验应改用更宽松的匹配（包含式判定）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
