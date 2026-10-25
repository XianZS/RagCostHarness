#!/usr/bin/env python3
"""按预算设定 num_ctx 的测速脚本（v3）

v2 的问题：num_ctx 固定为 32768，模型每次都为大窗口分配 KV 缓存，
在 16 GB 统一内存上触发换页，导致同一档位的墙钟出现 5 倍波动
（8K 档三次分别为 3.04s / 3.62s / 15.23s）。

v3 的改法：num_ctx = 预算 + 输出余量。这同时做两件事
  1. KV 缓存放小，消除换页造成的计时噪声
  2. 预算在模型层面被硬性强制，符合协议要求的"预算拦截"

每档跑 5 次，报告最小/中位/最大，波动一栏是判断稳定性的依据。

用法：
    python3 bench_budget.py
    python3 bench_budget.py --budgets 2048 4096 8192
    python3 bench_budget.py --model qwen2.5:3b-instruct
"""

import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:11434"
OUTPUT_ALLOWANCE = 512  # num_ctx = budget + 此值，留给输出

WORDS = (
    "atlas beacon cipher delta ember fathom gauge harbor igloo jigsaw kelp lantern "
    "meadow nectar orbit prism quarry ripple sable tundra umbra vessel willow xenon "
    "yarrow zephyr anvil basin cinder dune eclipse flint grove hollow inlet juniper "
    "kettle loam marrow niche opaque parchment quiver ravine sediment thistle upland "
    "verge wattle yonder zenith almond bramble cobble drapery elm furrow granite hedge "
    "ivory jetty kindle ledge mortar nimbus oyster pebble quill reed spruce trellis "
    "urn vane wharf yew zinnia basalt cobalt dovetail elder frost glade heath iris "
    "jade knoll larch mistle nook onyx plume quartz rill slate tumble umber vine whin "
    "axis brimstone cedar driftwood flagon girder hemp indigo jamb kiln lupine"
).split()


def _post(path, payload, timeout=3600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(path, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def check_server():
    try:
        _get("/api/tags")
        return True
    except (urllib.error.URLError, ConnectionError, OSError):
        return False


def has_model(name):
    try:
        tags = _get("/api/tags")
    except Exception:
        return False
    installed = {m.get("name", "") for m in tags.get("models", [])}
    return name in installed or any(
        n.split(":")[0] == name.split(":")[0] for n in installed
    )


def random_prompt(target_tokens, rng):
    n_words = int(target_tokens * 0.78)
    body = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return (
        "The passage below is a word list with no meaning. Ignore its content.\n\n"
        f"{body}\n\n"
        "Question: in about 100 words, explain what a retrieval augmented system "
        "conditions each response on.\nAnswer:"
    )


def one_call(model, prompt, num_predict, num_ctx):
    t0 = time.perf_counter()
    out = _post(
        "/api/generate",
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
        },
    )
    wall = time.perf_counter() - t0
    return {
        "wall_s": wall,
        "prompt_tokens": out.get("prompt_eval_count", 0) or 0,
        "completion_tokens": out.get("eval_count", 0) or 0,
        "load_s": (out.get("load_duration", 0) or 0) / 1e9,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:1.5b-instruct")
    ap.add_argument("--budgets", nargs="+", type=int, default=[2048, 4096, 8192])
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    if not check_server():
        print("x 连不上 Ollama。先运行：  ollama serve")
        return 1
    if not has_model(args.model):
        print(f"x 本机没有模型 {args.model}")
        print(f"  先运行：  ollama pull {args.model}")
        return 1

    rng = random.Random(20261024)
    print(f"模型 {args.model}    num_predict={args.num_predict}    每档 {args.repeats} 次")
    print("num_ctx = 预算 + %d，每次调用使用互不相同的随机提示\n" % OUTPUT_ALLOWANCE)
    print("建议先关掉浏览器等占用内存的程序，减少换页干扰\n")

    warm = one_call(args.model, random_prompt(args.budgets[0], rng), 64, args.budgets[0] + OUTPUT_ALLOWANCE)
    print(f"预热完成，加载 {warm['load_s']:.1f}s\n")

    rows = []
    for budget in args.budgets:
        num_ctx = budget + OUTPUT_ALLOWANCE
        # 提示按预算的 85% 造，留出模板与输出空间
        target = int(budget * 0.85)
        walls, ptoks, ctoks = [], [], []
        print(f"- 预算 {budget}  num_ctx={num_ctx}")
        for _ in range(args.repeats):
            r = one_call(args.model, random_prompt(target, rng), args.num_predict, num_ctx)
            walls.append(r["wall_s"])
            ptoks.append(r["prompt_tokens"])
            ctoks.append(r["completion_tokens"])
            print(
                f"    提示 {r['prompt_tokens']:>6} + 输出 {r['completion_tokens']:>4}"
                f"  墙钟 {r['wall_s']:6.2f}s"
            )
        med = statistics.median(walls)
        spread = (max(walls) - min(walls)) / med * 100 if med else 0
        rows.append(
            {
                "budget": budget,
                "num_ctx": num_ctx,
                "prompt_tokens": statistics.median(ptoks),
                "completion_tokens": statistics.median(ctoks),
                "min": min(walls),
                "median": med,
                "max": max(walls),
                "spread_pct": spread,
            }
        )
        verdict = "稳定" if spread < 30 else ("偏高" if spread < 100 else "不稳定")
        print(f"    中位 {med:.2f}s   极差 {spread:.0f}%   {verdict}\n")

    print("=" * 80)
    print(f"{'预算':>7}{'num_ctx':>9}{'提示token':>10}{'最小s':>8}{'中位s':>8}{'最大s':>8}{'极差%':>8}")
    for r in rows:
        print(
            f"{r['budget']:>7}{r['num_ctx']:>9}{r['prompt_tokens']:>10.0f}"
            f"{r['min']:>8.2f}{r['median']:>8.2f}{r['max']:>8.2f}{r['spread_pct']:>8.0f}"
        )

    designs = [
        ("最小  4策略x3档x40题", 4, 40, 1.0),
        ("推荐  5策略x3档x60题", 5, 60, 1.25),
        ("扩展  5策略x3档x100题", 5, 100, 1.25),
    ]
    print()
    print("=" * 80)
    print("按本次中位墙钟估算的总机器时间（三档预算）：\n")
    print(f"{'方案':<26}{'调用数':>8}{'小时':>9}")
    per_budget = {r["budget"]: r["median"] for r in rows}
    mean_call = sum(per_budget.values()) / len(per_budget)
    for name, strategies, nq, factor in designs:
        calls = strategies * len(per_budget) * nq * factor
        print(f"{name:<26}{int(calls):>8}{calls * mean_call / 3600:>9.1f}")
    print()
    print("另需索引构造成本（S3 聚类与摘要），见实验协议 7 节。")
    print("若某档极差超过 100%，先关掉占内存的程序重跑该档。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
