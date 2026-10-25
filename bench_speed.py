#!/usr/bin/env python3
"""M4 + Ollama 测速脚本 v2

v1 的两个缺陷已修正：
  1. 提示由重复文本构成，前缀被 KV 缓存命中，预填时长被严重低估
     → 现在每次调用生成互不相同的随机文本，杜绝前缀共享
  2. 模型只答约 20 token 就 EOS，解码占比失真
     → 现在明确要求约 120 词的回答，num_predict=256

以墙钟时间为准。API 返回的 prefill/decode 只作参考量打印。

用法：
    python3 bench_speed.py
    python3 bench_speed.py --model qwen2.5:3b-instruct
    python3 bench_speed.py --lens 2000 8000 16000 --repeats 3
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:11434"

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
    """生成与之前任何调用都不共享前缀的提示。1 token 约 0.78 词。"""
    n_words = int(target_tokens * 0.78)
    body = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return (
        "The passage below is a word list with no meaning. Ignore its content.\n\n"
        f"{body}\n\n"
        "Question: in about 120 words, describe how the cost of a language model "
        "call splits into a prefill term and a decode term, and which one dominates "
        "when the prompt is long but the answer is short.\nAnswer:"
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
    pc = out.get("prompt_eval_count", 0) or 0
    pd = out.get("prompt_eval_duration", 0) or 0
    ec = out.get("eval_count", 0) or 0
    ed = out.get("eval_duration", 0) or 0
    return {
        "wall_s": wall,
        "prompt_tokens": pc,
        "completion_tokens": ec,
        "api_prefill_tps": (pc / (pd / 1e9)) if pd else 0.0,
        "api_decode_tps": (ec / (ed / 1e9)) if ed else 0.0,
        "load_s": (out.get("load_duration", 0) or 0) / 1e9,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:1.5b-instruct")
    ap.add_argument("--lens", nargs="+", type=int, default=[2000, 8000, 16000])
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    if not check_server():
        print("x 连不上 Ollama。先运行：  ollama serve")
        return 1
    if not has_model(args.model):
        print(f"x 本机没有模型 {args.model}")
        print(f"  先运行：  ollama pull {args.model}")
        return 1

    rng = random.Random(20261024)
    print(f"模型 {args.model}    num_ctx={args.num_ctx}    num_predict={args.num_predict}")
    print("每次调用使用互不相同的随机提示，避免 KV 缓存复用\n")

    warm = one_call(args.model, random_prompt(args.lens[0], rng), args.num_predict, args.num_ctx)
    print(f"预热完成，加载 {warm['load_s']:.1f}s\n")

    rows = []
    for lens in args.lens:
        print(f"- 目标 {lens} tokens")
        vals = []
        for k in range(args.repeats):
            r = one_call(args.model, random_prompt(lens, rng), args.num_predict, args.num_ctx)
            vals.append(r)
            total_tok = r["prompt_tokens"] + r["completion_tokens"]
            print(
                f"    提示 {r['prompt_tokens']:>6} + 输出 {r['completion_tokens']:>4}"
                f"  墙钟 {r['wall_s']:6.2f}s"
                f"  有效吞吐 {total_tok / r['wall_s']:7.1f} tok/s"
                f"  (API 参考 预填 {r['api_prefill_tps']:.0f} / 解码 {r['api_decode_tps']:.1f})"
            )

        def mean(k):
            return sum(v[k] for v in vals) / len(vals)

        rows.append(
            {
                "target": lens,
                "prompt_tokens": mean("prompt_tokens"),
                "completion_tokens": mean("completion_tokens"),
                "wall_s": mean("wall_s"),
            }
        )
        print()

    print("=" * 78)
    print(f"{'提示token':>10}{'输出token':>10}{'墙钟秒':>10}{'有效吞吐 tok/s':>16}")
    for r in rows:
        eff = (r["prompt_tokens"] + r["completion_tokens"]) / r["wall_s"]
        print(
            f"{r['prompt_tokens']:>10.0f}{r['completion_tokens']:>10.0f}"
            f"{r['wall_s']:>10.2f}{eff:>16.1f}"
        )

    designs = [
        ("最小可行  4策略x3档x40题", 4, [2000, 8000, 16000], 40, 1.0),
        ("推荐      5策略x4档x60题", 5, [2000, 4000, 8000, 16000], 60, 1.25),
        ("完整      5策略x5档x100题", 5, [2000, 4000, 8000, 16000, 32000], 100, 1.25),
    ]
    by_len = {r["target"]: r for r in rows}

    def per_call(target):
        if target in by_len:
            return by_len[target]["wall_s"]
        near = min(by_len, key=lambda x: abs(x - target))
        r = by_len[near]
        return r["wall_s"] * (target / r["prompt_tokens"])

    print()
    print("=" * 78)
    print("按本次实测墙钟估算的总机器时间：\n")
    print(f"{'方案':<28}{'调用数':>8}{'小时':>9}")
    for name, strategies, budgets, nq, agentic_factor in designs:
        calls = strategies * len(budgets) * nq * agentic_factor
        secs = sum(per_call(b) for b in budgets) / len(budgets) * calls
        print(f"{name:<28}{int(calls):>8}{secs / 3600:>9.1f}")
    print()
    print("另需索引构造成本（S3 聚类与摘要），见实验协议 7 节。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
