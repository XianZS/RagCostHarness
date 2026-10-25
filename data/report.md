# 数据校验报告

- 查询数：57
- 语料段落数：321
- 语料总 token（估）：47429
- 题型分布：{'hotpot_comparison': 8, 'hotpot_bridge': 22, 'squad_single_hop': 27}

| 检查项 | 结果 | 说明 |
|---|---|---|
| V1 每个查询至少一个金标段落 | PASS | 缺失 0 条 [] |
| V2 金标段落都在语料库中 | PASS | 缺失 0 个 [] |
| V3 答案非空 | PASS | 缺失 0 条 [] |
| V4 题型至少两类且最大类不超 70% | PASS | 分布 {'hotpot_comparison': 8, 'hotpot_bridge': 22, 'squad_single_hop': 27}，最大类占 47% |
| V5 问题文本无重复 | PASS | 重复 0 条 [] |
| V6 语料规模足以让检索有意义（总 token 远大于单次预算） | PASS | 语料约 47429 tokens，语料库 321 段 |

全部通过，可以进入第二步。