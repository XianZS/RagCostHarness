# 实验说明（README）

本目录记录 2026-10-24 为论文《Knowledge Carriers in Retrieval-Augmented Generation: A Critical Survey of Design Space, Cost, and Failure》所做的全部实验。

---

## 一、实验与论文的对应关系

| 实验 | 脚本 | 产物 | 论文位置 | 支撑了什么 |
|---|---|---|---|---|
| 主实验（11 变体 × 2 预算 × 57 题）| `run_experiment.py` | `runs/results.jsonl` | 表 16、表 17，7.9.3、7.9.4 全部数字 | 六类成本可被填满；两项质量差异达显著 |
| 判定器交叉判定 | `exp_extra.py --exp judge` | `runs/judge.jsonl` | 7.9.5 | 两种措辞的判定一致率 96%，kappa 0.914 |
| 重测信度 | `exp_extra.py --exp retest` | `runs/retest.jsonl` | 7.9.5 | 重复测量翻转率 0% 与 2%，未复现文献的 41% |
| 位置效应 | `exp_extra.py --exp position` | `runs/position.jsonl` | 7.9.5 | 首/中/尾 74%/68%/72%，方向一致但未达显著 |
| 运维成本补测 | `exp_maint.py` | `runs/maint.jsonl` | 7.9.3、附录 B 的运维成本推算段 | 一次更新 7,416 token，日更一年为构造成本的 32.3 倍 |

### 论文中实验内容的位置

| 位置 | 内容 | 规模 |
|---|---|---|
| §7.9 框架的可行性检验 | 7.9.1 设定、7.9.2 记录与复现、7.9.3 六类成本均被填满、7.9.4 判据给出的裁决、7.9.5 检验的边界 | 约 1,800 字 |
| 附录 B 可行性检验的完整数据 | 表 16（各变体成本与质量）、表 17（检索诊断与四项探针）+ 运维成本推算过程 | 2 张表 |
| 摘要末段 | 一句「本文另以一次最小规模的实验检验该口径与判据可被实际执行」 | 1 句 |
| 引言导览 | 「第 7 章……并附一次可行性检验」 | 半句 |

### 每个数字对应哪张表、哪一节

| 数字 | 含义 | 出现位置 |
|---|---|---|
| 321 / 57 / 39 | 语料段数 / 题数 / 需检索子集 | 7.9.1、7.9.5、附录 B |
| 1,254 | 主实验调用总数 | 7.9.1 |
| 83,740 | 摘要层构造成本（token） | 7.9.3、7.9.4、附录 B |
| 46,425 | 导语摘要层构造成本 | 7.9.3、附录 B |
| 7,416 / 22.8 秒 | 一次更新事件的增量运维成本 | 7.9.3、附录 B |
| 2,706,840 / 32.3 倍 | 日更一年运维成本及其相对构造的倍数 | 7.9.3、附录 B |
| 3,361 → 5,496（+64%） | 智能体策略的查询成本变化 | 7.9.4 |
| p=0.016 / p=0.021 | 全文摘要层、LLM 重排两项的精确检验 | 7.9.4 |
| 8.6 / 17 | 两组之差的标准误 / 达到 95% 把握所需的最小差 | 7.9.4 |
| 0.914 / 96% | 两种措辞判定的一致率（Cohen kappa） | 7.9.5 |
| 74 / 68 / 72 | 位置效应三档的 contains | 7.9.5 |

### 框架组件与实验验证

论文第 2 章给出的框架有八个组件，下表逐个说明它们在实验里怎么被检验、结果如何、落在论文哪一处、以及由哪个脚本产出。

| # | 框架组件 | 实验中如何被验证 | 结果 | 位置 | 验证到的程度 | 代码文件 |
|---|---|---|---|---|---|---|
| ① | 四元组形式化 $S=(W,C,I,\pi)$ | 11 个变体各自映射到四元组，看是否都能落位 | 11 个全部落位；但 $W$ 只测了"不检索的固定模型"这一代理 | 7.9.1（p33）、表 16 | 四元组可落位，四个载体各有对应变体 | `run_experiment.py` |
| ② | 四种载体 | 每个载体至少构造一个变体 | $C$：s1、s2；$I$：s2、s3_good/bad/poor；$\pi$：s4、s5；$W$：s0（代理）| 表 16（p46）| 四个载体均有变体覆盖 | `run_experiment.py` |
| ③ | 四重约束 | 质量、成本、延迟、可信各给可测口径 | 质量 contains 与 EM；成本 token；延迟 ms；可信三分量：忠实性逐变体入表，鲁棒性 92%/95%、抗投毒 96%/91% | 表 16、7.9.4（p34）| 三项已测，忠实性逐变体入表 | `score.py` |
| ④ | 六类成本 | 逐类实测，看能否全部填满 | 六类全填：$C_{\text{build}}$ 83,740｜$C_{\text{ret}}$ 0 token｜$C_{\text{rank}}$ 1.76 s｜$C_{\text{gen}}$ 随策略｜$C_{\text{act}}$ 2,215｜$C_{\text{maint}}$ 7,416 | 7.9.3（p34）、表 16 | 六类全部实测，$C_{\text{maint}}$ 日更 32.3 倍 | `score.py`、`exp_maint.py` |
| ⑤ | $N^{*}$ 盈亏平衡判据 | 代入实测 $\Delta C_{\text{build}}$、$\Delta C_{\text{query}}$、$\Delta q$ | 给出 1 项显著否定（全文摘要层，$p=0.016$） | 7.9.4（p34） | 判据已代入实测值，给出 1 项显著否定 | `score.py` |
| ⑥ | 五分量评价向量 | 同一批输出按多分量同时报 | 表 16 现为七列：构造、查询、延迟、contains、忠实性、违规率 | 表 16（p46） | 六项已对齐，忠实性代表 $\Delta$Trust | `score.py` |
| ⑦ | 四级归因探针 P1–P4 | 逐个实现 | P1 召回 42%/77%；P2 重排召回 46%/23%；P3 金标 72%；P4 采纳 4%/9% | 表 17（p46） | 四探针全部实现并给出数值 | `score.py`、`exp_extra.py` |
| ⑧ | 质量口径自洽性 | 同一批输出换三种口径看排名 | 截断摘要层：忠实性第一（85%）、contains 第三（72%）、EM 第四（63%） | 7.9.4（p34）、表 16 | 三口径三排名，发现④的自有证据 | `score.py` |

---

## 二、环境配置

### 硬件与系统（本次实测环境）

| 项 | 值 |
|---|---|
| 机器 | Apple M4（10 核），16 GB 统一内存 |
| 系统 | macOS 26.6.2 |
| 磁盘 | 代码与数据 2.3 MB，模型 986 MB，运行产物 3.4 MB |

非 Apple 平台同样可跑，Ollama 支持 CUDA 与 CPU。上表是本报告所有耗时数据的来源机器。

### 软件版本

| 项 | 版本 |
|---|---|
| Python | 3.14.7 |
| Ollama | 0.40.8 |
| 模型 | `qwen2.5:1.5b-instruct`（Q4_K_M 量化，986 MB，参数 1.5B，上下文 32768，嵌入维度 1536）|

### 依赖：零第三方包

全部脚本只用 Python 标准库（`argparse` `collections` `json` `math` `os` `random` `re` `statistics` `string` `sys` `time` `urllib`）。

不需要 `pip install` 任何东西。 数据下载也走标准库的 `urllib`，不依赖 `datasets` 或 `requests`。

### 三步准备

```bash
# 1. 安装 Ollama
brew install ollama            # 或从 ollama.com 下载安装包

# 2. 启动服务（保持运行，可另开终端）
ollama serve

# 3. 拉取模型
ollama pull qwen2.5:1.5b-instruct
```

### 一条容易踩的实现细节

`num_ctx` 必须随预算设置，不能固定。

早期版本把 `num_ctx` 固定为 32768，于是 1,024 档的调用也按 32k 分配 KV cache，在 16 GB 机器上触发频繁换页，同一配置的耗时波动达 5 倍，token 数不变但时间数据不可用。

现版本按 `num_ctx = budget + 512` 设置，波动恢复到可接受范围。

### 耗时参考

| 阶段 | 调用次数 | 耗时 |
|---|---|---|
| 主实验 | 1,254 | 约 28 分钟 |
| 三组补充实验 | 1,539 | 约 12 分钟 |
| 运维成本补测 | 60 | 约 1 分钟 |
| 评分（不调模型）| 0 | 数秒 |

总计约 2,850 次调用、40 分钟，这是 M4 / 1.5B 的速度。更大的模型会显著变慢（3B 约慢一倍）。

---

## 三、跑了哪五组实验

### 1. 主实验（11 变体 × 2 预算 × 57 题）

```bash
./run_all.sh --reset
```

- 1,254 次模型调用，耗时 28 分钟
- 11 个变体：`s0_parametric` `s1_oracle` `s2_flat` `s3_good` `s3_bad` `s3_poor` `s4_agentic` `s5_rerank` `s6_recall` `s7_conflict` `s2_perturb`
- 两档输入预算：1,024 与 4,096 token
- 另含摘要层构建 642 次调用（`good` / `bad` / `poor` 各 321 条，第二次运行复用缓存）

产物：

| 文件 | 行数 | 说明 |
|---|---|---|
| `runs/results.jsonl` | 1,254 | 原始输出，不可再生，最重要 |
| `runs/build_costs.jsonl` | 26 | 构造成本，同一索引取历史最大值 |
| `runs/results_summary.csv` | 无 | 作图用汇总 |

### 2. 判定器交叉判定（两种措辞）

```bash
python3 exp_extra.py --exp judge
```

- 342 次调用（3 变体 × 57 题 × 2 个判定提示）
- 目的：检验两种措辞的判定提示是否一致
- 产物：`runs/judge.jsonl`（171 条） → 用于 §7.9.5 的「一致率 96%、kappa 0.914」

### 3. 重测信度（同一判定器重复测量）

```bash
python3 exp_extra.py --exp retest
```

- 1,026 次调用（171 条 × 2 个判定任务 × 3 次重复，每次不同随机种子）
- 目的：与文献报告的 41% 翻转率对照
- 产物：`runs/retest.jsonl`（171 条） → 用于 §7.9.5 的「翻转率 0% 与 2%」

### 4. 位置效应（金标证据在上下文中的位置）

```bash
python3 exp_extra.py --exp position
```

- 171 次调用（57 题 × 首部 / 中部 / 尾部）
- 证据集合按预算的 80% 裁剪，保证三种排列都不被截断
- 产物：`runs/position.jsonl`（171 条） → 用于 §7.9.5 的「74% / 68% / 72%，未达显著」

### 5. 新鲜度成本（补测 C_maint）

```bash
python3 exp_maint.py
```

- 60 次摘要调用（30 篇 × 2 次更新事件）
- 构造一次含 20 篇新增 + 10 篇修改的更新事件，测增量重建代价
- 产物：`runs/maint.jsonl`（2 条） → 用于 §7.9.3 的「7,416 token、日更 32.3 倍」

---

## 四、脚本清单

| 文件 | 作用 | 是否调模型 |
|---|---|---|
| `prepare_data.py` | 下载 HotpotQA + SQuAD，去重，6 项校验，生成 `data/*.jsonl` | 否 |
| `run_experiment.py` | 主实验：11 变体 × 2 预算，可断点续跑 | 是 |
| `score.py` | 离线评分：质量表、六类成本、$N^*$、探针、召回、失败分解 | 否 |
| `run_all.sh` | 一键脚本：5 道门禁 + 全量运行 + 自动备份 | 是 |
| `exp_extra.py` | 三个补充实验：`judge` / `retest` / `position` | 是 |
| `exp_maint.py` | 运维成本补测 | 是 |
| `../check_refs.py` | 参考文献核实（查 Crossref / arXiv），只读 bib，位在上级目录 | 否（联网） |
| `bench_speed.py` / `bench_budget.py` / `floor_check.py` | 早期速度与预算探针，已被主实验取代 | 是 |

---

## 五、数据文件

| 文件 | 行数 | 说明 |
|---|---|---|
| `data/corpus.jsonl` | 321 | 语料，取自 HotpotQA 干扰集与 SQuAD 验证集 |
| `data/queries.jsonl` | 57 | 查询，含金标答案与金标段落编号 |
| `data/summaries_good.jsonl` | 321 | 全文摘要层 |
| `data/summaries_bad.jsonl` | 321 | 导语摘要层（前 25%） |
| `data/summaries_poor.jsonl` | 321 | 截断摘要层（10 词） |
| `data/report.md` | 无 | 数据校验报告 |

---

## 六、复现顺序

```bash
cd experiment

# 1. 准备数据（只需一次）
python3 prepare_data.py

# 2. 主实验（约 28 分钟）
./run_all.sh --reset

# 3. 三个补充实验（约 12 分钟）
python3 exp_extra.py --exp judge
python3 exp_extra.py --exp retest
python3 exp_extra.py --exp position

# 4. 运维成本（约 2 分钟）
python3 exp_maint.py

# 5. 重新出报告（不调模型）
python3 score.py --paper
```

前置条件：本机 Ollama 在运行，且已 `ollama pull qwen2.5:1.5b-instruct`。
硬件： Mac mini M4（10 核 / 16 GB 统一内存）。

---

## 七、关键文件优先级

如果只能备份三样：

1. `runs/results.jsonl`：原始输出，不可再生
2. `data/summaries_*.jsonl`：摘要层，重建需 642 次调用
3. `runs/build_costs.jsonl`：构造成本记录

其余皆可由脚本重算。
