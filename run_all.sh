#!/usr/bin/env bash
# 一键跑完全部实验，并直接产出论文可用的数字块
#
# 覆盖范围（与论文的对应关系）
#   四种载体         s0 weights / s1 context / s2,s3,s5,s6 index / s4 actions
#   六类成本         分阶段计时，见 score.py 的「六类成本明细」
#   五分量向量       ΔQ / ΔC_build / ΔC_query / ΔL / ΔTrust，见「可信三分量」
#   四级归因探针     P1=s6  P2=s5  P3=s1  P4=s7，见「四级探针诊断」
#   成本方差         E / P95 / Var 与失败查询成本分布
#
# 用法
#   ./run_all.sh --check     只做门禁与静态检查，零模型调用
#   ./run_all.sh --smoke     3 题冒烟
#   ./run_all.sh             全量（断点续跑）
#   ./run_all.sh --reset     先备份并清空旧结果，再全量
#
set -u
cd "$(dirname "$0")" || exit 1

BUDGETS="1024 4096"
MODE="full"
RESET=0
for a in "$@"; do
  case "$a" in
    --smoke) MODE="smoke" ;;
    --check) MODE="check" ;;
    --reset) RESET=1 ;;
    *) echo "未知参数 $a。可用 --check --smoke --reset"; exit 2 ;;
  esac
done

mkdir -p runs
LOG="runs/log_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$LOG") 2>&1

echo "=============================================================="
echo " 一键实验  模式=$MODE  预算=$BUDGETS  重置=$RESET"
echo " 日志 $LOG"
echo "=============================================================="
echo

# ---------------- 门禁 1：Python 与语法 ----------------
command -v python3 >/dev/null 2>&1 || { echo "✗ 找不到 python3"; exit 1; }
echo "✓ $(python3 --version 2>&1)"
for f in prepare_data.py run_experiment.py score.py; do
  python3 -m py_compile "$f" 2>/dev/null || { echo "✗ $f 语法错误"; exit 1; }
done
echo "✓ 三个脚本语法通过"

# ---------------- 门禁 2：Ollama 与模型 ----------------
curl -s --max-time 5 http://localhost:11434/api/tags >/dev/null 2>&1 \
  || { echo "✗ 连不上 Ollama。先另开终端运行： ollama serve"; exit 1; }
echo "✓ Ollama 在运行"
MODEL="${MODEL:-qwen2.5:1.5b-instruct}"
curl -s --max-time 5 http://localhost:11434/api/tags | grep -q "${MODEL%%:*}" \
  || { echo "✗ 本机没有 $MODEL，先运行： ollama pull $MODEL"; exit 1; }
echo "✓ 模型 $MODEL"

# ---------------- 门禁 3：数据 ----------------
for f in data/corpus.jsonl data/queries.jsonl; do
  [ -s "$f" ] || { echo "✗ 缺少 $f，先运行： python3 prepare_data.py"; exit 1; }
done
QN=$(wc -l < data/queries.jsonl | tr -d ' ')
DN=$(wc -l < data/corpus.jsonl | tr -d ' ')
echo "✓ 数据  $QN 条查询 / $DN 段语料"

if [ -f data/report.md ] && grep -q "FAIL" data/report.md; then
  echo "✗ data/report.md 有 FAIL，先修数据"; grep FAIL data/report.md; exit 1
fi
echo "✓ 数据校验报告无 FAIL"

# ---------------- 门禁 4：旧记录字段版本 ----------------
if [ -s runs/results.jsonl ]; then
  STALE=$(python3 -c "
import json
n=0
for line in open('runs/results.jsonl'):
    line=line.strip()
    if line and 't_ret_s' not in json.loads(line): n+=1
print(n)")
  TOTAL=$(wc -l < runs/results.jsonl | tr -d ' ')
  echo "✓ 已有结果 $TOTAL 条，其中缺分阶段字段 $STALE 条"
  if [ "$STALE" != "0" ] && [ "$RESET" != "1" ]; then
    echo
    echo "⚠ 旧记录缺 t_ret_s 等分阶段字段，会让六类成本表出现 0。"
    echo "  建议 ./run_all.sh --reset 先备份再清空，保证论文数字同源。"
    echo
  fi
fi

if [ "$RESET" = "1" ] && [ -s runs/results.jsonl ]; then
  BAK="runs/results_$(date +%Y%m%d_%H%M).jsonl.bak"
  cp runs/results.jsonl "$BAK"
  rm -f runs/results.jsonl
  echo "✓ 已备份到 $BAK 并清空结果文件"
fi

# ---------------- 门禁 5：零调用自检 ----------------
echo
echo "--- 配置自检（不调用模型）---"
python3 run_experiment.py --dry-run --budgets $BUDGETS || { echo "✗ dry-run 未通过"; exit 1; }

if [ "$MODE" = "check" ]; then
  echo
  echo "门禁全部通过。去掉 --check 开始正式跑。"
  echo "预计：11 变体 x 2 档 x $QN 题 = $((11*2*QN)) 次生成，另加重排与摘要构建。"
  exit 0
fi

# ---------------- 冒烟 ----------------
if [ "$MODE" = "smoke" ]; then
  echo
  echo "--- 冒烟：3 题，全部 11 变体 ---"
  python3 run_experiment.py --limit 3 --budgets 1024 || { echo "✗ 冒烟失败"; exit 1; }
  echo
  python3 score.py | head -60
  echo
  echo "冒烟完成。确认无误后去掉 --smoke 跑全量。"
  exit 0
fi

# ---------------- 全量 ----------------
echo
echo "--- 全量开始：11 变体 x 2 档 x $QN 题 ---"
echo "  s3_poor 的摘要层首次运行会构建（约 $DN 次调用）"
echo
START=$(date +%s)
caffeinate -i python3 run_experiment.py --budgets $BUDGETS
RC=$?
END=$(date +%s)
echo
if [ $RC -ne 0 ]; then
  echo "✗ 主程序退出码 $RC。日志 $LOG"
  echo "  已完成的记录不会丢，重跑本脚本从断点继续。"
  exit $RC
fi
echo "✓ 主程序完成，耗时 $(( (END-START)/60 )) 分钟"

# ---------------- 出结果 ----------------
echo
echo "--- 评分与论文块 ---"
python3 score.py --paper

echo
echo "=============================================================="
echo " 产物"
echo "   runs/results.jsonl        原始输出（不可再生，请备份）"
echo "   runs/build_costs.jsonl    构造成本"
echo "   runs/results_summary.csv  作图用"
echo "   $LOG                      本次日志"
echo "=============================================================="
cp runs/results.jsonl "runs/results_$(date +%Y%m%d_%H%M).jsonl.bak" 2>/dev/null \
  && echo "已自动备份 runs/results_*.jsonl.bak"
