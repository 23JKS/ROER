#!/bin/bash
# IQL vs IQL+ROER 对比实验脚本

ENV_NAME=${1:-HalfCheetah-v2}
NUM_SEEDS=${2:-5}

echo "================================================"
echo "对比实验: IQL vs IQL+ROER"
echo "环境: $ENV_NAME"
echo "种子数量: $NUM_SEEDS"
echo "================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate roer

cd "$(dirname "$0")"

# IQL baseline
echo ""
echo "运行 IQL Baseline..."
for seed in $(seq 42 $((41 + NUM_SEEDS))); do
    echo "  - Seed $seed"
    python train_iql_roer.py \
        --env_name=$ENV_NAME \
        --seed=$seed \
        --use_roer=False \
        --save_dir=./results/iql_baseline/ &
    sleep 2
done

wait
echo "IQL Baseline 完成!"

# IQL+ROER
echo ""
echo "运行 IQL+ROER..."
for seed in $(seq 42 $((41 + NUM_SEEDS))); do
    echo "  - Seed $seed"
    python train_iql_roer.py \
        --env_name=$ENV_NAME \
        --seed=$seed \
        --use_roer=True \
        --roer_temp=4.0 \
        --roer_max_clip=50.0 \
        --roer_min_clip=10.0 \
        --save_dir=./results/iql_roer/ &
    sleep 2
done

wait
echo "IQL+ROER 完成!"

echo ""
echo "================================================"
echo "结果分析"
echo "================================================"

python << 'EOF'
import numpy as np
import glob

def analyze_method(pattern, name):
    files = glob.glob(pattern)
    
    if not files:
        print(f"\n{name}: 未找到结果文件")
        return None
    
    final_scores = []
    
    for f in files:
        try:
            data = np.loadtxt(f)
            if len(data) > 0:
                final_scores.append(data[-1, 1])
        except:
            pass
    
    if not final_scores:
        print(f"\n{name}: 无有效数据")
        return None
    
    print(f"\n{name}:")
    print(f"  种子数: {len(final_scores)}")
    print(f"  最终性能: {np.mean(final_scores):.1f} ± {np.std(final_scores):.1f}")
    print(f"  范围: [{np.min(final_scores):.1f}, {np.max(final_scores):.1f}]")
    print(f"  中位数: {np.median(final_scores):.1f}")
    
    return final_scores

baseline_scores = analyze_method(
    "./results/iql_baseline/*/eval_returns.txt",
    "IQL Baseline"
)

roer_scores = analyze_method(
    "./results/iql_roer/*/eval_returns.txt",
    "IQL+ROER"
)

if baseline_scores is not None and roer_scores is not None:
    improvement = (np.mean(roer_scores) - np.mean(baseline_scores)) / np.mean(baseline_scores) * 100
    print(f"\n相对提升: {improvement:+.1f}%")

EOF

echo ""
echo "================================================"
echo "实验完成！"
echo "================================================"

