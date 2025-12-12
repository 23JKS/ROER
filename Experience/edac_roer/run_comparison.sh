#!/bin/bash
# EDAC vs EDAC+ROER 对比实验脚本

ENV_NAME=${1:-HalfCheetah-v2}
NUM_SEEDS=${2:-5}

echo "================================================"
echo "对比实验: EDAC vs EDAC+ROER"
echo "环境: $ENV_NAME"
echo "种子数量: $NUM_SEEDS"
echo "================================================"

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate roer

cd "$(dirname "$0")"

# 运行EDAC baseline（不使用ROER）
echo ""
echo "运行 EDAC Baseline..."
for seed in $(seq 42 $((41 + NUM_SEEDS))); do
    echo "  - Seed $seed"
    python train_edac_roer.py \
        --env_name=$ENV_NAME \
        --seed=$seed \
        --use_roer=False \
        --save_dir=./results/edac_baseline/ &
    sleep 2
done

# 等待baseline完成
wait
echo "EDAC Baseline 完成!"

# 运行EDAC+ROER
echo ""
echo "运行 EDAC+ROER..."
for seed in $(seq 42 $((41 + NUM_SEEDS))); do
    echo "  - Seed $seed"
    python train_edac_roer.py \
        --env_name=$ENV_NAME \
        --seed=$seed \
        --use_roer=True \
        --roer_temp=4.0 \
        --roer_max_clip=50.0 \
        --roer_min_clip=10.0 \
        --save_dir=./results/edac_roer/ &
    sleep 2
done

# 等待完成
wait
echo "EDAC+ROER 完成!"

# 分析结果
echo ""
echo "================================================"
echo "分析结果..."
echo "================================================"

python << 'EOF'
import numpy as np
import glob
import os

def analyze_results(pattern, name):
    files = glob.glob(pattern)
    if not files:
        print(f"{name}: 未找到结果文件")
        return
    
    final_scores = []
    for f in files:
        try:
            data = np.loadtxt(f)
            if len(data) > 0:
                final_scores.append(data[-1, 1])
        except:
            pass
    
    if final_scores:
        print(f"\n{name}:")
        print(f"  样本数: {len(final_scores)}")
        print(f"  平均值: {np.mean(final_scores):.1f} ± {np.std(final_scores):.1f}")
        print(f"  最小值: {np.min(final_scores):.1f}")
        print(f"  最大值: {np.max(final_scores):.1f}")
        print(f"  中位数: {np.median(final_scores):.1f}")
    else:
        print(f"{name}: 无有效数据")

analyze_results("./results/edac_baseline/*/eval_returns.txt", "EDAC Baseline")
analyze_results("./results/edac_roer/*/eval_returns.txt", "EDAC+ROER")
EOF

echo ""
echo "================================================"
echo "对比实验完成！"
echo "================================================"

