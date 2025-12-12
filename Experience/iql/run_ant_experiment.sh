#!/bin/bash
# Ant-v2 IQL+ROER 完整实验脚本

NUM_SEEDS=${1:-5}

echo "================================================"
echo "Ant-v2 IQL+ROER 实验"
echo "种子数量: $NUM_SEEDS"
echo "================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate roer

cd "$(dirname "$0")"

# IQL+ROER（使用Ant-v2的参数）
echo ""
echo "运行 IQL+ROER (Ant-v2)..."
echo "参数: temp=1, max_clip=100, min_clip=10"

for seed in $(seq 42 $((41 + NUM_SEEDS))); do
    echo "  - IQL+ROER Seed $seed"
    python train_iql_roer.py \
        --env_name=Ant-v2 \
        --seed=$seed \
        --use_roer=True \
        --roer_temp=1.0 \
        --roer_max_clip=100.0 \
        --roer_min_clip=10.0 \
        --roer_per_beta=0.01 \
        --expectile=0.7 \
        --iql_beta=3.0 \
        --max_steps=1000000 \
        --save_dir=./results/ant_iql_roer/ \
        --tqdm=True &
    
    sleep 5
done

wait
echo "IQL+ROER 完成!"

# IQL baseline
echo ""
echo "运行 IQL Baseline (Ant-v2)..."

for seed in $(seq 42 $((41 + NUM_SEEDS))); do
    echo "  - IQL Baseline Seed $seed"
    python train_iql_roer.py \
        --env_name=Ant-v2 \
        --seed=$seed \
        --use_roer=False \
        --expectile=0.7 \
        --iql_beta=3.0 \
        --max_steps=1000000 \
        --save_dir=./results/ant_iql_baseline/ \
        --tqdm=True &
    
    sleep 5
done

wait
echo "IQL Baseline 完成!"

echo ""
echo "================================================"
echo "实验完成！查看结果..."
echo "================================================"

