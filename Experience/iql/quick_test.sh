#!/bin/bash
# IQL+ROER 快速测试脚本

echo "================================================"
echo "IQL+ROER 快速测试"
echo "================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate roer

cd "$(dirname "$0")"

# 短训练测试（10k步）
echo ""
echo "测试1: IQL baseline (10k steps)..."
python train_iql_roer.py \
    --env_name=HalfCheetah-v2 \
    --seed=42 \
    --use_roer=False \
    --max_steps=10000 \
    --start_training=1000 \
    --eval_interval=2000 \
    --save_dir=./test_results/iql_baseline/

echo ""
echo "测试2: IQL+ROER (10k steps)..."
python train_iql_roer.py \
    --env_name=HalfCheetah-v2 \
    --seed=42 \
    --use_roer=True \
    --roer_temp=4.0 \
    --max_steps=10000 \
    --start_training=1000 \
    --eval_interval=2000 \
    --save_dir=./test_results/iql_roer/

echo ""
echo "================================================"
echo "快速测试完成！"
echo "检查 ./test_results/ 查看结果"
echo "================================================"

