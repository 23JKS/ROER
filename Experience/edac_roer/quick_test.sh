#!/bin/bash
# 快速测试脚本 - 用于验证实现是否正确

echo "================================================"
echo "EDAC+ROER 快速测试"
echo "================================================"

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate roer

cd "$(dirname "$0")"

# 短训练测试（10k步）
echo ""
echo "测试1: EDAC baseline (10k steps)..."
python train_edac_roer.py \
    --env_name=HalfCheetah-v2 \
    --seed=42 \
    --use_roer=False \
    --max_steps=10000 \
    --start_training=1000 \
    --eval_interval=2000 \
    --save_dir=./test_results/edac_baseline/

echo ""
echo "测试2: EDAC+ROER (10k steps)..."
python train_edac_roer.py \
    --env_name=HalfCheetah-v2 \
    --seed=42 \
    --use_roer=True \
    --roer_temp=4.0 \
    --max_steps=10000 \
    --start_training=1000 \
    --eval_interval=2000 \
    --save_dir=./test_results/edac_roer/

echo ""
echo "================================================"
echo "快速测试完成！"
echo "检查 ./test_results/ 查看结果"
echo "================================================"

