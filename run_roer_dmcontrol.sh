#!/bin/bash
# ROER 复现脚本 - DM Control 环境
# 根据论文 Table 5 的超参数设置

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 如果有 GPU，设置 GPU ID

# DM Control 环境的超参数（根据论文 Table 5）
# 注意：DM Control 通常使用 temp=1 而不是 temp=4

# Fish-Swim
echo "训练 Fish-Swim..."
python train_online.py \
    --env_name=fish-swim \
    --max_steps=1000000 \
    --start_training=10000 \
    --seed=42 \
    --per_beta=0.01 \
    --gumbel_max_clip=7 \
    --temp=1 \
    --max_clip=50 \
    --min_clip=1 \
    --per_type=OER \
    --update_scheme=avg \
    --std_normalize=True \
    --track=False

# Hopper-stand
echo "训练 Hopper-stand..."
python train_online.py \
    --env_name=hopper-stand \
    --max_steps=1000000 \
    --start_training=10000 \
    --seed=42 \
    --per_beta=0.01 \
    --gumbel_max_clip=7 \
    --temp=1 \
    --max_clip=100 \
    --min_clip=1 \
    --per_type=OER \
    --update_scheme=avg \
    --std_normalize=True \
    --track=False

# Humanoid-run
echo "训练 Humanoid-run..."
python train_online.py \
    --env_name=humanoid-run \
    --max_steps=1000000 \
    --start_training=10000 \
    --seed=42 \
    --per_beta=0.01 \
    --gumbel_max_clip=7 \
    --temp=1 \
    --max_clip=100 \
    --min_clip=1 \
    --per_type=OER \
    --update_scheme=avg \
    --std_normalize=True \
    --track=False

# Humanoid-stand
echo "训练 Humanoid-stand..."
python train_online.py \
    --env_name=humanoid-stand \
    --max_steps=1000000 \
    --start_training=10000 \
    --seed=42 \
    --per_beta=0.01 \
    --gumbel_max_clip=7 \
    --temp=1 \
    --max_clip=100 \
    --min_clip=1 \
    --per_type=OER \
    --update_scheme=avg \
    --std_normalize=True \
    --track=False

# Quadruped-run
echo "训练 Quadruped-run..."
python train_online.py \
    --env_name=quadruped-run \
    --max_steps=1000000 \
    --start_training=10000 \
    --seed=42 \
    --per_beta=0.01 \
    --gumbel_max_clip=7 \
    --temp=1 \
    --max_clip=100 \
    --min_clip=10 \
    --per_type=OER \
    --update_scheme=avg \
    --std_normalize=True \
    --track=False

echo "所有 DM Control 环境训练完成！"


