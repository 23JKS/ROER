#!/bin/bash
# ROER 复现脚本 - MuJoCo 环境
# 根据论文 Table 5 的超参数设置

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 如果有 GPU，设置 GPU ID

# MuJoCo 环境的超参数（根据论文 Table 5 和消融研究）
# 格式: per_beta, gumbel_max_clip, temp, max_clip, min_clip

# Ant-v2
echo "训练 Ant-v2..."
python train_online.py \
    --env_name=Ant-v2 \
    --max_steps=1000000 \
    --start_training=10000 \
    --seed=42 \
    --per_beta=0.01 \
    --gumbel_max_clip=7 \
    --temp=4 \
    --max_clip=50 \
    --min_clip=10 \
    --per_type=OER \
    --update_scheme=avg \
    --std_normalize=True \
    --track=False

# HalfCheetah-v2
echo "训练 HalfCheetah-v2..."
python train_online.py \
    --env_name=HalfCheetah-v2 \
    --max_steps=1000000 \
    --start_training=10000 \
    --seed=42 \
    --per_beta=0.01 \
    --gumbel_max_clip=7 \
    --temp=4 \
    --max_clip=50 \
    --min_clip=10 \
    --per_type=OER \
    --update_scheme=avg \
    --std_normalize=True \
    --track=False

# Hopper-v2
echo "训练 Hopper-v2..."
python train_online.py \
    --env_name=Hopper-v2 \
    --max_steps=1000000 \
    --start_training=10000 \
    --seed=42 \
    --per_beta=0.01 \
    --gumbel_max_clip=7 \
    --temp=4 \
    --max_clip=50 \
    --min_clip=10 \
    --per_type=OER \
    --update_scheme=avg \
    --std_normalize=True \
    --track=False

# Humanoid-v2
echo "训练 Humanoid-v2..."
python train_online.py \
    --env_name=Humanoid-v2 \
    --max_steps=1000000 \
    --start_training=10000 \
    --seed=42 \
    --per_beta=0.01 \
    --gumbel_max_clip=7 \
    --temp=4 \
    --max_clip=50 \
    --min_clip=10 \
    --per_type=OER \
    --update_scheme=avg \
    --std_normalize=True \
    --track=False

# Walker2d-v2
echo "训练 Walker2d-v2..."
python train_online.py \
    --env_name=Walker2d-v2 \
    --max_steps=1000000 \
    --start_training=10000 \
    --seed=42 \
    --per_beta=0.01 \
    --gumbel_max_clip=7 \
    --temp=4 \
    --max_clip=50 \
    --min_clip=10 \
    --per_type=OER \
    --update_scheme=avg \
    --std_normalize=True \
    --track=False

echo "所有 MuJoCo 环境训练完成！"


