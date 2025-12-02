#!/bin/bash
# 复现 Hopper-v2 的 ROER 结果
# 根据论文 Table 5 和消融研究

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate roer

# 论文中的超参数设置（MuJoCo 环境）
# per_beta=0.01, gumbel_max_clip=7, temp=4, max_clip=50, min_clip=10

echo "开始训练 Hopper-v2 (ROER)..."

python train_online.py \
    --env_name=Hopper-v2 \
    --max_steps=1000000 \
    --start_training=10000 \
    --eval_interval=5000 \
    --eval_episodes=10 \
    --seed=42 \
    --per=True \
    --per_type=OER \
    --per_beta=0.01 \
    --gumbel_max_clip=7 \
    --temp=4 \
    --max_clip=50 \
    --min_clip=10 \
    --update_scheme=avg \
    --std_normalize=True \
    --track=False

echo "训练完成！结果保存在 ~/roer_output/"

