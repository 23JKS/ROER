#!/bin/bash
# 多种子训练脚本 - 用于统计结果
# 论文使用 20 个随机种子 (42-61)

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate roer

ENV_NAME=${1:-Hopper-v2}  # 默认环境
echo "训练环境: $ENV_NAME"
echo "使用 20 个随机种子 (42-61)"

# MuJoCo 环境的默认超参数
if [[ "$ENV_NAME" == *"v2"* ]]; then
    TEMP=4
    MAX_CLIP=50
    MIN_CLIP=10
else
    # DM Control 环境
    TEMP=1
    MAX_CLIP=50
    MIN_CLIP=1
fi

# 训练所有种子
for seed in 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61; do
    echo "========================================="
    echo "训练种子: $seed"
    echo "========================================="
    
    python train_online.py \
        --env_name=$ENV_NAME \
        --max_steps=1000000 \
        --start_training=10000 \
        --eval_interval=5000 \
        --eval_episodes=10 \
        --seed=$seed \
        --per=True \
        --per_type=OER \
        --per_beta=0.01 \
        --gumbel_max_clip=7 \
        --temp=$TEMP \
        --max_clip=$MAX_CLIP \
        --min_clip=$MIN_CLIP \
        --update_scheme=avg \
        --std_normalize=True \
        --track=False
    
    echo "种子 $seed 训练完成"
    echo ""
done

echo "所有种子训练完成！"
echo "结果保存在 ~/roer_output/tmp/evaluation/"


