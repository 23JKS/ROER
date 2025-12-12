#!/bin/bash
# Comparison script for different divergence types in ROER
# Compares: JS divergence, KL divergence (OER), and PER

ENV_NAME=${1:-"HalfCheetah-v2"}
SEED=${2:-42}
MAX_STEPS=${3:-1000000}

echo "=========================================="
echo "Comparing Divergence Methods on $ENV_NAME"
echo "Seed: $SEED, Max Steps: $MAX_STEPS"
echo "=========================================="

# Jensen-Shannon Divergence
echo ""
echo "Running JS Divergence..."
python train_online.py \
    --env_name=$ENV_NAME \
    --seed=$SEED \
    --max_steps=$MAX_STEPS \
    --per=True \
    --per_type=JS \
    --temp=4 \
    --min_clip=0.5 \
    --max_clip=50 \
    --gumbel_max_clip=7 \
    --js_min_weight=0.5 \
    --js_max_weight=2.0 \
    --log_loss=True \
    --update_scheme=exp \
    --save_dir=./results/js_divergence \
    --wandb_project_name=roer_comparison

# KL Divergence (OER)
echo ""
echo "Running KL Divergence (OER)..."
python train_online.py \
    --env_name=$ENV_NAME \
    --seed=$SEED \
    --max_steps=$MAX_STEPS \
    --per=True \
    --per_type=OER \
    --temp=4 \
    --min_clip=10 \
    --max_clip=50 \
    --gumbel_max_clip=7 \
    --log_loss=True \
    --update_scheme=exp \
    --save_dir=./results/kl_divergence \
    --wandb_project_name=roer_comparison

# Prioritized Experience Replay (PER)
echo ""
echo "Running PER..."
python train_online.py \
    --env_name=$ENV_NAME \
    --seed=$SEED \
    --max_steps=$MAX_STEPS \
    --per=True \
    --per_type=PER \
    --per_alpha=0.6 \
    --save_dir=./results/per \
    --wandb_project_name=roer_comparison

# Uniform (Baseline)
echo ""
echo "Running Uniform Baseline..."
python train_online.py \
    --env_name=$ENV_NAME \
    --seed=$SEED \
    --max_steps=$MAX_STEPS \
    --per=False \
    --save_dir=./results/uniform \
    --wandb_project_name=roer_comparison

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

