#!/bin/bash
# EDAC+ROER 超参数扫描脚本

ENV_NAME=${1:-HalfCheetah-v2}
SEED=${2:-42}

echo "================================================"
echo "超参数扫描: EDAC+ROER"
echo "环境: $ENV_NAME"
echo "种子: $SEED"
echo "================================================"

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate roer

cd "$(dirname "$0")"

# 1. 扫描 temperature (β)
echo ""
echo "扫描 temperature (β)..."
for temp in 0.5 1.0 2.0 4.0 8.0; do
    echo "  - temp=$temp"
    python train_edac_roer.py \
        --env_name=$ENV_NAME \
        --seed=$SEED \
        --use_roer=True \
        --roer_temp=$temp \
        --roer_max_clip=50 \
        --roer_min_clip=10 \
        --max_steps=500000 \
        --save_dir=./results/sweep_temp/temp_${temp}/ &
    sleep 2
done
wait

# 2. 扫描 max_clip
echo ""
echo "扫描 max_clip..."
for max_clip in 20 50 100 200; do
    echo "  - max_clip=$max_clip"
    python train_edac_roer.py \
        --env_name=$ENV_NAME \
        --seed=$SEED \
        --use_roer=True \
        --roer_temp=4.0 \
        --roer_max_clip=$max_clip \
        --roer_min_clip=10 \
        --max_steps=500000 \
        --save_dir=./results/sweep_maxclip/maxclip_${max_clip}/ &
    sleep 2
done
wait

# 3. 扫描 min_clip
echo ""
echo "扫描 min_clip..."
for min_clip in 1 5 10 20; do
    echo "  - min_clip=$min_clip"
    python train_edac_roer.py \
        --env_name=$ENV_NAME \
        --seed=$SEED \
        --use_roer=True \
        --roer_temp=4.0 \
        --roer_max_clip=50 \
        --roer_min_clip=$min_clip \
        --max_steps=500000 \
        --save_dir=./results/sweep_minclip/minclip_${min_clip}/ &
    sleep 2
done
wait

# 4. 扫描 diversity_coef
echo ""
echo "扫描 diversity_coef (EDAC)..."
for div_coef in 0.0 0.05 0.1 0.2 0.5; do
    echo "  - diversity_coef=$div_coef"
    python train_edac_roer.py \
        --env_name=$ENV_NAME \
        --seed=$SEED \
        --use_roer=True \
        --roer_temp=4.0 \
        --diversity_coef=$div_coef \
        --max_steps=500000 \
        --save_dir=./results/sweep_diversity/div_${div_coef}/ &
    sleep 2
done
wait

echo ""
echo "================================================"
echo "超参数扫描完成！"
echo "================================================"

# 分析结果
python << 'EOF'
import numpy as np
import glob
import os

def analyze_sweep(pattern, param_name):
    print(f"\n{param_name} 扫描结果:")
    print("-" * 50)
    
    results = {}
    for f in glob.glob(pattern):
        # 从路径中提取参数值
        param_val = os.path.basename(os.path.dirname(f))
        try:
            data = np.loadtxt(f)
            if len(data) > 0:
                final_score = data[-1, 1]
                results[param_val] = final_score
        except:
            pass
    
    if results:
        for k in sorted(results.keys()):
            print(f"  {k}: {results[k]:.1f}")
    else:
        print("  无有效数据")

analyze_sweep("./results/sweep_temp/*/eval_returns.txt", "Temperature")
analyze_sweep("./results/sweep_maxclip/*/eval_returns.txt", "Max Clip")
analyze_sweep("./results/sweep_minclip/*/eval_returns.txt", "Min Clip")
analyze_sweep("./results/sweep_diversity/*/eval_returns.txt", "Diversity Coef")
EOF

