# ROER 论文复现指南

本文档提供复现论文 "ROER: Regularized Optimal Experience Replay" 中 SAC+ROER 结果的详细指南。

## 论文信息

- **标题**: ROER: Regularized Optimal Experience Replay
- **作者**: Changling Li, Pulkit Agrawal, Zhang-Wei Hong, Joni Pajarinen
- **代码仓库**: https://github.com/XavierChanglingLi/Regularized-Optimal-Experience-Replay

## 环境设置

### 1. Conda 环境

确保已激活 `roer` 环境：

```bash
conda activate roer
```

### 2. 超参数说明

根据论文 Table 5 和消融研究，ROER 的关键超参数包括：

- **per_beta (λ)**: 0.01 - 收敛率，控制优先级更新的平滑度
- **gumbel_max_clip**: 7 - Gumbel 损失的裁剪值
- **temp (β)**: 
  - MuJoCo: 4（大多数任务）
  - DM Control: 1（大多数任务）
- **max_clip**: 
  - MuJoCo: 50（大多数任务）
  - DM Control: 50-100（根据任务）
- **min_clip**: 
  - MuJoCo: 10（大多数任务）
  - DM Control: 1（大多数任务，除了 quadruped-run 使用 10）

## 快速开始

### 单个环境训练

#### MuJoCo 环境示例（Hopper-v2）

```bash
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
```

#### DM Control 环境示例（Hopper-stand）

```bash
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
```

### 批量训练

使用提供的脚本批量训练所有环境：

```bash
# MuJoCo 环境
chmod +x run_roer_mujoco.sh
./run_roer_mujoco.sh

# DM Control 环境
chmod +x run_roer_dmcontrol.sh
./run_roer_dmcontrol.sh
```

## 论文中的超参数设置

### MuJoCo 环境（Table 5）

| 环境 | per_beta | gumbel_max_clip | temp | max_clip | min_clip |
|------|----------|-----------------|------|----------|----------|
| Ant-v2 | 0.01 | 7 | 4 | 50 | 10 |
| HalfCheetah-v2 | 0.01 | 7 | 4 | 50 | 10 |
| Hopper-v2 | 0.01 | 7 | 4 | 50 | 10 |
| Humanoid-v2 | 0.01 | 7 | 4 | 50 | 10 |
| Walker2d-v2 | 0.01 | 7 | 4 | 50 | 10 |

### DM Control 环境（Table 5）

| 环境 | per_beta | gumbel_max_clip | temp | max_clip | min_clip |
|------|----------|-----------------|------|----------|----------|
| Fish-Swim | 0.01 | 7 | 1 | 50 | 1 |
| Hopper-stand | 0.01 | 7 | 1 | 100 | 1 |
| Humanoid-run | 0.01 | 7 | 1 | 100 | 1 |
| Humanoid-stand | 0.01 | 7 | 1 | 100 | 1 |
| Quadruped-run | 0.01 | 7 | 1 | 100 | 10 |

## 训练设置

### 基本参数

- **训练步数**: 1,000,000 steps（论文中主要结果）
- **开始训练**: 10,000 steps（收集初始经验）
- **随机种子**: 论文使用 20 个随机种子进行统计
- **评估间隔**: 5,000 steps（默认）
- **评估回合数**: 10 episodes（默认）

### 多种子训练

为了复现论文中的统计结果，建议运行多个随机种子：

```bash
for seed in 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61; do
    python train_online.py \
        --env_name=Hopper-v2 \
        --max_steps=1000000 \
        --start_training=10000 \
        --seed=$seed \
        --per_beta=0.01 \
        --gumbel_max_clip=7 \
        --temp=4 \
        --max_clip=50 \
        --min_clip=10 \
        --per_type=OER \
        --update_scheme=avg \
        --std_normalize=True \
        --track=False
done
```

## 结果保存

训练结果保存在 `~/roer_output/` 目录下：

- **日志**: `~/roer_output/tmp/{timestamp}/tb/`
- **评估结果**: `~/roer_output/tmp/evaluation/`
- **模型检查点**: `~/roer_output/tmp/{timestamp}/{seed}/iter_{step}`

## 论文结果对比

根据论文，ROER 在以下任务上优于基线：

- **MuJoCo**: Ant-v2, HalfCheetah-v2, Hopper-v2, Humanoid-v2
- **DM Control**: Fish-Swim, Hopper-stand

## 注意事项

1. **训练时间**: 1M steps 的训练可能需要数小时到数天，取决于硬件配置
2. **GPU**: 虽然代码支持 CPU，但使用 GPU 会显著加速训练
3. **内存**: 确保有足够的 RAM（建议至少 16GB）
4. **MuJoCo**: 确保 MuJoCo 已正确安装并配置了 `LD_LIBRARY_PATH`

## 故障排除

如果遇到问题，请检查：

1. 环境是否正确激活：`conda activate roer`
2. MuJoCo 路径是否正确：`echo $LD_LIBRARY_PATH`
3. 依赖是否完整：`pip list | grep -E "jax|flax|gym|mujoco"`
4. 输出目录权限：确保 `~/roer_output/` 可写

## 参考

- 论文: Li et al. (2024). ROER: Regularized Optimal Experience Replay
- 代码: https://github.com/XavierChanglingLi/Regularized-Optimal-Experience-Replay

