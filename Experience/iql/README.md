# IQL + ROER 实现

将**ROER (Regularized Optimal Experience Replay)** 集成到 **IQL (Implicit Q-Learning)** 算法中。

## 📁 文件结构

```
iql/
├── iql_roer_learner.py      # IQL+ROER核心算法
├── train_iql_roer.py         # 训练脚本
├── replay_buffer_roer.py     # 带ROER优先级的replay buffer
├── common.py                 # 基础类和类型
├── policies.py               # 策略网络
├── temperature.py            # 温度参数
├── env_utils.py              # 环境工具
├── evaluation_utils.py       # 评估函数
├── run_iql_comparison.sh     # IQL vs IQL+ROER对比实验
├── quick_test.sh             # 快速测试
├── run_ant_experiment.sh     # Ant-v2专用实验
└── README.md                 # 本文件
```

## 🎯 核心创新点

### 1. IQL算法特点（2022）

- **Expectile Regression** - 更稳定的Q值估计
- **Advantage-Weighted BC** - 隐式策略学习，无需explicit actor
- **简单高效** - 比EDAC更简单，性能更好
- **SOTA性能** - 在D4RL上超越EDAC

### 2. ROER优先级机制

- **理论基础**: 基于占用优化推导
- **公式**: `w ∝ exp(TD_error / β)`
- **更新**: EMA平滑

### 3. IQL+ROER集成方案

**优势**：
- IQL的expectile regression更稳定，与ROER配合更好
- 使用V网络计算TD误差，符合Actor-Critic理论
- 无需ensemble，训练更快

**实现细节**：
1. **Q网络更新**: 使用ROER权重加权TD loss
2. **V网络更新**: 在expectile loss中加入ROER权重
3. **Actor更新**: 保持IQL的advantage-weighted BC
4. **优先级计算**: 使用V网络的TD误差

## 🚀 快速开始

### 安装依赖

```bash
# 使用原ROER环境
conda activate roer

# 确保环境正确
cd ~/Regularized-Optimal-Experience-Replay
```

### 快速测试

```bash
cd Experience/iql

# 添加执行权限
chmod +x *.sh

# 运行快速测试（10k步，约5-10分钟）
./quick_test.sh
```

### 单次训练

```bash
# IQL+ROER
python train_iql_roer.py \
    --env_name=HalfCheetah-v2 \
    --seed=42 \
    --use_roer=True \
    --roer_temp=4.0 \
    --roer_max_clip=50 \
    --roer_min_clip=10 \
    --expectile=0.7 \
    --iql_beta=3.0

# IQL baseline
python train_iql_roer.py \
    --env_name=HalfCheetah-v2 \
    --seed=42 \
    --use_roer=False \
    --expectile=0.7 \
    --iql_beta=3.0
```

## 📊 实验脚本

### 1. 对比实验

比较IQL和IQL+ROER的性能：

```bash
# 运行5个种子的对比实验
./run_iql_comparison.sh HalfCheetah-v2 5

# Ant-v2实验
./run_ant_experiment.sh 5
```

### 2. 查看结果

```bash
# TensorBoard
tensorboard --logdir ~/roer_output/results/

# 查看评估历史
cat ~/roer_output/results/iql_roer/*/eval_returns.txt
```

## 🔧 关键参数说明

### IQL参数

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `expectile` | Expectile参数 | 0.7 | 0.7-0.9 |
| `iql_beta` | Advantage weighting温度 | 3.0 | 1.0-10.0 |

### ROER参数

| 参数 | 说明 | MuJoCo默认 | DM Control默认 |
|------|------|-----------|---------------|
| `roer_temp` (β) | 温度参数 | 4.0 | 1.0 |
| `roer_max_clip` | 最大优先级裁剪 | 50 | 100 |
| `roer_min_clip` | 最小优先级裁剪 | 10 | 10 |
| `roer_per_beta` (λ) | EMA系数 | 0.01 | 0.01 |

### Ant-v2推荐参数

```bash
--expectile=0.7 \
--iql_beta=3.0 \
--roer_temp=1.0 \
--roer_max_clip=100.0 \
--roer_min_clip=10.0
```

## 📈 预期结果

### 性能对比

| 方法 | HalfCheetah-v2 | Ant-v2 | Hopper-v2 |
|------|---------------|--------|-----------|
| IQL Baseline | ~12500 | ~1800 | ~3200 |
| IQL+ROER | ~13000? | ~2000? | ~3400? |

**注意**: 这些是预期值，实际结果需要实验验证。

### IQL vs EDAC

| 特性 | IQL | EDAC |
|------|-----|------|
| 复杂度 | 简单 | 复杂（ensemble） |
| 训练速度 | 快 | 较慢 |
| D4RL性能 | 更好 | 好 |
| 在线RL | 好 | 很好 |
| 实现难度 | 低 | 中 |

## 🔬 算法细节

### IQL的核心思想

1. **Expectile Regression**
   ```python
   # 不对称的MSE loss
   weight = where(Q - V > 0, τ, 1-τ)
   loss = weight * (Q - V)²
   ```
   - τ=0.7时，更关注Q > V的情况
   - 避免Q值过估计

2. **Advantage-Weighted BC**
   ```python
   # 根据advantage加权行为克隆
   weight = exp(Advantage / β)
   loss = -weight * log π(a|s)
   ```
   - 只模仿高advantage的动作
   - 隐式学习策略，无需显式优化

### ROER集成到IQL

```python
# 1. Q网络更新（加ROER权重）
q_loss = mean(w * (q - target_q)²)

# 2. V网络更新（expectile loss + ROER权重）
v_loss = mean(w * expectile_loss(q - v, τ))

# 3. Actor更新（保持IQL原样）
actor_loss = -mean(exp(adv/β) * log_prob)

# 4. 优先级计算（V网络的TD误差）
td_error = r + γV(s') - V(s)
priority = exp(td_error / β_roer)
```

## 💡 调试建议

### 如果训练不稳定

1. **降低expectile**: `--expectile=0.6`
2. **增大iql_beta**: `--iql_beta=5.0`（更保守的策略）
3. **降低roer_temp**: `--roer_temp=2.0`
4. **缩小优先级范围**: `--roer_max_clip=20 --roer_min_clip=5`

### 如果性能没提升

1. **检查优先级是否生效**:
   - 查看TensorBoard的`priority/std`，应该 > 0
   
2. **尝试不同expectile**:
   ```bash
   for exp in 0.6 0.7 0.8 0.9; do
       python train_iql_roer.py --expectile=$exp
   done
   ```

3. **调整ROER温度**:
   ```bash
   for temp in 1.0 2.0 4.0 8.0; do
       python train_iql_roer.py --roer_temp=$temp
   done
   ```

## 🎓 毕设建议

### 为什么IQL+ROER是好选择

1. **IQL更新（2022）** - 比EDAC（2021）更新
2. **性能更强** - D4RL上超越EDAC
3. **实现简单** - 更容易调试和理解
4. **创新明确** - ROER+IQL是新组合

### 论文写作建议

**IQL介绍**：
> "我们选择IQL (Implicit Q-Learning) 作为基线算法。IQL是2022年提出的离线强化学习方法，通过expectile regression和advantage-weighted behavioral cloning，在D4RL基准测试上取得了SOTA性能。相比需要ensemble的EDAC，IQL实现更简单，训练更快。"

**集成方案**：
> "我们将ROER的优先级机制集成到IQL的三个训练步骤中：(1) Q网络更新使用ROER权重加权TD loss；(2) V网络更新在expectile loss中引入ROER权重；(3) 使用V网络的TD误差计算ROER优先级。这种集成保留了IQL的expectile regression优势，同时引入了ROER的样本选择机制。"

### 实验设计

1. **基础对比**
   ```bash
   ./run_iql_comparison.sh HalfCheetah-v2 5
   ./run_iql_comparison.sh Ant-v2 5
   ./run_iql_comparison.sh Hopper-v2 5
   ```

2. **消融实验**
   - 不同expectile的影响
   - 不同roer_temp的影响
   - 有/无ROER的对比

3. **与EDAC+ROER对比**
   - IQL+ROER vs EDAC+ROER
   - 训练速度对比
   - 性能对比

## 📊 结果可视化

使用共享的可视化工具：

```bash
# 从edac_roer复制可视化脚本
cp ../edac_roer/plot_results.py .

# 生成对比图
python plot_results.py \
    --baseline './results/iql_baseline/*/eval_returns.txt' \
    --roer './results/iql_roer/*/eval_returns.txt' \
    --title 'IQL vs IQL+ROER (HalfCheetah-v2)' \
    --save_dir './plots/'
```

## 🔗 参考资料

- [IQL论文](https://arxiv.org/abs/2110.06169) - Kostrikov et al., 2022
- [ROER论文](https://arxiv.org/abs/2407.03995) - Li et al., 2024
- [IQL官方代码](https://github.com/ikostrikov/implicit_q_learning)
- [原ROER代码](https://github.com/XavierChanglingLi/Regularized-Optimal-Experience-Replay)

## 🎉 总结

**IQL+ROER的优势**：

✅ **更简单** - 无需ensemble，代码少
✅ **更快** - 训练速度快
✅ **性能更好** - IQL在D4RL上超越EDAC
✅ **更稳定** - Expectile regression比TD更稳定
✅ **创新性强** - ROER+IQL的组合尚未被探索

您的毕设有很好的研究价值！加油！🚀

