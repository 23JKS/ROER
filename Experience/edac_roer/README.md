# EDAC + ROER 实现

将**ROER (Regularized Optimal Experience Replay)** 集成到 **EDAC (Ensemble-Diversified Actor-Critic)** 算法中。

## 📁 文件结构

```
edac_roer/
├── edac_roer_learner.py      # 核心算法实现
├── replay_buffer_roer.py     # 带ROER优先级的replay buffer  
├── train_edac_roer.py        # 训练脚本
├── configs/
│   └── edac_roer_default.py  # 默认配置
├── run_comparison.sh         # EDAC vs EDAC+ROER对比实验
├── hyperparameter_sweep.sh   # 超参数扫描
├── quick_test.sh             # 快速测试
├── run_ant_experiment.sh     # Ant-v2专用实验脚本
└── README.md                 # 本文件
```

## 🎯 核心创新点

### 1. EDAC算法特点
- **Ensemble Critics**: 使用10个独立的Q网络减少过估计
- **多样性正则化**: 鼓励不同critic给出不同预测
- **稳定性强**: 在离线和在线RL中都表现优异

### 2. ROER优先级机制
- **理论基础**: 基于占用优化推导的优先级权重
- **公式**: `w ∝ exp(TD_error / β)`
- **更新**: 指数移动平均（EMA）平滑

### 3. 集成方案
- **Critic Loss**: 使用ROER权重加权TD loss
- **TD误差计算**: 使用Value网络计算（Actor-Critic标准）
- **Ensemble**: 使用ensemble平均Q值计算TD误差

## 🚀 快速开始

### 安装依赖

```bash
# 使用原ROER环境
conda activate roer

# 如果缺少依赖
cd ~/Regularized-Optimal-Experience-Replay
pip install -r requirements.txt
```

### 快速测试

```bash
cd Experience/edac_roer

# 给脚本添加执行权限
chmod +x *.sh

# 运行快速测试（10k步）
./quick_test.sh
```

### 单次训练

```bash
# EDAC+ROER
python train_edac_roer.py \
    --env_name=HalfCheetah-v2 \
    --seed=42 \
    --use_roer=True \
    --roer_temp=4.0 \
    --roer_max_clip=50 \
    --roer_min_clip=10

# EDAC baseline（不使用ROER）
python train_edac_roer.py \
    --env_name=HalfCheetah-v2 \
    --seed=42 \
    --use_roer=False
```

## 📊 实验脚本

### 1. 对比实验

比较EDAC和EDAC+ROER的性能：

```bash
# 运行5个种子的对比实验
./run_comparison.sh HalfCheetah-v2 5

# 运行Ant-v2实验
./run_ant_experiment.sh 5
```

### 2. 超参数扫描

系统扫描关键超参数：

```bash
# 扫描temperature、max_clip、min_clip、diversity_coef
./hyperparameter_sweep.sh HalfCheetah-v2 42
```

## 🔧 关键参数说明

### ROER参数

| 参数 | 说明 | MuJoCo默认 | DM Control默认 |
|------|------|-----------|---------------|
| `roer_temp` (β) | 温度参数，控制正则化强度 | 4.0 | 1.0 |
| `roer_max_clip` | 优先级最大值 | 50 | 100 |
| `roer_min_clip` | 优先级最小值 | 10 | 1 |
| `roer_per_beta` (λ) | EMA系数 | 0.01 | 0.01 |
| `roer_std_normalize` | 是否标准化 | True | True |

### EDAC参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num_critics` | Ensemble大小 | 10 |
| `diversity_coef` | Critic多样性系数 | 0.1 |
| `eta` | Actor中Q标准差权重 | 1.0 |

### Ant-v2推荐参数

基于原ROER论文的设置：

```bash
--roer_temp=1.0 \
--roer_max_clip=100.0 \
--roer_min_clip=10.0 \
--roer_per_beta=0.01
```

## 📈 预期结果

### 性能对比

| 方法 | HalfCheetah-v2 | Ant-v2 | Hopper-v2 |
|------|---------------|--------|-----------|
| EDAC Baseline | ~12000 | ~1500 | ~3000 |
| EDAC+ROER | ~13000? | ~1800? | ~3200? |

**注意**: 这些是预期值，实际结果需要实验验证。

### 稳定性改进

ROER可能带来的改进：
- ✅ **减少种子方差**: 通过优先级平滑训练过程
- ✅ **加快收敛**: 重点学习高TD误差样本
- ✅ **更好的Q值估计**: Value网络提供更准确的TD误差

可能的风险：
- ⚠️ **超参数敏感**: temp、max_clip需要调节
- ⚠️ **计算开销**: 额外的Value网络和优先级计算

## 🔬 实验建议

### 毕设/论文实验设计

1. **Baseline对比**
   ```bash
   # 至少在3个环境上对比
   for env in HalfCheetah-v2 Ant-v2 Hopper-v2; do
       ./run_comparison.sh $env 5
   done
   ```

2. **消融实验**
   - EDAC vs EDAC+ROER
   - 不同temperature的影响
   - 不同ensemble大小的影响

3. **稳定性分析**
   - 多种子统计（至少5个）
   - 学习曲线对比
   - 优先级分布可视化

4. **计算效率分析**
   - 训练时间对比
   - 内存占用对比

## 📊 结果可视化

使用TensorBoard查看训练过程：

```bash
# 查看EDAC+ROER训练
tensorboard --logdir ~/roer_output/results/edac_roer/

# 对比两种方法
tensorboard --logdir ~/roer_output/results/ --port 6006
```

关键指标：
- `evaluation/return`: 评估回报
- `training/critic_loss`: Critic损失
- `training/diversity_loss`: 多样性损失
- `priority/mean`, `priority/std`: 优先级分布

## 🐛 调试建议

### 如果训练不稳定

1. **降低temp**: `--roer_temp=2.0` 或 `1.0`
2. **缩小优先级范围**: `--roer_max_clip=20 --roer_min_clip=5`
3. **增大EMA系数**: `--roer_per_beta=0.05`（更平滑）
4. **降低diversity系数**: `--diversity_coef=0.05`

### 如果性能没提升

1. **检查优先级是否生效**:
   - 查看 `priority/std`，应该 > 0
   - 查看 `priority/max` 和 `priority/min`，应该有明显差异

2. **尝试不同temp值**:
   ```bash
   for temp in 0.5 1.0 2.0 4.0; do
       python train_edac_roer.py --roer_temp=$temp --seed=42
   done
   ```

3. **对比EDAC baseline**: 确保baseline实现正确

## 📝 论文写作建议

如果用于毕设/论文，建议结构：

### 第3章：方法设计

```
3.1 EDAC算法回顾
    - Ensemble Critics
    - Diversity正则化
3.2 ROER优先级机制
    - 占用优化理论
    - 优先级计算公式
3.3 EDAC+ROER集成方案
    - Replay buffer设计
    - Critic loss加权
    - TD误差计算
3.4 算法伪代码
```

### 第4章：实验设计

```
4.1 实验设置
    - 环境：MuJoCo (Ant-v2, HalfCheetah-v2, Hopper-v2)
    - 对比方法：EDAC, EDAC+ROER
    - 随机种子：5-10个
4.2 评估指标
    - 最终性能
    - 收敛速度
    - 稳定性（方差）
4.3 消融实验
    - temperature影响
    - clip范围影响
4.4 计算效率分析
```

## 🎓 毕设价值点

这个实现可以支撑的毕设论述：

1. **创新性**: 首次将ROER应用到EDAC等SOTA算法
2. **理论性**: 结合占用优化和ensemble方法
3. **实验性**: 系统的对比和消融实验
4. **工程性**: 完整的代码实现和文档

## 📧 问题反馈

如有问题，可以：
1. 检查TensorBoard日志
2. 运行quick_test.sh验证基本功能
3. 参考原ROER和EDAC论文

## 🔗 参考资料

- [ROER论文](https://arxiv.org/abs/2407.03995)
- [EDAC论文](https://arxiv.org/abs/2110.01548)
- [原ROER代码](https://github.com/XavierChanglingLi/Regularized-Optimal-Experience-Replay)

