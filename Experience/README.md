# ROER集成到SOTA Off-Policy算法

本目录包含将**ROER (Regularized Optimal Experience Replay)** 集成到最新SOTA离线/在线强化学习算法的实现。

## 📁 目录结构

```
Experience/
├── edac_roer/           # EDAC + ROER 集成实现
│   ├── edac_roer_learner.py      # 核心算法实现
│   ├── replay_buffer_roer.py     # 带ROER优先级的replay buffer
│   ├── train_edac_roer.py        # 训练脚本
│   └── configs/                  # 配置文件
├── td3bc_roer/          # TD3+BC + ROER (待实现)
└── README.md            # 本文件
```

## 🎯 动机

原始ROER论文只在SAC算法上进行了验证。本项目旨在：

1. **探索ROER的泛化能力**：将ROER集成到更先进的off-policy算法（EDAC、TD3+BC等）
2. **提升SOTA性能**：验证ROER是否能进一步提升这些已经很强的算法
3. **稳定性改进**：EDAC等算法已经解决了Q值过估计问题，ROER可能进一步稳定训练
4. **毕设/研究方向**：这是一个尚未被充分探索的研究空间

## 🔬 核心思想

### ROER的优先级权重

ROER通过正则化占用优化推导出的优先级权重：

```
w(s,a) ∝ exp(TD_error / β)
```

其中：
- TD_error = r + γV(s') - V(s)（使用Value网络）
- β：温度参数，控制正则化强度

### 集成策略

**方式1：重加权损失（Reweighting Loss）**
- 保持均匀采样
- 在critic loss中用权重w加权
- 实现简单，稳定性好

**方式2：加权采样（Weighted Sampling）**
- 按优先级权重采样
- 需要重要性采样修正
- 更接近经典PER，但实现复杂

本实现采用**方式1**作为起点。

## 📊 EDAC + ROER

### EDAC简介

**EDAC (Ensemble-Diversified Actor-Critic)**:
- 使用ensemble critics减少Q值过估计
- 通过多样性正则化提升稳定性
- 在D4RL离线任务上SOTA

### 集成方案

1. **Critic更新**：在EDAC的ensemble critic loss中加入ROER权重
   ```python
   for critic in ensemble:
       loss_i = w * (q_i - target_q)^2
   ```

2. **优先级计算**：使用ensemble的平均Q值计算TD error
   ```python
   q_mean = mean([q1, q2, ..., qN])
   td_error = target_q - q_mean
   priority = exp(td_error / β)
   ```

3. **权重更新**：指数移动平均（EMA）
   ```python
   w_new = λ * exp(td_error/β) + (1-λ) * w_old
   ```

## 🚀 快速开始

### 安装依赖

```bash
# 使用原ROER环境
conda activate roer

# 如果需要额外依赖
pip install -r requirements.txt
```

### 运行EDAC+ROER

```bash
cd Experience/edac_roer

# 在线环境（MuJoCo）
python train_edac_roer.py \
    --env_name=HalfCheetah-v2 \
    --use_roer=True \
    --roer_temp=4.0 \
    --roer_max_clip=50 \
    --roer_min_clip=10

# 离线环境（D4RL）
python train_edac_roer.py \
    --env_name=halfcheetah-medium-v2 \
    --offline=True \
    --use_roer=True
```

## 📈 实验设计

### 对比实验

1. **Baseline**: EDAC（原始实现）
2. **EDAC+ROER**: 集成ROER优先级
3. **消融实验**：
   - 不同β (temp)值的影响
   - 不同max_clip/min_clip范围
   - EMA系数λ的影响

### 评估指标

- **最终性能**：1M步后的平均回报
- **收敛速度**：达到目标性能的步数
- **稳定性**：多种子方差
- **样本效率**：相同步数下的性能对比

## 📝 实现细节

### 关键修改点

1. **replay_buffer_roer.py**
   - 继承原Dataset类
   - 添加priority字段和update_priority方法
   - 保持均匀采样，但返回权重

2. **edac_roer_learner.py**
   - 基于EDAC架构
   - Critic loss加入w权重
   - 添加update_priority函数计算ROER权重

3. **train_edac_roer.py**
   - 训练循环
   - 每次update后更新priority
   - 记录权重分布统计

## 🎓 毕设建议

如果用于毕设，建议的章节结构：

1. **绪论**
   - 经验回放的重要性
   - PER/ROER背景
   - EDAC等SOTA算法

2. **相关工作**
   - 优先经验回放方法
   - Off-policy RL算法（EDAC、TD3+BC等）
   - 占用优化理论

3. **方法**
   - ROER理论回顾
   - EDAC算法简介
   - 集成方案设计

4. **实验**
   - 实验设置
   - 对比实验结果
   - 消融实验
   - 稳定性分析

5. **总结与展望**

## 🔗 参考资料

- [ROER论文](https://arxiv.org/abs/2407.03995)
- [EDAC论文](https://arxiv.org/abs/2110.01548)
- [TD3+BC论文](https://arxiv.org/abs/2106.06860)
- [原ROER代码](https://github.com/XavierChanglingLi/Regularized-Optimal-Experience-Replay)

## 📧 联系

如有问题，请参考原ROER论文或在GitHub开issue。

