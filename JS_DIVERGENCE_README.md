# Jensen-Shannon 散度在 ROER 中的完整实现

## 1. 理论基础

### 1.1 Jensen-Shannon 散度定义

Jensen-Shannon (JS) 散度是一个对称的散度度量，定义为：

$$\text{JSD}(P||Q) = \frac{1}{2}\text{KL}(P||M) + \frac{1}{2}\text{KL}(Q||M)$$

其中 $M = \frac{1}{2}(P + Q)$ 是 P 和 Q 的平均分布。

### 1.2 JS 散度作为 f-散度

JS 散度可以表示为 f-散度的形式，其中生成函数：

$$f(x) = -(x+1)\log\frac{x+1}{2} + x\log x$$

### 1.3 在 ROER 框架中的应用

根据 ROER 论文的理论框架，对于正则化 RL 目标：

$$\max_\pi J(\pi) - \alpha D_f(\rho^\pi || \rho^{\text{buffer}})$$

对于 TD 误差 $\delta = Q(s,a) - V(s)$，JS 散度对应的：

1. **价值函数损失**：
   $$L_{\text{JS}}(\delta) = \text{softplus}(\delta/\alpha) - \log 2$$
   其中 $\text{softplus}(x) = \log(1 + e^x)$

2. **优先级权重**（占用率比）：
   $$w = \sigma(\delta/\alpha) = \frac{1}{1 + e^{-\delta/\alpha}}$$
   这是 sigmoid 函数，相比 KL 散度的指数权重更加平滑和稳定。

## 2. 实现优势

与 KL 散度相比，JS 散度具有以下优势：

1. **更好的稳定性**：sigmoid 函数的输出范围在 [0, 1]，不会出现极大的权重值
2. **对称性**：JS 散度是对称的，对分布差异的处理更加平衡
3. **更平滑的梯度**：梯度更加平滑，可能减少训练不稳定性
4. **有界性**：JS 散度是有界的（最大值为 log 2），而 KL 散度可以无界
5. **对离群点更鲁棒**：sigmoid 函数对极端 TD 误差不敏感

## 3. 完整实现

### 3.1 损失函数（critic.py）

```python
def js_divergence_loss(diff, alpha, args):
    """
    Jensen-Shannon divergence loss for ROER.
    
    Args:
        diff: TD error (Q - V)
        alpha: temperature parameter (loss_temp)
        args: configuration arguments
    
    Returns:
        loss: JS divergence loss
        norm: normalization factor for logging
    """
    z = diff / alpha
    
    # Clip for numerical stability
    if args.gumbel_max_clip is not None:
        z = jnp.clip(z, -args.gumbel_max_clip, args.gumbel_max_clip)
    
    # JS divergence loss: softplus(z) - log(2)
    loss = jax.nn.softplus(z) - jnp.log(2.0)
    
    # Normalization: use mean of sigmoid
    norm = jnp.mean(jnp.maximum(0.5, jax.nn.sigmoid(z)))
    norm = jax.lax.stop_gradient(norm)
    
    loss = loss / norm
    
    return loss, norm
```

带裁剪的版本（用于线性外推）：

```python
def js_divergence_loss_clipped(diff, alpha, args):
    """
    JS divergence loss with linear extrapolation beyond clip.
    """
    x = diff / alpha
    z = jnp.clip(x, -args.gumbel_max_clip, args.gumbel_max_clip)
    
    # Base loss
    loss = jax.nn.softplus(z) - jnp.log(2.0)
    
    # Linear extrapolation for |x| > clip
    sigmoid_z = jax.nn.sigmoid(z)
    linear = (x - z) * sigmoid_z
    
    # Normalization
    norm = jnp.mean(jnp.maximum(0.5, sigmoid_z))
    norm = jax.lax.stop_gradient(norm)
    
    loss = (loss + linear) / norm
    
    return loss, norm
```

### 3.2 优先级更新（sac_learner.py）

```python
# Jensen-Shannon divergence
elif args.per_type == "JS":
    next_v = value(batch.next_observations)
    target_v = batch.rewards + discount * batch.masks * next_v
    current_v = value(batch.observations)
    td_error = target_v - current_v
    a = td_error / loss_temp
    
    # Clip for numerical stability
    a = jnp.clip(a, -args.max_clip, args.max_clip)
    
    if args.update_scheme == "exp":
        # Sigmoid priority: w = sigmoid(td_error / temp)
        sigmoid_a = jax.nn.sigmoid(a)
        
        # Scale to [js_min_weight, js_max_weight] range
        js_min = getattr(args, 'js_min_weight', 0.5)
        js_max = getattr(args, 'js_max_weight', 2.0)
        priority = js_min + sigmoid_a * (js_max - js_min)
        
        if args.std_normalize:
            priority = priority / jnp.mean(priority)
        
        exp_a = sigmoid_a  # For logging consistency
```

### 3.3 参数配置（train_online.py）

已添加的参数：
- `js_min_weight`: JS 最小权重（默认 0.5）
- `js_max_weight`: JS 最大权重（默认 2.0）

## 4. 使用方法

### 4.1 基本命令

MuJoCo 环境：
```bash
python train_online.py \
    --env_name=HalfCheetah-v2 \
    --seed=42 \
    --max_steps=1000000 \
    --per=True \
    --per_type=JS \
    --temp=4 \
    --min_clip=0.5 \
    --max_clip=50 \
    --gumbel_max_clip=7 \
    --js_min_weight=0.5 \
    --js_max_weight=2.0 \
    --log_loss=True \
    --update_scheme=exp
```

DM Control 环境：
```bash
python train_online.py \
    --env_name=hopper-stand \
    --seed=42 \
    --max_steps=1000000 \
    --per=True \
    --per_type=JS \
    --temp=1 \
    --min_clip=0.5 \
    --max_clip=100 \
    --gumbel_max_clip=7 \
    --js_min_weight=0.5 \
    --js_max_weight=2.0 \
    --log_loss=True \
    --update_scheme=exp
```

### 4.2 快速测试

使用提供的快速测试脚本：
```bash
bash test_js_quick.sh
```

### 4.3 完整对比实验

使用对比脚本测试 JS vs KL vs PER：
```bash
bash compare_divergences.sh HalfCheetah-v2 42 1000000
```

## 5. 超参数推荐

### 5.1 MuJoCo 环境推荐

| 环境 | temp (β) | js_min_weight | js_max_weight | gumbel_max_clip |
|------|----------|---------------|---------------|-----------------|
| HalfCheetah-v2 | 4 | 0.5 | 2.0 | 7 |
| Hopper-v2 | 4 | 0.5 | 2.0 | 7 |
| Ant-v2 | 4 | 0.5 | 2.0 | 7 |
| Walker2d-v2 | 4 | 0.5 | 2.0 | 7 |
| Humanoid-v2 | 4 | 0.5 | 1.5 | 7 |

### 5.2 DM Control 环境推荐

| 环境 | temp (β) | js_min_weight | js_max_weight | gumbel_max_clip |
|------|----------|---------------|---------------|-----------------|
| fish-swim | 1 | 0.5 | 2.0 | 7 |
| hopper-stand | 1 | 0.5 | 2.0 | 7 |
| humanoid-run | 1 | 0.5 | 1.5 | 7 |
| quadruped-run | 1 | 0.5 | 1.5 | 7 |

### 5.3 超参数说明

#### temp (β) - 损失温度
- **作用**：控制 KL 正则化器对 on-policy 和 off-policy 分布差异的惩罚强度
- **理论**：
  - 小 β：适合噪声数据，远离 on-policy 分布
  - 大 β：适合接近 on-policy 分布的数据
- **推荐**：
  - MuJoCo：通常使用 4
  - DM Control：通常使用 1
  - 困难任务（Humanoid）：可以尝试 1-2

#### js_min_weight - 最小优先级权重
- **作用**：sigmoid 函数映射后的最小值
- **理论**：防止样本被完全忽略，缓解经验遗忘
- **推荐**：0.5（sigmoid 的中点）

#### js_max_weight - 最大优先级权重
- **作用**：sigmoid 函数映射后的最大值
- **理论**：控制优先级分布的范围
- **对比**：KL 散度通常使用 50-100，JS 散度使用 1.5-2.5
- **推荐**：
  - 标准任务：2.0
  - 困难任务（Humanoid）：1.5（更小范围，更稳定）
  - 激进探索：2.5

#### gumbel_max_clip
- **作用**：对 TD 误差/温度的比值进行裁剪
- **理论**：防止数值溢出和极端权重
- **推荐**：7（与 KL 散度一致）

#### min_clip
- **作用**：优先级的全局最小值
- **理论**：确保所有样本都有被采样的机会
- **推荐**：
  - MuJoCo：0.5-1.0
  - DM Control：0.5-1.0

#### max_clip
- **作用**：对 a = td_error/temp 的裁剪上限
- **理论**：防止极端 TD 误差导致的数值问题
- **推荐**：
  - MuJoCo：50
  - DM Control：50-100

## 6. 调试和分析

### 6.1 关键指标监控

通过 WandB 或 TensorBoard 监控：
- `priority`：优先级权重的平均值
- `per_weight`：sigmoid(a) 的值
- `td_error_mean`：TD 误差的平均值
- `value_loss`：JS 散度损失
- `norm_min`：归一化因子

### 6.2 预期行为

**优先级分布**：
- JS 散度：更均匀，大部分权重在 [js_min_weight, js_max_weight] 范围内
- KL 散度：更极端，可能有很大和很小的权重

**训练稳定性**：
- JS 散度：梯度更平滑，训练更稳定
- KL 散度：可能有更大的梯度波动

**收敛速度**：
- JS 散度：可能稍慢，但更稳定
- KL 散度：可能更快，但不稳定

### 6.3 常见问题

**问题 1：优先级权重没有变化**
- 检查 `temp` 是否合适
- 检查 TD 误差是否被正确计算
- 确认 `per_type="JS"` 和 `per=True`

**问题 2：性能不如 KL 散度**
- 尝试增加 `js_max_weight`
- 尝试调整 `temp`（增加或减少）
- 检查是否使用 `log_loss=True`

**问题 3：训练不稳定**
- 减小 `js_max_weight`
- 增加 `min_clip`
- 减小 `temp`

## 7. 理论分析：JS vs KL

### 7.1 权重函数对比

| 特性 | KL 散度 | JS 散度 |
|------|---------|---------|
| 权重函数 | $w = e^{\delta/\alpha}$ | $w = \sigma(\delta/\alpha)$ |
| 范围 | $[0, \infty)$ | $[0, 1]$ |
| 对称性 | 非对称 | 对称 |
| 对极端值的敏感度 | 高 | 低 |
| 梯度 | $\frac{1}{\alpha}e^{\delta/\alpha}$ | $\frac{1}{\alpha}\sigma(\delta/\alpha)(1-\sigma(\delta/\alpha))$ |

### 7.2 适用场景

**KL 散度更适合**：
- 需要快速收敛的任务
- TD 误差分布相对均匀的环境
- 数据质量较高的情况

**JS 散度更适合**：
- 需要稳定训练的困难任务（如 Humanoid）
- TD 误差有极端值的环境
- 数据噪声较大的情况
- 需要防止经验遗忘的场景

### 7.3 数学性质

**JS 散度的优势**：
1. **有界性**：$0 \leq \text{JSD}(P||Q) \leq \log 2$
2. **对称性**：$\text{JSD}(P||Q) = \text{JSD}(Q||P)$
3. **平方根是距离**：$\sqrt{\text{JSD}(P||Q)}$ 满足三角不等式
4. **平滑性**：在整个定义域上都有良好的梯度

## 8. 实验建议

### 8.1 消融研究

建议进行以下消融实验：

1. **温度参数**：测试 temp ∈ [0.5, 1, 2, 4, 8]
2. **权重范围**：测试不同的 [js_min_weight, js_max_weight] 组合
3. **更新方案**：对比 `update_scheme="exp"` vs `"avg"`
4. **裁剪策略**：对比 `log_loss=True` vs `False`

### 8.2 对比实验

建议的对比基准：
1. Uniform（无优先级）
2. PER（传统优先级经验回放）
3. ROER with KL（原始 ROER）
4. ROER with JS（本实现）
5. LaBER（大批量经验回放）

### 8.3 评估指标

除了最终性能，还应关注：
- **学习曲线的平滑度**：方差和置信区间
- **样本效率**：达到特定性能所需的步数
- **稳定性**：多次运行的一致性
- **权重分布**：优先级权重的统计特性
- **价值估计**：Q 值和 V 值的准确性

## 9. 未来扩展方向

### 9.1 自适应温度
实现自适应 β 调整：
```python
# 伪代码
if replay_buffer_converges_to_on_policy:
    beta = beta * 1.1  # 增加温度
else:
    beta = beta * 0.9  # 减少温度
```

### 9.2 混合散度
结合 JS 和 KL 散度的优势：
```python
# 伪代码
alpha_kl = 0.3
alpha_js = 0.7
loss = alpha_kl * kl_loss + alpha_js * js_loss
```

### 9.3 其他 f-散度
- Chi-squared 散度
- Hellinger 距离
- Total Variation 距离
- Wasserstein 距离（需要不同的推导）

## 10. 参考文献

1. **ROER 原始论文**：Li et al., "ROER: Regularized Optimal Experience Replay", 2024
2. **Extreme Q-Learning**：Garg et al., "Extreme Q-Learning", 2023
3. **JS 散度理论**：Lin, "Divergence measures based on the Shannon entropy", 1991
4. **f-散度理论**：Csiszár, "Information-type measures of difference", 1967

## 11. 致谢

本实现基于 ROER 论文和 Extreme Q-Learning 的理论框架，扩展了 f-散度正则化器的选择。
