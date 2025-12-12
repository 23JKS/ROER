# Jensen-Shannon 散度在 ROER 中的实现

## 理论背景

### Jensen-Shannon 散度定义

Jensen-Shannon (JS) 散度是一个对称的散度度量，定义为：

$$\text{JSD}(P||Q) = \frac{1}{2}\text{KL}(P||M) + \frac{1}{2}\text{KL}(Q||M)$$

其中 $M = \frac{1}{2}(P + Q)$ 是 P 和 Q 的平均分布。

### JS 散度作为 f-散度

JS 散度可以表示为 f-散度的形式，其中：

$$f(x) = -(x+1)\log\frac{x+1}{2} + x\log x$$

对于 $x = \frac{dP}{dQ}$（Radon-Nikodym 导数），JS 散度可以写为：

$$\text{JSD}(P||Q) = \int f\left(\frac{dP}{dQ}\right) dQ$$

### 在 ROER 框架中的应用

根据 ROER 论文的理论框架，对于正则化 RL 目标：

$$\max_\pi J(\pi) - \alpha D_f(\rho^\pi || \rho^{\text{buffer}})$$

其对偶形式涉及到 f 的共轭函数 $f^*$。对于 JS 散度，我们需要计算相应的损失函数。

### JS 散度对应的损失函数

对于 TD 误差 $\delta = Q(s,a) - V(s)$，JS 散度对应的价值函数损失为：

$$L_{\text{JS}}(\delta) = -\log(2 - e^{\delta/\alpha}) - \log(1 + e^{-\delta/\alpha})$$

为了数值稳定性，我们可以使用等价形式：

$$L_{\text{JS}}(\delta) = \text{softplus}(\delta/\alpha) - \log 2$$

其中 $\text{softplus}(x) = \log(1 + e^x)$。

### 优先级权重

对应的占用率比（occupancy ratio）即优先级权重为：

$$w = \frac{1}{1 + e^{-\delta/\alpha}}$$

这是一个 sigmoid 函数，相比 KL 散度的指数权重 $w = e^{\delta/\alpha}$ 更加平滑和稳定。

## 实现优势

1. **稳定性更好**：sigmoid 函数的输出范围在 [0, 1]，不会出现极大的权重值
2. **对称性**：JS 散度是对称的，对分布差异的处理更加平衡
3. **梯度性质**：梯度更加平滑，可能减少训练不稳定性
4. **有界性**：JS 散度是有界的（最大值为 log 2），而 KL 散度可以无界

## 实现细节

### 损失函数实现

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
    # softplus(z) = log(1 + exp(z))
    loss = jax.nn.softplus(z) - jnp.log(2.0)
    
    # Normalization (optional, for consistency with KL version)
    norm = jnp.mean(jnp.maximum(1.0, jax.nn.sigmoid(z)))
    norm = jax.lax.stop_gradient(norm)
    
    loss = loss / norm
    
    return loss, norm
```

### 优先级更新实现

```python
def update_priority_js(batch, value, discount, loss_temp, args):
    """
    Update priority using JS divergence.
    
    Priority weight: w = sigmoid(td_error / alpha)
    """
    next_v = value(batch.next_observations)
    target_v = batch.rewards + discount * batch.masks * next_v
    current_v = value(batch.observations)
    td_error = target_v - current_v
    
    # JS divergence priority: sigmoid function
    a = td_error / loss_temp
    a = jnp.clip(a, -args.max_clip, args.max_clip)
    
    priority = jax.nn.sigmoid(a)
    
    # Scale to avoid too small priorities
    # Map [0, 1] to [min_clip, max_weight]
    priority = args.min_clip + priority * (args.js_max_weight - args.min_clip)
    
    return priority
```

## 超参数建议

基于理论分析，JS 散度的超参数设置建议：

- `temp` (β): 推荐范围 [1, 10]，通常比 KL 散度使用更大的值
- `min_clip`: 0.1 - 1.0
- `js_max_weight`: 2.0 - 10.0（相比 KL 的 50-100 更小）
- `gumbel_max_clip`: 7 - 10

## 实验计划

1. **基准测试**：在 MuJoCo 环境（HalfCheetah, Hopper, Ant）上测试
2. **消融研究**：
   - 比较 JS vs KL vs PER
   - 温度参数 β 的影响
   - 权重范围的影响
3. **难度环境**：在 Humanoid 和 DM Control 任务上验证
4. **稳定性分析**：记录权重分布、TD 误差、价值估计的方差

## 预期结果

基于 JS 散度的特性，预期：
- 在需要稳定训练的环境（如 Humanoid）中表现更好
- 权重分布更加均匀，减少极端权重
- 可能牺牲一些收敛速度，但获得更稳定的性能

