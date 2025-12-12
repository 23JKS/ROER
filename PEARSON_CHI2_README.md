# Pearson χ² 散度在 ROER 中的实现

本文档说明如何在 ROER 框架中使用 Pearson χ² 散度作为正则化器。

## 概述

Pearson χ² 散度是 f-散度的一种，其生成函数为：

\[
f(w) = \frac{1}{2}(w-1)^2
\]

### 优势

1. **数值稳定性**：二次形式，不会像指数函数那样爆炸
2. **计算效率**：实现简单，计算开销低
3. **高维任务优势**：对异常值更鲁棒，适合高维连续控制任务
4. **梯度平滑**：训练更稳定

## 理论推导

### 最优占用比率

对于 Pearson χ² 散度，最优占用比率为：

\[
w^*(s,a) = 1 + \frac{A(s,a) - \mathbb{E}_{d_b}[A]}{\beta}
\]

其中：
- \(A(s,a)\) 是优势函数
- \(\beta\) 是正则化温度参数
- \(d_b\) 是 off-policy 数据分布

用 TD 误差近似：

\[
w^*(s,a) \propto 1 + \frac{\delta(s,a)}{\beta^2}
\]

其中 \(\delta = Q(s,a) - V(s)\) 是 TD 误差。

### 凸共轭函数

Pearson χ² 的凸共轭函数为：

\[
f^*(y) = \frac{1}{2}y^2 + y
\]

其导数为：

\[
(f^*)'(y) = y + 1
\]

## 使用方法

### 基本用法

在训练脚本中，设置 `per_type='X2'`：

```bash
python train_online.py \
    --env_name=HalfCheetah-v2 \
    --per=True \
    --per_type=X2 \
    --min_clip=1 \
    --max_clip=50 \
    --gumbel_max_clip=7 \
    --temp=4
```

### 参数说明

- `--per_type=X2`：使用 Pearson χ² 散度
- `--temp`：损失温度参数（对应 \(\beta\)）
- `--min_clip`：最小优先级裁剪值（防止负权重）
- `--max_clip`：最大优先级裁剪值
- `--gumbel_max_clip`：损失函数裁剪值（用于数值稳定性）

### 与其他散度的对比

| 特性 | KL 散度 (OER) | JS 散度 | Pearson χ² |
|------|---------------|---------|------------|
| 权重形式 | \(\exp(\delta/\beta)\) | \(\sigma(\delta/\beta)\) | \(1 + \delta/\beta^2\) |
| 数值稳定性 | 需要裁剪 | 有界，较稳定 | 最稳定 |
| 计算复杂度 | 中等 | 中等 | 最低 |
| 高维任务 | 可能不稳定 | 较好 | 最好 |

## 实现细节

### 损失函数

```python
def conservative_loss(diff, alpha, args):
    """
    Pearson χ² divergence loss.
    
    Args:
        diff: TD error (Q - V)
        alpha: temperature parameter (loss_temp)
        args: configuration arguments
    
    Returns:
        loss: Pearson χ² divergence loss
        norm: normalization factor
    """
    z = diff / alpha
    if args.gumbel_max_clip is not None:
        z = jnp.clip(z, -args.gumbel_max_clip, args.gumbel_max_clip)
    
    loss = 0.5 * z ** 2
    norm = jnp.mean(jnp.ones_like(z))
    norm = jax.lax.stop_gradient(norm)
    loss = loss / norm
    
    return loss, norm
```

### 优先级计算

```python
# 在 update_priority 函数中
linear_priority = 1.0 + a / loss_temp
linear_priority = jnp.maximum(linear_priority, args.min_clip)
linear_priority = jnp.minimum(linear_priority, args.max_clip)
```

其中 `a = td_error / loss_temp`。

## 测试

运行测试脚本验证实现：

```bash
python test_pearson_chi2.py
```

测试包括：
1. 基本属性测试（对称性、零误差行为）
2. 裁剪行为测试
3. 与 KL 散度对比
4. 数值稳定性测试
5. 可视化

## 适用场景

Pearson χ² 散度特别适合：

1. **高维连续控制任务**（如 DM Control）
2. **稀疏奖励任务**（如 Antmaze）
3. **需要快速训练的场景**（计算效率高）
4. **大规模经验回放**（数值稳定性好）

## 参考文献

- ROER 论文：Regularized Optimal Experience Replay (2024)
- f-散度理论：Nachum et al. (2019)

## 注意事项

1. **权重裁剪**：虽然 Pearson χ² 数值稳定，但仍建议设置合理的 `min_clip` 和 `max_clip`
2. **温度参数**：`temp` 参数的选择会影响优先级分布，建议从 1-4 开始调参
3. **更新方案**：支持 `avg`（指数移动平均）和 `exp`（直接使用）两种更新方案

## 示例结果

在 MuJoCo 任务上的典型超参数：

```bash
# HalfCheetah-v2
--per_type=X2 --temp=4 --min_clip=1 --max_clip=50 --gumbel_max_clip=7

# Hopper-v2
--per_type=X2 --temp=1 --min_clip=1 --max_clip=100 --gumbel_max_clip=7
```

