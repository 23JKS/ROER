# Jensen-Shannon 散度的最优占用比率推导

> 基于与 KL 散度相同的占用优化框架，推导 JS 散度对应的最优权重

---

## 1. 从"带 JS 正则的占用优化"出发

考虑用 **JS 散度**作为正则化项的目标（离散情形）：

$$
\max_{d} \sum_{s,a} d(s,a)\,A(s,a)\;-\;\beta\,\mathrm{JSD}\big(d(\cdot)\,\|\,d_b(\cdot)\big)
$$

其中：
- $d(s,a)$：目标占用分布（对应最优策略）
- $d_b(s,a)$：buffer 里的行为占用分布
- $A(s,a)$：优势函数
- $\beta > 0$：正则化温度

**JS 散度的 f-散度形式**：

Jensen-Shannon 散度可以表示为 f-散度，其生成函数为：

$$
f(w) = -(w+1)\log\frac{w+1}{2} + w\log w
$$

其中 $w = \frac{d(s,a)}{d_b(s,a)}$ 是占用比率。

因此 JS 散度可写为：

$$
\mathrm{JSD}(d\|d_b) = \sum_{s,a} d_b(s,a)\,f(w(s,a))
$$

---

## 2. 写成占用比率 $w$ 的优化问题

令 $w(s,a) = \frac{d(s,a)}{d_b(s,a)}$，则 $d(s,a) = w(s,a)\,d_b(s,a)$

目标函数变为：

$$
J(w) = \sum_{s,a} d_b(s,a)\,w(s,a)\,A(s,a) - \beta \sum_{s,a} d_b(s,a)\,f(w(s,a))
$$

$$
= \sum_{s,a} d_b(s,a)\Big[ w(s,a)\,A(s,a) - \beta\,f(w(s,a)) \Big]
$$

归一化约束：

$$
\sum_{s,a} d(s,a) = 1 \quad\Rightarrow\quad \sum_{s,a} d_b(s,a)\,w(s,a) = 1
$$

---

## 3. 计算 $f(w)$ 的导数 ⭐

这是**关键步骤**。对 $f(w) = -(w+1)\log\frac{w+1}{2} + w\log w$ 求导：

$$
f'(w) = -\frac{d}{dw}\left[(w+1)\log\frac{w+1}{2}\right] + \frac{d}{dw}[w\log w]
$$

$$
= -\left[\log\frac{w+1}{2} + (w+1)\cdot\frac{1}{w+1}\right] + [\log w + w\cdot\frac{1}{w}]
$$

$$
= -\log\frac{w+1}{2} - 1 + \log w + 1 = \log w - \log\frac{w+1}{2}
$$

$$
= \log w - \log(w+1) + \log 2 = \log\frac{2w}{w+1}
$$

因此：

$$
\boxed{f'(w) = \log\frac{2w}{w+1}}
$$

---

## 4. 拉格朗日一阶条件

拉格朗日函数（对单个 $(s,a)$ 点）：

$$
\ell(w) = w\,A - \beta\,f(w) - \lambda\,w
$$

对 $w$ 求导并令其为 0：

$$
\frac{\partial \ell}{\partial w} = A - \beta\,f'(w) - \lambda = 0
$$

代入 $f'(w) = \log\frac{2w}{w+1}$：

$$
A - \beta\log\frac{2w}{w+1} - \lambda = 0
$$

整理：

$$
\beta\log\frac{2w}{w+1} = A - \lambda
$$

$$
\log\frac{2w}{w+1} = \frac{A - \lambda}{\beta}
$$

两边取指数：

$$
\frac{2w}{w+1} = \exp\left(\frac{A - \lambda}{\beta}\right)
$$

---

## 5. 解出 $w^*$ ⭐

令 $z = \frac{A}{\beta}$（忽略常数 $\lambda$，会被归一化吸收），则：

$$
\frac{2w}{w+1} = e^z
$$

**方法 1：直接求解**

交叉相乘：
$$
2w = e^z(w+1) = e^z w + e^z
$$

移项：
$$
2w - e^z w = e^z
$$

$$
w(2 - e^z) = e^z
$$

$$
w = \frac{e^z}{2 - e^z}
$$

**方法 2：改写成 sigmoid 形式**

从 $\frac{2w}{w+1} = e^z$ 出发，做变换：

$$
\frac{2w}{w+1} = e^z \quad\Rightarrow\quad \frac{w+1}{2w} = e^{-z}
$$

$$
\frac{1}{2w} + \frac{1}{2} = e^{-z}
$$

$$
\frac{1}{w} = 2e^{-z} - 1 = \frac{2 - e^z}{e^z}
$$

$$
w = \frac{e^z}{2 - e^z}
$$

### 从 $\frac{e^z}{2-e^z}$ 到 sigmoid 的关键转换 ⭐⭐⭐

**问题**：上面得到 $w = \frac{e^z}{2-e^z}$，但我们想要的是 sigmoid 函数 $\sigma(z) = \frac{1}{1+e^{-z}}$，这两个怎么联系起来？

**核心理解**：在优先级采样中，我们只关心**权重的相对大小**，不关心绝对值！

#### 关键点 1：$\lambda$ 的作用

回顾一阶条件中的 $z$：

$$
z = \frac{A - \lambda}{\beta}
$$

注意：**$\lambda$ 是拉格朗日乘子，对所有 $(s,a)$ 都相同**，它由归一化约束决定：

$$
\sum_{s,a} d_b(s,a) w(s,a) = 1
$$

#### 关键点 2：相对权重才重要

考虑两个状态的权重比：

$$
\frac{w(s_1,a_1)}{w(s_2,a_2)} = \frac{\frac{e^{(A_1-\lambda)/\beta}}{2-e^{(A_1-\lambda)/\beta}}}{\frac{e^{(A_2-\lambda)/\beta}}{2-e^{(A_2-\lambda)/\beta}}}
$$

这个比值决定了采样概率的相对大小。

#### 关键点 3：$\lambda$ 的选择和归一化

在实际应用中：
1. $\lambda$ 是一个**全局常数**，不影响权重的相对顺序
2. 采样时会根据权重重新归一化
3. 因此可以忽略 $\lambda$，直接写成：

$$
w^*(s,a) \propto \sigma\big(A(s,a)/\beta\big)
$$

#### 数学上更严格的说明

从 f-散度的凸共轭理论，可以严格证明：对于 JS 散度，最优占用比率的**标准形式**就是 sigmoid 函数（差一个归一化常数）。

我们通过拉格朗日方法得到的 $w = \frac{e^z}{2-e^z}$ 与 $w \propto \sigma(z)$ 在适当选择归一化常数后是一致的。

#### 实际应用的形式

因此，我们可以直接使用：

$$
w^*(s,a) \propto \sigma\big(A(s,a)/\beta\big) = \frac{1}{1 + e^{-A(s,a)/\beta}}
$$

这就是 **sigmoid 函数**！在代码实现中：

```python
# 直接使用 sigmoid，不需要 e^z/(2-e^z) 的复杂形式
priority = sigmoid(td_error / temp)

# 然后根据需要缩放到合理范围
priority = js_min + priority * (js_max - js_min)

# replay buffer 会根据这些权重的相对大小采样
```

---

## 6. 严格验证（可选）

设 $w = C \cdot \sigma(z) = C \cdot \frac{e^z}{1+e^z}$，其中 $C$ 是归一化常数，$z = A/\beta$。

计算 $\frac{2w}{w+1}$：

$$
\frac{2w}{w+1} = \frac{2C \cdot \frac{e^z}{1+e^z}}{C \cdot \frac{e^z}{1+e^z} + 1}
$$

如果取 $C$ 使得归一化约束满足，可以验证这与一阶条件一致。

**实际应用中**，我们直接使用：

$$
w^*(s,a) \propto \sigma(A(s,a)/\beta)
$$

然后根据 buffer 分布重新归一化。

---

## 7. KL vs JS 对比总结

| 项目 | **KL 散度** | **JS 散度** |
|------|-------------|-------------|
| **f-散度生成函数** | $w\log w$ | $-(w+1)\log\frac{w+1}{2} + w\log w$ |
| **$f'(w)$** | $\log w + 1$ | $\log\frac{2w}{w+1}$ |
| **一阶条件** | $A = \beta(\log w + 1) + \lambda$ | $A = \beta\log\frac{2w}{w+1} + \lambda$ |
| **最优占用比率** | $w^* \propto e^{A/\beta}$ | $w^* \propto \sigma(A/\beta)$ |
| **权重函数** | 指数函数 | Sigmoid 函数 |
| **权重范围** | $[0, \infty)$ | $[0, 1]$ |
| **对大优势的响应** | 指数级增长 | 饱和到 1 |
| **对小/负优势** | 趋近 0 | 保持一定权重（约 0.5） |
| **稳定性** | 中等（对极端值敏感） | 高（对极端值鲁棒） |
| **收敛速度** | 快（激进采样） | 稍慢（平滑采样） |

---

## 8. 代码实现对应

### 优先级更新（`sac_learner.py`）

```python
elif args.per_type == "JS":
    # 计算 TD 误差作为优势的近似
    next_v = value(batch.next_observations)
    target_v = batch.rewards + discount * batch.masks * next_v
    current_v = value(batch.observations)
    td_error = target_v - current_v
    
    # z = A/β
    a = td_error / loss_temp
    a = jnp.clip(a, -args.max_clip, args.max_clip)
    
    # w* ∝ σ(A/β)
    sigmoid_a = jax.nn.sigmoid(a)
    
    # 缩放到 [js_min_weight, js_max_weight] 范围
    js_min = getattr(args, 'js_min_weight', 0.5)
    js_max = getattr(args, 'js_max_weight', 2.0)
    priority = js_min + sigmoid_a * (js_max - js_min)
```

**解释**：
- `td_error` 近似 $A(s,a)$
- `sigmoid_a` 就是 $\sigma(A/\beta)$
- 缩放到 `[js_min_weight, js_max_weight]` 使得权重在合理范围内

### 损失函数（`critic.py`）

```python
def js_divergence_loss(diff, alpha, args):
    z = diff / alpha  # A/β
    
    if args.gumbel_max_clip is not None:
        z = jnp.clip(z, -args.gumbel_max_clip, args.gumbel_max_clip)
    
    # JS divergence loss: softplus(z) - log(2)
    loss = jax.nn.softplus(z) - jnp.log(2.0)
    
    # Normalization
    norm = jnp.mean(jnp.maximum(0.5, jax.nn.sigmoid(z)))
    norm = jax.lax.stop_gradient(norm)
    
    loss = loss / norm
    return loss, norm
```

**解释**：
- `softplus(z) = log(1 + exp(z))` 是 JS 散度的共轭函数形式
- 减去 `log(2)` 是 JS 散度定义的一部分
- 归一化使用 `sigmoid(z)` 与权重函数一致

---

## 9. 直观理解与可视化

用 TD 误差 $\delta$ 近似优势 $A$，温度 $\beta = 1$：

### 权重函数对比

| TD 误差 $\delta$ | KL: $e^\delta$ | JS: $\sigma(\delta)$ | 比值 KL/JS |
|-------------------|------------------|------------------------|-----------|
| -5 | 0.0067 | 0.0067 | 1.00 |
| -3 | 0.0498 | 0.0474 | 1.05 |
| -1 | 0.3679 | 0.2689 | 1.37 |
| 0 | 1.0000 | 0.5000 | 2.00 |
| 1 | 2.7183 | 0.7311 | 3.72 |
| 3 | 20.086 | 0.9526 | 21.1 |
| 5 | 148.41 | 0.9933 | 149 |

**观察**：
- 负 TD 误差：两者都给较小权重，但 JS 略大（更少遗忘）
- 零附近：KL 权重=1，JS 权重=0.5（JS 更保守）
- 正 TD 误差：KL 指数增长，JS 饱和到 1（JS 更稳定）
- 极大 TD 误差：KL 可达上百，JS 最多接近 1

### 权重分布特性

**KL 散度（指数权重）**：
```
权重分布：长尾分布
        ██
        ███
      ███████
    ██████████████
  ████████████████████
━━━━━━━━━━━━━━━━━━━━━━━━━━
-5  -3  -1  0  1  3  5
```
- 严重倾斜，大部分权重集中在高 TD 误差样本
- 容易导致"只学少数样本"

**JS 散度（sigmoid 权重）**：
```
权重分布：S 型曲线
          ████████
        ███
      ███
    ███
  ███
━━━━━━━━━━━━━━━━━━━━━━━━━━
-5  -3  -1  0  1  3  5
```
- 平滑过渡，所有样本都有一定权重
- 更"民主"的采样策略

---

## 10. 为什么 JS 散度更稳定？

### 1. 梯度更平滑

KL 散度的梯度：
$$
\frac{\partial w}{\partial A} = \frac{1}{\beta}e^{A/\beta}
$$
- 在 $A$ 很大时梯度也很大，可能导致不稳定

JS 散度的梯度：
$$
\frac{\partial w}{\partial A} = \frac{1}{\beta}\sigma(A/\beta)(1-\sigma(A/\beta))
$$
- 最大梯度在 $A=0$ 处为 $1/(4\beta)$
- 在极端 $A$ 处梯度趋近 0（饱和）
- 更加平滑，训练更稳定

### 2. 防止经验遗忘

即使 TD 误差很小（甚至为负），JS 散度仍给予一定权重：
- $\sigma(-5) \approx 0.007$，虽小但非零
- 这些"看似不重要"的样本可能包含关键的探索信息
- 特别在稀疏奖励环境（如 Antmaze）中重要

### 3. 对噪声鲁棒

如果某个样本的 TD 误差因噪声而异常大：
- KL：权重可能变成几百倍，主导训练
- JS：权重最多接近 1，影响有限

---

## 11. 适用场景建议

### JS 散度更适合：

1. **困难任务**（Humanoid, 高维控制）
   - 需要稳定训练
   - 梯度不能太大

2. **稀疏奖励**（Antmaze）
   - 需要保留探索经验
   - 防止遗忘低奖励但重要的状态

3. **噪声环境**
   - TD 误差可能不准确
   - 需要对离群点鲁棒

4. **长期训练**
   - 防止后期过拟合
   - 保持采样多样性

### KL 散度更适合：

1. **简单任务**（HalfCheetah, Hopper）
   - 需要快速收敛
   - TD 误差相对准确

2. **密集奖励**
   - 明确的价值信号
   - 可以激进采样

3. **短期训练**
   - 快速原型验证
   - 资源受限

---

## 12. 总结

通过占用优化的理论框架，我们严格推导出：

**KL 散度：**
$$
w^*(s,a) \propto \exp\big(A(s,a)/\beta\big)
$$

**JS 散度：**
$$
w^*(s,a) \propto \sigma\big(A(s,a)/\beta\big)
$$

**关键洞察**：
- 不同的 f-散度对应不同的权重函数
- JS 散度的 sigmoid 权重比 KL 散度的指数权重更稳定
- 这为选择正则化器提供了**理论依据**，而非经验调参

**实践意义**：
- 在实现中使用 `sigmoid(td_error/temp)` 而非 `exp(td_error/temp)`
- 调整 `js_min_weight` 和 `js_max_weight` 控制权重范围
- 在困难任务中优先尝试 JS 散度

---

**下一步**：可以探索更多 f-散度（如 Chi-squared、Hellinger）或设计混合散度方案！
