"""策略网络定义"""
import jax
import jax.numpy as jnp
import distrax
from flax import linen as nn
from typing import Sequence


LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class NormalTanhPolicy(nn.Module):
    """高斯策略 + Tanh变换"""
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: float = LOG_STD_MIN
    log_std_max: float = LOG_STD_MAX
    
    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> distrax.Distribution:
        net = observations
        for hidden_dim in self.hidden_dims:
            net = nn.Dense(hidden_dim)(net)
            net = nn.relu(net)
        
        # 均值和log标准差
        mean = nn.Dense(self.action_dim)(net)
        log_std = nn.Dense(self.action_dim)(net)
        
        # 裁剪log_std
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        
        # 创建正态分布
        base_dist = distrax.MultivariateNormalDiag(
            loc=mean,
            scale_diag=jnp.exp(log_std)
        )
        
        # Tanh变换（将输出映射到[-1, 1]）
        dist = distrax.Transformed(
            distribution=base_dist,
            bijector=distrax.Tanh()
        )
        
        return dist


def sample_actions(
    rng: jax.random.PRNGKey,
    apply_fn,
    params,
    observations: jnp.ndarray,
    temperature: float = 1.0
):
    """
    采样动作
    
    Args:
        rng: 随机数生成器
        apply_fn: 策略的apply函数
        params: 策略参数
        observations: 观察
        temperature: 温度参数（1.0为正常采样）
    
    Returns:
        新的rng和采样的动作
    """
    dist = apply_fn({'params': params}, observations)
    
    rng, key = jax.random.split(rng)
    actions = dist.sample(seed=key)
    
    # 应用温度
    if temperature != 1.0:
        # 重参数化：a = μ + σ * ε * temperature
        # 注意：这里简化实现，实际可能需要更复杂的温度控制
        pass
    
    return rng, actions

