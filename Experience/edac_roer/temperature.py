"""温度参数模块（用于SAC的自动熵调节）"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple
from common import Model, InfoDict, Params


class Temperature(nn.Module):
    """可学习的温度参数"""
    initial_temperature: float = 1.0
    
    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            'log_temp',
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature))
        )
        return jnp.exp(log_temp)


def update(
    temp: Model,
    entropy: float,
    target_entropy: float
) -> Tuple[Model, InfoDict]:
    """
    更新温度参数
    
    目标：max α * (H(π) - H_target)
    即让当前策略的熵接近目标熵
    
    Args:
        temp: 温度模型
        entropy: 当前策略的熵
        target_entropy: 目标熵
    
    Returns:
        新的温度模型和信息字典
    """
    def temperature_loss_fn(temp_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        temperature = temp.apply_fn({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy)
        
        return temp_loss, {
            'temperature': temperature,
            'temperature_loss': temp_loss
        }
    
    new_temp, info = temp.apply_gradient(temperature_loss_fn)
    return new_temp, info

