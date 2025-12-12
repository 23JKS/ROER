"""通用类型和函数定义"""
from typing import Any, Callable, Sequence, Tuple, Dict
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

# 类型别名
PRNGKey = Any
Params = Any
InfoDict = Dict[str, float]


class TrainState(train_state.TrainState):
    """扩展的训练状态，支持批量归一化等"""
    batch_stats: Any = None


class Model:
    """简化的模型封装"""
    
    def __init__(
        self,
        apply_fn: Callable,
        params: Params,
        tx: optax.GradientTransformation = None
    ):
        self.apply_fn = apply_fn
        self.params = params
        self.tx = tx
        if tx is not None:
            self.opt_state = tx.init(params)
        else:
            self.opt_state = None
    
    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        inputs: Sequence,
        tx: optax.GradientTransformation = None
    ) -> 'Model':
        """创建模型"""
        # 初始化参数
        key = inputs[0]
        variables = model_def.init(key, *inputs[1:])
        params = variables['params']
        
        return cls(model_def.apply, params, tx)
    
    def __call__(self, *args, **kwargs):
        """前向传播"""
        return self.apply_fn({'params': self.params}, *args, **kwargs)
    
    def apply_gradient(
        self,
        loss_fn: Callable[[Params], Tuple[jnp.ndarray, InfoDict]]
    ) -> Tuple['Model', InfoDict]:
        """应用梯度更新"""
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, info), grads = grad_fn(self.params)
        
        if self.tx is not None:
            updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
            new_params = optax.apply_updates(self.params, updates)
            return self.replace(params=new_params, opt_state=new_opt_state), info
        else:
            return self, info
    
    def replace(self, **kwargs):
        """替换属性"""
        new_model = Model(self.apply_fn, self.params, self.tx)
        for key, value in kwargs.items():
            setattr(new_model, key, value)
        return new_model

