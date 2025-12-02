import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

# Fix flax ShapedArray import issue for JAX 0.4.20
import jax
import jax.core
if not hasattr(jax, 'ShapedArray'):
    jax.ShapedArray = jax.core.ShapedArray

# Fix jax.xla compatibility issue for flax serialization
try:
    import jax._src.xla_bridge as xla_bridge
    if not hasattr(jax, 'xla'):
        # Create a mock xla module for compatibility
        class MockXLA:
            DeviceArray = jax.Array
        jax.xla = MockXLA()
except (ImportError, AttributeError):
    pass

import flax
import flax.linen as nn
import jax.numpy as jnp
import optax
from dataclasses import dataclass
import numpy as np

@dataclass
class Batch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    next_observations: np.ndarray
    priority: np.ndarray


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


# TODO: Replace with TrainState when it's ready
# https://github.com/google/flax/blob/master/docs/flip/1009-optimizer-api.md#train-state
@flax.struct.dataclass
class Model:
    step: int
    apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        _, params = variables.pop('params')
        # print("pop params", variables.pop('params'))

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def.apply,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn(*args, **kwargs)

    def apply_gradient(
            self,
            loss_fn: Optional[Callable[[Params], Any]] = None,
            grads: Optional[Any] = None,
            has_aux: bool = True) -> Union[Tuple['Model', Any], 'Model']:
        assert loss_fn is not None or grads is not None, \
                'Either a loss function or grads must be specified.'
        if grads is None:
            grad_fn = jax.grad(loss_fn, has_aux=has_aux)
            if has_aux:
                grads, aux = grad_fn(self.params)
            else:
                grads = grad_fn(self.params)
        else:
            assert has_aux, \
                    'When grads are provided, expects no aux outputs.'

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1,
                                 params=new_params,
                                 opt_state=new_opt_state)
        if has_aux:
            return new_model, aux
        else:
            return new_model

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Convert JAX device arrays to numpy arrays before serialization to avoid jax.xla compatibility issues
        # Use jax.device_get to move arrays from device to host, then convert to numpy
        try:
            params_host = jax.device_get(self.params)
            # Convert to numpy arrays to avoid jax.xla.DeviceArray issues in flax serialization
            params_numpy = jax.tree_map(lambda x: np.asarray(x) if isinstance(x, (jnp.ndarray, jax.Array)) else x, params_host)
        except (AttributeError, TypeError):
            # Fallback: convert to numpy arrays manually
            params_numpy = jax.tree_map(lambda x: np.asarray(x) if hasattr(x, '__array__') else x, self.params)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(params_numpy))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)