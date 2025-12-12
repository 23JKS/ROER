"""
IQL + ROER 核心算法实现

将ROER的优先级机制集成到IQL (Implicit Q-Learning)
"""
import functools
from typing import Sequence, Tuple, Optional
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from common import InfoDict, Model, Params, PRNGKey
from replay_buffer_roer import Batch
import policies
import temperature as temp_module


def expectile_loss(diff: jnp.ndarray, expectile: float = 0.7) -> jnp.ndarray:
    """
    IQL的核心：Expectile回归损失
    
    当expectile=0.5时，等价于MSE
    当expectile=0.7时，更关注正的diff（即V(s) < target_Q的情况）
    
    Args:
        diff: Q - V
        expectile: expectile参数，通常0.7或0.8
    
    Returns:
        expectile loss
    """
    weight = jnp.where(diff > 0, expectile, 1 - expectile)
    return weight * (diff ** 2)


def compute_roer_priority(
    td_error: jnp.ndarray,
    old_priority: np.ndarray,
    loss_temp: float,
    per_beta: float,
    max_clip: float,
    min_clip: float,
    std_normalize: bool = True
) -> np.ndarray:
    """
    计算ROER优先级权重
    
    Args:
        td_error: TD误差
        old_priority: 旧的优先级
        loss_temp: 温度参数β
        per_beta: EMA系数λ
        max_clip: 最大裁剪值
        min_clip: 最小裁剪值
        std_normalize: 是否标准化
    
    Returns:
        新的优先级权重
    """
    # 计算 exp(TD_error / β)
    a = td_error / loss_temp
    exp_a = jnp.minimum(jnp.exp(a), max_clip)
    exp_a = jnp.maximum(exp_a, 1.0)
    
    # 标准化
    if std_normalize:
        exp_a = exp_a / jnp.mean(old_priority * exp_a)
    
    # EMA更新
    priority = (per_beta * exp_a + (1 - per_beta)) * old_priority
    
    # 下界裁剪
    priority = jnp.maximum(priority, min_clip)
    
    return np.asarray(priority)


class QNetwork(nn.Module):
    """Q网络"""
    hidden_dims: Sequence[int]
    
    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        x = inputs
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        q = nn.Dense(1)(x)
        return q.squeeze(-1)


class DoubleQNetwork(nn.Module):
    """双Q网络（TD3风格）"""
    hidden_dims: Sequence[int]
    
    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Q1
        inputs = jnp.concatenate([observations, actions], axis=-1)
        x1 = inputs
        for i, hidden_dim in enumerate(self.hidden_dims):
            x1 = nn.Dense(hidden_dim, name=f'q1_fc{i}')(x1)
            x1 = nn.relu(x1)
        q1 = nn.Dense(1, name='q1_out')(x1).squeeze(-1)
        
        # Q2
        x2 = inputs
        for i, hidden_dim in enumerate(self.hidden_dims):
            x2 = nn.Dense(hidden_dim, name=f'q2_fc{i}')(x2)
            x2 = nn.relu(x2)
        q2 = nn.Dense(1, name='q2_out')(x2).squeeze(-1)
        
        return q1, q2


class ValueNetwork(nn.Module):
    """V网络"""
    hidden_dims: Sequence[int]
    
    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = observations
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        v = nn.Dense(1)(x)
        return v.squeeze(-1)


def update_q(
    critic: Model,
    target_value: Model,
    batch: Batch,
    discount: float
) -> Tuple[Model, InfoDict]:
    """
    更新Q网络
    
    使用标准的TD learning（加ROER权重）
    """
    next_v = target_value(batch.next_observations)
    target_q = batch.rewards + discount * batch.masks * next_v
    
    # ROER权重
    w = batch.priority
    
    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply({'params': critic_params}, 
                             batch.observations, 
                             batch.actions)
        
        # 加权MSE loss
        critic_loss = jnp.mean(w * ((q1 - target_q) ** 2 + (q2 - target_q) ** 2))
        
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1_mean': q1.mean(),
            'q2_mean': q2.mean(),
            'target_q_mean': target_q.mean()
        }
    
    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


def update_v(
    value: Model,
    critic: Model,
    batch: Batch,
    expectile: float = 0.7
) -> Tuple[Model, InfoDict]:
    """
    更新V网络
    
    使用IQL的expectile regression（加入ROER权重）
    """
    # 计算target Q（两个Q网络的最小值）
    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    
    # ROER权重
    w = batch.priority
    
    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, batch.observations)
        
        # Expectile loss（IQL的核心）
        diff = q - v
        loss = expectile_loss(diff, expectile)
        
        # 加权（ROER）
        value_loss = jnp.mean(w * loss)
        
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'q_v_diff': diff.mean(),
            'q_v_diff_abs': jnp.abs(diff).mean()
        }
    
    new_value, info = value.apply_gradient(value_loss_fn)
    return new_value, info


def update_actor(
    actor: Model,
    critic: Model,
    value: Model,
    batch: Batch,
    beta: float = 3.0,
    clip_score: float = 100.0
) -> Tuple[Model, InfoDict]:
    """
    更新Actor网络
    
    使用IQL的advantage-weighted behavioral cloning
    """
    # 计算advantage
    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    v = value(batch.observations)
    adv = q - v
    
    # Advantage weighting（IQL的核心）
    exp_adv = jnp.exp(adv * beta)
    exp_adv = jnp.minimum(exp_adv, clip_score)
    
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params}, batch.observations)
        log_prob = dist.log_prob(batch.actions)
        
        # Advantage-weighted BC
        actor_loss = -(exp_adv * log_prob).mean()
        
        return actor_loss, {
            'actor_loss': actor_loss,
            'adv_mean': adv.mean(),
            'adv_max': adv.max(),
            'adv_min': adv.min(),
            'exp_adv_mean': exp_adv.mean()
        }
    
    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def compute_td_error_for_priority(
    value: Model,
    batch: Batch,
    discount: float
) -> jnp.ndarray:
    """
    计算用于ROER优先级的TD误差
    
    使用V网络的TD误差：δ = r + γV(s') - V(s)
    """
    current_v = value(batch.observations)
    next_v = value(batch.next_observations)
    target_v = batch.rewards + discount * batch.masks * next_v
    td_error = target_v - current_v
    return td_error


def target_update(model: Model, target_model: Model, tau: float) -> Model:
    """软更新目标网络"""
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau),
        model.params,
        target_model.params
    )
    return target_model.replace(params=new_target_params)


@functools.partial(jax.jit, static_argnames=['update_target'])
def _update_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    value: Model,
    target_value: Model,
    batch: Batch,
    discount: float,
    tau: float,
    expectile: float,
    beta: float,
    update_target: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    """
    JIT编译的更新函数（完整的IQL训练步）
    """
    rng, key = jax.random.split(rng)
    
    # 1. 更新Q网络
    new_critic, critic_info = update_q(
        critic, target_value, batch, discount
    )
    
    # 2. 更新V网络
    new_value, value_info = update_v(
        value, new_critic, batch, expectile
    )
    
    # 3. 更新Actor
    new_actor, actor_info = update_actor(
        actor, new_critic, new_value, batch, beta
    )
    
    # 4. 软更新目标网络
    if update_target:
        new_target_value = target_update(new_value, target_value, tau)
    else:
        new_target_value = target_value
    
    # 5. 计算TD误差（用于下一轮的priority更新）
    td_error = compute_td_error_for_priority(
        new_value, batch, discount
    )
    
    return rng, new_actor, new_critic, new_value, new_target_value, {
        **critic_info,
        **value_info,
        **actor_info,
        'td_error': td_error
    }


class IQLROERLearner:
    """
    IQL + ROER 学习器
    
    结合了：
    - IQL的expectile regression和advantage-weighted BC
    - ROER的优先级经验回放
    """
    
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        target_update_period: int = 1,
        # IQL特定参数
        expectile: float = 0.7,
        beta: float = 3.0,
        # ROER相关参数
        loss_temp: float = 1.0,
        roer_per_beta: float = 0.01,
        roer_max_clip: float = 100.0,
        roer_min_clip: float = 10.0,
        roer_std_normalize: bool = True
    ):
        """
        初始化IQL+ROER学习器
        
        Args:
            expectile: IQL的expectile参数（0.7或0.8）
            beta: advantage weighting的温度参数
            loss_temp: ROER温度参数β
            roer_per_beta: ROER的EMA系数λ
            其他参数同标准RL
        """
        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount
        self.expectile = expectile
        self.beta = beta
        
        # ROER参数
        self.loss_temp = loss_temp
        self.roer_per_beta = roer_per_beta
        self.roer_max_clip = roer_max_clip
        self.roer_min_clip = roer_min_clip
        self.roer_std_normalize = roer_std_normalize
        
        # 初始化随机数生成器
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
        
        action_dim = actions.shape[-1]
        
        # 1. Actor网络
        actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim)
        actor = Model.create(
            actor_def,
            inputs=[actor_key, observations],
            tx=optax.adam(learning_rate=actor_lr)
        )
        
        # 2. 双Q网络（TD3风格）
        critic_def = DoubleQNetwork(hidden_dims)
        critic = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions],
            tx=optax.adam(learning_rate=critic_lr)
        )
        
        # 3. V网络
        value_def = ValueNetwork(hidden_dims)
        value = Model.create(
            value_def,
            inputs=[value_key, observations],
            tx=optax.adam(learning_rate=value_lr)
        )
        target_value = Model.create(
            value_def,
            inputs=[value_key, observations]
        )
        
        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_value = target_value
        self.rng = rng
        self.step = 0
    
    def sample_actions(
        self,
        observations: np.ndarray,
        temperature: float = 1.0
    ) -> np.ndarray:
        """采样动作"""
        rng, actions = policies.sample_actions(
            self.rng,
            self.actor.apply_fn,
            self.actor.params,
            observations,
            temperature
        )
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    
    def update(self, batch: Batch) -> InfoDict:
        """
        执行一次完整更新
        
        Returns:
            包含loss信息和新的priority的字典
        """
        self.step += 1
        update_target = (self.step % self.target_update_period == 0)
        
        # JIT编译的更新
        (self.rng, new_actor, new_critic, new_value,
         new_target_value, info) = _update_jit(
            self.rng,
            self.actor,
            self.critic,
            self.value,
            self.target_value,
            batch,
            self.discount,
            self.tau,
            self.expectile,
            self.beta,
            update_target
        )
        
        # 更新模型
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_value = new_target_value
        
        # 计算新的priority
        td_error = info['td_error']
        new_priority = compute_roer_priority(
            td_error,
            batch.priority,
            self.loss_temp,
            self.roer_per_beta,
            self.roer_max_clip,
            self.roer_min_clip,
            self.roer_std_normalize
        )
        
        info['priority'] = new_priority
        return info
    
    def save(self, save_path: str):
        """保存模型"""
        import pickle
        state = {
            'actor': self.actor,
            'critic': self.critic,
            'value': self.value,
            'step': self.step
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, load_path: str):
        """加载模型"""
        import pickle
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        self.actor = state['actor']
        self.critic = state['critic']
        self.value = state['value']
        self.step = state['step']

