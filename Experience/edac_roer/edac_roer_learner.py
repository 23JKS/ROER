"""
EDAC + ROER 核心算法实现

将ROER的优先级机制集成到EDAC (Ensemble-Diversified Actor-Critic)
"""
import functools
from typing import Sequence, Tuple, Optional
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

# 从父目录导入
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from common import InfoDict, Model, Params, PRNGKey
from replay_buffer_roer import Batch
import policies
import temperature as temp_module


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
    
    # 标准化（可选）
    if std_normalize:
        exp_a = exp_a / jnp.mean(old_priority * exp_a)
    
    # EMA更新：w_new = λ * exp_a * w_old + (1-λ) * w_old
    priority = (per_beta * exp_a + (1 - per_beta)) * old_priority
    
    # 下界裁剪
    priority = jnp.maximum(priority, min_clip)
    
    return np.asarray(priority)


class EnsembleCritic(nn.Module):
    """EDAC的Ensemble Critic网络"""
    hidden_dims: Sequence[int]
    num_critics: int = 10  # EDAC默认使用10个critics
    
    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """
        返回所有critic的Q值
        
        Returns:
            shape: (num_critics, batch_size)
        """
        # 拼接obs和action
        inputs = jnp.concatenate([observations, actions], axis=-1)
        
        # 每个critic独立的MLP
        q_values = []
        for i in range(self.num_critics):
            net = inputs
            for hidden_dim in self.hidden_dims:
                net = nn.Dense(hidden_dim, name=f'critic_{i}_fc_{hidden_dim}')(net)
                net = nn.relu(net)
            q = nn.Dense(1, name=f'critic_{i}_output')(net)
            q_values.append(q.squeeze(-1))
        
        return jnp.stack(q_values, axis=0)  # (num_critics, batch_size)


class ValueCritic(nn.Module):
    """Value网络，用于ROER的TD误差计算"""
    hidden_dims: Sequence[int]
    
    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        net = observations
        for hidden_dim in self.hidden_dims:
            net = nn.Dense(hidden_dim)(net)
            net = nn.relu(net)
        value = nn.Dense(1)(net)
        return value.squeeze(-1)


def update_ensemble_critic(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    backup_entropy: bool,
    diversity_coef: float = 0.1
) -> Tuple[Model, InfoDict]:
    """
    更新EDAC的ensemble critic
    
    包含：
    1. 标准的TD loss
    2. Ensemble多样性正则化
    3. ROER的样本权重
    """
    # 采样下一个动作
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    
    # 获取所有target critic的Q值
    next_q_all = target_critic(batch.next_observations, next_actions)  # (num_critics, batch)
    
    # 使用最小Q值（减少过估计）
    next_q = jnp.min(next_q_all, axis=0)
    
    # 计算target
    target_q = batch.rewards + discount * batch.masks * next_q
    if backup_entropy:
        target_q -= discount * batch.masks * temp() * next_log_probs
    
    # ROER权重
    w = batch.priority
    
    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # 所有critic的Q值预测
        q_all = critic.apply({'params': critic_params}, 
                            batch.observations, 
                            batch.actions)  # (num_critics, batch)
        
        # 1. TD loss（加权）
        td_errors = q_all - target_q[None, :]  # (num_critics, batch)
        td_loss = jnp.mean(w * td_errors**2)  # 使用ROER权重
        
        # 2. Ensemble多样性正则化
        # 鼓励不同critic给出不同的预测
        q_std = jnp.std(q_all, axis=0)  # 每个样本上的标准差
        diversity_loss = -jnp.mean(q_std)  # 负号：鼓励大方差
        
        # 总损失
        total_loss = td_loss + diversity_coef * diversity_loss
        
        return total_loss, {
            'critic_loss': td_loss,
            'diversity_loss': diversity_loss,
            'q_mean': jnp.mean(q_all),
            'q_std_mean': jnp.mean(q_std),
            'target_q': jnp.mean(target_q)
        }
    
    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


def update_value(
    critic: Model,
    value: Model,
    batch: Batch,
    loss_temp: float,
    discount: float,
    gumbel_max_clip: float = 7.0,
    use_ensemble: bool = True
) -> Tuple[Model, InfoDict]:
    """
    更新Value网络（用于ROER的TD误差计算）
    
    使用Gumbel rescale loss（来自ROER论文）
    """
    obs = batch.observations
    acts = batch.actions
    
    # 使用ensemble的平均Q值
    q_all = critic(obs, acts)  # (num_critics, batch)
    if use_ensemble:
        q = jnp.mean(q_all, axis=0)  # ensemble平均
    else:
        q = jnp.min(q_all, axis=0)  # 或者用最小值
    
    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, obs)
        
        # Gumbel rescale loss (ROER的核心)
        diff = q - v
        z = diff / loss_temp
        z = jnp.minimum(z, gumbel_max_clip)
        
        # exp(z) - z - 1
        loss = jnp.exp(z) - z - 1
        
        # 归一化
        norm = jnp.mean(jnp.maximum(1, jnp.exp(z)))
        norm = jax.lax.stop_gradient(norm)
        loss = loss / norm
        
        value_loss = jnp.mean(loss)
        
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': jnp.mean(v),
            'q_mean': jnp.mean(q),
            'norm': norm
        }
    
    new_value, info = value.apply_gradient(value_loss_fn)
    return new_value, info


def compute_td_error_for_priority(
    critic: Model,
    value: Model,
    batch: Batch,
    discount: float,
    use_ensemble: bool = True
) -> jnp.ndarray:
    """
    计算用于ROER优先级的TD误差
    
    使用Value网络的TD误差：δ = r + γV(s') - V(s)
    """
    # 当前状态的V值
    current_v = value(batch.observations)
    
    # 下一状态的V值
    next_v = value(batch.next_observations)
    
    # TD误差
    target_v = batch.rewards + discount * batch.masks * next_v
    td_error = target_v - current_v
    
    return td_error


def update_actor(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    temp: Model,
    batch: Batch,
    eta: float = 1.0  # EDAC的diversity权重
) -> Tuple[Model, InfoDict]:
    """
    更新Actor（策略网络）
    
    使用ensemble Q值，加入diversity鼓励探索
    """
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params}, batch.observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        
        # 所有critic的Q值
        q_all = critic(batch.observations, actions)  # (num_critics, batch)
        
        # EDAC: 使用平均Q值
        q_mean = jnp.mean(q_all, axis=0)
        
        # EDAC diversity: 鼓励在Q值方差大的地方探索
        q_std = jnp.std(q_all, axis=0)
        
        # Actor loss: -Q + α*logπ - η*std(Q)
        actor_loss = (temp() * log_probs - q_mean - eta * q_std).mean()
        
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'q_actor': q_mean.mean(),
            'q_std_actor': q_std.mean()
        }
    
    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def target_update(model: Model, target_model: Model, tau: float) -> Model:
    """软更新目标网络"""
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau),
        model.params,
        target_model.params
    )
    return target_model.replace(params=new_target_params)


@functools.partial(jax.jit, static_argnames=['backup_entropy', 'update_target'])
def _update_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    value: Model,
    target_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    tau: float,
    loss_temp: float,
    target_entropy: float,
    backup_entropy: bool,
    update_target: bool,
    diversity_coef: float,
    eta: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:
    """
    JIT编译的更新函数（完整的训练步）
    """
    rng, key = jax.random.split(rng)
    
    # 1. 更新Ensemble Critic
    new_critic, critic_info = update_ensemble_critic(
        key, actor, critic, target_critic, temp, batch,
        discount, backup_entropy, diversity_coef
    )
    
    # 2. 更新Value网络（用于ROER）
    new_value, value_info = update_value(
        new_critic, value, batch, loss_temp, discount
    )
    
    # 3. 更新Actor
    new_actor, actor_info = update_actor(
        key, actor, new_critic, temp, batch, eta
    )
    
    # 4. 更新温度
    new_temp, temp_info = temp_module.update(
        temp, actor_info['entropy'], target_entropy
    )
    
    # 5. 软更新目标网络
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic
    
    # 6. 计算TD误差（用于下一轮的priority更新）
    td_error = compute_td_error_for_priority(
        new_critic, new_value, batch, discount
    )
    
    return rng, new_actor, new_critic, new_value, new_target_critic, new_temp, {
        **critic_info,
        **value_info,
        **actor_info,
        **temp_info,
        'td_error': td_error
    }


class EDACROERLearner:
    """
    EDAC + ROER 学习器
    
    结合了：
    - EDAC的ensemble critics和diversity正则化
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
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        num_critics: int = 10,
        discount: float = 0.99,
        tau: float = 0.005,
        target_update_period: int = 1,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        init_temperature: float = 1.0,
        # ROER相关参数
        loss_temp: float = 1.0,
        roer_per_beta: float = 0.01,
        roer_max_clip: float = 100.0,
        roer_min_clip: float = 10.0,
        roer_std_normalize: bool = True,
        # EDAC相关参数
        diversity_coef: float = 0.1,
        eta: float = 1.0
    ):
        """
        初始化EDAC+ROER学习器
        
        Args:
            num_critics: Ensemble中critic的数量
            diversity_coef: Critic多样性正则化系数
            eta: Actor中Q标准差的权重
            loss_temp: ROER温度参数β
            roer_per_beta: ROER的EMA系数λ
            其他参数同SAC
        """
        action_dim = actions.shape[-1]
        
        # 设置target entropy
        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy
        
        self.backup_entropy = backup_entropy
        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount
        self.diversity_coef = diversity_coef
        self.eta = eta
        
        # ROER参数
        self.loss_temp = loss_temp
        self.roer_per_beta = roer_per_beta
        self.roer_max_clip = roer_max_clip
        self.roer_min_clip = roer_min_clip
        self.roer_std_normalize = roer_std_normalize
        
        # 初始化随机数生成器
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, temp_key = jax.random.split(rng, 5)
        
        # 1. Actor网络
        actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim)
        actor = Model.create(
            actor_def,
            inputs=[actor_key, observations],
            tx=optax.adam(learning_rate=actor_lr)
        )
        
        # 2. Ensemble Critic网络
        critic_def = EnsembleCritic(hidden_dims, num_critics=num_critics)
        critic = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions],
            tx=optax.adam(learning_rate=critic_lr)
        )
        target_critic = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions]
        )
        
        # 3. Value网络（用于ROER）
        value_def = ValueCritic(hidden_dims)
        value = Model.create(
            value_def,
            inputs=[value_key, observations],
            tx=optax.adam(learning_rate=value_lr)
        )
        
        # 4. 温度参数
        temp = Model.create(
            temp_module.Temperature(init_temperature),
            inputs=[temp_key],
            tx=optax.adam(learning_rate=temp_lr)
        )
        
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.value = value
        self.temp = temp
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
         new_target_critic, new_temp, info) = _update_jit(
            self.rng,
            self.actor,
            self.critic,
            self.value,
            self.target_critic,
            self.temp,
            batch,
            self.discount,
            self.tau,
            self.loss_temp,
            self.target_entropy,
            self.backup_entropy,
            update_target,
            self.diversity_coef,
            self.eta
        )
        
        # 更新模型
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic
        self.temp = new_temp
        
        # 计算新的priority（使用ROER）
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
            'temp': self.temp,
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
        self.temp = state['temp']
        self.step = state['step']

