"""
ROER优先级Replay Buffer实现
支持均匀采样 + 重加权损失的方式
"""
from typing import Optional, Union, Tuple
import gym
import numpy as np


class Batch:
    """数据批次类"""
    def __init__(self, observations, actions, rewards, masks, 
                 next_observations, priority, indx):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.next_observations = next_observations
        self.priority = priority
        self.indx = indx


class ReplayBufferROER:
    """
    带ROER优先级的Replay Buffer
    
    采样策略：均匀随机采样
    权重策略：通过priority字段提供样本权重，用于loss加权
    覆盖策略：FIFO（先进先出）环形缓冲区
    """
    
    def __init__(self, 
                 observation_space: gym.spaces.Box,
                 action_space: Union[gym.spaces.Discrete, gym.spaces.Box], 
                 capacity: int):
        """
        Args:
            observation_space: 观察空间
            action_space: 动作空间
            capacity: 缓冲区容量
        """
        self.capacity = capacity
        self.size = 0
        self.insert_index = 0
        
        # 预分配内存
        self.observations = np.empty((capacity, *observation_space.shape),
                                    dtype=observation_space.dtype)
        self.actions = np.empty((capacity, *action_space.shape),
                               dtype=action_space.dtype)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.masks = np.empty((capacity,), dtype=np.float32)
        self.next_observations = np.empty((capacity, *observation_space.shape),
                                         dtype=observation_space.dtype)
        
        # 优先级初始化为1（均匀）
        self.priority = np.ones((capacity,), dtype=np.float32)
    
    def insert(self, 
               observation: np.ndarray, 
               action: np.ndarray,
               reward: float, 
               mask: float,
               next_observation: np.ndarray):
        """插入一条经验，使用FIFO策略覆盖旧数据"""
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.next_observations[self.insert_index] = next_observation
        
        # 新插入的数据优先级初始化为当前平均值（避免极端）
        if self.size > 0:
            self.priority[self.insert_index] = np.mean(self.priority[:self.size])
        else:
            self.priority[self.insert_index] = 1.0
        
        # 环形缓冲区索引更新
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Batch:
        """
        均匀随机采样
        
        注意：这里是均匀采样，不按priority加权采样概率
        priority只作为batch的一个字段返回，用于后续loss加权
        """
        # 均匀随机采样索引
        indx = np.random.randint(self.size, size=batch_size)
        
        return Batch(
            observations=self.observations[indx],
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            masks=self.masks[indx],
            next_observations=self.next_observations[indx],
            priority=self.priority[indx],  # 权重用于loss加权
            indx=indx
        )
    
    def update_priority(self, indices: np.ndarray, priorities: np.ndarray):
        """
        更新指定样本的优先级
        
        Args:
            indices: 样本索引
            priorities: 新的优先级值
        """
        self.priority[indices] = priorities
    
    def get_priority_stats(self) -> dict:
        """获取优先级分布统计"""
        valid_priority = self.priority[:self.size]
        return {
            'mean': np.mean(valid_priority),
            'std': np.std(valid_priority),
            'min': np.min(valid_priority),
            'max': np.max(valid_priority),
            'median': np.median(valid_priority)
        }
    
    def __len__(self):
        return self.size


class WeightedReplayBufferROER(ReplayBufferROER):
    """
    加权采样版本的ROER Replay Buffer（可选实现）
    
    按priority加权采样，需要重要性采样修正
    更接近经典PER，但实现更复杂
    """
    
    def sample_weighted(self, batch_size: int, beta: float = 1.0) -> Batch:
        """
        按优先级加权采样
        
        Args:
            batch_size: 批次大小
            beta: 重要性采样修正系数 [0,1]
        """
        valid_priority = self.priority[:self.size]
        
        # 归一化为采样概率
        probs = valid_priority / np.sum(valid_priority)
        
        # 按概率采样
        indx = np.random.choice(self.size, size=batch_size, p=probs)
        
        # 计算重要性采样权重
        # w_i = (N * P(i))^(-beta)
        N = self.size
        weights = (N * probs[indx]) ** (-beta)
        weights /= np.max(weights)  # 归一化到[0,1]
        
        return Batch(
            observations=self.observations[indx],
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            masks=self.masks[indx],
            next_observations=self.next_observations[indx],
            priority=weights,  # 这里是IS权重，不是原始priority
            indx=indx
        )

