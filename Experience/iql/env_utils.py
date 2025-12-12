"""环境工具函数"""
import gym
import numpy as np


def make_env(env_name: str, seed: int) -> gym.Env:
    """
    创建环境并设置随机种子
    
    Args:
        env_name: 环境名称
        seed: 随机种子
    
    Returns:
        gym.Env: 配置好的环境
    """
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def wrap_env(env: gym.Env) -> gym.Env:
    """
    为环境添加包装器（如果需要）
    
    Args:
        env: 原始环境
    
    Returns:
        包装后的环境
    """
    # 这里可以添加各种wrapper，例如：
    # - RecordEpisodeStatistics
    # - NormalizeObservation
    # - NormalizeReward
    return env

