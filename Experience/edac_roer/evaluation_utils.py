"""评估工具函数"""
import numpy as np
import gym
from typing import Dict


def evaluate(
    agent,
    env: gym.Env,
    num_episodes: int,
    temperature: float = 1.0
) -> Dict[str, float]:
    """
    评估agent性能
    
    Args:
        agent: 训练的agent
        env: 评估环境
        num_episodes: 评估回合数
        temperature: 采样温度（1.0为正常采样）
    
    Returns:
        包含统计信息的字典
    """
    returns = []
    lengths = []
    
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        episode_return = 0.0
        episode_length = 0
        
        while not done:
            action = agent.sample_actions(observation, temperature=temperature)
            observation, reward, done, info = env.step(action)
            episode_return += reward
            episode_length += 1
        
        returns.append(episode_return)
        lengths.append(episode_length)
    
    return {
        'return': np.mean(returns),
        'return_std': np.std(returns),
        'return_min': np.min(returns),
        'return_max': np.max(returns),
        'length': np.mean(lengths)
    }

