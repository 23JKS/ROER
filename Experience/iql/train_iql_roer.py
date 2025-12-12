"""
IQL + ROER 训练脚本

用法：
    python train_iql_roer.py --env_name=HalfCheetah-v2 --use_roer=True
"""
import os
import sys
import random
import datetime
import time
import numpy as np
import gym
import tqdm
from absl import app, flags
from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from iql_roer_learner import IQLROERLearner
from replay_buffer_roer import ReplayBufferROER
from evaluation_utils import evaluate
from env_utils import make_env

FLAGS = flags.FLAGS

# 环境设置
flags.DEFINE_string('env_name', 'HalfCheetah-v2', '环境名称')
flags.DEFINE_integer('seed', 42, '随机种子')
flags.DEFINE_integer('max_steps', int(1e6), '训练总步数')
flags.DEFINE_integer('start_training', int(1e4), '开始训练的步数')
flags.DEFINE_integer('batch_size', 256, '批次大小')
flags.DEFINE_integer('eval_episodes', 10, '评估回合数')
flags.DEFINE_integer('eval_interval', 5000, '评估间隔')
flags.DEFINE_integer('log_interval', 1000, '日志间隔')

# IQL参数
flags.DEFINE_float('expectile', 0.7, 'IQL的expectile参数')
flags.DEFINE_float('iql_beta', 3.0, 'Advantage weighting温度')

# ROER参数
flags.DEFINE_boolean('use_roer', True, '是否使用ROER优先级')
flags.DEFINE_float('roer_temp', 4.0, 'ROER温度参数β')
flags.DEFINE_float('roer_per_beta', 0.01, 'ROER的EMA系数λ')
flags.DEFINE_float('roer_max_clip', 50.0, 'ROER最大裁剪值')
flags.DEFINE_float('roer_min_clip', 10.0, 'ROER最小裁剪值')
flags.DEFINE_boolean('roer_std_normalize', True, 'ROER是否标准化')

# 学习率
flags.DEFINE_float('actor_lr', 3e-4, 'Actor学习率')
flags.DEFINE_float('critic_lr', 3e-4, 'Critic学习率')
flags.DEFINE_float('value_lr', 3e-4, 'Value学习率')

# 其他
flags.DEFINE_float('discount', 0.99, '折扣因子')
flags.DEFINE_float('tau', 0.005, '目标网络更新系数')
flags.DEFINE_integer('capacity', int(1e6), 'Replay buffer容量')
flags.DEFINE_string('save_dir', './results/iql_roer/', '结果保存目录')
flags.DEFINE_boolean('tqdm', True, '是否显示进度条')


def main(_):
    # 设置随机种子
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    
    # 创建保存目录
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    algo_name = 'iql_roer' if FLAGS.use_roer else 'iql_baseline'
    save_dir = os.path.join(
        os.path.expanduser('~'),
        'roer_output',
        FLAGS.save_dir.lstrip('./'),
        algo_name,
        f'{FLAGS.env_name}_seed{FLAGS.seed}_{ts_str}'
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # TensorBoard
    summary_writer = SummaryWriter(
        os.path.join(save_dir, 'tb'),
        write_to_disk=True
    )
    
    # 创建环境
    env = make_env(FLAGS.env_name, FLAGS.seed)
    eval_env = make_env(FLAGS.env_name, FLAGS.seed + 42)
    
    # 创建IQL+ROER学习器
    agent = IQLROERLearner(
        seed=FLAGS.seed,
        observations=env.observation_space.sample()[np.newaxis],
        actions=env.action_space.sample()[np.newaxis],
        actor_lr=FLAGS.actor_lr,
        critic_lr=FLAGS.critic_lr,
        value_lr=FLAGS.value_lr,
        discount=FLAGS.discount,
        tau=FLAGS.tau,
        expectile=FLAGS.expectile,
        beta=FLAGS.iql_beta,
        loss_temp=FLAGS.roer_temp,
        roer_per_beta=FLAGS.roer_per_beta,
        roer_max_clip=FLAGS.roer_max_clip,
        roer_min_clip=FLAGS.roer_min_clip,
        roer_std_normalize=FLAGS.roer_std_normalize
    )
    
    # 创建Replay Buffer
    replay_buffer = ReplayBufferROER(
        env.observation_space,
        env.action_space,
        FLAGS.capacity
    )
    
    # 训练循环
    observation, done = env.reset(), False
    best_eval_returns = -np.inf
    eval_returns = []
    
    print(f"\n{'='*60}")
    print(f"IQL+ROER 训练开始")
    print(f"环境: {FLAGS.env_name}")
    print(f"使用ROER: {FLAGS.use_roer}")
    print(f"IQL参数: expectile={FLAGS.expectile}, beta={FLAGS.iql_beta}")
    print(f"ROER参数: temp={FLAGS.roer_temp}, max_clip={FLAGS.roer_max_clip}")
    print(f"保存路径: {save_dir}")
    print(f"{'='*60}\n")
    
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        
        # 1. 收集经验
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        
        next_observation, reward, done, info = env.step(action)
        
        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0
        
        replay_buffer.insert(observation, action, reward, mask, next_observation)
        observation = next_observation
        
        if done:
            observation, done = env.reset(), False
            if 'episode' in info:
                for k, v in info['episode'].items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
        
        # 2. 训练更新
        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)
            
            # 更新优先级
            if FLAGS.use_roer and 'priority' in update_info:
                priority = update_info['priority']
                replay_buffer.update_priority(batch.indx, priority)
            
            # 记录日志
            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    if k == 'priority' or k == 'td_error':
                        continue
                    if isinstance(v, (int, float, np.number)):
                        summary_writer.add_scalar(f'training/{k}', v, i)
                    elif hasattr(v, 'ndim') and v.ndim == 0:
                        summary_writer.add_scalar(f'training/{k}', v, i)
                
                if FLAGS.use_roer:
                    priority_stats = replay_buffer.get_priority_stats()
                    for k, v in priority_stats.items():
                        summary_writer.add_scalar(f'priority/{k}', v, i)
                
                summary_writer.flush()
        
        # 3. 评估
        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)
            
            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/{k}', v, i)
            
            if eval_stats['return'] >= best_eval_returns:
                best_eval_returns = eval_stats['return']
                agent.save(os.path.join(save_dir, 'best_model.pkl'))
            
            summary_writer.add_scalar('evaluation/best_return', best_eval_returns, i)
            summary_writer.flush()
            
            eval_returns.append((i, eval_stats['return']))
            np.savetxt(
                os.path.join(save_dir, 'eval_returns.txt'),
                eval_returns,
                fmt=['%d', '%.1f'],
                header='step return'
            )
            
            print(f"\nStep {i}: Eval Return = {eval_stats['return']:.1f}, Best = {best_eval_returns:.1f}")
    
    print(f"\n训练完成！最佳回报: {best_eval_returns:.1f}")
    print(f"结果保存在: {save_dir}")


if __name__ == '__main__':
    app.run(main)

