"""
Training script for Jensen-Shannon Divergence ROER
"""
from jax import config
import os
from typing import Tuple

import os
import random
import datetime
import gym
import numpy as np
import tqdm
import time
import sys
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
from dataclasses import dataclass

from sac_learner import SACLearner
from dataset_utils import D4RLDataset, ReplayBuffer
from evaluation import evaluate
from env_utils import make_env

import warnings

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_string('eval_save_dir', './tmp/evaluation/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('updates_per_step', 1, 'Gradient updates per step.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('track', True, 'Track experiments with Weights and Biases.')
flags.DEFINE_string('wandb_project_name', "jaxrl-js-divergence", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', None, "the entity (team) of wandb's project")

# common
flags.DEFINE_float('actor_lr', 3e-4, 'actor learning rate')
flags.DEFINE_float('critic_lr', 3e-4, 'critic learning rate')
flags.DEFINE_float('value_lr', 3e-4, 'value learning rate')
flags.DEFINE_float('temp_lr', 3e-4, 'temperature learning rate')
flags.DEFINE_float('discount', 0.99, 'discount value')
flags.DEFINE_float('tau', 0.005, 'value of tau')

# JS divergence specific parameters
flags.DEFINE_boolean('per', True, 'Use prioritized experience replay')
flags.DEFINE_string('per_type', 'JS', 'Type of prioritization: JS, OER, PER, or UNIFORM')
flags.DEFINE_float('temp', 4.0, 'Loss temperature (beta in paper)')
flags.DEFINE_float('per_alpha', 1.0, 'Priority exponent alpha')
flags.DEFINE_float('per_beta', 1.0, 'Importance sampling beta')
flags.DEFINE_string('update_scheme', 'exp', 'Priority update scheme: exp or avg')

# JS divergence specific clipping
flags.DEFINE_float('min_clip', 0.5, 'Minimum priority clip for JS')
flags.DEFINE_float('max_clip', 50, 'Maximum exponential clip')
flags.DEFINE_float('gumbel_max_clip', 7, 'Gumbel loss clip')
flags.DEFINE_float('js_min_weight', 0.5, 'JS minimum weight for scaling sigmoid')
flags.DEFINE_float('js_max_weight', 2.0, 'JS maximum weight for scaling sigmoid')

# Other parameters
flags.DEFINE_boolean('log_loss', True, 'Use log loss variant')
flags.DEFINE_boolean('std_normalize', False, 'Normalize by standard deviation')
flags.DEFINE_boolean('noise', False, 'Add noise to actions')
flags.DEFINE_float('noise_std', 0.1, 'Noise standard deviation')
flags.DEFINE_boolean('grad_pen', False, 'Use gradient penalty')
flags.DEFINE_float('lambda_gp', 10.0, 'Gradient penalty coefficient')


def main(_):
    warnings.filterwarnings("ignore")
    
    # Create experiment name
    exp_name = f"js_divergence_{FLAGS.env_name}_temp{FLAGS.temp}_seed{FLAGS.seed}"
    run_name = f"{exp_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup directories
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    save_dir = os.path.join(FLAGS.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup wandb
    if FLAGS.track:
        import wandb
        wandb.init(
            project=FLAGS.wandb_project_name,
            entity=FLAGS.wandb_entity,
            config=vars(FLAGS),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    # Setup tensorboard
    summary_writer = SummaryWriter(os.path.join(save_dir, 'tb', run_name))
    
    # Set random seeds
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    
    # Create environment
    env = make_env(FLAGS.env_name, FLAGS.seed)
    eval_env = make_env(FLAGS.env_name, FLAGS.seed + 100)
    
    # Get observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Environment: {FLAGS.env_name}")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Using JS Divergence with temp={FLAGS.temp}")
    print(f"JS weight range: [{FLAGS.js_min_weight}, {FLAGS.js_max_weight}]")
    
    # Initialize agent
    kwargs = dict(FLAGS.__flags)
    agent = SACLearner(
        FLAGS.seed,
        env.observation_space.sample()[np.newaxis],
        env.action_space.sample()[np.newaxis],
        actor_lr=FLAGS.actor_lr,
        critic_lr=FLAGS.critic_lr,
        value_lr=FLAGS.value_lr,
        temp_lr=FLAGS.temp_lr,
        discount=FLAGS.discount,
        tau=FLAGS.tau,
        loss_temp=FLAGS.temp,
        args=FLAGS
    )
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(obs_dim, action_dim, capacity=int(1e6))
    
    # Training loop
    observation, done = env.reset(), False
    episode_return = 0
    episode_step = 0
    
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        
        # Collect experience
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        
        next_observation, reward, done, info = env.step(action)
        
        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0
        
        replay_buffer.insert(observation, action, reward, mask, float(done),
                           next_observation)
        observation = next_observation
        
        episode_return += reward
        episode_step += 1
        
        if done:
            observation, done = env.reset(), False
            summary_writer.add_scalar('training/return', episode_return, i)
            summary_writer.add_scalar('training/episode_length', episode_step, i)
            
            if FLAGS.track:
                wandb.log({
                    'training/return': episode_return,
                    'training/episode_length': episode_step,
                    'global_step': i
                })
            
            episode_return = 0
            episode_step = 0
        
        # Train agent
        if i >= FLAGS.start_training:
            for _ in range(FLAGS.updates_per_step):
                batch = replay_buffer.sample(FLAGS.batch_size)
                update_info = agent.update(batch, policy_update=True, per_type=FLAGS.per_type)
                
                if i % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        summary_writer.add_scalar(f'training/{k}', v, i)
                        
                    if FLAGS.track:
                        wandb_log = {f'training/{k}': v for k, v in update_info.items()}
                        wandb_log['global_step'] = i
                        wandb.log(wandb_log)
        
        # Evaluation
        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)
            
            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/{k}', v, i)
            
            print(f"\nStep {i}: Evaluation return = {eval_stats['return']:.2f}")
            
            if FLAGS.track:
                wandb_log = {f'evaluation/{k}': v for k, v in eval_stats.items()}
                wandb_log['global_step'] = i
                wandb.log(wandb_log)
    
    # Save final model
    agent.save(os.path.join(save_dir, 'final_model'))
    print(f"\nTraining completed. Model saved to {save_dir}")


if __name__ == '__main__':
    app.run(main)

