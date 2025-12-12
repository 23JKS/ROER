"""
EDAC+ROER 默认配置
"""
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    
    # 算法类型
    config.algo = 'edac_roer'
    
    # 网络结构
    config.hidden_dims = (256, 256)
    config.num_critics = 10  # EDAC ensemble大小
    
    # 学习率
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.value_lr = 3e-4
    config.temp_lr = 3e-4
    
    # RL基础参数
    config.discount = 0.99
    config.tau = 0.005
    config.target_update_period = 1
    config.backup_entropy = True
    config.init_temperature = 1.0
    config.target_entropy = None  # 自动设置为 -action_dim/2
    
    # EDAC特定参数
    config.diversity_coef = 0.1  # Critic多样性正则化
    config.eta = 1.0  # Actor中Q标准差的权重
    
    # ROER参数
    config.loss_temp = 4.0  # β: 温度参数（MuJoCo默认4.0，DM Control默认1.0）
    config.roer_per_beta = 0.01  # λ: EMA系数
    config.roer_max_clip = 50.0  # 最大优先级裁剪
    config.roer_min_clip = 10.0  # 最小优先级裁剪
    config.roer_std_normalize = True  # 是否标准化
    config.gumbel_max_clip = 7.0  # Gumbel loss裁剪
    
    return config


def get_mujoco_config():
    """MuJoCo环境专用配置"""
    config = get_config()
    config.loss_temp = 4.0
    config.roer_max_clip = 50.0
    config.roer_min_clip = 10.0
    return config


def get_dmcontrol_config():
    """DM Control环境专用配置"""
    config = get_config()
    config.loss_temp = 1.0
    config.roer_max_clip = 100.0
    config.roer_min_clip = 1.0
    return config

