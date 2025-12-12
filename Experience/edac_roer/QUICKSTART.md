# EDAC+ROER 快速入门指南

本指南帮助您快速上手EDAC+ROER的实现和实验。

## 📦 安装检查

```bash
# 1. 激活环境
conda activate roer

# 2. 检查环境
python -c "import jax; import flax; import gym; print('环境OK')"

# 3. 进入目录
cd ~/Regularized-Optimal-Experience-Replay/Experience/edac_roer
```

## 🏃 5分钟快速测试

最快的验证方式：

```bash
# 运行10k步快速测试（约5-10分钟）
./quick_test.sh
```

这将运行EDAC baseline和EDAC+ROER各10k步，验证实现是否正确。

## 🎯 单次完整训练

### 在HalfCheetah-v2上训练

```bash
# EDAC+ROER（使用ROER优先级）
python train_edac_roer.py \
    --env_name=HalfCheetah-v2 \
    --seed=42 \
    --use_roer=True \
    --roer_temp=4.0 \
    --roer_max_clip=50 \
    --roer_min_clip=10 \
    --max_steps=1000000

# EDAC baseline（不使用ROER）
python train_edac_roer.py \
    --env_name=HalfCheetah-v2 \
    --seed=42 \
    --use_roer=False \
    --max_steps=1000000
```

### 在Ant-v2上训练（使用论文参数）

```bash
python train_edac_roer.py \
    --env_name=Ant-v2 \
    --seed=42 \
    --use_roer=True \
    --roer_temp=1.0 \
    --roer_max_clip=100 \
    --roer_min_clip=10 \
    --max_steps=1000000
```

## 📊 对比实验（推荐用于毕设）

### 方案1：简单对比（2个方法 × 5个种子）

```bash
# 在HalfCheetah-v2上运行对比实验
./run_comparison.sh HalfCheetah-v2 5
```

这将自动运行：
- EDAC baseline × 5个种子
- EDAC+ROER × 5个种子
- 自动统计和分析结果

### 方案2：Ant-v2专用实验

```bash
# Ant-v2完整实验
./run_ant_experiment.sh 5
```

### 方案3：超参数扫描

```bash
# 系统扫描temperature、max_clip等参数
./hyperparameter_sweep.sh HalfCheetah-v2 42
```

## 📈 查看结果

### 方法1：TensorBoard

```bash
# 查看训练过程
tensorboard --logdir ~/roer_output/results/

# 在浏览器打开 http://localhost:6006
```

关键指标：
- `evaluation/return`: 评估回报（最重要）
- `training/critic_loss`: Critic损失
- `training/diversity_loss`: EDAC多样性损失
- `priority/mean`, `priority/std`: ROER优先级分布

### 方法2：文本结果

```bash
# 查看评估回报历史
cat ~/roer_output/results/edac_roer/*/eval_returns.txt
```

### 方法3：可视化脚本

```bash
# 生成对比图
python plot_results.py \
    --baseline './results/edac_baseline/*/eval_returns.txt' \
    --roer './results/edac_roer/*/eval_returns.txt' \
    --title 'HalfCheetah-v2' \
    --save_dir './plots/'
```

## 🔧 常见参数调整

### 如果训练不稳定

```bash
# 降低temperature（减少正则化强度）
--roer_temp=2.0  # 从4.0降到2.0

# 缩小优先级范围
--roer_max_clip=20 --roer_min_clip=5

# 增大EMA系数（更平滑的优先级更新）
--roer_per_beta=0.05  # 从0.01增到0.05
```

### 如果想更激进的优先级

```bash
# 增大temperature
--roer_temp=8.0

# 扩大优先级范围
--roer_max_clip=100 --roer_min_clip=1
```

### 调整EDAC参数

```bash
# 改变ensemble大小
--num_critics=5  # 或15（默认10）

# 调整多样性系数
--diversity_coef=0.05  # 或0.2（默认0.1）
```

## 📁 结果保存位置

所有结果默认保存在：

```
~/roer_output/results/
├── edac_baseline/
│   └── HalfCheetah-v2_seed42_2024-12-09_10-30-00/
│       ├── tb/              # TensorBoard日志
│       ├── eval_returns.txt # 评估历史
│       └── best_model.pkl   # 最佳模型
└── edac_roer/
    └── HalfCheetah-v2_seed42_2024-12-09_11-00-00/
        └── ...
```

## 🐛 常见问题

### 1. wandb登录提示

如果看到wandb登录提示，进程会停止。解决方法：

**方法A：禁用wandb**（推荐）
```bash
# 训练时添加 --track=False
python train_edac_roer.py --env_name=Ant-v2 --track=False
```

**方法B：配置wandb**
```bash
# 一次性设置
export WANDB_MODE=disabled
```

### 2. GPU/CPU选择

代码会自动检测GPU。如果想强制使用CPU：

```bash
export JAX_PLATFORM_NAME=cpu
python train_edac_roer.py ...
```

### 3. 内存不足

如果内存不足，可以：

```bash
# 减小batch size
--batch_size=128  # 默认256

# 减小replay buffer容量
--capacity=500000  # 默认1000000

# 减少critic数量
--num_critics=5  # 默认10
```

### 4. 脚本权限问题

```bash
# 添加执行权限
chmod +x *.sh
```

## 🎓 毕设实验时间表

### 第1天：环境测试
```bash
./quick_test.sh
```
确保代码能跑通。

### 第2-3天：单环境对比
```bash
./run_comparison.sh HalfCheetah-v2 5
```
获得第一组实验数据。

### 第4-7天：多环境实验
```bash
for env in HalfCheetah-v2 Ant-v2 Hopper-v2 Walker2d-v2; do
    ./run_comparison.sh $env 5 &
done
```

### 第8-9天：超参数扫描
```bash
./hyperparameter_sweep.sh HalfCheetah-v2 42
```

### 第10天：结果分析和可视化
```bash
python plot_results.py ...
```

## 💡 实验建议

1. **先小后大**：从quick_test开始，确保能跑通
2. **单种子验证**：先用单个种子调试参数
3. **并行训练**：多个种子可以并行（注意CPU/GPU资源）
4. **及时保存**：定期备份results目录
5. **记录日志**：记录每次实验的参数和结果

## 📞 获取帮助

1. **查看TensorBoard**：实时监控训练
2. **检查日志文件**：`~/roer_output/results/*/tb/`
3. **参考README.md**：详细文档
4. **查看原论文**：理论依据

## 🚀 下一步

完成快速测试后，可以：

1. 阅读完整的 [README.md](./README.md)
2. 查看 [edac_roer_learner.py](./edac_roer_learner.py) 了解实现细节
3. 运行完整对比实验
4. 自定义实验参数

