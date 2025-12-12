"""结果可视化脚本"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from typing import List, Tuple
import argparse


def load_results(pattern: str) -> List[np.ndarray]:
    """加载结果文件"""
    files = glob.glob(pattern)
    results = []
    for f in files:
        try:
            data = np.loadtxt(f)
            if len(data) > 0:
                results.append(data)
        except:
            pass
    return results


def plot_learning_curves(
    baseline_pattern: str,
    roer_pattern: str,
    title: str = "EDAC vs EDAC+ROER",
    save_path: str = None
):
    """
    绘制学习曲线对比图
    
    Args:
        baseline_pattern: baseline结果文件匹配模式
        roer_pattern: ROER结果文件匹配模式
        title: 图标题
        save_path: 保存路径（None则显示）
    """
    baseline_results = load_results(baseline_pattern)
    roer_results = load_results(roer_pattern)
    
    if not baseline_results and not roer_results:
        print("未找到结果文件！")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制baseline
    if baseline_results:
        all_steps = [r[:, 0] for r in baseline_results]
        all_returns = [r[:, 1] for r in baseline_results]
        
        # 对齐步数（取最小长度）
        min_len = min(len(s) for s in all_steps)
        steps = all_steps[0][:min_len]
        returns = np.array([r[:min_len] for r in all_returns])
        
        mean_return = np.mean(returns, axis=0)
        std_return = np.std(returns, axis=0)
        
        ax.plot(steps, mean_return, label='EDAC Baseline', linewidth=2)
        ax.fill_between(
            steps,
            mean_return - std_return,
            mean_return + std_return,
            alpha=0.3
        )
    
    # 绘制ROER
    if roer_results:
        all_steps = [r[:, 0] for r in roer_results]
        all_returns = [r[:, 1] for r in roer_results]
        
        min_len = min(len(s) for s in all_steps)
        steps = all_steps[0][:min_len]
        returns = np.array([r[:min_len] for r in all_returns])
        
        mean_return = np.mean(returns, axis=0)
        std_return = np.std(returns, axis=0)
        
        ax.plot(steps, mean_return, label='EDAC+ROER', linewidth=2)
        ax.fill_between(
            steps,
            mean_return - std_return,
            mean_return + std_return,
            alpha=0.3
        )
    
    ax.set_xlabel('Training Steps', fontsize=14)
    ax.set_ylabel('Average Return', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像保存至: {save_path}")
    else:
        plt.show()


def plot_final_performance(
    baseline_pattern: str,
    roer_pattern: str,
    title: str = "Final Performance",
    save_path: str = None
):
    """绘制最终性能对比柱状图"""
    baseline_results = load_results(baseline_pattern)
    roer_results = load_results(roer_pattern)
    
    baseline_final = [r[-1, 1] for r in baseline_results] if baseline_results else []
    roer_final = [r[-1, 1] for r in roer_results] if roer_results else []
    
    if not baseline_final and not roer_final:
        print("未找到结果文件！")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = []
    means = []
    stds = []
    
    if baseline_final:
        methods.append('EDAC Baseline')
        means.append(np.mean(baseline_final))
        stds.append(np.std(baseline_final))
    
    if roer_final:
        methods.append('EDAC+ROER')
        means.append(np.mean(roer_final))
        stds.append(np.std(roer_final))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=10, alpha=0.8)
    
    # 添加数值标签
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s, f'{m:.1f}±{s:.1f}', ha='center', va='bottom', fontsize=12)
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=14)
    ax.set_ylabel('Final Average Return', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像保存至: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True, help='Baseline结果路径模式')
    parser.add_argument('--roer', type=str, required=True, help='ROER结果路径模式')
    parser.add_argument('--title', type=str, default='EDAC vs EDAC+ROER', help='图标题')
    parser.add_argument('--save_dir', type=str, default=None, help='图像保存目录')
    
    args = parser.parse_args()
    
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        curve_path = os.path.join(args.save_dir, 'learning_curves.png')
        bar_path = os.path.join(args.save_dir, 'final_performance.png')
    else:
        curve_path = None
        bar_path = None
    
    # 绘制学习曲线
    plot_learning_curves(
        args.baseline,
        args.roer,
        title=f"{args.title} - Learning Curves",
        save_path=curve_path
    )
    
    # 绘制最终性能
    plot_final_performance(
        args.baseline,
        args.roer,
        title=f"{args.title} - Final Performance",
        save_path=bar_path
    )


if __name__ == '__main__':
    # 示例用法
    print("可视化结果工具")
    print("\n用法示例:")
    print("python plot_results.py \\")
    print("    --baseline './results/edac_baseline/*/eval_returns.txt' \\")
    print("    --roer './results/edac_roer/*/eval_returns.txt' \\")
    print("    --title 'HalfCheetah-v2' \\")
    print("    --save_dir './plots/'")
    print("\n")
    
    main()

