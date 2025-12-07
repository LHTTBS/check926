import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict

# ---------------------- 1. 解析TXT日志数据（复用之前的稳定解析逻辑） ----------------------
def parse_training_log(txt_path):
    metrics = defaultdict(list)
    required_keys = [
        'epochs', 'train_loss', 'val_loss', 'learning_rate',
        'stance_acc', 'stance_f1', 'harm_acc', 'harm_f1',
        'fair_acc', 'fair_f1', 'intent_macro_f1',
        'intent_political_f1', 'intent_public_f1'
    ]
    for key in required_keys:
        metrics[key] = []

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except Exception as e:
        print(f"日志读取失败: {str(e)}")
        return None

    # 适配你的日志格式的正则（若日志指标名有差异，修改此处即可）
    epoch_pattern = r"Epoch (\d+)/\d+.*?训练损失: ([\d.]+).*?验证损失: ([\d.]+).*?当前学习率: ([\d.e+-]+).*?stance_accuracy: ([\d.]+).*?stance_f1: ([\d.]+).*?harmfulness_accuracy: ([\d.]+).*?harmfulness_f1: ([\d.]+).*?fairness_accuracy: ([\d.]+).*?fairness_f1: ([\d.]+).*?intent_macro_f1: ([\d.]+).*?intent_Political_f1: ([\d.]+).*?intent_Public_f1: ([\d.]+)"
    matches = re.findall(epoch_pattern, log_content, re.DOTALL)

    if not matches:
        print("未匹配到日志数据，请检查正则表达式与日志格式是否一致")
        return None

    for match in matches:
        metrics['epochs'].append(int(match[0]))
        metrics['train_loss'].append(float(match[1]))
        metrics['val_loss'].append(float(match[2]))
        metrics['learning_rate'].append(float(match[3]))
        metrics['stance_acc'].append(float(match[4]))
        metrics['stance_f1'].append(float(match[5]))
        metrics['harm_acc'].append(float(match[6]))
        metrics['harm_f1'].append(float(match[7]))
        metrics['fair_acc'].append(float(match[8]))
        metrics['fair_f1'].append(float(match[9]))
        metrics['intent_macro_f1'].append(float(match[10]))
        metrics['intent_political_f1'].append(float(match[11]))
        metrics['intent_public_f1'].append(float(match[12]))

    return metrics

# ---------------------- 2. 生成与示例图一致的可视化 ----------------------
def visualize_like_sample(metrics, save_path='sample_style_plot.png'):
    if not metrics or len(metrics['epochs']) == 0:
        print("无数据可可视化")
        return

    # 中文显示配置
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['lines.linewidth'] = 2  # 匹配示例图的线条粗细

    # 创建2行3列子图（与示例图布局完全一致）
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.tight_layout(pad=3.5)  # 调整子图间距

    # ---------- 子图1（左上）：训练损失+验证损失 ----------
    ax1 = axes[0, 0]
    ax1.plot(metrics['epochs'], metrics['train_loss'], color='#1f77b4', label='训练损失')
    ax1.plot(metrics['epochs'], metrics['val_loss'], color='#ff7f0e', label='验证损失')
    ax1.set_title('训练/验证损失趋势', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('损失值', fontsize=10)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(alpha=0.3)

    # ---------- 子图2（右上）：学习率变化 ----------
    ax2 = axes[0, 1]
    ax2.plot(metrics['epochs'], metrics['learning_rate'], color='#2ca02c', label='学习率')
    ax2.set_title('学习率变化曲线', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('学习率', fontsize=10)
    ax2.set_yscale('log')  # 示例图学习率为对数刻度
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(alpha=0.3)

    # ---------- 子图3（左中）：Stance任务指标 ----------
    ax3 = axes[0, 2]
    ax3.plot(metrics['epochs'], metrics['stance_acc'], color='#d62728', label='Stance准确率')
    ax3.plot(metrics['epochs'], metrics['stance_f1'], color='#9467bd', label='Stance F1')
    ax3.set_title('Stance任务性能', fontsize=12)
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('分数', fontsize=10)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(alpha=0.3)

    # ---------- 子图4（右中）：Harm任务指标 ----------
    ax4 = axes[1, 0]
    ax4.plot(metrics['epochs'], metrics['harm_acc'], color='#8c564b', label='Harm准确率')
    ax4.plot(metrics['epochs'], metrics['harm_f1'], color='#e377c2', label='Harm F1')
    ax4.set_title('Harm任务性能', fontsize=12)
    ax4.set_xlabel('Epoch', fontsize=10)
    ax4.set_ylabel('分数', fontsize=10)
    ax4.legend(fontsize=9, loc='upper right')
    ax4.grid(alpha=0.3)

    # ---------- 子图5（左下）：Fair任务指标 ----------
    ax5 = axes[1, 1]
    ax5.plot(metrics['epochs'], metrics['fair_acc'], color='#7f7f7f', label='Fair准确率')
    ax5.plot(metrics['epochs'], metrics['fair_f1'], color='#bcbd22', label='Fair F1')
    ax5.set_title('Fair任务性能', fontsize=12)
    ax5.set_xlabel('Epoch', fontsize=10)
    ax5.set_ylabel('分数', fontsize=10)
    ax5.legend(fontsize=9, loc='upper right')
    ax5.grid(alpha=0.3)

    # ---------- 子图6（右下）：Intent任务指标 ----------
    ax6 = axes[1, 2]
    ax6.plot(metrics['epochs'], metrics['intent_macro_f1'], color='#17becf', label='Intent Macro F1')
    ax6.plot(metrics['epochs'], metrics['intent_political_f1'], color='#ffbb78', label='Political F1')
    ax6.plot(metrics['epochs'], metrics['intent_public_f1'], color='#98df8a', label='Public F1')
    ax6.set_title('Intent任务F1分数', fontsize=12)
    ax6.set_xlabel('Epoch', fontsize=10)
    ax6.set_ylabel('F1分数', fontsize=10)
    ax6.legend(fontsize=9, loc='upper right')
    ax6.grid(alpha=0.3)

    # 保存并显示
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"图表已保存为: {save_path}")

# ---------------------- 3. 运行代码 ----------------------
if __name__ == "__main__":
    # 替换为你的TXT日志文件路径（与脚本同目录可直接写文件名）
    LOG_PATH = r"outputs\training_log.txt"
    
    # 解析日志
    log_data = parse_training_log(LOG_PATH)
    
    # 生成示例风格的图
    if log_data:
        visualize_like_sample(log_data)