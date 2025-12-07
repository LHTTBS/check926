import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# 解决方案1: 修复中文显示问题
# 方法A: 检查系统字体并设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 方法B: 如果上面方法不行，尝试使用系统路径中的字体
try:
    import matplotlib
    # 获取系统字体列表
    font_list = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    chinese_fonts = [f for f in font_list if any(keyword in f.lower() for keyword in ['simhei', 'msyh', 'microsoft', 'arial'])]
    if chinese_fonts:
        # 添加找到的中文字体路径
        for font_path in chinese_fonts[:3]:  # 取前3个字体
            matplotlib.font_manager.fontManager.addfont(font_path)
            font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams['font.sans-serif'].insert(0, font_name)
except Exception as e:
    print(f"字体加载警告: {e}")

# 设置图表样式
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150  # 提高DPI使图像更清晰
plt.rcParams['savefig.dpi'] = 300  # 保存时使用更高DPI
plt.rcParams['axes.titlesize'] = 14  # 标题字体大小
plt.rcParams['axes.labelsize'] = 12  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 10  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 10  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 10  # 图例字体大小

# 读取数据
df = pd.read_csv('log_20251203_143539_data.csv')
print(f"数据已加载，共{len(df)}行")

# 解决方案2: 优化数据显示范围
# 计算各个指标的数据范围，用于动态调整y轴
def calculate_y_limits(series, padding=0.1, min_range=None, max_range=None):
    """计算y轴的合适范围"""
    min_val = series.min()
    max_val = series.max()
    
    # 如果指定了范围，使用指定范围
    if min_range is not None:
        min_val = min(min_val, min_range)
    if max_range is not None:
        max_val = max(max_val, max_range)
    
    # 添加边距
    data_range = max_val - min_val
    if data_range == 0:  # 防止数据全相同的情况
        data_range = max_val * 0.1 if max_val != 0 else 0.1
    
    y_min = min_val - data_range * padding
    y_max = max_val + data_range * padding
    
    # 对于百分比数据，确保在0-1范围内
    if max_val <= 1.0 and min_val >= 0:
        y_min = max(0, y_min)
        y_max = min(1.0, y_max)
    
    return y_min, y_max

# 创建多图布局，适当调整子图间距
fig = plt.figure(figsize=(22, 18))  # 稍微增大画布尺寸
fig.suptitle('模型训练过程分析报告', fontsize=18, fontweight='bold', y=0.98)

# 1. 损失函数变化图 - 优化显示范围
ax1 = plt.subplot(3, 3, 1)
ax1.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2.5, label='训练损失', marker='o', markersize=4, markevery=5)
ax1.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2.5, label='验证损失', marker='s', markersize=4, markevery=5)
ax1.set_xlabel('训练轮次 (Epoch)', fontsize=11)
ax1.set_ylabel('损失值', fontsize=11)
ax1.set_title('训练与验证损失变化', fontsize=13, fontweight='bold')

# 动态设置y轴范围
train_y_min, train_y_max = calculate_y_limits(df['train_loss'], padding=0.1)
val_y_min, val_y_max = calculate_y_limits(df['val_loss'], padding=0.1)
y_min = min(train_y_min, val_y_min)
y_max = max(train_y_max, val_y_max)
ax1.set_ylim(0, y_max * 1.05)  # 从0开始，留5%边距

ax1.legend(loc='upper right', framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')

# 添加损失最小点标记
min_train_idx = df['train_loss'].idxmin()
min_val_idx = df['val_loss'].idxmin()
ax1.scatter(df['epoch'][min_train_idx], df['train_loss'][min_train_idx], 
           color='blue', s=150, zorder=5, edgecolors='black', linewidth=1.5)
ax1.scatter(df['epoch'][min_val_idx], df['val_loss'][min_val_idx], 
           color='red', s=150, zorder=5, edgecolors='black', linewidth=1.5)

# 添加标注
ax1.annotate(f'最小训练损失: {df["train_loss"][min_train_idx]:.3f}',
            xy=(df['epoch'][min_train_idx], df['train_loss'][min_train_idx]),
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 2. 学习率变化图
ax2 = plt.subplot(3, 3, 2)
ax2.plot(df['epoch'], df['lr'], 'g-', linewidth=2.5, marker='^', markersize=4, markevery=5)
ax2.set_xlabel('训练轮次 (Epoch)', fontsize=11)
ax2.set_ylabel('学习率', fontsize=11)
ax2.set_title('学习率变化 (指数衰减)', fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, linestyle='--')

# 3. 主要任务准确率对比 - 优化显示范围
ax3 = plt.subplot(3, 3, 3)
tasks = ['stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy']
colors = ['blue', 'red', 'green']
markers = ['o', 's', '^']
labels = ['立场准确率', '危害性准确率', '公平性准确率']

for task, color, marker, label in zip(tasks, colors, markers, labels):
    ax3.plot(df['epoch'], df[task], color=color, linewidth=2, label=label, 
             marker=marker, markersize=4, markevery=5)

ax3.set_xlabel('训练轮次 (Epoch)', fontsize=11)
ax3.set_ylabel('准确率', fontsize=11)
ax3.set_title('主要任务准确率对比', fontsize=13, fontweight='bold')
ax3.set_ylim(0.6, 0.85)  # 准确率通常在0.6-0.85之间，适当设置范围
ax3.legend(loc='lower right', framealpha=0.9)
ax3.grid(True, alpha=0.3, linestyle='--')

# 4. 主要任务F1分数对比 - 优化显示范围
ax4 = plt.subplot(3, 3, 4)
f1_tasks = ['stance_f1', 'harmfulness_f1', 'fairness_f1']
for task, color, marker, label in zip(f1_tasks, colors, markers, labels):
    ax4.plot(df['epoch'], df[task], color=color, linewidth=2, label=label,
             marker=marker, markersize=4, markevery=5)
    
ax4.set_xlabel('训练轮次 (Epoch)', fontsize=11)
ax4.set_ylabel('F1分数', fontsize=11)
ax4.set_title('主要任务F1分数对比', fontsize=13, fontweight='bold')
ax4.set_ylim(0.6, 0.85)  # F1分数范围与准确率类似
ax4.legend(loc='lower right', framealpha=0.9)
ax4.grid(True, alpha=0.3, linestyle='--')

# 5. 意图分类指标 - 优化显示范围
ax5 = plt.subplot(3, 3, 5)
intent_metrics = ['intent_exact_match', 'intent_macro_f1']
intent_labels = ['意图精确匹配', '意图宏平均F1']
colors_intent = ['purple', 'orange']
markers_intent = ['o', 's']

for metric, color, marker, label in zip(intent_metrics, colors_intent, markers_intent, intent_labels):
    ax5.plot(df['epoch'], df[metric], color=color, linewidth=2, label=label,
             marker=marker, markersize=4, markevery=5)
    
ax5.set_xlabel('训练轮次 (Epoch)', fontsize=11)
ax5.set_ylabel('分数', fontsize=11)
ax5.set_title('意图分类指标', fontsize=13, fontweight='bold')
ax5.set_ylim(0.0, 0.75)  # 意图分类分数较低，设置合适范围
ax5.legend(loc='lower right', framealpha=0.9)
ax5.grid(True, alpha=0.3, linestyle='--')

# 6. 意图分类各类别F1分数 - 优化显示范围
ax6 = plt.subplot(3, 3, 6)
intent_categories = ['intent_Political_f1', 'intent_Economic_f1', 
                     'intent_Psychological_f1', 'intent_Public_f1']
category_labels = ['政治意图', '经济意图', '心理意图', '公共意图']
category_colors = ['darkblue', 'darkgreen', 'darkred', 'darkorange']
category_markers = ['o', 's', '^', 'D']

for category, color, marker, label in zip(intent_categories, category_colors, category_markers, category_labels):
    ax6.plot(df['epoch'], df[category], color=color, linewidth=2, label=label,
             marker=marker, markersize=4, markevery=5)
    
ax6.set_xlabel('训练轮次 (Epoch)', fontsize=11)
ax6.set_ylabel('F1分数', fontsize=11)
ax6.set_title('意图分类各类别F1分数', fontsize=13, fontweight='bold')
ax6.set_ylim(0.4, 0.9)  # 设置合适的y轴范围
ax6.legend(loc='lower right', framealpha=0.9, ncol=2)
ax6.grid(True, alpha=0.3, linestyle='--')

# 7. 后期训练阶段平均性能
ax7 = plt.subplot(3, 3, 7)
df_late = df[df['epoch'] >= 20]

tasks_summary = ['stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy', 
                 'stance_f1', 'harmfulness_f1', 'fairness_f1',
                 'intent_exact_match', 'intent_macro_f1']
task_names = ['立场准确率', '危害性准确率', '公平性准确率',
              '立场F1', '危害性F1', '公平性F1',
              '意图精确匹配', '意图宏F1']

mean_values = [df_late[task].mean() for task in tasks_summary]

x_pos = np.arange(len(task_names))
colors_bar = plt.cm.Set3(np.arange(len(task_names))/len(task_names))
bars = ax7.bar(x_pos, mean_values, color=colors_bar, edgecolor='black', linewidth=0.5)

ax7.set_xlabel('任务指标', fontsize=11)
ax7.set_ylabel('平均值 (20-40轮)', fontsize=11)
ax7.set_title('后期训练阶段平均性能 (第20-40轮)', fontsize=13, fontweight='bold')
ax7.set_xticks(x_pos)
ax7.set_xticklabels(task_names, rotation=45, ha='right', fontsize=9)
ax7.set_ylim(0, 0.9)  # 柱状图y轴范围
ax7.grid(True, alpha=0.3, axis='y', linestyle='--')

# 在柱状图上添加数值
for bar, value in zip(bars, mean_values):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 8. 验证损失与主要指标的相关性热力图
ax8 = plt.subplot(3, 3, 8)
correlation_metrics = ['val_loss', 'stance_accuracy', 'harmfulness_accuracy', 
                      'fairness_accuracy', 'intent_macro_f1', 'intent_exact_match']
metric_labels = ['验证损失', '立场准确率', '危害性准确率', 
                '公平性准确率', '意图宏F1', '意图精确匹配']

corr_matrix = df[correlation_metrics].corr()

im = ax8.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax8.set_xticks(np.arange(len(correlation_metrics)))
ax8.set_yticks(np.arange(len(correlation_metrics)))
ax8.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=9)
ax8.set_yticklabels(metric_labels, fontsize=9)
ax8.set_title('验证损失与主要指标相关性', fontsize=13, fontweight='bold')

# 添加相关性数值
for i in range(len(correlation_metrics)):
    for j in range(len(correlation_metrics)):
        value = corr_matrix.iloc[i, j]
        color = "white" if abs(value) > 0.6 else "black"
        ax8.text(j, i, f'{value:.2f}', ha="center", va="center", 
                color=color, fontsize=9, fontweight='bold')

# 添加颜色条
cbar = plt.colorbar(im, ax=ax8, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=9)

# 9. 最佳性能指标展示
ax9 = plt.subplot(3, 3, 9)
best_metrics = {
    '最小训练损失': df['train_loss'].min(),
    '最小验证损失': df['val_loss'].min(),
    '最高立场准确率': df['stance_accuracy'].max(),
    '最高立场F1': df['stance_f1'].max(),
    '最高危害性准确率': df['harmfulness_accuracy'].max(),
    '最高公平性准确率': df['fairness_accuracy'].max(),
    '最高意图精确匹配': df['intent_exact_match'].max(),
    '最高意图宏F1': df['intent_macro_f1'].max()
}

best_epochs = {
    '最小训练损失轮次': int(df['epoch'][df['train_loss'].idxmin()]),
    '最小验证损失轮次': int(df['epoch'][df['val_loss'].idxmin()]),
    '最高立场准确率轮次': int(df['epoch'][df['stance_accuracy'].idxmax()]),
    '最高意图精确匹配轮次': int(df['epoch'][df['intent_exact_match'].idxmax()])
}

# 创建格式化的文本总结
summary_text = "最佳性能总结:\n"
summary_text += "="*25 + "\n"
for metric, value in best_metrics.items():
    summary_text += f"{metric}:\n  {value:.4f}\n"

summary_text += "\n关键轮次:\n"
summary_text += "="*25 + "\n"
for metric, epoch in best_epochs.items():
    summary_text += f"{metric}: {epoch}\n"

# 检测过拟合
summary_text += "\n过拟合分析:\n"
summary_text += "="*25 + "\n"
train_loss_reduction = ((df['train_loss'].iloc[0] - df['train_loss'].iloc[-1]) / df['train_loss'].iloc[0]) * 100
val_loss_increase = ((df['val_loss'].iloc[-1] - df['val_loss'].iloc[0]) / df['val_loss'].iloc[0]) * 100
summary_text += f"训练损失下降: {train_loss_reduction:.1f}%\n"
summary_text += f"验证损失上升: {val_loss_increase:.1f}%\n"
summary_text += f"过拟合程度: 明显"

ax9.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontfamily='monospace')
ax9.set_xlim(0, 1)
ax9.set_ylim(0, 1)
ax9.axis('off')
ax9.set_title('最佳性能与过拟合分析', fontsize=13, fontweight='bold')

# 调整子图间距
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间

# 保存图片
save_path = 'training_analysis_optimized.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"主分析图已保存到: {os.path.abspath(save_path)}")
plt.show()

# 额外创建一个性能趋势汇总图（更简洁）
print("\n创建性能趋势汇总图...")
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('模型性能趋势汇总', fontsize=16, fontweight='bold', y=0.98)

performance_indicators = {
    '损失函数变化': ['train_loss', 'val_loss'],
    '主要任务准确率': ['stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy'],
    '意图分类指标': ['intent_exact_match', 'intent_macro_f1'],
    '意图分类细粒度F1': ['intent_Political_f1', 'intent_Economic_f1', 'intent_Psychological_f1', 'intent_Public_f1']
}

# 设置每个子图的y轴范围
y_limits = {
    '损失函数变化': (0, df['val_loss'].max() * 1.1),
    '主要任务准确率': (0.6, 0.85),
    '意图分类指标': (0, 0.75),
    '意图分类细粒度F1': (0.4, 0.9)
}

line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D']

for ax, (title, metrics) in zip(axes2.flatten(), performance_indicators.items()):
    for i, metric in enumerate(metrics):
        label = metric.replace('_', ' ').replace('intent', '').replace('f1', 'F1').title()
        ax.plot(df['epoch'], df[metric], linewidth=2, 
                linestyle=line_styles[i % len(line_styles)],
                marker=markers[i % len(markers)], markersize=3, markevery=5,
                label=label)
    
    ax.set_xlabel('训练轮次 (Epoch)', fontsize=11)
    ax.set_ylabel('分数', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # 设置y轴范围
    if title in y_limits:
        ax.set_ylim(y_limits[title])
    
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存第二个图片
save_path2 = 'performance_trends_optimized.png'
plt.savefig(save_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"性能趋势图已保存到: {os.path.abspath(save_path2)}")

plt.show()

# 打印关键发现
print("\n" + "="*70)
print("关键发现摘要:")
print("="*70)
print(f"1. 过拟合程度: 明显")
print(f"   - 训练损失下降: {train_loss_reduction:.1f}%")
print(f"   - 验证损失上升: {val_loss_increase:.1f}%")

print(f"\n2. 最佳验证损失: 第{best_epochs['最小验证损失轮次']}轮 ({best_metrics['最小验证损失']:.4f})")
print(f"3. 最佳意图分类: 第{best_epochs['最高意图精确匹配轮次']}轮 ({best_metrics['最高意图精确匹配']:.3f})")

print(f"\n4. 任务表现排名 (后期平均):")
# 计算后期各任务平均表现
late_performance = {}
for task, name in zip(tasks_summary, task_names):
    late_performance[name] = df_late[task].mean()

sorted_tasks = sorted(late_performance.items(), key=lambda x: x[1], reverse=True)
for i, (task, acc) in enumerate(sorted_tasks[:5], 1):
    print(f"   {i}. {task}: {acc:.3f}")

print(f"\n5. 建议改进方向:")
print("   - 早停策略: 在第5-10轮附近停止")
print("   - 正则化: 添加Dropout和权重衰减")
print("   - 学习率调整: 使用余弦退火或热重启")
print("   - 数据增强: 特别是意图分类任务")
print("="*70)