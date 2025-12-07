import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 设置全局字体大小
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300
})

# 读取数据
df = pd.read_csv('log_20251203_143539_data.csv')
print(f"数据已加载，共{len(df)}轮训练数据")

# ==================== 方案一：6图布局 ====================

print("\n创建6图布局分析报告...")

# 创建6个子图的布局
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
fig1.suptitle('模型训练分析报告 - 6图布局', fontsize=16, fontweight='bold', y=0.98)

# 图1：损失函数变化（训练损失+验证损失）
ax1 = axes1[0, 0]
ax1.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='训练损失', marker='o', markersize=3, markevery=5)
ax1.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, label='验证损失', marker='s', markersize=3, markevery=5)
ax1.set_xlabel('训练轮次')
ax1.set_ylabel('损失值')
ax1.set_title('训练与验证损失变化', fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
# 标记最小值
min_val_idx = df['val_loss'].idxmin()
ax1.scatter(df['epoch'][min_val_idx], df['val_loss'][min_val_idx], 
           color='red', s=80, zorder=5, label=f'最佳验证点')
ax1.legend(loc='upper right')

# 图2：学习率变化
ax2 = axes1[0, 1]
ax2.plot(df['epoch'], df['lr'], 'g-', linewidth=2, marker='^', markersize=3, markevery=5)
ax2.set_xlabel('训练轮次')
ax2.set_ylabel('学习率')
ax2.set_title('学习率衰减曲线', fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# 图3：主要任务性能对比（准确率）
ax3 = axes1[0, 2]
tasks = ['stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy']
labels = ['立场检测', '危害性检测', '公平性检测']
colors = ['blue', 'red', 'green']
markers = ['o', 's', '^']

for task, label, color, marker in zip(tasks, labels, colors, markers):
    ax3.plot(df['epoch'], df[task], color=color, linewidth=1.5, label=label, 
             marker=marker, markersize=3, markevery=5)
ax3.set_xlabel('训练轮次')
ax3.set_ylabel('准确率')
ax3.set_title('三个主要任务准确率对比', fontweight='bold')
ax3.set_ylim(0.6, 0.85)
ax3.legend(loc='lower right')
ax3.grid(True, alpha=0.3)

# 图4：意图分类性能
ax4 = axes1[1, 0]
intent_metrics = ['intent_exact_match', 'intent_macro_f1']
intent_labels = ['意图精确匹配', '意图宏平均F1']
colors_intent = ['purple', 'orange']

for metric, label, color in zip(intent_metrics, intent_labels, colors_intent):
    ax4.plot(df['epoch'], df[metric], color=color, linewidth=2, label=label)
ax4.set_xlabel('训练轮次')
ax4.set_ylabel('分数')
ax4.set_title('意图分类性能', fontweight='bold')
ax4.set_ylim(0.0, 0.75)
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)

# 图5：意图分类细粒度分析
ax5 = axes1[1, 1]
intent_categories = ['intent_Political_f1', 'intent_Economic_f1', 
                     'intent_Psychological_f1', 'intent_Public_f1']
category_labels = ['政治意图', '经济意图', '心理意图', '公共意图']
category_colors = ['darkblue', 'darkgreen', 'darkred', 'darkorange']

for category, label, color in zip(intent_categories, category_labels, category_colors):
    ax5.plot(df['epoch'], df[category], color=color, linewidth=1.5, label=label)
ax5.set_xlabel('训练轮次')
ax5.set_ylabel('F1分数')
ax5.set_title('意图分类细粒度分析', fontweight='bold')
ax5.set_ylim(0.4, 0.9)
ax5.legend(loc='lower right', fontsize=8)
ax5.grid(True, alpha=0.3)

# 图6：后期训练阶段平均性能（柱状图）
ax6 = axes1[1, 2]
# 取最后10轮的数据计算平均值
df_late = df.tail(10)

tasks_summary = ['stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy']
task_names = ['立场检测', '危害性检测', '公平性检测']

mean_values = [df_late[task].mean() for task in tasks_summary]
colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c']

x_pos = np.arange(len(task_names))
bars = ax6.bar(x_pos, mean_values, color=colors_bar, edgecolor='black', linewidth=0.5)

ax6.set_xlabel('任务')
ax6.set_ylabel('平均准确率 (最后10轮)')
ax6.set_title('后期训练阶段性能', fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(task_names)
ax6.set_ylim(0.6, 0.85)
ax6.grid(True, alpha=0.3, axis='y')

# 在柱状图上添加数值
for bar, value in zip(bars, mean_values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{value:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('training_analysis_6plots.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ==================== 方案二：4图布局 ====================

print("\n创建4图布局分析报告...")

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('模型训练综合分析 - 4图布局', fontsize=16, fontweight='bold', y=0.98)

# 图1：损失函数与学习率（合并图）
ax1_4 = axes2[0, 0]
# 创建双y轴
ax1_4_twin = ax1_4.twinx()

# 绘制损失函数（左轴）
line1 = ax1_4.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='训练损失')
line2 = ax1_4.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, label='验证损失')
ax1_4.set_xlabel('训练轮次')
ax1_4.set_ylabel('损失值', color='black')
ax1_4.tick_params(axis='y', labelcolor='black')
ax1_4.set_title('损失函数与学习率变化', fontweight='bold')
ax1_4.grid(True, alpha=0.3)

# 绘制学习率（右轴）
line3 = ax1_4_twin.plot(df['epoch'], df['lr'], 'g--', linewidth=1.5, label='学习率')
ax1_4_twin.set_ylabel('学习率', color='green')
ax1_4_twin.tick_params(axis='y', labelcolor='green')
ax1_4_twin.set_yscale('log')

# 合并图例
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1_4.legend(lines, labels, loc='upper right')

# 图2：所有任务准确率对比
ax2_4 = axes2[0, 1]
# 绘制四个准确率指标
acc_metrics = ['stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy', 'intent_exact_match']
acc_labels = ['立场检测', '危害性检测', '公平性检测', '意图分类']
acc_colors = ['blue', 'red', 'green', 'purple']
acc_markers = ['o', 's', '^', 'D']

for metric, label, color, marker in zip(acc_metrics, acc_labels, acc_colors, acc_markers):
    ax2_4.plot(df['epoch'], df[metric], color=color, linewidth=1.5, label=label,
               marker=marker, markersize=3, markevery=5)
    
ax2_4.set_xlabel('训练轮次')
ax2_4.set_ylabel('准确率')
ax2_4.set_title('所有任务准确率对比', fontweight='bold')
ax2_4.set_ylim(0.0, 0.85)
ax2_4.legend(loc='lower right')
ax2_4.grid(True, alpha=0.3)

# 图3：所有任务F1分数对比
ax3_4 = axes2[1, 0]
# 绘制四个F1指标
f1_metrics = ['stance_f1', 'harmfulness_f1', 'fairness_f1', 'intent_macro_f1']
f1_labels = ['立场检测', '危害性检测', '公平性检测', '意图分类']
f1_colors = ['blue', 'red', 'green', 'orange']

for metric, label, color in zip(f1_metrics, f1_labels, f1_colors):
    ax3_4.plot(df['epoch'], df[metric], color=color, linewidth=1.5, label=label)
    
ax3_4.set_xlabel('训练轮次')
ax3_4.set_ylabel('F1分数')
ax3_4.set_title('所有任务F1分数对比', fontweight='bold')
ax3_4.set_ylim(0.0, 0.85)
ax3_4.legend(loc='lower right')
ax3_4.grid(True, alpha=0.3)

# 图4：性能总结与相关性分析
ax4_4 = axes2[1, 1]
# 创建热力图
correlation_metrics = ['val_loss', 'stance_accuracy', 'harmfulness_accuracy', 
                      'fairness_accuracy', 'intent_macro_f1']
metric_labels = ['验证损失', '立场准确率', '危害性准确率', 
                '公平性准确率', '意图宏F1']

corr_matrix = df[correlation_metrics].corr()

# 创建热力图
im = ax4_4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax4_4.set_xticks(np.arange(len(correlation_metrics)))
ax4_4.set_yticks(np.arange(len(correlation_metrics)))
ax4_4.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=9)
ax4_4.set_yticklabels(metric_labels, fontsize=9)
ax4_4.set_title('指标相关性热力图', fontweight='bold')

# 添加相关性数值
for i in range(len(correlation_metrics)):
    for j in range(len(correlation_metrics)):
        value = corr_matrix.iloc[i, j]
        color = "white" if abs(value) > 0.6 else "black"
        ax4_4.text(j, i, f'{value:.2f}', ha="center", va="center", 
                  color=color, fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('training_analysis_4plots.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ==================== 关键指标总结图 ====================

print("\n创建关键指标总结图...")

fig3 = plt.figure(figsize=(15, 10))
fig3.suptitle('模型训练关键指标总结', fontsize=16, fontweight='bold', y=0.98)

# 创建3个子图
gs = fig3.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.25, wspace=0.3)

# 左上：损失函数与过拟合分析
ax1_sum = fig3.add_subplot(gs[0, 0])
ax1_sum.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='训练损失')
ax1_sum.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, label='验证损失')
ax1_sum.fill_between(df['epoch'], df['train_loss'], df['val_loss'], 
                     color='gray', alpha=0.2, label='过拟合区域')
ax1_sum.set_xlabel('训练轮次')
ax1_sum.set_ylabel('损失值')
ax1_sum.set_title('损失函数与过拟合分析', fontweight='bold')
ax1_sum.legend(loc='upper right')
ax1_sum.grid(True, alpha=0.3)

# 计算过拟合程度
overfit_gap = df['val_loss'] - df['train_loss']
max_overfit_idx = overfit_gap.idxmax()
ax1_sum.axvline(x=df['epoch'][max_overfit_idx], color='gray', linestyle='--', alpha=0.5)
ax1_sum.text(df['epoch'][max_overfit_idx], ax1_sum.get_ylim()[1]*0.9, 
            f'最大过拟合点\n(第{int(df["epoch"][max_overfit_idx])}轮)', 
            ha='center', fontsize=8)

# 中上：最佳性能对比
ax2_sum = fig3.add_subplot(gs[0, 1])
# 计算每个任务的最佳性能
tasks_best = {
    '立场检测': df['stance_accuracy'].max(),
    '危害性检测': df['harmfulness_accuracy'].max(),
    '公平性检测': df['fairness_accuracy'].max(),
    '意图分类': df['intent_exact_match'].max()
}

x_pos = np.arange(len(tasks_best))
colors_best = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

bars = ax2_sum.bar(x_pos, list(tasks_best.values()), color=colors_best, edgecolor='black', linewidth=0.5)
ax2_sum.set_xlabel('任务')
ax2_sum.set_ylabel('最佳准确率')
ax2_sum.set_title('各任务最佳性能', fontweight='bold')
ax2_sum.set_xticks(x_pos)
ax2_sum.set_xticklabels(tasks_best.keys())
ax2_sum.set_ylim(0, 1.0)
ax2_sum.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, value in zip(bars, tasks_best.values()):
    height = bar.get_height()
    ax2_sum.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)

# 右上：训练进度分析
ax3_sum = fig3.add_subplot(gs[0, 2])
# 将训练分为三个阶段
early_idx = len(df) // 3
mid_idx = 2 * len(df) // 3

stages = ['早期(1-13轮)', '中期(14-26轮)', '后期(27-40轮)']
stance_stage_acc = [
    df['stance_accuracy'].iloc[:early_idx].mean(),
    df['stance_accuracy'].iloc[early_idx:mid_idx].mean(),
    df['stance_accuracy'].iloc[mid_idx:].mean()
]
harm_stage_acc = [
    df['harmfulness_accuracy'].iloc[:early_idx].mean(),
    df['harmfulness_accuracy'].iloc[early_idx:mid_idx].mean(),
    df['harmfulness_accuracy'].iloc[mid_idx:].mean()
]

x = np.arange(len(stages))
width = 0.35

bars1 = ax3_sum.bar(x - width/2, stance_stage_acc, width, label='立场检测', color='#1f77b4')
bars2 = ax3_sum.bar(x + width/2, harm_stage_acc, width, label='危害性检测', color='#ff7f0e')

ax3_sum.set_xlabel('训练阶段')
ax3_sum.set_ylabel('平均准确率')
ax3_sum.set_title('不同训练阶段性能对比', fontweight='bold')
ax3_sum.set_xticks(x)
ax3_sum.set_xticklabels(stages)
ax3_sum.legend()
ax3_sum.grid(True, alpha=0.3, axis='y')

# 左下：详细指标表格
ax4_sum = fig3.add_subplot(gs[1, :])  # 跨三列
ax4_sum.axis('off')

# 创建总结表格
summary_data = []
summary_data.append(['最佳验证损失', f"{df['val_loss'].min():.4f}", f"第{int(df['epoch'][df['val_loss'].idxmin()])}轮"])
summary_data.append(['最佳训练损失', f"{df['train_loss'].min():.4f}", f"第{int(df['epoch'][df['train_loss'].idxmin()])}轮"])
summary_data.append(['最佳立场准确率', f"{df['stance_accuracy'].max():.4f}", f"第{int(df['epoch'][df['stance_accuracy'].idxmax()])}轮"])
summary_data.append(['最佳意图分类', f"{df['intent_exact_match'].max():.4f}", f"第{int(df['epoch'][df['intent_exact_match'].idxmax()])}轮"])
summary_data.append(['过拟合程度', f"{(overfit_gap.max()/df['train_loss'].min()*100):.1f}%", "验证/训练损失比率"])
summary_data.append(['学习率衰减', f"{(df['lr'].iloc[0]/df['lr'].iloc[-1]):.0f}倍", f"{df['lr'].iloc[0]:.1e}→{df['lr'].iloc[-1]:.1e}"])

# 创建表格
table = ax4_sum.table(cellText=summary_data,
                      colLabels=['指标', '数值', '备注'],
                      colColours=['lightgray', 'lightgray', 'lightgray'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# 设置表格标题
ax4_sum.set_title('训练关键指标总结', fontweight='bold', fontsize=12, pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('training_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("\n" + "="*70)
print("图表已生成完成！")
print("="*70)
print(f"1. 6图布局分析: training_analysis_6plots.png")
print(f"2. 4图布局分析: training_analysis_4plots.png")
print(f"3. 关键指标总结: training_summary.png")
print("\n关键发现:")
print(f"- 过拟合明显: 验证损失从{df['val_loss'].iloc[0]:.1f}增加到{df['val_loss'].iloc[-1]:.1f}")
print(f"- 最佳验证轮次: 第{int(df['epoch'][df['val_loss'].idxmin()])}轮")
print(f"- 最佳意图分类: 第{int(df['epoch'][df['intent_exact_match'].idxmax()])}轮 ({df['intent_exact_match'].max():.3f})")
print("="*70)