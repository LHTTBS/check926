import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.font_manager as fm  # 引入字体管理模块

# 读取数据
df = pd.read_csv('log_20251203_143539_data.csv')

# ==================== 关键：强制中文显示设置 ====================
# 1. 查找系统中可用的中文字体（打印可用字体列表，方便调试）
available_fonts = [f.name for f in fm.findSystemFonts()]
print("可用字体列表：", [f for f in available_fonts if 'hei' in f.lower() or 'song' in f.lower() or 'microsoft' in f.lower()])

# 2. 多系统兼容的字体设置（按优先级排序）
plt.rcParams["font.family"] = [
    "SimHei",         # Windows常用
    "WenQuanYi Micro Hei",  # Linux常用
    "Heiti TC",       # Mac常用
    "Microsoft YaHei",# 微软雅黑
    "Arial Unicode MS"# 跨平台备选
]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 3. 全局字体大小调整（避免文字拥挤）
plt.rcParams['axes.titlesize'] = 11    # 标题
plt.rcParams['axes.labelsize'] = 10    # 坐标轴标签
plt.rcParams['xtick.labelsize'] = 9    # x轴刻度
plt.rcParams['ytick.labelsize'] = 9    # y轴刻度
plt.rcParams['legend.fontsize'] = 9    # 图例
plt.rcParams['font.size'] = 9          # 全局基础字体

# 设置样式
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.05)  # 微调上下文字体

# ==================== 调整布局避免拥挤 ====================
fig = plt.figure(figsize=(22, 18))  # 增大画布
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # 增加子图间距

# 1. 损失函数变化图（不变，继承全局字体设置）
ax1 = plt.subplot(3, 3, 1)
ax1.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='训练损失')
ax1.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, label='验证损失')
ax1.set_xlabel('训练轮次 (Epoch)')
ax1.set_ylabel('损失值')
ax1.set_title('训练与验证损失变化')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 添加损失最小点标记
min_train_idx = df['train_loss'].idxmin()
min_val_idx = df['val_loss'].idxmin()
ax1.scatter(df['epoch'][min_train_idx], df['train_loss'][min_train_idx], 
           color='blue', s=80, zorder=5, label=f'最小训练损失: {df["train_loss"][min_train_idx]:.3f}')
ax1.scatter(df['epoch'][min_val_idx], df['val_loss'][min_val_idx], 
           color='red', s=80, zorder=5, label=f'最小验证损失: {df["val_loss"][min_val_idx]:.3f}')
ax1.legend()

# 2. 学习率变化图
ax2 = plt.subplot(3, 3, 2)
ax2.plot(df['epoch'], df['lr'], 'g-', linewidth=2)
ax2.set_xlabel('训练轮次 (Epoch)')
ax2.set_ylabel('学习率')
ax2.set_title('学习率变化 (指数衰减)')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# 3. 主要任务准确率对比
ax3 = plt.subplot(3, 3, 3)
tasks = ['stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy']
colors = ['blue', 'red', 'green']
labels = ['立场准确率', '危害性准确率', '公平性准确率']
for task, color, label in zip(tasks, colors, labels):
    ax3.plot(df['epoch'], df[task], color=color, linewidth=2, label=label)
ax3.set_xlabel('训练轮次 (Epoch)')
ax3.set_ylabel('准确率')
ax3.set_title('主要任务准确率对比')
ax3.legend(loc='best')  # 自动选择最佳图例位置
ax3.grid(True, alpha=0.3)

# 4. 主要任务F1分数对比（同上，略）
ax4 = plt.subplot(3, 3, 4)
f1_tasks = ['stance_f1', 'harmfulness_f1', 'fairness_f1']
for task, color, label in zip(f1_tasks, colors, labels):
    ax4.plot(df['epoch'], df[task], color=color, linewidth=2, label=label)
ax4.set_xlabel('训练轮次 (Epoch)')
ax4.set_ylabel('F1分数')
ax4.set_title('主要任务F1分数对比')
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3)

# 5. 意图分类指标
ax5 = plt.subplot(3, 3, 5)
intent_metrics = ['intent_exact_match', 'intent_macro_f1']
intent_labels = ['意图精确匹配', '意图宏平均F1']
colors_intent = ['purple', 'orange']
for metric, color, label in zip(intent_metrics, colors_intent, intent_labels):
    ax5.plot(df['epoch'], df[metric], color=color, linewidth=2, label=label)
ax5.set_xlabel('训练轮次 (Epoch)')
ax5.set_ylabel('分数')
ax5.set_title('意图分类指标')
ax5.legend(loc='best')
ax5.grid(True, alpha=0.3)

# 6. 意图分类各类别F1分数
ax6 = plt.subplot(3, 3, 6)
intent_categories = ['intent_Political_f1', 'intent_Economic_f1', 
                     'intent_Psychological_f1', 'intent_Public_f1']
category_labels = ['政治意图', '经济意图', '心理意图', '公共意图']
category_colors = ['darkblue', 'darkgreen', 'darkred', 'darkorange']
for category, color, label in zip(intent_categories, category_colors, category_labels):
    ax6.plot(df['epoch'], df[category], color=color, linewidth=2, label=label)
ax6.set_xlabel('训练轮次 (Epoch)')
ax6.set_ylabel('F1分数')
ax6.set_title('意图分类各类别F1分数')
ax6.legend(loc='best', ncol=2)  # 图例分2列显示，避免拥挤
ax6.grid(True, alpha=0.3)

# 7. 第20轮后性能汇总（重点解决x轴标签拥挤）
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
bars = ax7.bar(x_pos, mean_values, color=plt.cm.tab20(np.arange(len(task_names))/len(task_names)))
ax7.set_xlabel('任务指标')
ax7.set_ylabel('平均值 (20-40轮)')
ax7.set_title('后期训练阶段平均性能')
ax7.set_xticks(x_pos)
# 关键：x轴标签旋转+锚定，避免重叠
ax7.set_xticklabels(task_names, rotation=45, ha='right', rotation_mode='anchor')
ax7.grid(True, alpha=0.3, axis='y')
# 柱状图数值调整
for bar, value in zip(bars, mean_values):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{value:.3f}', ha='center', va='bottom', fontsize=8)

# 8. 热力图（解决标签拥挤）
ax8 = plt.subplot(3, 3, 8)
correlation_metrics = ['val_loss', 'stance_accuracy', 'harmfulness_accuracy', 
                      'fairness_accuracy', 'intent_macro_f1', 'intent_exact_match']
corr_matrix = df[correlation_metrics].corr()
im = ax8.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax8.set_xticks(np.arange(len(correlation_metrics)))
ax8.set_yticks(np.arange(len(correlation_metrics)))
# 缩小热力图标签字体
ax8.set_xticklabels(['验证损失', '立场准确率', '危害性准确率', 
                    '公平性准确率', '意图宏F1', '意图精确匹配'], 
                   rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
ax8.set_yticklabels(['验证损失', '立场准确率', '危害性准确率', 
                    '公平性准确率', '意图宏F1', '意图精确匹配'], fontsize=8)
ax8.set_title('验证损失与主要指标相关性')
# 热力图数值调整
for i in range(len(correlation_metrics)):
    for j in range(len(correlation_metrics)):
        text = ax8.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", fontsize=7,
                       color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white")

# 9. 最佳性能总结（文本布局优化）
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
    '最小训练损失轮次': df['epoch'][df['train_loss'].idxmin()],
    '最小验证损失轮次': df['epoch'][df['val_loss'].idxmin()],
    '最高立场准确率轮次': df['epoch'][df['stance_accuracy'].idxmax()],
    '最高意图精确匹配轮次': df['epoch'][df['intent_exact_match'].idxmax()]
}
# 文本布局调整，使用更紧凑的格式
summary_text = "最佳性能总结:\n\n"
for metric, value in best_metrics.items():
    summary_text += f"{metric}: {value:.4f}\n"
summary_text += "\n关键轮次:\n"
for metric, epoch in best_epochs.items():
    summary_text += f"{metric}: {int(epoch)}\n"
# 调整文本位置和背景，确保清晰
ax9.text(0.05, 0.95, summary_text, fontsize=10, 
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.9, pad=10))
ax9.set_xlim(0, 1)
ax9.set_ylim(0, 1)
ax9.axis('off')
ax9.set_title('最佳性能总结', fontsize=12)

# 保存和显示（关键：bbox_inches确保文字不被截断）
plt.tight_layout()
plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 性能趋势汇总图（同上优化）
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 14))
plt.subplots_adjust(wspace=0.25, hspace=0.3)
performance_indicators = {
    '损失函数': ['train_loss', 'val_loss'],
    '主要任务准确率': ['stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy'],
    '意图分类指标': ['intent_exact_match', 'intent_macro_f1'],
    '意图分类细粒度F1': ['intent_Political_f1', 'intent_Economic_f1', 'intent_Psychological_f1', 'intent_Public_f1']
}
for ax, (title, metrics) in zip(axes2.flatten(), performance_indicators.items()):
    for metric in metrics:
        label = metric.replace('intent_', '').replace('_f1', ' F1').replace('_', ' ').title()
        ax.plot(df['epoch'], df[metric], linewidth=2, label=label)
    ax.set_xlabel('训练轮次 (Epoch)')
    ax.set_ylabel('分数')
    ax.set_title(title)
    ax.legend(loc='best', ncol=1 if len(metrics)<=3 else 2)  # 图例根据数量调整列数
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('performance_trends.png', dpi=300, bbox_inches='tight')
plt.show()