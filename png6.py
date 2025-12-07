import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set chart style
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
sns.set_style("whitegrid")

# Read data
df = pd.read_csv('log_20251203_143539_data.csv')
print(f"Data loaded: {len(df)} epochs")

# ==================== Option 1: 6-plot layout ====================

print("\nCreating 6-plot layout...")

fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
fig1.suptitle('Training Analysis - 6 Plots', fontsize=16, fontweight='bold', y=0.98)

# Plot 1: Loss functions
ax1 = axes1[0, 0]
ax1.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Train Loss', marker='o', markersize=3, markevery=5)
ax1.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, label='Val Loss', marker='s', markersize=3, markevery=5)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training & Validation Loss', fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Learning rate
ax2 = axes1[0, 1]
ax2.plot(df['epoch'], df['lr'], 'g-', linewidth=2, marker='^', markersize=3, markevery=5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Learning Rate Decay', fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# Plot 3: Main tasks accuracy
ax3 = axes1[0, 2]
tasks = ['stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy']
labels = ['Stance', 'Harmfulness', 'Fairness']
colors = ['blue', 'red', 'green']
markers = ['o', 's', '^']

for task, label, color, marker in zip(tasks, labels, colors, markers):
    ax3.plot(df['epoch'], df[task], color=color, linewidth=1.5, label=label, 
             marker=marker, markersize=3, markevery=5)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Accuracy')
ax3.set_title('Main Tasks Accuracy', fontweight='bold')
ax3.set_ylim(0.6, 0.85)
ax3.legend(loc='lower right')
ax3.grid(True, alpha=0.3)

# Plot 4: Intent classification
ax4 = axes1[1, 0]
intent_metrics = ['intent_exact_match', 'intent_macro_f1']
intent_labels = ['Exact Match', 'Macro F1']
colors_intent = ['purple', 'orange']

for metric, label, color in zip(intent_metrics, intent_labels, colors_intent):
    ax4.plot(df['epoch'], df[metric], color=color, linewidth=2, label=label)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Score')
ax4.set_title('Intent Classification', fontweight='bold')
ax4.set_ylim(0.0, 0.75)
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)

# Plot 5: Intent categories
ax5 = axes1[1, 1]
intent_categories = ['intent_Political_f1', 'intent_Economic_f1', 
                     'intent_Psychological_f1', 'intent_Public_f1']
category_labels = ['Political', 'Economic', 'Psychological', 'Public']
category_colors = ['darkblue', 'darkgreen', 'darkred', 'darkorange']

for category, label, color in zip(intent_categories, category_labels, category_colors):
    ax5.plot(df['epoch'], df[category], color=color, linewidth=1.5, label=label)
ax5.set_xlabel('Epoch')
ax5.set_ylabel('F1 Score')
ax5.set_title('Intent Categories F1', fontweight='bold')
ax5.set_ylim(0.4, 0.9)
ax5.legend(loc='lower right', fontsize=8)
ax5.grid(True, alpha=0.3)

# Plot 6: Late-stage performance
ax6 = axes1[1, 2]
df_late = df.tail(10)
tasks_summary = ['stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy']
task_names = ['Stance', 'Harmfulness', 'Fairness']
mean_values = [df_late[task].mean() for task in tasks_summary]
colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c']

x_pos = np.arange(len(task_names))
bars = ax6.bar(x_pos, mean_values, color=colors_bar, edgecolor='black', linewidth=0.5)

ax6.set_xlabel('Task')
ax6.set_ylabel('Avg Accuracy (Last 10 epochs)')
ax6.set_title('Late-stage Performance', fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(task_names)
ax6.set_ylim(0.6, 0.85)
ax6.grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars, mean_values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{value:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('training_analysis_6plots_en.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ==================== Option 2: 4-plot layout ====================

print("\nCreating 4-plot layout...")

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Training Analysis - 4 Plots', fontsize=16, fontweight='bold', y=0.98)

# Plot 1: Loss & LR (dual axis)
ax1_4 = axes2[0, 0]
ax1_4_twin = ax1_4.twinx()

line1 = ax1_4.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Train Loss')
line2 = ax1_4.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, label='Val Loss')
ax1_4.set_xlabel('Epoch')
ax1_4.set_ylabel('Loss', color='black')
ax1_4.tick_params(axis='y', labelcolor='black')
ax1_4.set_title('Loss & Learning Rate', fontweight='bold')
ax1_4.grid(True, alpha=0.3)

line3 = ax1_4_twin.plot(df['epoch'], df['lr'], 'g--', linewidth=1.5, label='LR')
ax1_4_twin.set_ylabel('Learning Rate', color='green')
ax1_4_twin.tick_params(axis='y', labelcolor='green')
ax1_4_twin.set_yscale('log')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1_4.legend(lines, labels, loc='upper right')

# Plot 2: All tasks accuracy
ax2_4 = axes2[0, 1]
acc_metrics = ['stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy', 'intent_exact_match']
acc_labels = ['Stance', 'Harmfulness', 'Fairness', 'Intent']
acc_colors = ['blue', 'red', 'green', 'purple']
acc_markers = ['o', 's', '^', 'D']

for metric, label, color, marker in zip(acc_metrics, acc_labels, acc_colors, acc_markers):
    ax2_4.plot(df['epoch'], df[metric], color=color, linewidth=1.5, label=label,
               marker=marker, markersize=3, markevery=5)
    
ax2_4.set_xlabel('Epoch')
ax2_4.set_ylabel('Accuracy')
ax2_4.set_title('All Tasks Accuracy', fontweight='bold')
ax2_4.set_ylim(0.0, 0.85)
ax2_4.legend(loc='lower right')
ax2_4.grid(True, alpha=0.3)

# Plot 3: All tasks F1
ax3_4 = axes2[1, 0]
f1_metrics = ['stance_f1', 'harmfulness_f1', 'fairness_f1', 'intent_macro_f1']
f1_labels = ['Stance', 'Harmfulness', 'Fairness', 'Intent']
f1_colors = ['blue', 'red', 'green', 'orange']

for metric, label, color in zip(f1_metrics, f1_labels, f1_colors):
    ax3_4.plot(df['epoch'], df[metric], color=color, linewidth=1.5, label=label)
    
ax3_4.set_xlabel('Epoch')
ax3_4.set_ylabel('F1 Score')
ax3_4.set_title('All Tasks F1 Score', fontweight='bold')
ax3_4.set_ylim(0.0, 0.85)
ax3_4.legend(loc='lower right')
ax3_4.grid(True, alpha=0.3)

# Plot 4: Correlation heatmap
ax4_4 = axes2[1, 1]
correlation_metrics = ['val_loss', 'stance_accuracy', 'harmfulness_accuracy', 
                      'fairness_accuracy', 'intent_macro_f1']
metric_labels = ['Val Loss', 'Stance Acc', 'Harm Acc', 
                'Fairness Acc', 'Intent F1']

corr_matrix = df[correlation_metrics].corr()
im = ax4_4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax4_4.set_xticks(np.arange(len(correlation_metrics)))
ax4_4.set_yticks(np.arange(len(correlation_metrics)))
ax4_4.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=9)
ax4_4.set_yticklabels(metric_labels, fontsize=9)
ax4_4.set_title('Metrics Correlation', fontweight='bold')

for i in range(len(correlation_metrics)):
    for j in range(len(correlation_metrics)):
        value = corr_matrix.iloc[i, j]
        color = "white" if abs(value) > 0.6 else "black"
        ax4_4.text(j, i, f'{value:.2f}', ha="center", va="center", 
                  color=color, fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('training_analysis_4plots_en.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ==================== Key metrics summary ====================

print("\nCreating key metrics summary...")

fig3 = plt.figure(figsize=(15, 10))
fig3.suptitle('Training Key Metrics Summary', fontsize=16, fontweight='bold', y=0.98)

gs = fig3.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.25, wspace=0.3)

# Subplot 1: Loss & overfitting
ax1_sum = fig3.add_subplot(gs[0, 0])
ax1_sum.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Train Loss')
ax1_sum.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, label='Val Loss')
ax1_sum.fill_between(df['epoch'], df['train_loss'], df['val_loss'], 
                     color='gray', alpha=0.2, label='Overfitting Gap')
ax1_sum.set_xlabel('Epoch')
ax1_sum.set_ylabel('Loss')
ax1_sum.set_title('Loss & Overfitting', fontweight='bold')
ax1_sum.legend(loc='upper right')
ax1_sum.grid(True, alpha=0.3)

# Subplot 2: Best performance
ax2_sum = fig3.add_subplot(gs[0, 1])
tasks_best = {
    'Stance': df['stance_accuracy'].max(),
    'Harmfulness': df['harmfulness_accuracy'].max(),
    'Fairness': df['fairness_accuracy'].max(),
    'Intent': df['intent_exact_match'].max()
}

x_pos = np.arange(len(tasks_best))
colors_best = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

bars = ax2_sum.bar(x_pos, list(tasks_best.values()), color=colors_best, edgecolor='black', linewidth=0.5)
ax2_sum.set_xlabel('Task')
ax2_sum.set_ylabel('Best Accuracy')
ax2_sum.set_title('Best Performance per Task', fontweight='bold')
ax2_sum.set_xticks(x_pos)
ax2_sum.set_xticklabels(tasks_best.keys())
ax2_sum.set_ylim(0, 1.0)
ax2_sum.grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars, tasks_best.values()):
    height = bar.get_height()
    ax2_sum.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)

# Subplot 3: Training stages
ax3_sum = fig3.add_subplot(gs[0, 2])
early_idx = len(df) // 3
mid_idx = 2 * len(df) // 3

stages = ['Early', 'Mid', 'Late']
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

bars1 = ax3_sum.bar(x - width/2, stance_stage_acc, width, label='Stance', color='#1f77b4')
bars2 = ax3_sum.bar(x + width/2, harm_stage_acc, width, label='Harmfulness', color='#ff7f0e')

ax3_sum.set_xlabel('Training Stage')
ax3_sum.set_ylabel('Avg Accuracy')
ax3_sum.set_title('Performance by Training Stage', fontweight='bold')
ax3_sum.set_xticks(x)
ax3_sum.set_xticklabels(stages)
ax3_sum.legend()
ax3_sum.grid(True, alpha=0.3, axis='y')

# Subplot 4: Summary table
ax4_sum = fig3.add_subplot(gs[1, :])
ax4_sum.axis('off')

summary_data = []
summary_data.append(['Best Val Loss', f"{df['val_loss'].min():.4f}", f"Epoch {int(df['epoch'][df['val_loss'].idxmin()])}"])
summary_data.append(['Best Train Loss', f"{df['train_loss'].min():.4f}", f"Epoch {int(df['epoch'][df['train_loss'].idxmin()])}"])
summary_data.append(['Best Stance Acc', f"{df['stance_accuracy'].max():.4f}", f"Epoch {int(df['epoch'][df['stance_accuracy'].idxmax()])}"])
summary_data.append(['Best Intent Acc', f"{df['intent_exact_match'].max():.4f}", f"Epoch {int(df['epoch'][df['intent_exact_match'].idxmax()])}"])
summary_data.append(['Overfitting', f"{(df['val_loss'].iloc[-1]/df['val_loss'].iloc[0]):.1f}x", "Val loss increase"])
summary_data.append(['LR Decay', f"{(df['lr'].iloc[0]/df['lr'].iloc[-1]):.0f}x", f"{df['lr'].iloc[0]:.1e}→{df['lr'].iloc[-1]:.1e}"])

table = ax4_sum.table(cellText=summary_data,
                      colLabels=['Metric', 'Value', 'Note'],
                      colColours=['lightgray', 'lightgray', 'lightgray'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
ax4_sum.set_title('Key Metrics Summary', fontweight='bold', fontsize=12, pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('training_summary_en.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("\n" + "="*70)
print("Charts generated successfully!")
print("="*70)
print(f"1. 6-plot layout: training_analysis_6plots_en.png")
print(f"2. 4-plot layout: training_analysis_4plots_en.png")
print(f"3. Key metrics summary: training_summary_en.png")
print("\nKey findings:")
print(f"- Overfitting: Val loss increased from {df['val_loss'].iloc[0]:.1f} to {df['val_loss'].iloc[-1]:.1f}")
print(f"- Best val loss at epoch: {int(df['epoch'][df['val_loss'].idxmin()])}")
print(f"- Best intent accuracy: {df['intent_exact_match'].max():.3f} at epoch {int(df['epoch'][df['intent_exact_match'].idxmax()])}")
print("="*70)