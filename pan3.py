import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read data
df = pd.read_csv('log_20251203_143539_data.csv')

# Set font and plot style for English display
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Create subplot layout
fig = plt.figure(figsize=(20, 16))

# 1. Loss function change plot
ax1 = plt.subplot(3, 3, 1)
ax1.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Training Loss')
ax1.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, label='Validation Loss')
ax1.set_xlabel('Training Epochs')
ax1.set_ylabel('Loss Value')
ax1.set_title('Training vs Validation Loss Changes')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add markers for minimum loss points
min_train_idx = df['train_loss'].idxmin()
min_val_idx = df['val_loss'].idxmin()
ax1.scatter(df['epoch'][min_train_idx], df['train_loss'][min_train_idx], 
           color='blue', s=100, zorder=5, label=f'Min Train Loss: {df["train_loss"][min_train_idx]:.3f}')
ax1.scatter(df['epoch'][min_val_idx], df['val_loss'][min_val_idx], 
           color='red', s=100, zorder=5, label=f'Min Val Loss: {df["val_loss"][min_val_idx]:.3f}')
ax1.legend()

# 2. Learning rate change plot
ax2 = plt.subplot(3, 3, 2)
ax2.plot(df['epoch'], df['lr'], 'g-', linewidth=2)
ax2.set_xlabel('Training Epochs')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Learning Rate Changes (Exponential Decay)')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# 3. Main task accuracy comparison
ax3 = plt.subplot(3, 3, 3)
tasks = ['stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy']
colors = ['blue', 'red', 'green']
labels = ['Stance Accuracy', 'Harmfulness Accuracy', 'Fairness Accuracy']

for task, color, label in zip(tasks, colors, labels):
    ax3.plot(df['epoch'], df[task], color=color, linewidth=2, label=label)
ax3.set_xlabel('Training Epochs')
ax3.set_ylabel('Accuracy')
ax3.set_title('Main Task Accuracy Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Main task F1 score comparison
ax4 = plt.subplot(3, 3, 4)
f1_tasks = ['stance_f1', 'harmfulness_f1', 'fairness_f1']
for task, color, label in zip(f1_tasks, colors, labels):
    ax4.plot(df['epoch'], df[task], color=color, linewidth=2, label=label)
ax4.set_xlabel('Training Epochs')
ax4.set_ylabel('F1 Score')
ax4.set_title('Main Task F1 Score Comparison')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Intent classification metrics
ax5 = plt.subplot(3, 3, 5)
intent_metrics = ['intent_exact_match', 'intent_macro_f1']
intent_labels = ['Intent Exact Match', 'Intent Macro F1']
colors_intent = ['purple', 'orange']

for metric, color, label in zip(intent_metrics, colors_intent, intent_labels):
    ax5.plot(df['epoch'], df[metric], color=color, linewidth=2, label=label)
ax5.set_xlabel('Training Epochs')
ax5.set_ylabel('Score')
ax5.set_title('Intent Classification Metrics')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. F1 scores for intent classification categories
ax6 = plt.subplot(3, 3, 6)
intent_categories = ['intent_Political_f1', 'intent_Economic_f1', 
                     'intent_Psychological_f1', 'intent_Public_f1']
category_labels = ['Political Intent', 'Economic Intent', 'Psychological Intent', 'Public Intent']
category_colors = ['darkblue', 'darkgreen', 'darkred', 'darkorange']

for category, color, label in zip(intent_categories, category_colors, category_labels):
    ax6.plot(df['epoch'], df[category], color=color, linewidth=2, label=label)
ax6.set_xlabel('Training Epochs')
ax6.set_ylabel('F1 Score')
ax6.set_title('F1 Scores by Intent Category')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Performance summary after epoch 20 (to avoid early fluctuations)
ax7 = plt.subplot(3, 3, 7)
# Select data from epoch 20 onwards
df_late = df[df['epoch'] >= 20]

# Calculate average performance for each task after epoch 20
tasks_summary = ['stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy', 
                 'stance_f1', 'harmfulness_f1', 'fairness_f1',
                 'intent_exact_match', 'intent_macro_f1']
task_names = ['Stance Accuracy', 'Harmfulness Accuracy', 'Fairness Accuracy',
              'Stance F1', 'Harmfulness F1', 'Fairness F1',
              'Intent Exact Match', 'Intent Macro F1']

mean_values = [df_late[task].mean() for task in tasks_summary]

x_pos = np.arange(len(task_names))
bars = ax7.bar(x_pos, mean_values, color=plt.cm.tab20(np.arange(len(task_names))/len(task_names)))
ax7.set_xlabel('Task Metrics')
ax7.set_ylabel('Average Value (Epochs 20-40)')
ax7.set_title('Average Performance in Late Training Phase (Epochs 20-40)')
ax7.set_xticks(x_pos)
ax7.set_xticklabels(task_names, rotation=45, ha='right')
ax7.grid(True, alpha=0.3, axis='y')

# Add values on top of bars
for bar, value in zip(bars, mean_values):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{value:.3f}', ha='center', va='bottom', fontsize=9)

# 8. Heatmap of correlation between validation loss and key metrics
ax8 = plt.subplot(3, 3, 8)
correlation_metrics = ['val_loss', 'stance_accuracy', 'harmfulness_accuracy', 
                      'fairness_accuracy', 'intent_macro_f1', 'intent_exact_match']
corr_matrix = df[correlation_metrics].corr()

# Create heatmap
im = ax8.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax8.set_xticks(np.arange(len(correlation_metrics)))
ax8.set_yticks(np.arange(len(correlation_metrics)))
ax8.set_xticklabels(['Validation Loss', 'Stance Accuracy', 'Harmfulness Accuracy', 
                    'Fairness Accuracy', 'Intent Macro F1', 'Intent Exact Match'], rotation=45, ha='right')
ax8.set_yticklabels(['Validation Loss', 'Stance Accuracy', 'Harmfulness Accuracy', 
                    'Fairness Accuracy', 'Intent Macro F1', 'Intent Exact Match'])
ax8.set_title('Correlation: Validation Loss vs Key Metrics')

# Add correlation values
for i in range(len(correlation_metrics)):
    for j in range(len(correlation_metrics)):
        text = ax8.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white")

# 9. Best performance metrics display
ax9 = plt.subplot(3, 3, 9)
best_metrics = {
    'Minimum Training Loss': df['train_loss'].min(),
    'Minimum Validation Loss': df['val_loss'].min(),
    'Maximum Stance Accuracy': df['stance_accuracy'].max(),
    'Maximum Stance F1': df['stance_f1'].max(),
    'Maximum Harmfulness Accuracy': df['harmfulness_accuracy'].max(),
    'Maximum Fairness Accuracy': df['fairness_accuracy'].max(),
    'Maximum Intent Exact Match': df['intent_exact_match'].max(),
    'Maximum Intent Macro F1': df['intent_macro_f1'].max()
}

# Find epochs where best performance was achieved
best_epochs = {
    'Epoch of Min Train Loss': df['epoch'][df['train_loss'].idxmin()],
    'Epoch of Min Val Loss': df['epoch'][df['val_loss'].idxmin()],
    'Epoch of Max Stance Accuracy': df['epoch'][df['stance_accuracy'].idxmax()],
    'Epoch of Max Intent Match': df['epoch'][df['intent_exact_match'].idxmax()]
}

# Create text summary
summary_text = "Best Performance Summary:\n\n"
for metric, value in best_metrics.items():
    summary_text += f"{metric}: {value:.4f}\n"

summary_text += "\nKey Epochs:\n"
for metric, epoch in best_epochs.items():
    summary_text += f"{metric}: {int(epoch)}\n"

ax9.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center')
ax9.set_xlim(0, 1)
ax9.set_ylim(0, 1)
ax9.axis('off')
ax9.set_title('Best Performance Summary', fontsize=12)

plt.tight_layout()
plt.savefig('training_analysis_english.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional performance trend summary plot
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))

# Create performance trend indicators
performance_indicators = {
    'Loss Functions': ['train_loss', 'val_loss'],
    'Main Task Accuracy': ['stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy'],
    'Intent Classification Metrics': ['intent_exact_match', 'intent_macro_f1'],
    'Fine-grained Intent F1 Scores': ['intent_Political_f1', 'intent_Economic_f1', 'intent_Psychological_f1', 'intent_Public_f1']
}

for ax, (title, metrics) in zip(axes2.flatten(), performance_indicators.items()):
    for metric in metrics:
        # Format label by replacing underscores and capitalizing
        label = metric.replace('intent_', '').replace('_f1', ' F1').replace('_', ' ').title()
        ax.plot(df['epoch'], df[metric], linewidth=2, label=label)
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('performance_trends_english.png', dpi=300, bbox_inches='tight')
plt.show()