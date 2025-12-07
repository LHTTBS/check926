import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class DMINTLogAnalyzer:
    def __init__(self, log_path):
        """
        DMINT训练日志分析器
        
        Args:
            log_path: 训练日志文件路径
        """
        self.log_path = Path(log_path)
        self.df = None
        self.best_epoch = None
        
    def parse_log(self):
        """解析训练日志文件"""
        print(f"正在解析日志文件: {self.log_path}")
        
        # 初始化数据列表
        epochs = []
        train_losses = []
        val_losses = []
        learning_rates = []
        
        # 指标字典
        metrics = {
            'stance_accuracy': [], 'stance_f1': [],
            'harmfulness_accuracy': [], 'harmfulness_f1': [],
            'fairness_accuracy': [], 'fairness_f1': [],
            'intent_exact_match': [], 'intent_macro_f1': [],
            'intent_Political_f1': [], 'intent_Economic_f1': [],
            'intent_Psychological_f1': [], 'intent_Public_f1': []
        }
        
        # 读取文件
        with open(self.log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_epoch = None
        in_validation = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # 匹配epoch开始
            epoch_match = re.match(r'Epoch (\d+)/\d+', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                print(f"找到Epoch {current_epoch}")
                continue
            
            # 匹配训练损失（批次级别）
            if 'Batch' in line and 'Loss:' in line:
                loss_match = re.search(r'Loss:\s*([\d.]+)', line)
                if loss_match and current_epoch and len(epochs) < current_epoch:
                    # 我们只关心每个epoch的最终损失，所以跳过批次损失
                    pass
                continue
            
            # 匹配epoch完成
            if 'Epoch' in line and '完成:' in line:
                # 获取epoch编号
                epoch_num_match = re.search(r'Epoch (\d+)', line)
                if epoch_num_match:
                    current_epoch = int(epoch_num_match.group(1))
                    epochs.append(current_epoch)
                continue
            
            # 匹配训练损失
            if '训练损失:' in line:
                loss_match = re.search(r'训练损失:\s*([\d.]+)', line)
                if loss_match:
                    train_losses.append(float(loss_match.group(1)))
                continue
            
            # 匹配验证损失
            if '验证损失:' in line:
                loss_match = re.search(r'验证损失:\s*([\d.]+)', line)
                if loss_match:
                    val_losses.append(float(loss_match.group(1)))
                continue
            
            # 匹配学习率
            if '当前学习率:' in line or '学习率:' in line:
                lr_match = re.search(r'当前学习率:\s*([\d.eE+-]+)', line) or \
                          re.search(r'学习率:\s*([\d.eE+-]+)', line)
                if lr_match:
                    lr_str = lr_match.group(1)
                    # 处理科学计数法
                    if 'e' in lr_str or 'E' in lr_str:
                        learning_rates.append(float(lr_str))
                    else:
                        learning_rates.append(float(lr_str))
                continue
            
            # 匹配验证指标开始
            if '验证指标:' in line:
                in_validation = True
                continue
            
            # 解析验证指标
            if in_validation:
                # 匹配各种指标
                for metric in metrics.keys():
                    pattern = rf'{metric}:\s*([\d.]+)'
                    match = re.search(pattern, line)
                    if match:
                        metrics[metric].append(float(match.group(1)))
                
                # 检查是否结束验证指标块
                if line.startswith('🎉') or '保存最佳模型' in line or line.startswith('================================'):
                    in_validation = False
        
        # 创建DataFrame
        data = {'epoch': epochs}
        
        # 检查数据长度一致性
        min_len = len(epochs)
        print(f"找到 {min_len} 个epoch")
        
        # 处理损失数据
        if len(train_losses) < min_len:
            train_losses.extend([None] * (min_len - len(train_losses)))
        if len(val_losses) < min_len:
            val_losses.extend([None] * (min_len - len(val_losses)))
        if len(learning_rates) < min_len:
            learning_rates.extend([None] * (min_len - len(learning_rates)))
        
        data.update({
            'train_loss': train_losses[:min_len],
            'val_loss': val_losses[:min_len],
            'lr': learning_rates[:min_len]
        })
        
        # 添加所有指标
        for metric, values in metrics.items():
            if len(values) < min_len:
                values.extend([None] * (min_len - len(values)))
            data[metric] = values[:min_len]
        
        self.df = pd.DataFrame(data)
        
        # 清理数据（去除NaN）
        self.df = self.df.dropna(subset=['val_loss'])
        
        # 找到最佳epoch（验证损失最低）
        if not self.df.empty:
            best_idx = self.df['val_loss'].idxmin()
            self.best_epoch = int(self.df.loc[best_idx, 'epoch'])
            print(f"最佳epoch: {self.best_epoch} (验证损失: {self.df.loc[best_idx, 'val_loss']:.4f})")
        
        return self.df
    
    def plot_analysis(self, save_path=None):
        """生成训练分析图表"""
        if self.df is None or self.df.empty:
            print("没有可用的数据，请先调用parse_log()")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DMINT多任务学习训练分析', fontsize=16, fontweight='bold')
        
        # 获取最佳epoch
        best_epoch = self.best_epoch
        best_epoch_data = self.df[self.df['epoch'] == best_epoch]
        
        # 1. 学习率曲线
        ax1 = axes[0, 0]
        ax1.plot(self.df['epoch'], self.df['lr'], '^-', color='#556270', 
                linewidth=2, markersize=6)
        ax1.set_yscale('log')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Learning Rate')
        ax1.set_title('学习率变化曲线', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 训练/验证损失
        ax2 = axes[0, 1]
        ax2.plot(self.df['epoch'], self.df['train_loss'], 'o-', color='#FF6B6B',
                linewidth=2, markersize=6, label='训练损失')
        ax2.plot(self.df['epoch'], self.df['val_loss'], 's-', color='#4ECDC4',
                linewidth=2, markersize=6, label='验证损失')
        
        # 标注过拟合区域（从epoch 5开始）
        if len(self.df) >= 5:
            ax2.axvspan(5, max(self.df['epoch']), alpha=0.2, color='red', label='过拟合区域')
        
        # 标注最佳epoch
        if best_epoch_data is not None and not best_epoch_data.empty:
            best_val_loss = best_epoch_data['val_loss'].values[0]
            ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, 
                       linewidth=1.5, label=f'最佳epoch ({best_epoch})')
            ax2.plot(best_epoch, best_val_loss, 'r*', markersize=12)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('训练与验证损失趋势', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Stance任务
        ax3 = axes[0, 2]
        if 'stance_accuracy' in self.df.columns and 'stance_f1' in self.df.columns:
            ax3.plot(self.df['epoch'], self.df['stance_accuracy'], 'o-', 
                    color='#00B8A9', linewidth=2, markersize=6, label='准确率')
            ax3.plot(self.df['epoch'], self.df['stance_f1'], 's-', 
                    color='#F6416C', linewidth=2, markersize=6, label='F1分数')
            if best_epoch:
                ax3.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax3.set_ylim([0.5, 0.85])
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.set_title('Stance任务性能', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Harmfulness任务
        ax4 = axes[1, 0]
        if 'harmfulness_accuracy' in self.df.columns and 'harmfulness_f1' in self.df.columns:
            ax4.plot(self.df['epoch'], self.df['harmfulness_accuracy'], 'o-', 
                    color='#00B8A9', linewidth=2, markersize=6, label='准确率')
            ax4.plot(self.df['epoch'], self.df['harmfulness_f1'], 's-', 
                    color='#F6416C', linewidth=2, markersize=6, label='F1分数')
            if best_epoch:
                ax4.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax4.set_ylim([0.55, 0.75])
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.set_title('Harmfulness任务性能', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Fairness任务
        ax5 = axes[1, 1]
        if 'fairness_accuracy' in self.df.columns and 'fairness_f1' in self.df.columns:
            ax5.plot(self.df['epoch'], self.df['fairness_accuracy'], 'o-', 
                    color='#00B8A9', linewidth=2, markersize=6, label='准确率')
            ax5.plot(self.df['epoch'], self.df['fairness_f1'], 's-', 
                    color='#F6416C', linewidth=2, markersize=6, label='F1分数')
            if best_epoch:
                ax5.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax5.set_ylim([0.65, 0.85])
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Score')
        ax5.set_title('Fairness任务性能', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Intent任务
        ax6 = axes[1, 2]
        intent_metrics = ['intent_macro_f1', 'intent_Political_f1', 'intent_Public_f1']
        colors = ['#6A67CE', '#00B8A9', '#F6416C']
        markers = ['o', '^', 'v']
        labels = ['Macro F1', 'Political F1', 'Public F1']
        
        for i, metric in enumerate(intent_metrics):
            if metric in self.df.columns:
                ax6.plot(self.df['epoch'], self.df[metric], 
                        marker=markers[i], linestyle='-', color=colors[i],
                        linewidth=2, markersize=6, label=labels[i])
        
        # 及格线
        ax6.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='及格线')
        
        if best_epoch:
            ax6.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('F1 Score')
        ax6.set_title('Intent任务F1分数', fontsize=12, fontweight='bold')
        ax6.set_ylim([-0.05, 0.9])
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.show()
        
        return fig
    
    def generate_report(self):
        """生成分析报告"""
        if self.df is None:
            return None
        
        report = {
            'epochs': len(self.df),
            'best_epoch': self.best_epoch,
            'best_val_loss': float(self.df['val_loss'].min()),
            'overfit_start': 5,  # 假设从epoch 5开始过拟合
            'metrics_summary': {}
        }
        
        # 计算各任务最佳性能
        if self.best_epoch:
            best_data = self.df[self.df['epoch'] == self.best_epoch].iloc[0]
            
            tasks = {
                'stance': ['accuracy', 'f1'],
                'harmfulness': ['accuracy', 'f1'],
                'fairness': ['accuracy', 'f1'],
                'intent': ['macro_f1']
            }
            
            for task, metrics_list in tasks.items():
                report['metrics_summary'][task] = {}
                for metric in metrics_list:
                    col_name = f"{task}_{metric}"
                    if col_name in best_data:
                        report['metrics_summary'][task][metric] = float(best_data[col_name])
        
        return report


def main():
    """主函数"""
    # 创建分析器
    analyzer = DMINTLogAnalyzer('outputs\ing_log.txt')
    
    # 解析日志
    df = analyzer.parse_log()
    
    if df is not None:
        print("\n" + "="*60)
        print("训练日志解析完成")
        print("="*60)
        
        print(f"\n数据形状: {df.shape}")
        print(f"Epoch范围: {df['epoch'].min()} - {df['epoch'].max()}")
        print(f"最佳Epoch: {analyzer.best_epoch}")
        
        print("\n前3个Epoch的数据:")
        print(df.head(3))
        
        print("\n关键指标摘要:")
        print(f"最低训练损失: {df['train_loss'].min():.4f} (Epoch {df['train_loss'].idxmin()+1})")
        print(f"最低验证损失: {df['val_loss'].min():.4f} (Epoch {analyzer.best_epoch})")
        
        if analyzer.best_epoch:
            best_row = df[df['epoch'] == analyzer.best_epoch].iloc[0]
            print(f"\n最佳Epoch ({analyzer.best_epoch}) 关键指标:")
            print(f"  Stance Accuracy: {best_row['stance_accuracy']:.4f}")
            print(f"  Harmfulness F1: {best_row['harmfulness_f1']:.4f}")
            print(f"  Fairness F1: {best_row['fairness_f1']:.4f}")
            print(f"  Intent Macro F1: {best_row['intent_macro_f1']:.4f}")
        
        # 生成可视化图表
        analyzer.plot_analysis('DMINT_training_analysis.png')
        
        # 生成报告
        report = analyzer.generate_report()
        print("\n" + "="*60)
        print("分析报告")
        print("="*60)
        
        for key, value in report.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()