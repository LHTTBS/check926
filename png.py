import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

class DMINTLogAnalyzer:
    def __init__(self, log_file=None):
        """
        DMINT训练日志分析器
        
        Args:
            log_file: 训练日志文件路径，如果为None则自动查找
        """
        if log_file is None:
            # 自动查找可能的日志文件
            log_files = glob.glob('*.txt') + glob.glob('logs/*.txt') + glob.glob('**/*.txt', recursive=True)
            if log_files:
                self.log_file = log_files[0]  # 使用第一个找到的txt文件
                print(f"自动选择日志文件: {self.log_file}")
            else:
                raise FileNotFoundError("未找到日志文件")
        else:
            self.log_file = log_file
            
        self.df = None
        self.best_epoch = None
        
    def parse_log(self):
        """解析训练日志文件"""
        print(f"正在解析日志文件: {self.log_file}")
        
        if not os.path.exists(self.log_file):
            raise FileNotFoundError(f"文件不存在: {self.log_file}")
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"日志文件大小: {len(content):,} 字符")
        
        # 更灵活的解析方式 - 查找所有epoch的数据
        # 使用更简单的模式匹配每个epoch的数据块
        epoch_blocks = re.split(r'Epoch \d+/\d+\s*[-]+\n', content)[1:]  # 跳过第一个空的部分
        
        print(f"找到 {len(epoch_blocks)} 个epoch的数据块")
        
        data = []
        
        for i, block in enumerate(epoch_blocks):
            epoch_num = i + 1
            
            # 初始化字典
            epoch_data = {'epoch': epoch_num}
            
            # 提取训练损失
            train_loss_match = re.search(r'训练损失:\s*([\d.]+)', block)
            if train_loss_match:
                epoch_data['train_loss'] = float(train_loss_match.group(1))
            
            # 提取验证损失
            val_loss_match = re.search(r'验证损失:\s*([\d.]+)', block)
            if val_loss_match:
                epoch_data['val_loss'] = float(val_loss_match.group(1))
            
            # 提取学习率
            lr_match = re.search(r'当前学习率:\s*([\d.eE+-]+)', block)
            if lr_match:
                lr_str = lr_match.group(1)
                try:
                    epoch_data['lr'] = float(lr_str)
                except:
                    epoch_data['lr'] = 2e-5  # 默认值
            
            # 提取所有指标
            metric_patterns = [
                ('stance_accuracy', r'stance_accuracy:\s*([\d.]+)'),
                ('stance_f1', r'stance_f1:\s*([\d.]+)'),
                ('harmfulness_accuracy', r'harmfulness_accuracy:\s*([\d.]+)'),
                ('harmfulness_f1', r'harmfulness_f1:\s*([\d.]+)'),
                ('fairness_accuracy', r'fairness_accuracy:\s*([\d.]+)'),
                ('fairness_f1', r'fairness_f1:\s*([\d.]+)'),
                ('intent_exact_match', r'intent_exact_match:\s*([\d.]+)'),
                ('intent_macro_f1', r'intent_macro_f1:\s*([\d.]+)'),
                ('intent_Political_f1', r'intent_Political_f1:\s*([\d.]+)'),
                ('intent_Economic_f1', r'intent_Economic_f1:\s*([\d.]+)'),
                ('intent_Psychological_f1', r'intent_Psychological_f1:\s*([\d.]+)'),
                ('intent_Public_f1', r'intent_Public_f1:\s*([\d.]+)')
            ]
            
            for metric_name, pattern in metric_patterns:
                match = re.search(pattern, block)
                if match:
                    try:
                        epoch_data[metric_name] = float(match.group(1))
                    except:
                        epoch_data[metric_name] = 0.0
                else:
                    epoch_data[metric_name] = 0.0
            
            data.append(epoch_data)
        
        self.df = pd.DataFrame(data)
        
        if not self.df.empty:
            self.df = self.df.sort_values('epoch').reset_index(drop=True)
            best_idx = self.df['val_loss'].idxmin()
            self.best_epoch = int(self.df.loc[best_idx, 'epoch'])
            print(f"最佳epoch: {self.best_epoch} (验证损失: {self.df.loc[best_idx, 'val_loss']:.4f})")
        
        return self.df
    
    def plot_analysis(self, save_path=None, show=True):
        """生成并显示/保存分析图表"""
        if self.df is None or self.df.empty:
            print("错误: 没有数据可用于绘图")
            return None
        
        print(f"准备绘制 {len(self.df)} 个epoch的可视化图表...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle(f'DMINT多任务学习训练分析 (共{len(self.df)}个Epoch)', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 子图1: 学习率变化
        ax1 = axes[0, 0]
        ax1.plot(self.df['epoch'], self.df['lr'], '^-', color='#556270', 
                linewidth=2, markersize=6, label='学习率')
        ax1.set_yscale('log')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Learning Rate', fontsize=12)
        ax1.set_title('学习率变化曲线', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10)
        
        # 子图2: 训练/验证损失
        ax2 = axes[0, 1]
        ax2.plot(self.df['epoch'], self.df['train_loss'], 'o-', color='#FF6B6B',
                linewidth=2, markersize=6, label='训练损失')
        ax2.plot(self.df['epoch'], self.df['val_loss'], 's-', color='#4ECDC4',
                linewidth=2, markersize=6, label='验证损失')
        
        # 标注过拟合区域（从epoch 5开始）
        if len(self.df) >= 5:
            ax2.axvspan(5, self.df['epoch'].max(), alpha=0.2, color='red', 
                       label='过拟合区域')
        
        # 标注最佳epoch
        if self.best_epoch is not None:
            best_val_loss = self.df.loc[self.df['epoch'] == self.best_epoch, 'val_loss'].values[0]
            ax2.axvline(x=self.best_epoch, color='red', linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'最佳epoch ({self.best_epoch})')
            ax2.plot(self.best_epoch, best_val_loss, 'r*', markersize=15, 
                    markeredgecolor='black', markeredgewidth=1)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('训练与验证损失趋势', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # 子图3: Stance任务性能
        ax3 = axes[0, 2]
        ax3.plot(self.df['epoch'], self.df['stance_accuracy'], 'o-', 
                color='#00B8A9', linewidth=2, markersize=6, label='准确率')
        ax3.plot(self.df['epoch'], self.df['stance_f1'], 's-', 
                color='#F6416C', linewidth=2, markersize=6, label='F1分数')
        
        if self.best_epoch is not None:
            ax3.axvline(x=self.best_epoch, color='red', linestyle='--', 
                       linewidth=2, alpha=0.5)
            # 标记最佳点的值
            best_stance_acc = self.df.loc[self.df['epoch'] == self.best_epoch, 'stance_accuracy'].values[0]
            best_stance_f1 = self.df.loc[self.df['epoch'] == self.best_epoch, 'stance_f1'].values[0]
            ax3.annotate(f'{best_stance_acc:.3f}', 
                        xy=(self.best_epoch, best_stance_acc),
                        xytext=(self.best_epoch+1, best_stance_acc),
                        fontsize=9)
            ax3.annotate(f'{best_stance_f1:.3f}', 
                        xy=(self.best_epoch, best_stance_f1),
                        xytext=(self.best_epoch+1, best_stance_f1-0.02),
                        fontsize=9)
        
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_title('Stance任务性能', fontsize=14, fontweight='bold')
        ax3.set_ylim([0.5, 0.85])
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # 子图4: Harmfulness任务性能
        ax4 = axes[1, 0]
        ax4.plot(self.df['epoch'], self.df['harmfulness_accuracy'], 'o-', 
                color='#00B8A9', linewidth=2, markersize=6, label='准确率')
        ax4.plot(self.df['epoch'], self.df['harmfulness_f1'], 's-', 
                color='#F6416C', linewidth=2, markersize=6, label='F1分数')
        
        if self.best_epoch is not None:
            ax4.axvline(x=self.best_epoch, color='red', linestyle='--', 
                       linewidth=2, alpha=0.5)
            # 标记最佳点的值
            best_harm_acc = self.df.loc[self.df['epoch'] == self.best_epoch, 'harmfulness_accuracy'].values[0]
            best_harm_f1 = self.df.loc[self.df['epoch'] == self.best_epoch, 'harmfulness_f1'].values[0]
            ax4.annotate(f'{best_harm_acc:.3f}', 
                        xy=(self.best_epoch, best_harm_acc),
                        xytext=(self.best_epoch+1, best_harm_acc),
                        fontsize=9)
            ax4.annotate(f'{best_harm_f1:.3f}', 
                        xy=(self.best_epoch, best_harm_f1),
                        xytext=(self.best_epoch+1, best_harm_f1-0.02),
                        fontsize=9)
        
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_title('Harmfulness任务性能', fontsize=14, fontweight='bold')
        ax4.set_ylim([0.55, 0.75])
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        # 子图5: Fairness任务性能
        ax5 = axes[1, 1]
        ax5.plot(self.df['epoch'], self.df['fairness_accuracy'], 'o-', 
                color='#00B8A9', linewidth=2, markersize=6, label='准确率')
        ax5.plot(self.df['epoch'], self.df['fairness_f1'], 's-', 
                color='#F6416C', linewidth=2, markersize=6, label='F1分数')
        
        if self.best_epoch is not None:
            ax5.axvline(x=self.best_epoch, color='red', linestyle='--', 
                       linewidth=2, alpha=0.5)
            # 标记最佳点的值
            best_fair_acc = self.df.loc[self.df['epoch'] == self.best_epoch, 'fairness_accuracy'].values[0]
            best_fair_f1 = self.df.loc[self.df['epoch'] == self.best_epoch, 'fairness_f1'].values[0]
            ax5.annotate(f'{best_fair_acc:.3f}', 
                        xy=(self.best_epoch, best_fair_acc),
                        xytext=(self.best_epoch+1, best_fair_acc),
                        fontsize=9)
            ax5.annotate(f'{best_fair_f1:.3f}', 
                        xy=(self.best_epoch, best_fair_f1),
                        xytext=(self.best_epoch+1, best_fair_f1-0.02),
                        fontsize=9)
        
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('Score', fontsize=12)
        ax5.set_title('Fairness任务性能', fontsize=14, fontweight='bold')
        ax5.set_ylim([0.65, 0.85])
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3, linestyle='--')
        
        # 子图6: Intent任务F1分数
        ax6 = axes[1, 2]
        
        # 绘制主要的三条曲线
        if 'intent_macro_f1' in self.df.columns:
            ax6.plot(self.df['epoch'], self.df['intent_macro_f1'], 'o-', 
                    color='#6A67CE', linewidth=2, markersize=6, label='Macro F1')
        
        if 'intent_Political_f1' in self.df.columns:
            ax6.plot(self.df['epoch'], self.df['intent_Political_f1'], '^-', 
                    color='#00B8A9', linewidth=2, markersize=6, label='Political F1')
        
        if 'intent_Public_f1' in self.df.columns:
            ax6.plot(self.df['epoch'], self.df['intent_Public_f1'], 'v-', 
                    color='#F6416C', linewidth=2, markersize=6, label='Public F1')
        
        # 添加其他F1作为参考
        if 'intent_Economic_f1' in self.df.columns:
            ax6.plot(self.df['epoch'], self.df['intent_Economic_f1'], 'd--', 
                    color='#FFA500', linewidth=1.5, markersize=5, alpha=0.8, label='Economic F1')
        
        if 'intent_Psychological_f1' in self.df.columns:
            ax6.plot(self.df['epoch'], self.df['intent_Psychological_f1'], 'x--', 
                    color='#9370DB', linewidth=1.5, markersize=5, alpha=0.8, label='Psychological F1')
        
        # 及格线
        ax6.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='及格线')
        
        if self.best_epoch is not None:
            ax6.axvline(x=self.best_epoch, color='red', linestyle='--', 
                       linewidth=2, alpha=0.5)
            # 标记最佳点的值
            best_macro_f1 = self.df.loc[self.df['epoch'] == self.best_epoch, 'intent_macro_f1'].values[0]
            ax6.plot(self.best_epoch, best_macro_f1, 'r*', markersize=15, 
                    markeredgecolor='black', markeredgewidth=1)
            ax6.annotate(f'{best_macro_f1:.3f}', 
                        xy=(self.best_epoch, best_macro_f1),
                        xytext=(self.best_epoch+1, best_macro_f1),
                        fontsize=9)
        
        ax6.set_xlabel('Epoch', fontsize=12)
        ax6.set_ylabel('F1 Score', fontsize=12)
        ax6.set_title('Intent任务核心F1分数', fontsize=14, fontweight='bold')
        ax6.set_ylim([0.1, 0.9])
        ax6.legend(loc='lower right', fontsize=10)
        ax6.grid(True, alpha=0.3, linestyle='--')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"图表已保存为: {save_path}")
        
        # 显示图表
        if show:
            plt.show()
        else:
            print("提示: 图表已保存但未显示。您可以在文件管理器中打开PNG文件查看。")
        
        return fig
    
    def save_to_csv(self, filename=None):
        """保存解析的数据到CSV文件"""
        if self.df is None or self.df.empty:
            print("没有数据可保存")
            return False
        
        if filename is None:
            # 基于日志文件名生成CSV文件名
            base_name = os.path.basename(self.log_file)
            csv_name = base_name.replace('.txt', '_data.csv')
            filename = csv_name
        
        self.df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"数据已保存到CSV文件: {filename}")
        return True
    
    def print_summary(self):
        """打印数据摘要"""
        if self.df is None or self.df.empty:
            print("没有数据可显示")
            return
        
        print("\n" + "="*80)
        print("DMINT训练数据摘要")
        print("="*80)
        
        print(f"数据形状: {self.df.shape}")
        print(f"Epoch范围: {self.df['epoch'].min()} - {self.df['epoch'].max()}")
        print(f"最佳Epoch: {self.best_epoch}")
        
        if self.best_epoch is not None:
            best_row = self.df[self.df['epoch'] == self.best_epoch].iloc[0]
            print(f"\n最佳Epoch ({self.best_epoch}) 的关键指标:")
            print(f"  训练损失: {best_row['train_loss']:.4f}")
            print(f"  验证损失: {best_row['val_loss']:.4f}")
            print(f"  学习率: {best_row['lr']:.2e}")
            print(f"  Stance准确率: {best_row['stance_accuracy']:.4f}")
            print(f"  Harmfulness F1: {best_row['harmfulness_f1']:.4f}")
            print(f"  Fairness F1: {best_row['fairness_f1']:.4f}")
            print(f"  Intent Macro F1: {best_row['intent_macro_f1']:.4f}")
            
            print(f"\nIntent任务细粒度F1分数:")
            print(f"  Political: {best_row['intent_Political_f1']:.4f}")
            print(f"  Economic: {best_row['intent_Economic_f1']:.4f}")
            print(f"  Psychological: {best_row['intent_Psychological_f1']:.4f}")
            print(f"  Public: {best_row['intent_Public_f1']:.4f}")
        
        print(f"\n训练过程分析:")
        print(f"  过拟合起始点: Epoch 5")
        print(f"  验证损失最低点: Epoch {self.best_epoch}")
        
        if self.best_epoch is not None:
            final_val_loss = self.df['val_loss'].iloc[-1]
            best_val_loss = self.df.loc[self.df['epoch'] == self.best_epoch, 'val_loss'].values[0]
            if best_val_loss > 0:
                overfit_percent = ((final_val_loss - best_val_loss) / best_val_loss) * 100
                print(f"  最终验证损失: {final_val_loss:.4f}")
                print(f"  过拟合程度: {overfit_percent:.1f}%")
        
        print("\n数据预览:")
        print(self.df[['epoch', 'train_loss', 'val_loss', 'stance_accuracy', 'intent_macro_f1']].head(5))
        print("="*80 + "\n")


def main():
    """主函数"""
    print("DMINT训练日志分析工具")
    print("="*60)
    
    # 指定您的日志文件路径
    log_file = "logs\\log_20251203_143539.txt"
    
    # 检查文件是否存在
    if not os.path.exists(log_file):
        print(f"错误: 文件 '{log_file}' 不存在!")
        print("正在搜索可能的日志文件...")
        
        # 搜索所有可能的日志文件
        log_files = glob.glob('*.txt') + glob.glob('logs/*.txt') + glob.glob('**/*.txt', recursive=True)
        if log_files:
            print("找到以下日志文件:")
            for i, f in enumerate(log_files):
                print(f"  [{i+1}] {f}")
            
            if len(log_files) == 1:
                log_file = log_files[0]
                print(f"自动选择: {log_file}")
            else:
                try:
                    choice = input(f"请选择要分析的日志文件 (1-{len(log_files)}): ")
                    idx = int(choice) - 1
                    if 0 <= idx < len(log_files):
                        log_file = log_files[idx]
                    else:
                        log_file = log_files[0]
                except:
                    log_file = log_files[0]
                    print(f"使用默认文件: {log_file}")
        else:
            print("未找到任何日志文件!")
            return
    
    print(f"\n分析文件: {log_file}")
    
    try:
        # 创建分析器
        analyzer = DMINTLogAnalyzer(log_file)
        
        # 解析日志
        df = analyzer.parse_log()
        
        if df is not None and not df.empty:
            # 打印摘要
            analyzer.print_summary()
            
            # 保存到CSV
            csv_name = os.path.basename(log_file).replace('.txt', '_data.csv')
            analyzer.save_to_csv(csv_name)
            
            # 生成图表
            png_name = os.path.basename(log_file).replace('.txt', '_analysis.png')
            print(f"\n正在生成图表并保存为: {png_name}")
            
            # 创建图表
            fig = analyzer.plot_analysis(save_path=png_name, show=True)
            
            # 生成附加图表
            print("\n生成附加分析图表...")
            generate_additional_plots(analyzer, os.path.basename(log_file).replace('.txt', ''))
            
            print("\n分析完成!")
            print(f"生成的文件:")
            print(f"  1. 数据文件: {csv_name}")
            print(f"  2. 主分析图表: {png_name}")
            print(f"  3. 附加图表: 多个PNG文件")
            
        else:
            print("未能解析出有效数据")
            
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()


def generate_additional_plots(analyzer, base_name):
    """生成附加分析图表"""
    if analyzer.df is None or analyzer.df.empty:
        return
    
    df = analyzer.df
    
    # 图表1: 所有任务准确率对比
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    tasks = ['stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    labels = ['Stance准确率', 'Harmfulness准确率', 'Fairness准确率']
    
    for i, task in enumerate(tasks):
        if task in df.columns:
            ax1.plot(df['epoch'], df[task], color=colors[i], linewidth=2, label=labels[i])
    
    if analyzer.best_epoch is not None:
        ax1.axvline(x=analyzer.best_epoch, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'最佳Epoch ({analyzer.best_epoch})')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('准确率', fontsize=12)
    ax1.set_title('所有任务准确率对比', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{base_name}_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 图表2: 所有任务F1分数对比
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    tasks_f1 = ['stance_f1', 'harmfulness_f1', 'fairness_f1', 'intent_macro_f1']
    colors_f1 = ['#F6416C', '#00B8A9', '#96CEB4', '#6A67CE']
    labels_f1 = ['Stance F1', 'Harmfulness F1', 'Fairness F1', 'Intent Macro F1']
    
    for i, task in enumerate(tasks_f1):
        if task in df.columns:
            ax2.plot(df['epoch'], df[task], color=colors_f1[i], linewidth=2, label=labels_f1[i])
    
    if analyzer.best_epoch is not None:
        ax2.axvline(x=analyzer.best_epoch, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('F1分数', fontsize=12)
    ax2.set_title('所有任务F1分数对比', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{base_name}_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 图表3: 损失比率分析
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    # 计算训练/验证损失比率
    loss_ratio = df['train_loss'] / df['val_loss']
    ax3.plot(df['epoch'], loss_ratio, 'o-', color='#556270', 
            linewidth=2, markersize=5, label='训练损失/验证损失')
    
    # 添加参考线
    ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='平衡线')
    
    # 标注过拟合区域
    if len(df) >= 5:
        ax3.axvspan(5, df['epoch'].max(), alpha=0.2, color='red', label='过拟合区域')
    
    if analyzer.best_epoch is not None:
        ax3.axvline(x=analyzer.best_epoch, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7)
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('损失比率', fontsize=12)
    ax3.set_title('损失比率与过拟合分析', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{base_name}_loss_ratio.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"已生成附加图表:")
    print(f"  - {base_name}_accuracy_comparison.png")
    print(f"  - {base_name}_f1_comparison.png")
    print(f"  - {base_name}_loss_ratio.png")


if __name__ == "__main__":
    main()