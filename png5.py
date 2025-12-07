import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class DMINTTrainingAnalyzer:
    def __init__(self, log_file_path='training_log.txt'):
        """
        初始化训练日志分析器
        
        Args:
            log_file_path: 训练日志文件路径，默认为'training_log.txt'
        """
        self.log_file_path = log_file_path
        self.df = None
        self.best_epoch = None
        
    def parse_log_file(self):
        """解析训练日志文件，提取所有epoch的数据"""
        print(f"正在解析日志文件: {self.log_file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(self.log_file_path):
            print(f"错误: 文件 '{self.log_file_path}' 不存在!")
            print("请检查文件路径，或者将文件复制到当前目录")
            return None
        
        # 读取整个日志文件
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"日志文件大小: {len(content)} 字符")
        
        # 首先尝试不同的解析方法
        # 方法1: 查找所有epoch完成的部分
        epoch_patterns = [
            # 模式1: Epoch X 完成: ... 验证指标:
            r'Epoch (\d+) 完成:\s*\n\s*训练损失:\s*([\d.]+)\s*\n\s*验证损失:\s*([\d.]+)\s*\n\s*当前学习率:\s*([\d.eE+-]+).*?验证指标:\s*\n(.*?)\n\n',
            # 模式2: 更宽松的匹配
            r'Epoch (\d+) 完成:(.*?)验证指标:(.*?)(?=\n\n|\nEpoch \d|$)'
        ]
        
        # 初始化数据列表
        data = []
        
        # 尝试第一种模式
        matches = re.findall(epoch_patterns[0], content, re.DOTALL)
        
        if matches:
            print(f"使用模式1找到 {len(matches)} 个epoch的数据")
            
            for match in matches:
                epoch_num = int(match[0])
                train_loss = float(match[1])
                val_loss = float(match[2])
                lr_str = match[3]
                
                # 解析学习率
                try:
                    if 'e' in lr_str or 'E' in lr_str:
                        lr = float(lr_str)
                    else:
                        lr = float(lr_str)
                except:
                    lr = 0.0
                
                # 解析指标部分
                metrics_text = match[4]
                
                # 提取所有指标
                metrics = {}
                metric_patterns = [
                    (r'stance_accuracy:\s*([\d.]+)', 'stance_accuracy'),
                    (r'stance_f1:\s*([\d.]+)', 'stance_f1'),
                    (r'harmfulness_accuracy:\s*([\d.]+)', 'harmfulness_accuracy'),
                    (r'harmfulness_f1:\s*([\d.]+)', 'harmfulness_f1'),
                    (r'fairness_accuracy:\s*([\d.]+)', 'fairness_accuracy'),
                    (r'fairness_f1:\s*([\d.]+)', 'fairness_f1'),
                    (r'intent_exact_match:\s*([\d.]+)', 'intent_exact_match'),
                    (r'intent_macro_f1:\s*([\d.]+)', 'intent_macro_f1'),
                    (r'intent_Political_f1:\s*([\d.]+)', 'intent_Political_f1'),
                    (r'intent_Economic_f1:\s*([\d.]+)', 'intent_Economic_f1'),
                    (r'intent_Psychological_f1:\s*([\d.]+)', 'intent_Psychological_f1'),
                    (r'intent_Public_f1:\s*([\d.]+)', 'intent_Public_f1')
                ]
                
                for pattern, key in metric_patterns:
                    m = re.search(pattern, metrics_text)
                    if m:
                        metrics[key] = float(m.group(1))
                    else:
                        metrics[key] = 0.0  # 默认值
                
                # 创建数据行
                row = {
                    'epoch': epoch_num,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': lr,
                    **metrics
                }
                data.append(row)
        
        # 如果第一种模式没找到数据，尝试第二种方法：直接提取关键信息
        if not data:
            print("尝试第二种解析方法...")
            
            # 查找所有epoch完成的部分
            epoch_sections = re.split(r'============================================================', content)
            
            for section in epoch_sections:
                # 查找epoch编号
                epoch_match = re.search(r'Epoch (\d+)/\d+', section)
                if not epoch_match:
                    continue
                
                epoch_num = int(epoch_match.group(1))
                
                # 查找训练损失
                train_loss_match = re.search(r'训练损失:\s*([\d.]+)', section)
                train_loss = float(train_loss_match.group(1)) if train_loss_match else 0.0
                
                # 查找验证损失
                val_loss_match = re.search(r'验证损失:\s*([\d.]+)', section)
                val_loss = float(val_loss_match.group(1)) if val_loss_match else 0.0
                
                # 查找学习率
                lr_match = re.search(r'当前学习率:\s*([\d.eE+-]+)', section)
                if lr_match:
                    lr_str = lr_match.group(1)
                    try:
                        lr = float(lr_str)
                    except:
                        lr = 0.0
                else:
                    lr = 0.0
                
                # 提取指标
                metrics = {}
                metric_names = [
                    'stance_accuracy', 'stance_f1',
                    'harmfulness_accuracy', 'harmfulness_f1',
                    'fairness_accuracy', 'fairness_f1',
                    'intent_exact_match', 'intent_macro_f1',
                    'intent_Political_f1', 'intent_Economic_f1',
                    'intent_Psychological_f1', 'intent_Public_f1'
                ]
                
                for metric in metric_names:
                    pattern = rf'{metric}:\s*([\d.]+)'
                    match = re.search(pattern, section)
                    if match:
                        metrics[metric] = float(match.group(1))
                    else:
                        metrics[metric] = 0.0
                
                row = {
                    'epoch': epoch_num,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': lr,
                    **metrics
                }
                data.append(row)
        
        if data:
            print(f"成功解析 {len(data)} 个epoch的数据")
            
            # 创建DataFrame并按epoch排序
            self.df = pd.DataFrame(data)
            self.df = self.df.sort_values('epoch').reset_index(drop=True)
            
            # 找到最佳epoch（验证损失最低）
            if not self.df.empty:
                best_idx = self.df['val_loss'].idxmin()
                self.best_epoch = int(self.df.loc[best_idx, 'epoch'])
                print(f"最佳epoch: {self.best_epoch} (验证损失: {self.df.loc[best_idx, 'val_loss']:.4f})")
            
            return self.df
        else:
            print("警告: 没有找到任何epoch数据!")
            print("日志前1000字符:")
            print(content[:1000])
            return None
    
    def create_visualization(self, save_path='DMINT_training_analysis.png'):
        """创建训练分析可视化图表"""
        if self.df is None or self.df.empty:
            print("错误: 没有数据可用于可视化")
            return
        
        print(f"创建可视化图表，包含 {len(self.df)} 个epoch的数据...")
        
        # 创建2x3的子图布局
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DMINT多任务学习模型训练分析', fontsize=16, fontweight='bold')
        
        # 子图1: 学习率变化
        ax1 = axes[0, 0]
        ax1.plot(self.df['epoch'], self.df['lr'], '^-', color='#556270', 
                linewidth=2, markersize=6, label='学习率')
        ax1.set_yscale('log')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('学习率 (log scale)')
        ax1.set_title('学习率变化曲线', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 子图2: 训练和验证损失
        ax2 = axes[0, 1]
        ax2.plot(self.df['epoch'], self.df['train_loss'], 'o-', color='#FF6B6B',
                linewidth=2, markersize=6, label='训练损失')
        ax2.plot(self.df['epoch'], self.df['val_loss'], 's-', color='#4ECDC4',
                linewidth=2, markersize=6, label='验证损失')
        
        # 标注过拟合区域（从epoch 5开始）
        if len(self.df) >= 5:
            ax2.axvspan(5, self.df['epoch'].max(), alpha=0.2, color='red', label='过拟合区域')
        
        # 标注最佳epoch
        if self.best_epoch is not None:
            best_val_loss = self.df[self.df['epoch'] == self.best_epoch]['val_loss'].values[0]
            ax2.axvline(x=self.best_epoch, color='red', linestyle='--', 
                       linewidth=1.5, alpha=0.7, label=f'最佳epoch ({self.best_epoch})')
            ax2.plot(self.best_epoch, best_val_loss, 'r*', markersize=12)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('损失值')
        ax2.set_title('训练损失与验证损失趋势', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3: Stance任务
        ax3 = axes[0, 2]
        ax3.plot(self.df['epoch'], self.df['stance_accuracy'], 'o-', 
                color='#00B8A9', linewidth=2, markersize=6, label='准确率')
        ax3.plot(self.df['epoch'], self.df['stance_f1'], 's-', 
                color='#F6416C', linewidth=2, markersize=6, label='F1分数')
        
        if self.best_epoch is not None:
            ax3.axvline(x=self.best_epoch, color='red', linestyle='--', 
                       linewidth=1.5, alpha=0.5)
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('分数')
        ax3.set_title('Stance任务性能', fontsize=12, fontweight='bold')
        ax3.set_ylim([0.5, 0.85])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 子图4: Harmfulness任务
        ax4 = axes[1, 0]
        ax4.plot(self.df['epoch'], self.df['harmfulness_accuracy'], 'o-', 
                color='#00B8A9', linewidth=2, markersize=6, label='准确率')
        ax4.plot(self.df['epoch'], self.df['harmfulness_f1'], 's-', 
                color='#F6416C', linewidth=2, markersize=6, label='F1分数')
        
        if self.best_epoch is not None:
            ax4.axvline(x=self.best_epoch, color='red', linestyle='--', 
                       linewidth=1.5, alpha=0.5)
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('分数')
        ax4.set_title('Harmfulness任务性能', fontsize=12, fontweight='bold')
        ax4.set_ylim([0.55, 0.75])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 子图5: Fairness任务
        ax5 = axes[1, 1]
        ax5.plot(self.df['epoch'], self.df['fairness_accuracy'], 'o-', 
                color='#00B8A9', linewidth=2, markersize=6, label='准确率')
        ax5.plot(self.df['epoch'], self.df['fairness_f1'], 's-', 
                color='#F6416C', linewidth=2, markersize=6, label='F1分数')
        
        if self.best_epoch is not None:
            ax5.axvline(x=self.best_epoch, color='red', linestyle='--', 
                       linewidth=1.5, alpha=0.5)
        
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('分数')
        ax5.set_title('Fairness任务性能', fontsize=12, fontweight='bold')
        ax5.set_ylim([0.65, 0.85])
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 子图6: Intent任务F1分数
        ax6 = axes[1, 2]
        # 绘制三条主要曲线
        ax6.plot(self.df['epoch'], self.df['intent_macro_f1'], 'o-', 
                color='#6A67CE', linewidth=2, markersize=6, label='Macro F1')
        ax6.plot(self.df['epoch'], self.df['intent_Political_f1'], '^-', 
                color='#00B8A9', linewidth=2, markersize=6, label='Political F1')
        ax6.plot(self.df['epoch'], self.df['intent_Public_f1'], 'v-', 
                color='#F6416C', linewidth=2, markersize=6, label='Public F1')
        
        # 添加其他F1作为参考
        if 'intent_Economic_f1' in self.df.columns:
            ax6.plot(self.df['epoch'], self.df['intent_Economic_f1'], 'd--', 
                    color='#FFA500', linewidth=1.5, markersize=5, alpha=0.7, label='Economic F1')
        
        if 'intent_Psychological_f1' in self.df.columns:
            ax6.plot(self.df['epoch'], self.df['intent_Psychological_f1'], 'x--', 
                    color='#9370DB', linewidth=1.5, markersize=5, alpha=0.7, label='Psychological F1')
        
        # 及格线
        ax6.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='及格线')
        
        if self.best_epoch is not None:
            ax6.axvline(x=self.best_epoch, color='red', linestyle='--', 
                       linewidth=1.5, alpha=0.5)
        
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('F1分数')
        ax6.set_title('Intent任务核心F1分数', fontsize=12, fontweight='bold')
        ax6.set_ylim([-0.05, 0.9])
        ax6.legend(loc='lower right', fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存为: {save_path}")
        
        return fig
    
    def generate_summary_report(self):
        """生成训练分析摘要报告"""
        if self.df is None or self.df.empty:
            print("错误: 没有数据可用于生成报告")
            return
        
        print("\n" + "="*70)
        print("DMINT训练分析报告")
        print("="*70)
        
        print(f"\n训练概况:")
        print(f"  - 总Epoch数: {len(self.df)}")
        print(f"  - Epoch范围: {self.df['epoch'].min()} - {self.df['epoch'].max()}")
        print(f"  - 最佳Epoch: {self.best_epoch}")
        
        # 获取最佳epoch的数据
        best_row = self.df[self.df['epoch'] == self.best_epoch].iloc[0] if self.best_epoch else None
        
        if best_row is not None:
            print(f"\n最佳模型性能 (Epoch {self.best_epoch}):")
            print(f"  - 训练损失: {best_row['train_loss']:.4f}")
            print(f"  - 验证损失: {best_row['val_loss']:.4f}")
            print(f"  - 学习率: {best_row['lr']:.2e}")
            
            print(f"\n任务性能:")
            print(f"  Stance - 准确率: {best_row['stance_accuracy']:.4f}, F1: {best_row['stance_f1']:.4f}")
            print(f"  Harmfulness - 准确率: {best_row['harmfulness_accuracy']:.4f}, F1: {best_row['harmfulness_f1']:.4f}")
            print(f"  Fairness - 准确率: {best_row['fairness_accuracy']:.4f}, F1: {best_row['fairness_f1']:.4f}")
            print(f"  Intent - Macro F1: {best_row['intent_macro_f1']:.4f}")
            
            print(f"\nIntent任务细粒度F1:")
            print(f"  Political: {best_row['intent_Political_f1']:.4f}")
            print(f"  Economic: {best_row['intent_Economic_f1']:.4f}")
            print(f"  Psychological: {best_row['intent_Psychological_f1']:.4f}")
            print(f"  Public: {best_row['intent_Public_f1']:.4f}")
        
        # 过拟合分析
        if len(self.df) > 5:
            print(f"\n过拟合分析:")
            print(f"  - 建议过拟合起始点: Epoch 5")
            print(f"  - 验证损失最低点: Epoch {self.best_epoch}")
            
            if best_row is not None:
                final_val_loss = self.df['val_loss'].iloc[-1]
                overfit_percent = ((final_val_loss - best_row['val_loss']) / best_row['val_loss']) * 100
                print(f"  - 最终验证损失: {final_val_loss:.4f}")
                print(f"  - 过拟合程度: {overfit_percent:.1f}%")
        
        print("\n建议:")
        print("  1. 使用验证损失最低的epoch作为最终模型")
        print("  2. 考虑使用早停策略避免过拟合")
        print("  3. 对表现较差的任务进行针对性优化")
        print("="*70 + "\n")
    
    def save_data_to_csv(self, filename='training_data.csv'):
        """将解析的数据保存到CSV文件"""
        if self.df is not None and not self.df.empty:
            self.df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"数据已保存到: {filename}")
            return True
        else:
            print("没有数据可保存")
            return False

# 主函数
def main():
    """主程序入口"""
    print("DMINT训练日志分析工具 v2.0")
    print("="*60)
    
    # 获取当前目录下的文件列表
    print("当前目录下的文件:")
    files = os.listdir('.')
    for f in files:
        if f.endswith('.txt'):
            print(f"  - {f}")
    
    # 尝试常见的日志文件名
    possible_files = ['logs\log_20251203_143539.txt']
    
    log_file = None
    for f in possible_files:
        if os.path.exists(f):
            log_file = f
            print(f"\n找到日志文件: {log_file}")
            break
    
    if log_file is None:
        # 让用户输入文件名
        log_file = input("请输入日志文件名: ").strip()
        if not os.path.exists(log_file):
            print(f"文件 '{log_file}' 不存在!")
            return
    
    # 创建分析器
    analyzer = DMINTTrainingAnalyzer(log_file)
    
    # 解析日志
    df = analyzer.parse_log_file()
    
    if df is not None and not df.empty:
        print(f"\n成功解析数据，共有 {len(df)} 个epoch")
        print("数据列:", list(df.columns))
        print("\n前3行数据:")
        print(df.head(3))
        
        # 生成摘要报告
        analyzer.generate_summary_report()
        
        # 保存数据到CSV
        analyzer.save_data_to_csv()
        
        # 创建可视化图表
        fig = analyzer.create_visualization('DMINT_training_analysis.png')
        
        # 显示图表
        plt.show()
        
        print("\n分析完成!")
    else:
        print("\n未能解析出有效数据，请检查日志文件格式")

if __name__ == "__main__":
    main()