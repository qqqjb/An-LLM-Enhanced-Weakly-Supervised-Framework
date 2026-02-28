import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# 添加路径以导入模型
sys.path.append('/home/lab1015/qjb/VadCLIP-main')

from model import CLIPVAD
import xd_option
from utils.tools import get_prompt_text

class ModelComplexityEvaluator:
    def __init__(self, device="cuda:1"):
        self.device = device
        self.args = xd_option.parser.parse_args()
        
        # 标签映射
        self.label_map = dict({
            'A': '{Normal} scene, {Usual} activity, {Ordinary} behavior, {Routine} action, {Typical} situation',
            'B1': '{Physical} fight, {Violent} conflict, {Aggressive} confrontation, {Hostile} encounter, {Brutal} assault',
            'B2': '{Gun} violence, {Firearm} discharge, {Shooting} incident, {Gunfire} exchange, {Bullets} flying',
            'B4': '{Violent} protest, {Civil} unrest, {Mob} violence, {Chaotic} riot, {Public} disturbance',
            'B5': '{Physical} abuse, {Violent} mistreatment, {Cruel} assault, {Brutal} attack, {Intentional} harm',
            'B6': '{Vehicle} collision, {Traffic} accident, {Car} crash, {Auto} wreck, {Road} disaster',
            'G': '{Explosive} blast, {Powerful} detonation, {Fiery} eruption, {Devastating} explosion, {Bombarding} bang'
        })
        
        # 要测试的CLIP模型
        self.clip_models = ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14"]
        
        # 准备输入数据
        self.prepare_dummy_input()
        
    def prepare_dummy_input(self):
        """准备虚拟输入数据"""
        self.batch_size = 1
        self.visual_length = self.args.visual_length
        self.visual_width = self.args.visual_width
        
        self.dummy_visual = torch.randn(self.batch_size, self.visual_length, self.visual_width).to(self.device)
        self.dummy_padding_mask = None
        self.dummy_text = get_prompt_text(self.label_map)
        self.dummy_lengths = torch.tensor([self.visual_length]).to(self.device)
        
    def count_parameters(self, model):
        """统计模型参数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        clip_params = sum(p.numel() for p in model.clipmodel.parameters())
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'clip': clip_params,
            'others': total_params - clip_params
        }
        
    def calculate_model_flops(self, model):
        d = self.visual_width  # 512
        L = self.visual_length  # 256
        embed_dim = self.args.embed_dim  # 768
        heads = self.args.visual_head  # 1
        layers = self.args.visual_layers  # 1
        text_len = len(self.dummy_text)  # 7
        
        flops = 0
        
        # 1. Frame Position Embeddings (查找操作，基本忽略)
        pos_flops = 0  # 嵌入查找不算FLOPs
        
        # 2. Temporal Transformer 
        for layer_idx in range(layers):
            # QKV projection: 3 * d * d * L
            qkv_flops = 3 * d * d * L
            
            # Attention computation: 2 * L^2 * d
            attn_flops = 2 * (L ** 2) * d
            
            # Output projection: d * d * L  
            proj_flops = d * d * L
            
            # MLP in transformer: 2 * d * 4d * L
            transformer_mlp_flops = 2 * d * (4 * d) * L
            
            layer_flops = qkv_flops + attn_flops + proj_flops + transformer_mlp_flops
            flops += layer_flops
        
        print(f"Temporal Transformer FLOPs: {flops/1e9:.3f}G")
        
        # 3. Graph Convolution Network
        gc_flops = 0
        width = d // 2  # 256
        
        # adj4 computation (相似度矩阵计算)
        adj_computation = L * L * d  # x.matmul(x.permute) 
        adj_normalization = L * L    # 归一化和softmax
        
        # Graph convolutions
        gc_flops += d * width * L      # GC1: (512, 256) × L
        gc_flops += width * width * L  # GC2: (256, 256) × L  
        gc_flops += d * width * L      # GC3: (512, 256) × L
        gc_flops += width * width * L  # GC4: (256, 256) × L
        gc_flops += (2 * width) * d * L  # concat + linear: (512, 512) × L
        
        gc_flops += adj_computation + adj_normalization
        
        print(f"Graph Convolution FLOPs: {gc_flops/1e9:.3f}G")
        flops += gc_flops
        
        # 4. MLP modules (修正)
        mlp_flops = 0
        
        # MLP2: visual processing (visual_features + mlp2(visual_features))
        mlp2_flops = 2 * d * (4 * d) * L  # 对visual features的MLP
        
        # MLP1: text processing (text_features + mlp1(text_features))  
        # text_features维度是 [batch, text_len, d] 在visual-text interaction后
        mlp1_flops = 2 * d * (4 * d) * text_len  # 对text features的MLP
        
        mlp_flops = mlp1_flops + mlp2_flops
        
        print(f"MLP FLOPs: {mlp_flops/1e9:.3f}G")
        flops += mlp_flops
        
        # 5. Visual-Text Interaction
        interaction_flops = 0
        
        # Classifier: self.classifier(visual_features + self.mlp2(visual_features))
        classifier_flops = d * 1 * L
        
        # Visual attention: logits_attn @ visual_features
        # logits_attn shape: [batch, 1, L], visual_features: [batch, L, d]
        visual_attn_flops = 1 * L * d
        
        # Text features expansion and addition (基本忽略，只是张量操作)
        
        # Feature normalization (基本忽略，计算量很小)
        
        # Final similarity computation: visual_features_norm @ text_features_norm
        # [batch, L, d] @ [batch, d, text_len] = [batch, L, text_len]
        similarity_flops = L * d * text_len
        
        interaction_flops = classifier_flops + visual_attn_flops + similarity_flops
        
        print(f"Visual-Text Interaction FLOPs: {interaction_flops/1e9:.3f}G")
        flops += interaction_flops
        
        return flops
        
    def get_clip_flops(self, clip_model_name):
        """获取不同CLIP模型的大概FLOPs"""
        # 这些是估算值，基于模型架构
        clip_flops = {
            "RN50": 4.1e9,      # ~4.1G FLOPs
            "RN101": 7.8e9,     # ~7.8G FLOPs
            "RN50x4": 16.4e9,   # ~16.4G FLOPs  
            "RN50x16": 65.6e9,  # ~65.6G FLOPs
            "RN50x64": 262.4e9, # ~262.4G FLOPs
            "ViT-B/32": 4.4e9,  # ~4.4G FLOPs
            "ViT-B/16": 17.6e9, # ~17.6G FLOPs
            "ViT-L/14": 81.1e9, # ~81.1G FLOPs
        }
        return clip_flops.get(clip_model_name, 0)
    
    def measure_inference_time(self, model, num_runs=100, warmup_runs=10):
        """测量推理时间"""
        model.eval()
        
        # 预热
        print(f"Warming up for {warmup_runs} runs...")
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(self.dummy_visual, self.dummy_padding_mask, 
                         self.dummy_text, self.dummy_lengths)
        
        # GPU同步
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        
        # 正式测试
        print(f"Measuring inference time for {num_runs} runs...")
        times = []
        for i in range(num_runs):
            if i % 20 == 0:
                print(f"Progress: {i}/{num_runs}")
                
            start_time = time.time()
            with torch.no_grad():
                _ = model(self.dummy_visual, self.dummy_padding_mask, 
                         self.dummy_text, self.dummy_lengths)
            
            if self.device.startswith('cuda'):
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
    
    def measure_memory_usage(self, model):
        """测量GPU内存使用"""
        if not self.device.startswith('cuda'):
            return {'allocated': 0, 'reserved': 0}
            
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model.eval()
        with torch.no_grad():
            _ = model(self.dummy_visual, self.dummy_padding_mask, 
                     self.dummy_text, self.dummy_lengths)
        
        allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.max_memory_reserved() / 1024**3    # GB
        
        return {'allocated': allocated, 'reserved': reserved}
    
    def evaluate_single_model(self, clip_model_name):
        """评估单个模型"""
        print(f"\n{'='*60}")
        print(f"Evaluating {clip_model_name}")
        print(f"{'='*60}")
        
        try:
            # 创建模型
            model = CLIPVAD(
                self.args.classes_num, 
                self.args.embed_dim, 
                self.args.visual_length, 
                self.args.visual_width, 
                self.args.visual_head, 
                self.args.visual_layers, 
                self.args.attn_window, 
                self.args.prompt_prefix, 
                self.args.prompt_postfix, 
                self.device
            )
            
            # 替换CLIP模型 - 添加错误处理
            import clip
            try:
                model.clipmodel, _ = clip.load(clip_model_name, self.device)
            except Exception as e:
                print(f"Failed to load CLIP model {clip_model_name}: {e}")
                return None
                
            for clip_param in model.clipmodel.parameters():
                clip_param.requires_grad = False
            
            model.to(self.device)
            
            # 参数量统计
            param_info = self.count_parameters(model)
            print(f"Total parameters: {param_info['total']/1e6:.2f}M")
            print(f"Trainable parameters: {param_info['trainable']/1e6:.2f}M")
            print(f"CLIP parameters: {param_info['clip']/1e6:.2f}M")
            print(f"Other parameters: {param_info['others']/1e6:.2f}M")
            
            # FLOPs计算
            model_flops = self.calculate_model_flops(model)
            clip_flops = self.get_clip_flops(clip_model_name)
            total_flops = model_flops + clip_flops
            
            print(f"Model FLOPs (without CLIP): {model_flops/1e9:.2f}G")
            print(f"CLIP FLOPs (estimated): {clip_flops/1e9:.2f}G")
            print(f"Total FLOPs: {total_flops/1e9:.2f}G")
            
            # 内存使用测量
            memory_info = self.measure_memory_usage(model)
            print(f"GPU Memory - Allocated: {memory_info['allocated']:.2f}GB, Reserved: {memory_info['reserved']:.2f}GB")
            
            # 推理时间测量
            time_info = self.measure_inference_time(model, num_runs=50)  # 减少运行次数以加快测试
            print(f"Inference time: {time_info['mean']:.2f}±{time_info['std']:.2f}ms")
            print(f"Time range: {time_info['min']:.2f}ms - {time_info['max']:.2f}ms")
            
            # 清理内存
            del model
            torch.cuda.empty_cache()
            
            return {
                'model': clip_model_name,
                'total_params': param_info['total'],
                'trainable_params': param_info['trainable'],
                'clip_params': param_info['clip'],
                'other_params': param_info['others'],
                'model_flops': model_flops,
                'clip_flops': clip_flops,
                'total_flops': total_flops,
                'inference_time_mean': time_info['mean'],
                'inference_time_std': time_info['std'],
                'memory_allocated': memory_info['allocated'],
                'memory_reserved': memory_info['reserved']
            }
            
        except Exception as e:
            print(f"Error evaluating {clip_model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_all_models(self):
        """评估所有模型"""
        results = []
        
        print("Starting evaluation of all CLIP models...")
        print(f"Input shape: {self.dummy_visual.shape}")
        print(f"Device: {self.device}")
        
        for i, clip_model in enumerate(self.clip_models):
            print(f"\nProgress: {i+1}/{len(self.clip_models)}")
            result = self.evaluate_single_model(clip_model)
            if result:
                results.append(result)
        
        return results
    
    def save_results(self, results, filename="model_complexity_results.csv"):
        """保存结果到CSV"""
        if not results:
            print("No results to save!")
            return None
            
        df = pd.DataFrame(results)
        
        # 转换单位
        df['Total Params (M)'] = df['total_params'] / 1e6
        df['Trainable Params (M)'] = df['trainable_params'] / 1e6
        df['CLIP Params (M)'] = df['clip_params'] / 1e6
        df['Other Params (M)'] = df['other_params'] / 1e6
        df['Model FLOPs (G)'] = df['model_flops'] / 1e9
        df['CLIP FLOPs (G)'] = df['clip_flops'] / 1e9
        df['Total FLOPs (G)'] = df['total_flops'] / 1e9
        df['Inference Time (ms)'] = df['inference_time_mean']
        df['Time Std (ms)'] = df['inference_time_std']
        df['Memory (GB)'] = df['memory_allocated']
        
        # 选择要保存的列
        save_cols = ['model', 'Total Params (M)', 'Trainable Params (M)', 
                    'CLIP Params (M)', 'Other Params (M)', 'Model FLOPs (G)',
                    'CLIP FLOPs (G)', 'Total FLOPs (G)', 'Inference Time (ms)', 
                    'Time Std (ms)', 'Memory (GB)']
        
        df[save_cols].to_csv(filename, index=False, float_format='%.3f')
        print(f"\nResults saved to {filename}")
        
        return df[save_cols]
    
    def plot_results(self, df):
        """绘制结果图表"""
        if df is None or len(df) == 0:
            print("No data to plot!")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 参数量对比
        axes[0, 0].bar(df['model'], df['Total Params (M)'], color='skyblue')
        axes[0, 0].set_title('Total Parameters (M)', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # FLOPs对比
        axes[0, 1].bar(df['model'], df['Total FLOPs (G)'], color='lightgreen')
        axes[0, 1].set_title('Total FLOPs (G)', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 推理时间对比
        axes[0, 2].bar(df['model'], df['Inference Time (ms)'], color='lightcoral')
        axes[0, 2].set_title('Inference Time (ms)', fontsize=12)
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # FLOPs分解
        width = 0.35
        x = np.arange(len(df))
        axes[1, 0].bar(x - width/2, df['Model FLOPs (G)'], width, label='Model FLOPs', color='orange')
        axes[1, 0].bar(x + width/2, df['CLIP FLOPs (G)'], width, label='CLIP FLOPs', color='purple')
        axes[1, 0].set_title('FLOPs Breakdown (G)', fontsize=12)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(df['model'], rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 内存使用
        axes[1, 1].bar(df['model'], df['Memory (GB)'], color='gold')
        axes[1, 1].set_title('GPU Memory Usage (GB)', fontsize=12)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 效率对比散点图
        scatter = axes[1, 2].scatter(df['Total FLOPs (G)'], df['Inference Time (ms)'], 
                                   s=df['Total Params (M)']*3, alpha=0.7, c=range(len(df)), cmap='viridis')
        for i, model in enumerate(df['model']):
            axes[1, 2].annotate(model, 
                               (df['Total FLOPs (G)'].iloc[i], df['Inference Time (ms)'].iloc[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 2].set_xlabel('Total FLOPs (G)')
        axes[1, 2].set_ylabel('Inference Time (ms)')
        axes[1, 2].set_title('Efficiency vs Performance\n(Bubble size = Params)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_complexity_comparison.png', dpi=300, bbox_inches='tight')
        print("Plot saved to model_complexity_comparison.png")
        plt.show()
    
    def print_summary_table(self, df):
        """打印汇总表格"""
        if df is None or len(df) == 0:
            print("No data to display!")
            return
            
        print("\n" + "="*120)
        print("MODEL COMPLEXITY COMPARISON SUMMARY")
        print("="*120)
        
        # 设置pandas显示选项
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 15)
        
        print(df.round(3).to_string(index=False))
        print("="*120)
        
        # 打印一些统计信息
        print(f"\nSTATISTICS:")
        print(f"Lightest model: {df.loc[df['Total Params (M)'].idxmin(), 'model']} ({df['Total Params (M)'].min():.1f}M params)")
        print(f"Heaviest model: {df.loc[df['Total Params (M)'].idxmax(), 'model']} ({df['Total Params (M)'].max():.1f}M params)")
        print(f"Fastest model: {df.loc[df['Inference Time (ms)'].idxmin(), 'model']} ({df['Inference Time (ms)'].min():.1f}ms)")
        print(f"Slowest model: {df.loc[df['Inference Time (ms)'].idxmax(), 'model']} ({df['Inference Time (ms)'].max():.1f}ms)")
        print(f"Most efficient (lowest FLOPs): {df.loc[df['Total FLOPs (G)'].idxmin(), 'model']} ({df['Total FLOPs (G)'].min():.1f}G)")
        print(f"Least efficient (highest FLOPs): {df.loc[df['Total FLOPs (G)'].idxmax(), 'model']} ({df['Total FLOPs (G)'].max():.1f}G)")

def main():
    # 设置设备 - 改为 cuda:1
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device.startswith('cuda'):
        gpu_id = int(device.split(':')[1])
        print(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f}GB")
    
    # 创建评估器
    evaluator = ModelComplexityEvaluator(device)
    
    # 评估所有模型
    print("Starting comprehensive model complexity evaluation...")
    results = evaluator.evaluate_all_models()
    
    if not results:
        print("No results obtained!")
        return
    
    print(f"\nSuccessfully evaluated {len(results)} models")
    
    # 保存结果
    df = evaluator.save_results(results)
    
    # 打印汇总表格
    evaluator.print_summary_table(df)
    
    # 绘制图表
    evaluator.plot_results(df)
    
    print(f"\nEvaluation completed! Check the following files:")
    print("- model_complexity_results.csv: Detailed results")
    print("- model_complexity_comparison.png: Visualization")

if __name__ == "__main__":
    main()