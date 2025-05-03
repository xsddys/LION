#!/usr/bin/env python
# 评估生成的点云样本与参考点云数据集（本地Windows版本）

import os
import sys
import argparse
import shutil
sys.path.append('.')
import torch
from loguru import logger
from utils.eval_helper import compute_score

def main():
    parser = argparse.ArgumentParser(description='点云样本评估工具')
    parser.add_argument('--samples', type=str, default=None, 
                        help='生成的样本文件路径（.pt文件）')
    parser.add_argument('--ref', type=str, default='./datasets/test_data/ref_val_cow.pt',
                        help='参考点云数据集路径')
    parser.add_argument('--norm_box', action='store_true',
                        help='是否进行归一化框处理')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='评估时的批次大小')
    
    args = parser.parse_args()
    
    # 如果没有指定样本文件，尝试查找样本文件
    if args.samples is None:
        # 尝试在常见位置查找samples文件
        potential_paths = [
            "./samples_71999s1H994c9diet.pt",  # 当前目录
            "./eval/samples_71999s1H994c9diet.pt",  # eval子目录
            "../eval/samples_71999s1H994c9diet.pt"  # 上级目录的eval子目录
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                args.samples = path
                logger.info(f"自动找到样本文件: {path}")
                break
        
        if args.samples is None:
            logger.error("未找到样本文件，请使用--samples参数指定样本文件路径")
            return
    
    # 检查文件是否存在
    if not os.path.exists(args.samples):
        logger.error(f"找不到生成的样本文件: {args.samples}")
        return
    
    if not os.path.exists(args.ref):
        logger.error(f"找不到参考点云数据集: {args.ref}")
        return
    
    # 创建结果目录
    results_dir = "./eval_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 复制样本文件到结果目录（如果不在当前目录）
    local_samples_path = os.path.join(results_dir, os.path.basename(args.samples))
    if args.samples != local_samples_path:
        try:
            shutil.copy2(args.samples, local_samples_path)
            logger.info(f"已复制样本文件到: {local_samples_path}")
        except Exception as e:
            logger.warning(f"复制样本文件失败: {e}")
            local_samples_path = args.samples
    
    logger.info(f"开始评估生成的样本: {args.samples}")
    logger.info(f"参考数据集: {args.ref}")
    logger.info(f"归一化框处理: {'启用' if args.norm_box else '禁用'}")
    
    try:
        # 使用compute_score函数评估生成的样本
        results = compute_score(
            output_name=args.samples,
            ref_name=args.ref,
            batch_size_test=args.batch_size,  # 根据您的GPU内存调整批次大小
            device_str='cuda' if torch.cuda.is_available() else 'cpu',
            accelerated_cd=True,
            norm_box=args.norm_box,
            dataset="cow",
            hash="eval",
            step="71.999k",
            epoch="evaluation"
        )
        
        # 输出评估结果
        logger.info("评估完成，结果如下:")
        logger.info(f"MinMatDis (MMD) | CD: {results['lgan_mmd-CD']:.6f} | EMD: {results['lgan_mmd-EMD']:.6f}")
        logger.info(f"Coverage (COV) | CD: {results['lgan_cov-CD']:.6f} | EMD: {results['lgan_cov-EMD']:.6f}")
        logger.info(f"1NN-Accuracy | CD: {results['1-NN-CD-acc']:.6f} | EMD: {results['1-NN-EMD-acc']:.6f}")
        logger.info(f"Jensen-Shannon Divergence (JSD): {results['jsd']:.6f}")
        
        # 保存结果到文本文件
        result_file = os.path.join(results_dir, "evaluation_results.txt")
        with open(result_file, 'w') as f:
            f.write(f"样本文件: {args.samples}\n")
            f.write(f"参考数据集: {args.ref}\n")
            f.write(f"归一化框处理: {'启用' if args.norm_box else '禁用'}\n\n")
            f.write("评估结果:\n")
            f.write(f"MinMatDis (MMD) | CD: {results['lgan_mmd-CD']:.6f} | EMD: {results['lgan_mmd-EMD']:.6f}\n")
            f.write(f"Coverage (COV) | CD: {results['lgan_cov-CD']:.6f} | EMD: {results['lgan_cov-EMD']:.6f}\n")
            f.write(f"1NN-Accuracy | CD: {results['1-NN-CD-acc']:.6f} | EMD: {results['1-NN-EMD-acc']:.6f}\n")
            f.write(f"Jensen-Shannon Divergence (JSD): {results['jsd']:.6f}\n")
        
        logger.info(f"结果已保存到: {result_file}")
        
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 