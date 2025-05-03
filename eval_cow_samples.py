#!/usr/bin/env python
# 评估生成的点云样本与参考点云数据集

import os
import sys
sys.path.append('.')
import torch
from loguru import logger
from utils.eval_helper import compute_score

def main():
    # 设置生成的样本和参考数据集路径
    samples_path = "/data/intern1_siqichen/lion_exp/0426/cow/cow_prior/eval/samples_71999s1H994c9diet.pt"
    ref_path = "./datasets/test_data/ref_val_cow.pt"
    
    # 检查文件是否存在
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"找不到生成的样本文件: {samples_path}")
    
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"找不到参考点云数据集: {ref_path}")
    
    logger.info(f"开始评估生成的样本: {samples_path}")
    logger.info(f"参考数据集: {ref_path}")
    
    # 使用compute_score函数评估生成的样本
    # norm_box=False表示不进行归一化框处理，对于cow类别可能适合这种设置
    # 如果需要归一化框处理，请将norm_box设置为True
    results = compute_score(
        output_name=samples_path,
        ref_name=ref_path,
        batch_size_test=10,  # 根据您的GPU内存调整批次大小
        device_str='cuda' if torch.cuda.is_available() else 'cpu',
        accelerated_cd=True,
        norm_box=False,  # 根据需要调整
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

if __name__ == "__main__":
    main() 