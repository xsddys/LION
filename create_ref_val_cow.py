#!/usr/bin/env python
# 创建用于评估的参考点云数据集

import os
import torch
import numpy as np
from loguru import logger
import glob

# 设置数据目录路径
DATA_DIR = "./data/MESH/cow/test"
OUTPUT_PATH = "./datasets/test_data/ref_val_cow.pt"

def main():
    # 确保数据目录存在
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"找不到测试数据目录: {DATA_DIR}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # 获取所有NPY文件
    npy_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"在 {DATA_DIR} 中找不到NPY文件")
    
    logger.info(f"共找到 {len(npy_files)} 个点云文件用于参考数据集")
    
    # 读取点云数据
    ref_pcs = []
    for npy_file in npy_files:
        points = np.load(npy_file)
        ref_pcs.append(points[np.newaxis, ...])  # 添加批次维度
    
    # 合并点云数据
    ref_pcs = np.concatenate(ref_pcs, axis=0)
    logger.info(f"参考点云数据形状: {ref_pcs.shape}")
    
    # 计算全局均值和标准差
    all_points_mean = ref_pcs.reshape(-1, 3).mean(axis=0).reshape(1, 1, 3)
    all_points_std = ref_pcs.reshape(-1).std(axis=0).reshape(1, 1, 1)
    
    logger.info(f"全局均值: {all_points_mean.reshape(-1)}")
    logger.info(f"全局标准差: {all_points_std.reshape(-1)}")
    
    # 标准化点云数据
    ref_pcs = (ref_pcs - all_points_mean) / all_points_std
    
    # 转换为张量
    ref_pcs = torch.tensor(ref_pcs, dtype=torch.float32)
    all_points_mean = torch.tensor(all_points_mean, dtype=torch.float32)
    all_points_std = torch.tensor(all_points_std, dtype=torch.float32)
    
    # 保存参考数据集
    torch.save({
        'ref': ref_pcs,
        'mean': all_points_mean,
        'std': all_points_std
    }, OUTPUT_PATH)
    
    logger.info(f"参考数据集已保存到: {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 