#!/usr/bin/env python
# 创建一个自定义数据集，将PLY格式点云转换为LION训练所需的NPY格式

import os
import numpy as np
import open3d as o3d
import argparse
import glob
from tqdm import tqdm
import shutil


def standardize_bbox(pcl):
    """将点云标准化到[-0.5, 0.5]范围内"""
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result


def process_ply_file(ply_file, output_dir, n_points=10000):
    """处理单个PLY文件，转换为NPY格式"""
    # 读取PLY文件
    print(f"处理文件: {ply_file}")
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)
    
    # 如果点数过多则采样
    if len(points) > n_points:
        indices = np.random.choice(len(points), n_points, replace=False)
        points = points[indices]
    
    # 如果点数不足，则复制点到达到所需数量
    elif len(points) < n_points:
        # 计算需要复制的次数
        repeats = n_points // len(points) + 1
        repeated_points = np.repeat(points, repeats, axis=0)
        # 然后随机选择需要的点数
        indices = np.random.choice(len(repeated_points), n_points, replace=False)
        points = repeated_points[indices]
    
    # 标准化点云
    points = standardize_bbox(points)
    
    # 获取文件名（不含扩展名）作为保存的NPY文件名
    filename = os.path.splitext(os.path.basename(ply_file))[0]
    output_file = os.path.join(output_dir, f"{filename}.npy")
    
    # 保存为NPY格式
    np.save(output_file, points)
    print(f"已保存到: {output_file}")
    
    return filename


def create_dataset_structure(source_dir, output_base_dir, category_name, n_points=10000, train_ratio=0.8, val_ratio=0.1):
    """创建符合ShapeNet格式的数据集结构"""
    
    # 创建目录结构
    synset_id = f"cow_{category_name}"  # 可以自定义一个synset_id
    
    train_dir = os.path.join(output_base_dir, synset_id, "train")
    val_dir = os.path.join(output_base_dir, synset_id, "val")
    test_dir = os.path.join(output_base_dir, synset_id, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 获取所有PLY文件
    ply_files = glob.glob(os.path.join(source_dir, "*.ply"))
    print(f"找到{len(ply_files)}个PLY文件")
    
    # 创建临时目录存储处理后的文件
    temp_dir = os.path.join(output_base_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 处理所有PLY文件
    processed_files = []
    for ply_file in tqdm(ply_files):
        filename = process_ply_file(ply_file, temp_dir, n_points)
        processed_files.append(filename)
    
    # 随机分配到训练、验证和测试集
    np.random.shuffle(processed_files)
    n_files = len(processed_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
    train_files = processed_files[:n_train]
    val_files = processed_files[n_train:n_train+n_val]
    test_files = processed_files[n_train+n_val:]
    
    # 移动文件到对应目录
    for filename in train_files:
        src = os.path.join(temp_dir, f"{filename}.npy")
        dst = os.path.join(train_dir, f"{filename}.npy")
        shutil.copy(src, dst)
    
    for filename in val_files:
        src = os.path.join(temp_dir, f"{filename}.npy")
        dst = os.path.join(val_dir, f"{filename}.npy")
        shutil.copy(src, dst)
    
    for filename in test_files:
        src = os.path.join(temp_dir, f"{filename}.npy")
        dst = os.path.join(test_dir, f"{filename}.npy")
        shutil.copy(src, dst)
    
    # 删除临时目录
    shutil.rmtree(temp_dir)
    
    print(f"数据集创建完成！")
    print(f"训练集: {len(train_files)}个文件")
    print(f"验证集: {len(val_files)}个文件")
    print(f"测试集: {len(test_files)}个文件")


def update_data_path(synset_id):
    """更新data_path.py文件以包含新数据集的路径"""
    data_path_file = "datasets/data_path.py"
    
    with open(data_path_file, 'r') as f:
        content = f.read()
    
    # 检查是否已经有指向该目录的路径
    if f"'./data/ShapeNetCore.v2.PC15k/'" in content:
        # 替换为更新后的路径列表
        new_content = content.replace(
            "'./data/ShapeNetCore.v2.PC15k/'", 
            "'./data/ShapeNetCore.v2.PC15k/',\n        './data/CustomShapeNet/'"
        )
        
        with open(data_path_file, 'w') as f:
            f.write(new_content)
        
        print(f"已更新 {data_path_file} 以包含新数据集的路径")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将PLY格式点云转换为LION训练所需的NPY格式")
    parser.add_argument("--source_dir", type=str, required=True, help="源PLY文件目录")
    parser.add_argument("--output_dir", type=str, default="./data/CustomShapeNet", help="输出目录")
    parser.add_argument("--category", type=str, default="custom", help="类别名称")
    parser.add_argument("--n_points", type=int, default=10000, help="每个点云的点数")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    
    args = parser.parse_args()
    
    create_dataset_structure(
        args.source_dir, 
        args.output_dir, 
        args.category, 
        args.n_points, 
        args.train_ratio, 
        args.val_ratio
    )
    
    # 更新数据路径
    synset_id = f"cow_{args.category}"
    update_data_path(synset_id)
    
    print("\n使用方法:")
    print(f"1. 将生成的数据放在 {args.output_dir} 目录下")
    print(f"2. 在训练LION模型时，使用以下参数:")
    print(f"   --config configs/text2shape/custom.yml")
    print(f"   cfg.data.shapenet_id={synset_id}") 