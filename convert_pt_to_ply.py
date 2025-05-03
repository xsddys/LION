#!/usr/bin/env python
# 将点云.pt文件转换为.ply格式

import os
import torch
import numpy as np
import argparse
from loguru import logger
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud(points, output_path=None):
    """可视化点云并保存图像"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置一致的视角和范围
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def save_point_cloud_as_ply(points, output_path):
    """将点云保存为.ply文件"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(output_path, pcd)
    return output_path

def process_pt_file(pt_file_path, output_dir, start_idx=0, end_idx=None, visualize=False):
    """处理.pt文件并转换为.ply格式"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 图像输出目录
    if visualize:
        img_dir = os.path.join(output_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
    
    # 读取.pt文件
    logger.info(f"读取点云文件: {pt_file_path}")
    try:
        # 尝试直接加载为张量
        point_clouds = torch.load(pt_file_path)
        
        # 检查是否是字典格式
        if isinstance(point_clouds, dict):
            if 'ref' in point_clouds:
                point_clouds = point_clouds['ref']
            else:
                logger.warning(f"未找到点云数据，已知键: {list(point_clouds.keys())}")
                return
        
        # 转换为numpy数组
        if isinstance(point_clouds, torch.Tensor):
            point_clouds = point_clouds.detach().cpu().numpy()
        
        # 检查维度是否正确
        if len(point_clouds.shape) != 3 or point_clouds.shape[2] < 3:
            logger.error(f"点云数据格式错误: {point_clouds.shape}，预期为(B, N, 3+)")
            return
        
        # 获取点云数量
        num_point_clouds = point_clouds.shape[0]
        logger.info(f"共有 {num_point_clouds} 个点云")
        
        # 设置处理范围
        if end_idx is None:
            end_idx = num_point_clouds
        
        start_idx = max(0, min(start_idx, num_point_clouds-1))
        end_idx = max(start_idx+1, min(end_idx, num_point_clouds))
        
        logger.info(f"处理点云 {start_idx} 到 {end_idx-1}")
        
        # 处理每个点云
        for i in range(start_idx, end_idx):
            # 获取当前点云
            points = point_clouds[i]
            
            # 只使用XYZ坐标
            if points.shape[1] > 3:
                points = points[:, :3]
            
            # 检查点云是否为空
            if np.isnan(points).any() or len(points) == 0:
                logger.warning(f"点云 {i} 包含NaN值或为空，跳过")
                continue
            
            # 保存为PLY文件
            ply_path = os.path.join(output_dir, f"point_cloud_{i:04d}.ply")
            save_point_cloud_as_ply(points, ply_path)
            logger.info(f"已保存: {ply_path}")
            
            # 可视化点云
            if visualize:
                img_path = os.path.join(img_dir, f"point_cloud_{i:04d}.png")
                visualize_point_cloud(points, img_path)
                logger.info(f"已生成可视化图像: {img_path}")
        
        # 还创建一个合并的点云文件(最多取前10个点云)
        if end_idx - start_idx > 1:
            num_to_combine = min(10, end_idx - start_idx)
            combined_points = []
            offset = 0
            offset_step = 2.0  # 每个点云之间的间隔
            
            for i in range(start_idx, start_idx + num_to_combine):
                pc = point_clouds[i].copy()
                pc[:, 0] += offset  # 在X轴上偏移
                combined_points.append(pc)
                offset += offset_step
            
            combined_points = np.concatenate(combined_points, axis=0)
            combined_ply_path = os.path.join(output_dir, "combined_point_clouds.ply")
            save_point_cloud_as_ply(combined_points, combined_ply_path)
            logger.info(f"已保存合并点云: {combined_ply_path}")
            
            if visualize:
                combined_img_path = os.path.join(img_dir, "combined_point_clouds.png")
                visualize_point_cloud(combined_points, combined_img_path)
                logger.info(f"已生成合并点云可视化图像: {combined_img_path}")
        
        logger.info("转换完成!")
        
    except Exception as e:
        logger.error(f"处理文件时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="将点云.pt文件转换为.ply格式")
    parser.add_argument("--input", type=str, required=True, 
                        help="输入的.pt文件路径")
    parser.add_argument("--output_dir", type=str, default="./ply_point_clouds",
                        help="输出.ply文件的目录")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="起始点云索引")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="结束点云索引")
    parser.add_argument("--visualize", action="store_true",
                        help="是否生成点云可视化图像")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return
    
    # 处理文件
    process_pt_file(
        args.input, 
        args.output_dir, 
        args.start_idx, 
        args.end_idx,
        args.visualize
    )

if __name__ == "__main__":
    main() 