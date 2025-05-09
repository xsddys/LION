#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import argparse
import numpy as np
from default_config import cfg
from loguru import logger
from trainers.hvae_trainer import Trainer
from utils.utils import Writer
from utils.vis_helper import visualize_point_clouds_3d
from utils.data_helper import normalize_point_clouds
from utils.eval_helper import compute_NLL_metric
from PIL import Image
import torchvision

def save_point_cloud_as_ply(points, filename, normals=None):
    """Save point cloud data as PLY file for visualization in MeshLab
    
    Args:
        points: Point cloud data with shape (N, 3)
        filename: Output filename
        normals: Point normals if available
    """
    try:
        import open3d as o3d
        # Create point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Add normals if available
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # Save as PLY file
        o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
        logger.info(f"Point cloud saved to: {filename}")
    except ImportError:
        logger.warning("Open3D not available. Saving PLY file using manual method.")
        # Manual PLY file creation
        with open(filename, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if normals is not None:
                f.write("property float nx\n")
                f.write("property float ny\n")
                f.write("property float nz\n")
            f.write("end_header\n")
            
            # Write vertices
            if normals is not None:
                for i in range(len(points)):
                    f.write(f"{points[i][0]} {points[i][1]} {points[i][2]} ")
                    f.write(f"{normals[i][0]} {normals[i][1]} {normals[i][2]}\n")
            else:
                for i in range(len(points)):
                    f.write(f"{points[i][0]} {points[i][1]} {points[i][2]}\n")
        
        logger.info(f"Point cloud saved to: {filename}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='VAE Model Evaluation Tool')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Model checkpoint path')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of evaluation samples')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Custom output directory (default: checkpoint_dir/eval)')
    parser.add_argument('--mode', type=str, choices=['recon', 'generate', 'both'], default='both',
                       help='Evaluation mode: reconstruction, generation or both')
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # Infer config file path from checkpoint path
    checkpoint_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(os.path.dirname(checkpoint_dir), 'cfg.yml')
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Load configuration
    cfg.merge_from_file(config_path)
    
    # Set distributed training parameters (single GPU mode)
    args.local_rank = 0
    args.global_rank = 0
    args.distributed = False
    args.global_size = 1
    
    # Create output directory
    eval_dir = args.output_dir if args.output_dir else os.path.join(checkpoint_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Initialize experiment writer
    # Create a Comet experiment instance if .comet_api file exists
    exp = None
    if os.path.exists('.comet_api'):
        try:
            from comet_ml import Experiment, OfflineExperiment
            import json
            comet_args = json.load(open('.comet_api', 'r'))
            exp = Experiment(display_summary_level=0, **comet_args)
            exp.set_name(f"eval_{os.path.basename(checkpoint_dir)}")
        except Exception as e:
            logger.warning(f"Failed to create Comet experiment: {e}")
            exp = None
    
    # Create writer with experiment
    writer = Writer(rank=0, save=eval_dir, exp=exp)
    
    # Initialize trainer and load model
    trainer = Trainer(cfg, args)
    trainer.set_writer(writer)
    
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    trainer.resume(args.checkpoint)
    trainer.model.eval()
    
    # Reconstruction evaluation
    if args.mode in ['recon', 'both']:
        logger.info("==== RECONSTRUCTION EVALUATION ====")
        
        # Evaluate reconstruction performance (metrics)
        logger.info("Computing reconstruction metrics...")
        results = trainer.eval_nll(trainer.step)
        
        # Visualize reconstruction results
        logger.info("Generating reconstruction visualizations...")
        
        # Standard visualization through trainer (may not work well)
        trainer.vis_sample(writer=trainer.writer, step=trainer.step)
        
        # Manually generate and save visualizations
        with torch.no_grad():
            # Get validation data
            val_x = trainer.val_x
            
            # Generate reconstructions
            output = trainer.model.recont(val_x)
            recon_x = output['final_pred']
            
            # Visualize original and reconstructions
            num_vis = min(10, val_x.shape[0])
            img_list = []
            
            recon_dir = os.path.join(eval_dir, 'reconstruction')
            os.makedirs(recon_dir, exist_ok=True)
            
            # Create new folder for reconstruction PLY files
            recon_ply_dir = os.path.join(eval_dir, 'reconstruction_ply')
            os.makedirs(recon_ply_dir, exist_ok=True)
            
            for i in range(num_vis):
                # Combined visualization (side by side)
                combined_points = normalize_point_clouds([val_x[i], recon_x[i]])
                combined_img = visualize_point_clouds_3d(combined_points, ['original', 'reconstruction'])
                
                # Save individual visualization
                vis_path = os.path.join(recon_dir, f'recon_sample_{i}.png')
                Image.fromarray(combined_img.transpose(1, 2, 0).astype(np.uint8)).save(vis_path)
                
                # Save original and reconstructed point clouds as PLY files
                original_points = combined_points[0].cpu().numpy()
                reconstructed_points = combined_points[1].cpu().numpy()
                
                # Save as PLY files
                original_ply_path = os.path.join(recon_ply_dir, f'original_sample_{i}.ply')
                recon_ply_path = os.path.join(recon_ply_dir, f'recon_sample_{i}.ply')
                
                save_point_cloud_as_ply(original_points, original_ply_path)
                save_point_cloud_as_ply(reconstructed_points, recon_ply_path)
                
                # Add to grid images
                img_list.append(torch.from_numpy(combined_img).float() / 255.0)
            
            # Create a grid of all visualizations
            grid = torchvision.utils.make_grid(img_list, nrow=2)
            grid_path = os.path.join(recon_dir, 'reconstruction_grid.png')
            torchvision.utils.save_image(grid, grid_path)
            
            # Check if latent points visualization is available
            if 'vis/latent_pts' in output:
                latent_dir = os.path.join(eval_dir, 'latent_space')
                os.makedirs(latent_dir, exist_ok=True)
                
                # Create new folder for latent space PLY files
                latent_ply_dir = os.path.join(eval_dir, 'latent_space_ply')
                os.makedirs(latent_ply_dir, exist_ok=True)
                
                latent_pts = output['vis/latent_pts']
                for i in range(num_vis):
                    points = latent_pts[i]
                    points = normalize_point_clouds([points])[0]
                    img = visualize_point_clouds_3d([points], [f'latent_{i}'])
                    
                    # Save image visualization
                    latent_path = os.path.join(latent_dir, f'latent_sample_{i}.png')
                    Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8)).save(latent_path)
                    
                    # Save as PLY file
                    latent_ply_path = os.path.join(latent_ply_dir, f'latent_sample_{i}.ply')
                    latent_points = points.cpu().numpy()
                    save_point_cloud_as_ply(latent_points, latent_ply_path)
    
    # Generation from latent space
    if args.mode in ['generate', 'both']:
        logger.info("==== GENERATION FROM LATENT SPACE ====")
        gen_dir = os.path.join(eval_dir, 'generation')
        os.makedirs(gen_dir, exist_ok=True)
        
        # Create new folder for generated point cloud PLY files
        gen_ply_dir = os.path.join(eval_dir, 'generation_ply')
        os.makedirs(gen_ply_dir, exist_ok=True)
        
        with torch.no_grad():
            # Get number of samples to generate
            num_gen_samples = args.num_samples
            
            # Call the model's sample method to generate new point clouds
            logger.info(f"Generating {num_gen_samples} new point cloud samples...")
            
            # Generate samples using the trainer's sample method 
            # B3N format for generated samples
            # 1. Call model's sample method to generate point clouds
            generated_samples = trainer.sample(num_shapes=num_gen_samples, 
                                              num_points=cfg.data.tr_max_sample_points,
                                              device_str=args.device)
            
            # 2. Format conversion (B3N -> BN3)
            # Convert from B3N to BN3 format for visualization
            generated_samples = generated_samples.permute(0, 2, 1).contiguous()
            
            # Visualize generated samples
            gen_img_list = []
            
            for i in range(num_gen_samples):
                # Process and visualize generated point cloud
                points = normalize_point_clouds([generated_samples[i]])[0]
                img = visualize_point_clouds_3d([points], [f'generated_{i}'])
                
                # Save individual visualization
                gen_path = os.path.join(gen_dir, f'generated_sample_{i}.png')
                Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8)).save(gen_path)
                
                # Save as PLY file
                gen_ply_path = os.path.join(gen_ply_dir, f'generated_sample_{i}.ply')
                gen_points = points.cpu().numpy()
                save_point_cloud_as_ply(gen_points, gen_ply_path)
                
                # Save raw point cloud data
                pc_path = os.path.join(gen_dir, f'generated_sample_{i}.pt')
                torch.save(generated_samples[i].cpu(), pc_path)
                
                # Add to grid images
                gen_img_list.append(torch.from_numpy(img).float() / 255.0)
            
            # Create a grid of all generated samples
            gen_grid = torchvision.utils.make_grid(gen_img_list, nrow=4)
            gen_grid_path = os.path.join(gen_dir, 'generation_grid.png')
            torchvision.utils.save_image(gen_grid, gen_grid_path)
            
            # Save all generated samples in a single file
            all_samples_path = os.path.join(gen_dir, 'all_generated_samples.pt')
            torch.save(generated_samples.cpu(), all_samples_path)
            
            # Additionally save all generated samples in a single combined PLY file (optional, for overall visualization)
            all_samples_ply_path = os.path.join(gen_ply_dir, 'all_generated_samples.ply')
            all_points = []
            for i in range(num_gen_samples):
                # Normalize each point cloud and add offset to separate them in space
                sample_points = normalize_point_clouds([generated_samples[i]])[0].cpu().numpy()
                # Add offset to separate them in 3D space
                offset = np.array([i * 2.5, 0, 0])  # Separate along x-axis
                all_points.append(sample_points + offset)
            
            # If too many generated samples, just use the first 10 for the combined PLY
            if num_gen_samples > 10:
                all_points = all_points[:10]
            
            if len(all_points) > 0:
                combined_points = np.vstack(all_points)
                save_point_cloud_as_ply(combined_points, all_samples_ply_path)
    
    logger.info(f"Evaluation complete! Results saved in: {eval_dir}")
    if args.mode in ['recon', 'both']:
        logger.info(f"Reconstruction results in: {os.path.join(eval_dir, 'reconstruction')}")
        logger.info(f"Reconstruction PLY files in: {os.path.join(eval_dir, 'reconstruction_ply')}")
    if args.mode in ['generate', 'both']:
        logger.info(f"Generated samples in: {os.path.join(eval_dir, 'generation')}")
        logger.info(f"Generated PLY files in: {os.path.join(eval_dir, 'generation_ply')}")

if __name__ == '__main__':
    main()