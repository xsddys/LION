#!/bin/bash
#SBATCH -J cow_vae       # 设置作业名为cow_vae
#SBATCH -N 1              # 使用1个计算节点
#SBATCH -c 2              # 每个任务分配2个CPU核心(从4减少到2)
#SBATCH --gres=gpu:1      # 每个节点分配1个GPU
#SBATCH -w inspur0        # 指定在inspur0节点上运行
#SBATCH -o logs/vae_%j.out     # 标准输出文件，%j代表作业ID
#SBATCH -e logs/vae_%j.err     # 标准错误文件

# 确保日志目录存在
mkdir -p logs

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1

# 设置使用的GPU数量为1
NUM_GPUS=1
# 是否从检查点恢复训练 (0:不恢复, 1:恢复)
RESUME=0

# 打印一些信息，方便调试
echo "作业ID: $SLURM_JOB_ID 运行在节点 $SLURM_NODELIST"
echo "运行时间: $(date)"
echo "使用数据集: mesh, 类别: cow, GPU数量: $NUM_GPUS"
echo "是否恢复训练: $RESUME (0:不恢复, 1:恢复)"
echo "分配CPU核心数: 2 (SBATCH设置)"
echo "GPU信息:"
nvidia-smi

# 运行训练脚本，确保传递参数
echo "运行cow数据集训练脚本..."
bash script/mesh_script/train_vae_cow.sh $NUM_GPUS $RESUME

# 打印结束信息
echo "训练完成时间: $(date)" 