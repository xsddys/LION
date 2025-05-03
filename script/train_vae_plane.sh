#!/bin/bash

# 检查参数
if [ -z "$1" ]; then
    echo "Require NGPU input; "
    exit
fi

# 第二个参数控制是否恢复训练（可选），默认为0（不恢复）
RESUME=${2:-0}

DATA_DIR="./data/ShapeNetCore.v2.PC15k"
DATA=" ddpm.input_dim 3 data.cates airplane data.dataset_type pointflow data.data_dir $DATA_DIR data.normalize_global True"
NGPU=$1 # 
num_node=1
BS=16 
total_bs=$(( $NGPU * $BS ))
if (( $total_bs > 128 )); then 
    echo "[WARNING] total batch_size larger than 128 may lead to unstable training, please reduce the size"
    exit
fi
# 创建输出目录
OUTPUT_DIR="/data/intern1_siqichen/lion_exp"
mkdir -p $OUTPUT_DIR

# 确保基本路径正确设置
DATE=$(date +%m%d)
EXP_NAME="airplane_vae"

# 如果不恢复训练，则添加时间戳确保路径唯一
if [ "$RESUME" -eq 0 ]; then
    TIMESTAMP=$(date +%H%M%S)
    EXP_NAME="${EXP_NAME}_${TIMESTAMP}"
    echo "开始新的训练，实验名称为: $EXP_NAME"
else
    echo "从已有检查点恢复训练，实验名称为: $EXP_NAME"
    # 这里可以设置特定的恢复路径，如果需要
    # PRETRAINED_PATH="$OUTPUT_DIR/具体路径/checkpoints/snapshot"
fi

EXP_PATH="$OUTPUT_DIR/$DATE/airplane/${EXP_NAME}"
mkdir -p "$EXP_PATH"

ENT="python train_dist.py --num_process_per_node $NGPU "
kl=0.5
lr=1e-3
latent=1
skip_weight=0.01 
sigma_offset=6.0
loss='l1_sum'

# 减少CPU核心使用量
NUM_WORKERS=2

# 根据是否恢复训练设置不同的命令
if [ "$RESUME" -eq 0 ]; then
    # 新的训练
    CMD="$ENT --exp_root $OUTPUT_DIR ddpm.num_steps 1 ddpm.ema 0"
else
    # 恢复训练
    # 如果有特定的路径则使用特定路径
    # CMD="$ENT --exp_root $OUTPUT_DIR --resume --pretrained $PRETRAINED_PATH ddpm.num_steps 1 ddpm.ema 0"
    # 否则使用默认路径
    CMD="$ENT --exp_root $OUTPUT_DIR --resume ddpm.num_steps 1 ddpm.ema 0"
fi

# 执行命令
$CMD \
    exp_name ${DATE}/airplane/${EXP_NAME} \
    log_dir ${EXP_PATH} \
    save_dir ${EXP_PATH} \
    log_name ${EXP_PATH} \
    ddpm.num_steps 1 ddpm.ema 0 \
    trainer.opt.vae_lr_warmup_epochs 0 \
    latent_pts.ada_mlp_init_scale 0.1 \
    sde.kl_const_coeff_vada 1e-7 \
    trainer.anneal_kl 1 sde.kl_max_coeff_vada $kl \
    sde.kl_anneal_portion_vada 0.5 \
    shapelatent.log_sigma_offset $sigma_offset latent_pts.skip_weight $skip_weight \
    trainer.opt.beta2 0.99 \
    data.num_workers $NUM_WORKERS \
    ddpm.loss_weight_emd 1.0 \
    trainer.epochs 8000 data.random_subsample 1 \
    viz.viz_freq -400 viz.log_freq -1 viz.val_freq 200 \
    data.batch_size $BS viz.save_freq 1000 \
    trainer.type 'trainers.hvae_trainer' \
    model_config default shapelatent.model 'models.vae_adain' \
    shapelatent.decoder_type 'models.latent_points_ada.LatentPointDecPVC' \
    shapelatent.encoder_type 'models.latent_points_ada.PointTransPVC' \
    latent_pts.style_encoder 'models.shapelatent_modules.PointNetPlusEncoder' \
    shapelatent.prior_type normal \
    shapelatent.latent_dim $latent trainer.opt.lr $lr \
    shapelatent.kl_weight ${kl} \
    shapelatent.decoder_num_points 2048 \
    data.tr_max_sample_points 2048 data.te_max_sample_points 2048 \
    ddpm.loss_type $loss cmt ${EXP_NAME} \
    $DATA viz.viz_order [2,0,1] data.recenter_per_shape False data.normalize_global True