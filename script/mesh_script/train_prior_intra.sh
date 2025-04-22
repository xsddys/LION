if [ -z "$1" ]
    then
    echo "Require NGPU input; "
    exit
fi
loss="mse_sum"
NGPU=$1 ## 1 #8
num_node=1
mem=32
BS=8 
lr=2e-4
ENT="python train_dist.py --num_process_per_node $NGPU "
train_vae=False
#cmt="lion"
#ckpt="./lion_ckpt/unconditional/car/checkpoints/vae_only.pt"
cmt="intra_prior"
ckpt="/data/intern1_siqichen/lion_exp/0411/intra/vae_intra/checkpoints/vae_only.pt"
# 创建输出目录
OUTPUT_DIR="/data/intern1_siqichen/lion_exp"
mkdir -p $OUTPUT_DIR

$ENT \
    --config "./config/intra_prior_cfg.yml" \
    latent_pts.pvd_mse_loss 1 \
    vis_latent_point 1 \
    num_val_samples 16 \
    ddpm.ema 1 \
    ddpm.use_bn False ddpm.use_gn True \
    ddpm.time_dim 64 \
    ddpm.beta_T 0.02 \
    sde.vae_checkpoint $ckpt \
    sde.learning_rate_dae $lr sde.learning_rate_min_dae $lr \
    trainer.epochs 5000 \
    sde.num_channels_dae 1024 \
    sde.dropout 0.3 \
    latent_pts.style_prior 'models.score_sde.resnet.PriorSEDrop' \
    sde.prior_model 'models.latent_points_ada_localprior.PVCNN2Prior' \
    sde.train_vae $train_vae \
    sde.embedding_scale 1.0 \
    viz.save_freq 500 \
    viz.viz_freq -200 viz.log_freq -1 viz.val_freq -2000 \
    data.batch_size $BS \
    trainer.type 'trainers.train_2prior' \
    cmt $cmt \
    data.cates intra data.dataset_type mesh data.data_dir ./data/MESH \
    data.nclass 1 \
    data.tr_max_sample_points 2048 data.te_max_sample_points 2048 \
    log_dir $OUTPUT_DIR/intra/intra_prior \
    save_dir $OUTPUT_DIR/intra/intra_prior
