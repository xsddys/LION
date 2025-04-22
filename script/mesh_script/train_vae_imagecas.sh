if [ -z "$1" ]
    then
    echo "Require NGPU input; "
    exit
fi

# 确保数据目录存在
DATA_DIR="./data/MESH"
if [ ! -d "$DATA_DIR" ]; then
    echo "错误：数据目录 $DATA_DIR 不存在！"
    exit 1
fi

if [ ! -d "$DATA_DIR/imagecas" ]; then
    echo "错误：数据类别目录 $DATA_DIR/imagecas 不存在！"
    exit 1
fi


# 创建输出目录
OUTPUT_DIR="/data/intern1_siqichen/lion_exp"
mkdir -p $OUTPUT_DIR

# 确保基本路径正确设置
DATE=$(date +%m%d)
EXP_NAME="imagecas_vae"
EXP_PATH="$OUTPUT_DIR/$DATE/imagecas/${EXP_NAME}"
mkdir -p "$EXP_PATH"

DATA=" ddpm.input_dim 3 data.cates imagecas data.dataset_type mesh data.data_dir $DATA_DIR data.normalize_global True data.recenter_per_shape False data.normalize_per_shape False"
NGPU=$1 # 
num_node=1
BS=16
total_bs=$(( $NGPU * $BS ))
if (( $total_bs > 128 )); then 
    echo "[WARNING] total batch_size larger than 128 may lead to unstable training, please reduce the size"
    exit
fi

ENT="python train_dist.py --num_process_per_node $NGPU "
kl=0.5  
lr=1e-3
latent=1
skip_weight=0.01 
sigma_offset=6.0
loss='l1_sum'

# 明确指定实验根目录
$ENT --exp_root $OUTPUT_DIR ddpm.num_steps 1 ddpm.ema 0 \
    exp_name ${DATE}/imagecas/${EXP_NAME} \
    log_dir ${EXP_PATH} \
    save_dir ${EXP_PATH} \
    log_name ${EXP_PATH} \
    trainer.opt.vae_lr_warmup_epochs 0 \
    latent_pts.ada_mlp_init_scale 0.1 \
    sde.kl_const_coeff_vada 1e-7 \
    trainer.anneal_kl 1 sde.kl_max_coeff_vada $kl \
    sde.kl_anneal_portion_vada 0.5 \
    shapelatent.log_sigma_offset $sigma_offset latent_pts.skip_weight $skip_weight \
    trainer.opt.beta2 0.99 \
    data.num_workers 2 \
    ddpm.loss_weight_emd 1.0 \
    trainer.epochs 3000 data.random_subsample 1 \
    viz.viz_freq -400 viz.log_freq -1 viz.val_freq 200 \
    data.batch_size $BS viz.save_freq 500 \
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
    $DATA viz.viz_order [2,0,1] data.recenter_per_shape False data.normalize_global True \
    data.nclass 1
