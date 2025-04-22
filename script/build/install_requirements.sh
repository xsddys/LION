#!/bin/bash

echo "开始安装LION项目依赖..."

# 安装系统级依赖
echo "安装系统级依赖..."
apt-get update
apt-get install -y \
    build-essential \
    ninja-build \
    gcc \
    g++ \
    cmake \
    git

# 创建新的conda环境
conda create -n lion_env python=3.8 -y

# 激活环境
source activate lion_env

# 降级setuptools以解决兼容性问题
pip install setuptools==57.5.0

# 安装基础包
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit=11.1 -c pytorch -c nvidia -y

# 安装ninja
pip install ninja

# 安装必要的pip包
pip install --no-cache-dir \
    diffusers==0.11.1 \
    huggingface-hub==0.11.1 \
    clip \
    einops==0.4.0 \
    pytorch3d==0.3.0 \
    open3d==0.15.2 \
    trimesh==3.10.1 \
    pytorch-lightning==1.5.1 \
    wandb==0.12.10 \
    matplotlib==3.5.1 \
    scipy==1.8.0 \
    scikit-learn==1.0.2 \
    scikit-image==0.19.1 \
    opencv-python==4.5.5.64 \
    kornia==0.6.6 \
    omegaconf==2.2.2 \
    yacs==0.1.8 \
    about-time==3.1.1 \
    addict==2.4.0 \
    aiohttp==3.10.11 \
    antlr4-python3-runtime==4.9.3 \
    deprecation==2.1.0 \
    docker-pycreds==0.4.0 \
    fvcore==0.1.5.post20221221 \
    gitdb==4.0.12 \
    gitpython==3.1.44 \
    imageio==2.35.1 \
    iopath==0.1.10 \
    joblib==1.4.2 \
    loguru==0.7.3 \
    networkx==3.1 \
    pandas==2.0.3 \
    pathtools==0.1.0 \
    portalocker==3.0.0 \
    promise==2.3 \
    pydeprecate==0.3.1 \
    pyquaternion==0.9.9 \
    pywavelets==1.4.1 \
    regex==2024.11.6 \
    sentry-sdk==2.25.1 \
    shortuuid==1.0.13 \
    smmap==5.0.2 \
    tabulate==0.9.0 \
    termcolor==2.4.0 \
    threadpoolctl==3.5.0 \
    tifffile==2023.7.10 \
    tomlkit==0.13.2 \
    torchmetrics==1.5.2 \
    yaspin==2.5.0

# 尝试安装calmsize
pip install --no-deps calmsize==0.1.3
if [ $? -ne 0 ]; then
    echo "尝试使用源码安装calmsize..."
    pip install git+https://github.com/javadba/calmsize.git
fi

# 检查CUDA是否可用
python -c "import torch; print('CUDA是否可用:', torch.cuda.is_available())"

echo "依赖安装完成！"
echo "请使用 'conda activate lion_env' 激活环境" 