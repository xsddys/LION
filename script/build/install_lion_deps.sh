#!/bin/bash

echo "开始安装LION项目依赖..."
echo "确保已经激活了lion38环境"

# 使用清华镜像源加速下载
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 错误处理函数
handle_error() {
    echo "安装过程中出现错误，请检查上面的错误信息。"
    exit 1
}

# 设置错误处理
set -e
trap handle_error ERR

# 降级setuptools以解决兼容性问题
echo "安装setuptools 57.5.0..."
pip install setuptools==57.5.0

# 分批安装基础依赖，避免一次安装太多包导致的问题
echo "安装第一批依赖..."
pip install --no-cache-dir \
    about-time==3.1.1 \
    absl-py==1.0.0 \
    addict==2.4.0 \
    aiohappyeyeballs==2.4.4 \
    aiohttp==3.10.11 \
    aiosignal==1.3.1 \
    antlr4-python3-runtime==4.9.3 \
    anyio==3.4.0 \
    argon2-cffi==21.1.0 \
    async-timeout==5.0.1 \
    attrs==21.2.0 \
    babel==2.9.1 \
    backcall==0.2.0 \
    bleach==4.1.0 \
    cachetools==4.2.4 

echo "安装第二批依赖..."
pip install --no-cache-dir \
    calmsize==0.1.3 \
    chardet==4.0.0 \
    click==8.1.8 \
    comet-ml==3.49.7 \
    configobj==5.0.9 \
    configparser==7.1.0 \
    cycler==0.11.0 \
    debugpy==1.5.1 \
    decorator==5.1.0 \
    defusedxml==0.7.1 \
    deprecation==2.1.0 \
    diffusers==0.11.1 \
    docker-pycreds==0.4.0 \
    dulwich==0.22.4 \
    einops==0.4.0 

echo "安装第三批依赖..."
pip install --no-cache-dir \
    entrypoints==0.3 \
    env-yaml==0.0.3 \
    everett==3.1.0 \
    filelock==3.16.1 \
    fonttools==4.28.2 \
    frozenlist==1.5.0 \
    fsspec==2025.3.0 \
    ftfy==6.2.3 \
    future==1.0.0 \
    fvcore==0.1.5.post20221221 \
    gitdb==4.0.12 \
    gitpython==3.1.44 \
    google-auth==2.3.3 \
    google-auth-oauthlib==0.4.6 \
    grpcio==1.42.0 

echo "安装第四批依赖..."
pip install --no-cache-dir \
    huggingface-hub==0.11.1 \
    imageio==2.35.1 \
    importlib-metadata==4.8.2 \
    importlib-resources==5.4.0 \
    iopath==0.1.10 \
    ipykernel==6.5.1 \
    ipython==7.29.0 \
    ipython-genutils==0.2.0 \
    ipywidgets==7.6.5 \
    jedi==0.18.1 \
    jinja2==3.0.3 \
    joblib==1.4.2 \
    json5==0.9.6 \
    jsonschema==4.2.1 \
    jupyter-client==7.1.0 

echo "安装第五批依赖..."
pip install --no-cache-dir \
    jupyter-core==4.9.1 \
    jupyter-packaging==0.12.3 \
    jupyter-server==1.12.0 \
    jupyterlab==3.2.4 \
    jupyterlab-language-pack-zh-cn==3.2.post2 \
    jupyterlab-pygments==0.1.2 \
    jupyterlab-server==2.8.2 \
    jupyterlab-widgets==1.0.2 \
    kiwisolver==1.3.2 \
    kornia==0.6.6 \
    lightning-utilities==0.11.9 \
    loguru==0.7.3 \
    markdown==3.3.6 \
    markdown-it-py==3.0.0 \
    markupsafe==2.0.1 

echo "安装第六批依赖..."
pip install --no-cache-dir \
    matplotlib==3.5.1 \
    matplotlib-inline==0.1.3 \
    mdurl==0.1.2 \
    mistune==0.8.4 \
    multidict==6.1.0 \
    nbclassic==0.3.4 \
    nbclient==0.5.9 \
    nbconvert==6.3.0 \
    nbformat==5.1.3 \
    nest-asyncio==1.5.1 \
    networkx==3.1 \
    ninja==1.11.1.4 \
    notebook==6.4.6 \
    numpy==1.21.4 \
    oauthlib==3.1.1 

echo "安装第七批依赖..."
pip install --no-cache-dir \
    omegaconf==2.2.2 \
    packaging==21.3 \
    pandas==2.0.3 \
    pandocfilters==1.5.0 \
    parso==0.8.2 \
    pathtools==0.1.2 \
    pexpect==4.8.0 \
    pickleshare==0.7.5 \
    pillow==8.4.0 \
    portalocker==3.0.0 \
    prometheus-client==0.12.0 \
    promise==2.3 \
    prompt-toolkit==3.0.22 \
    propcache==0.2.0 \
    protobuf==3.19.1 

echo "安装第八批依赖..."
pip install --no-cache-dir \
    psutil==7.0.0 \
    ptyprocess==0.7.0 \
    pyasn1==0.4.8 \
    pyasn1-modules==0.2.8 \
    pydeprecate==0.3.1 \
    pygments==2.19.1 \
    pyparsing==3.0.6 \
    pyquaternion==0.9.9 \
    pyrsistent==0.18.0 \
    python-box==6.1.0 \
    python-dateutil==2.8.2 \
    pytz==2021.3 \
    pywavelets==1.4.1 \
    pyyaml==6.0.2 \
    pyzmq==22.3.0 

echo "安装第九批依赖..."
pip install --no-cache-dir \
    regex==2024.11.6 \
    requests-oauthlib==1.3.0 \
    requests-toolbelt==1.0.0 \
    rich==14.0.0 \
    rsa==4.8 \
    scikit-image==0.19.1 \
    scikit-learn==1.0.2 \
    scipy==1.8.0 \
    semantic-version==2.10.0 \
    send2trash==1.8.0 \
    sentry-sdk==2.25.1 \
    setuptools-scm==6.3.2 \
    shortuuid==1.0.13 \
    simplejson==3.20.1 \
    smmap==5.0.2 

echo "安装第十批依赖..."
pip install --no-cache-dir \
    sniffio==1.2.0 \
    subprocess32==3.5.4 \
    supervisor==4.2.2 \
    tabulate==0.9.0 \
    tensorboard==2.7.0 \
    tensorboard-data-server==0.6.1 \
    tensorboard-plugin-wit==1.8.0 \
    termcolor==2.4.0 \
    terminado==0.12.1 \
    testpath==0.5.0 \
    threadpoolctl==3.5.0 \
    tifffile==2023.7.10 \
    tomli==1.2.2 \
    tomlkit==0.13.2 \
    torchmetrics==1.5.2 

echo "安装第十一批依赖..."
pip install --no-cache-dir \
    tornado==6.1 \
    traitlets==5.1.1 \
    typing-extensions==4.13.1 \
    tzdata==2025.2 \
    wandb==0.12.0 \
    wcwidth==0.2.13 \
    webencodings==0.5.1 \
    websocket-client==1.2.1 \
    werkzeug==2.0.2 \
    widgetsnbextension==3.5.2 \
    wrapt==1.17.2 \
    wurlitzer==3.1.1 \
    yacs==0.1.8 \
    yarl==1.15.2 \
    yaspin==2.5.0 \
    zipp==3.6.0

echo "安装较难安装的包..."
# 安装open3d和trimesh
pip install --no-cache-dir open3d==0.15.2
pip install --no-cache-dir trimesh==3.10.1

# 安装pytorch3d - 可能需要额外依赖项
echo "安装pytorch3d==0.3.0..."
pip install --no-cache-dir pytorch3d==0.3.0 || {
    echo "pytorch3d安装失败，尝试从源码安装..."
    pip install --no-cache-dir 'git+https://github.com/facebookresearch/pytorch3d.git@v0.3.0'
}

# 安装pytorch-lightning
echo "安装pytorch-lightning==1.5.1..."
pip install --no-cache-dir pytorch-lightning==1.5.1

# 安装opencv-python
echo "安装opencv-python==4.5.5.64..."
pip install --no-cache-dir opencv-python==4.5.5.64

# 安装CLIP (OpenAI CLIP模型)
echo "安装CLIP..."
pip install git+https://github.com/openai/CLIP.git

# 检查安装状态
echo "检查PyTorch安装:"
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA是否可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda if torch.cuda.is_available() else 'NA')"

echo "检查主要依赖项..."
python -c "
try:
    import numpy
    print('✓ numpy已安装')
except ImportError:
    print('✗ numpy安装失败')

try:
    import scipy
    print('✓ scipy已安装')
except ImportError:
    print('✗ scipy安装失败')

try:
    import matplotlib
    print('✓ matplotlib已安装')
except ImportError:
    print('✗ matplotlib安装失败')

try:
    import open3d
    print('✓ open3d已安装')
except ImportError:
    print('✗ open3d安装失败')

try:
    import pytorch3d
    print('✓ pytorch3d已安装')
except ImportError:
    print('✗ pytorch3d安装失败')

try:
    import trimesh
    print('✓ trimesh已安装')
except ImportError:
    print('✗ trimesh安装失败')

try:
    import diffusers
    print('✓ diffusers已安装')
except ImportError:
    print('✗ diffusers安装失败')
"

echo "依赖安装完成！请尝试运行项目。" 