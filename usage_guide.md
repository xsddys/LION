# VAE训练任务使用指南

## Slurm作业调度系统简介

Slurm是一个用于Linux集群的作业调度系统，用于管理计算资源和任务调度。主要命令包括：

- `sbatch`：提交批处理作业
- `squeue`：查看作业队列
- `scancel`：取消作业
- `sinfo`：查看集群节点状态

## CPU和GPU资源管理

### 控制CPU核心使用量

训练脚本中有两处会影响CPU使用量：

1. **Slurm提交脚本中的设置**:
```bash
#SBATCH -c 2  # 每个任务分配2个CPU核心
```
这个设置是Slurm的请求，但并不会强制限制您的任务使用的CPU数量。

2. **数据加载的工作线程**:
在训练脚本中的参数：
```bash
data.num_workers 2  # 数据加载使用的线程数
```
这个参数直接控制了PyTorch `DataLoader`使用的工作线程数，是实际占用CPU核心的主要因素。

### 查看CPU使用情况

您可以使用以下命令查看作业的资源使用情况：

1. **使用`sstat`命令查看正在运行的作业资源使用**:
```bash
sstat --format=AveCPU,MaxRSS,MaxDiskRead,MaxDiskWrite,NTasks -j <作业ID>
```

2. **使用`sacct`命令查看已完成作业的资源使用**:
```bash
sacct -j <作业ID> --format=JobID,JobName,MaxRSS,MaxVMSize,NNodes,NTasks,CPUTime,TotalCPU,Elapsed
```

3. **使用`top`命令实时监控**:
```bash
top -u <用户名>
```

4. **使用`htop`查看更友好的界面**（如果已安装）:
```bash
htop -u <用户名>
```

### 优化CPU使用

如果需要减少CPU使用量，您可以：

1. 降低`data.num_workers`参数值（如从4降到2或1）
2. 在提交脚本中减少`-c`参数请求的CPU核心数
3. 避免在训练脚本中使用多进程或多线程处理

## 如何使用脚本提交VAE训练任务

### 1. 可用的提交脚本

我们提供了以下几个Slurm脚本来训练VAE模型：

- `submit_vae.sh`：标准训练脚本，使用默认参数
- `mesh_submit_vae.sh`：专门用于mesh数据集的训练脚本
- `submit_vae_low_kl.sh`：优化后的训练脚本，降低了KL权重，调整了学习率

### 2. 提交作业

```bash
# 提交标准训练作业
sbatch submit_vae.sh

# 提交mesh数据集训练（默认使用cow类别）
sbatch mesh_submit_vae.sh

# 提交mesh数据集训练并指定类别
sbatch mesh_submit_vae.sh intra
sbatch mesh_submit_vae.sh march
sbatch mesh_submit_vae.sh imagecas

# 提交优化参数的训练作业
sbatch submit_vae_low_kl.sh
```

### 3. 控制是否从检查点恢复训练

现在，训练脚本支持控制是否从检查点恢复训练。您可以通过修改提交脚本中的`RESUME`参数来控制：

```bash
# 在submit_vae.sh中修改此参数
# 设置为0表示不恢复训练（每次使用新的输出路径）
# 设置为1表示从已有检查点恢复训练
RESUME=0  # 默认不恢复
```

也可以直接调用训练脚本，并传入恢复参数：

```bash
# 不恢复训练（新开始）
bash script/train_vae.sh 1 0

# 从检查点恢复训练
bash script/train_vae.sh 1 1

# mesh数据集脚本同理
bash script/mesh_script/train_vae_cow.sh 1 0  # 不恢复
bash script/mesh_script/train_vae_cow.sh 1 1  # 恢复
```

当`RESUME=0`时，脚本会自动在实验名称后添加时间戳，确保每次训练都使用新的输出路径，不会加载之前的检查点。

### 4. 修改训练参数

您可以通过编辑脚本文件来修改训练参数：

```bash
# 编辑标准训练脚本
nano submit_vae.sh

# 编辑mesh专用训练脚本
nano mesh_submit_vae.sh

# 编辑优化参数的训练脚本
nano submit_vae_low_kl.sh
```

主要参数包括：
- `DATASET_TYPE`：数据集类型，可选值：pointflow，mesh
- `CATEGORY`：数据集类别，例如：car, cow, intra, march, imagecas
- `NUM_GPUS`：使用的GPU数量，通常设置为1
- `RESUME`：是否从检查点恢复训练（0:不恢复，1:恢复）
- `NUM_WORKERS`：数据加载使用的线程数，影响CPU使用量
- `KL_WEIGHT`：KL散度权重（在优化脚本中）
- `LEARNING_RATE`：学习率（在优化脚本中）
- `EPOCHS`：训练轮数（在优化脚本中）

### 5. 查看作业状态

```bash
# 查看队列中的所有作业
squeue

# 查看特定用户的作业
squeue -u <用户名>
```

### 6. 查看作业输出

所有作业的输出都保存在logs目录下：

```bash
# 查看标准输出
cat logs/vae_<作业ID>.out

# 查看标准错误
cat logs/vae_<作业ID>.err

# 查看mesh训练的输出
cat logs/mesh_vae_<作业ID>.out
```

### 7. 取消作业

```bash
# 取消指定作业ID的作业
scancel <作业ID>

# 取消用户的所有作业
scancel -u <用户名>
```

## 常见错误与解决方案

### 1. "Require NGPU input;"错误

如果脚本报错"Require NGPU input;"，这表示训练脚本没有接收到GPU数量参数。

解决方案：
- 确保提交脚本中设置了`NUM_GPUS`变量（默认为1）
- 确保在调用训练脚本时传递了此参数，例如：`bash script/train_vae.sh $NUM_GPUS`

### 2. 数据目录不存在

如果报错"错误：数据目录 $DATA_DIR 不存在！"，表示找不到数据目录。

解决方案：
- 确认数据目录的路径是否正确
- 检查`./data/MESH/`或`./data/ShapeNetCore.v2.PC15k/`目录是否存在
- 确保数据类别目录（如`cow`、`intra`等）存在

### 3. 无法获取GPU资源

如果作业排队很长时间不执行，可能是因为没有足够的GPU资源。

解决方案：
- 使用`sinfo`命令查看节点状态
- 查看`nvidia-smi`命令输出，检查GPU使用情况
- 尝试在不同节点上运行，修改脚本中的`#SBATCH -w <节点名>`参数

### 4. 自动从检查点恢复

如果您发现训练自动从之前的检查点恢复，但您希望开始新的训练：

解决方案：
- 确保提交脚本中设置`RESUME=0`
- 脚本将自动在实验名称后添加时间戳，创建新的输出路径
- 如果您仍然希望使用同一个实验名称，可以手动删除之前的输出目录，或修改`EXP_NAME`

### 5. CPU资源使用过多

如果服务器管理员提示您使用了过多的CPU资源：

解决方案：
- 修改`submit_vae.sh`中的`#SBATCH -c`参数，降低请求的CPU核心数
- 更重要的是，降低训练脚本中的`data.num_workers`参数值
- 检查并确保没有在代码中启动过多的后台进程

## VAE训练中的常见问题

### 问题：Loss先减小然后大幅增加

如您在先前训练中观察到的，Loss先减小然后大幅增加到10000+，这通常是由以下原因导致的：

1. **KL散度权重过大**：默认的KL权重(0.5)可能导致模型过度关注使隐空间分布接近正态分布，而牺牲了重构质量。
   - 解决方案：使用`submit_vae_low_kl.sh`脚本，它将KL权重降低到0.01。

2. **学习率不合适**：学习率过高可能导致模型在后期训练中发散。
   - 解决方案：优化脚本中已将学习率从1e-3降低到5e-4。

3. **过拟合**：长时间训练可能导致模型过拟合。
   - 解决方案：优化脚本将训练轮数从3000减少到1000，并增加了更频繁的保存检查点。

### 关于VAE的Loss解释

VAE的损失函数由两部分组成：
1. **重构损失**：衡量通过编码器-解码器重建输入数据的质量
2. **KL散度损失**：促使隐空间分布接近标准正态分布

总损失 = 重构损失 + KL散度权重 × KL散度损失

在训练过程中，过高的KL权重可能导致重构质量下降，而过低的KL权重可能导致隐空间分布不理想。需要在两者之间找到平衡点。

## 最佳实践建议

1. **多保存检查点**：确保在训练过程中保存多个检查点，而不仅仅是最终模型。

2. **监控验证Loss**：观察验证集上的Loss变化，在Loss开始上升时考虑停止训练。

3. **KL权重调整**：尝试不同的KL权重（例如：0.5, 0.1, 0.01, 0.001）找到最适合您数据的值。

4. **提前停止**：如果发现Loss开始增加，可以提前停止训练并使用最佳检查点。

5. **合理选择GPU**：根据`nvidia-smi`的输出，选择负载较轻的GPU进行训练。您可以修改脚本中的`CUDA_VISIBLE_DEVICES`参数，例如设置为`export CUDA_VISIBLE_DEVICES=0`表示使用第一块GPU。

6. **确保每次训练的独立性**：当不想从检查点恢复时，设置`RESUME=0`确保每次训练都是全新开始。

7. **控制CPU使用量**：
   - 减少`data.num_workers`参数值
   - 降低`#SBATCH -c`参数请求的CPU核心数
   - 监控CPU使用情况，避免长时间占用过多资源 