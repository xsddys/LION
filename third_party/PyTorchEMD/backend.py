import os
import time
import torch
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(os.path.join(_src_path, 'build_dynamic')):
    os.makedirs(os.path.join(_src_path, 'build_dynamic'))
    
# 设置CUDA架构标志
arch_flags = []
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    # 确保架构值在合理范围内
    if capability[0] >= 8:
        arch = "80"  # 使用较保守的架构版本
    else:
        arch = ''.join(map(str, capability))
    arch_flags = ['-arch=sm_' + arch]
    
tic = time.time() 
emd_cuda_dynamic = load(name='emd_ext',
                extra_cflags=['-O3', '-std=c++17'],
                extra_cuda_cflags=arch_flags,  # 添加CUDA架构标志
                ## build_directory=os.path.join(_src_path, 'build_dynamic'),
                verbose=True,
                sources=[
                    os.path.join(_src_path, f) for f in [
                        'cuda/emd.cpp',
                        'cuda/emd_kernel.cu',
                    ]
                ])
print('load emd_ext time: {:.3f}s'.format(time.time() - tic))
__all__ = ['emd_cuda_dynamic']
