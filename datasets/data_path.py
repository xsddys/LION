# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import os


def get_path(dataname=None):
    dataset_path = {}
    dataset_path['pointflow'] = [
        './data/ShapeNetCore.v2.PC15k/',
        './datasets/NPY/'
    ]
    dataset_path['clip_forge_image'] = [
            './data/shapenet_render/'
            ]
    dataset_path['mesh'] = [
            './data/MESH/'
            ]

    if dataname is None:
        return dataset_path
    else:
        assert(
            dataname in dataset_path), f'not found {dataname}, only: {list(dataset_path.keys())}'
        for p in dataset_path[dataname]:
            print(f'searching: {dataname}, get: {p}')
            if os.path.exists(p):
                print(f'[数据路径] 成功找到{dataname}数据集路径: {p}')
                return p
        # 如果没有找到有效路径，直接返回第一个路径，避免返回None
        default_path = dataset_path[dataname][0]
        print(f"[警告] 未找到可用的{dataname}数据路径，使用默认路径: {default_path}")
        # 检查默认路径是否存在，如果不存在则尝试创建
        if not os.path.exists(default_path):
            try:
                os.makedirs(default_path)
                print(f"[信息] 已创建默认数据路径: {default_path}")
            except Exception as e:
                print(f"[错误] 创建默认路径失败: {str(e)}")
        return default_path


def get_cache_path():
    cache_list = ['/workspace/data_cache_local/data_stat/',
                  '/workspace/data_cache/data_stat/']
    for p in cache_list:
        if os.path.exists(p):
            return p
    ValueError(
        f'all path not found for {cache_list}, please double check: or edit the datasets/data_path.py ')
