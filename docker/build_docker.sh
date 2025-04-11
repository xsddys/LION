#!/bin/bash
set -x  # 启用调试输出
docker build --no-cache -t nvcr.io/nvidian/lion_env:0 -f ./docker/Dockerfile .