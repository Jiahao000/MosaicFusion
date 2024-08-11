#!/usr/bin/env bash

ANN_PATH=$1
IMG_PATH=$2
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-29500}
PY_ARGS=${@:3}

CUDA_VISIBLE_DEVICES=$GPU_ID OMP_NUM_THREADS=1 \
    torchrun --nproc_per_node=1 --master_port=$PORT \
    sam_iou_metric.py --ann_path ${ANN_PATH} --img_path ${IMG_PATH} \
    ${PY_ARGS}
