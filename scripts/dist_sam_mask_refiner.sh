#!/usr/bin/env bash

ANN_PATH=$1
IMG_PATH=$2
OUT_ANN_PATH=$3
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-29500}
PY_ARGS=${@:4}

CUDA_VISIBLE_DEVICES=$GPU_ID OMP_NUM_THREADS=1 \
    torchrun --nproc_per_node=1 --master_port=$PORT \
    sam_mask_refiner.py --ann_path ${ANN_PATH} --img_path ${IMG_PATH} --output_ann_path ${OUT_ANN_PATH} \
    ${PY_ARGS}
