#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
ANN_PATH=$3
IMG_PATH=$4
OUT_ANN_PATH=$5
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:6}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u sam_mask_refiner.py --ann_path ${ANN_PATH} --img_path ${IMG_PATH} --output_ann_path ${OUT_ANN_PATH} \
    ${PY_ARGS}
