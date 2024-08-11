#!/usr/bin/env bash

PROMPT=$1
WORK_DIR=$2
LOG_NAME=$3
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-29500}
PY_ARGS=${@:4}

if ! [ -d "$WORK_DIR" ]; then
   mkdir -p $WORK_DIR
fi

CUDA_VISIBLE_DEVICES=$GPU_ID OMP_NUM_THREADS=1 \
    torchrun --nproc_per_node=1 --master_port=$PORT \
    text2seg.py --prompt "${PROMPT}" --fg_def --output_dir ${WORK_DIR} \
    ${PY_ARGS} \
    2>&1 | tee $WORK_DIR/$LOG_NAME.log > /dev/null &

echo "tail -f $WORK_DIR/$LOG_NAME.log"
