#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
PROMPT=$3
WORK_DIR=$4
LOG_NAME=$5
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:6}

if ! [ -d "$WORK_DIR" ]; then
   mkdir -p $WORK_DIR
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u text2seg.py --prompt "${PROMPT}" --fg_def --output_dir ${WORK_DIR} \
    ${PY_ARGS} \
    2>&1 | tee $WORK_DIR/$LOG_NAME.log > /dev/null &

echo "tail -f $WORK_DIR/$LOG_NAME.log"
