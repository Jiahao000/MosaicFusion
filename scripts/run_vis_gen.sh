#!/usr/bin/env bash

ANN_PATH=$1
IMG_PATH=$2
OUTPUT_DIR=$3
PY_ARGS=${@:4}
# --show_boxes --show_segms --show_classes --save_grid

if ! [ -d "$OUTPUT_DIR" ]; then
   mkdir -p $OUTPUT_DIR
fi

python vis_gen.py --ann_path ${ANN_PATH} --img_path ${IMG_PATH} --save_path ${OUTPUT_DIR} ${PY_ARGS}
