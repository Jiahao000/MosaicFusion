#!/usr/bin/env bash

INPUT_DIR=$1
OUTPUT_DIR=$2
PY_ARGS=${@:3}

if ! [ -d "$OUTPUT_DIR" ]; then
   mkdir -p $OUTPUT_DIR
fi

python seg2ann.py --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} ${PY_ARGS}
