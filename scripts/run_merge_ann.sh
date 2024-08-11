#!/usr/bin/env bash

LVIS=$1
MOSAIC=$2
OUTPUT=$3

python merge_ann.py --lvis_path ${LVIS} --mosaic_path ${MOSAIC} --save_path ${OUTPUT}
