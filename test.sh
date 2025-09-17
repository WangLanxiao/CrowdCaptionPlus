#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python ./train_net.py --num-gpus 1 \
--config-file ./configs/region_caption/jmdc/jmdc.yaml \
--eval-only  \
DECODE_STRATEGY.BEAM_SIZE 3  \
MODEL.WEIGHTS  ./work_dirs/jmdc/model.pth \
MODEL.BILINEAR.ENCODE.LAYERS 4 \
OUTPUT_DIR ./work_dirs/jmdc \
INFERENCE.NAME 'RegionMPEvaler'
