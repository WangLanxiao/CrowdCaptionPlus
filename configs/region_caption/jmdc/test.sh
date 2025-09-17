#!/usr/bin/env bash

#PTH_PATH="/data1/wlx/project/202209_xmodaler_dif_pers/work_dirs/jmdc_V1000/model_Epoch_00"
#for i in `seq 50 1 62`
#do
#  let "ii=$i*2"
#  let "j=$ii*112-1"
#  echo "$PTH_PATH""$ii"_Iter_00"$j".pth
#  CUDA_VISIBLE_DEVICES=3 python ./train_net.py --num-gpus 1 \
#  --config-file ./configs/region_caption/jmdc/jmdc.yaml \
#  --eval-only DECODE_STRATEGY.BEAM_SIZE 3 MODEL.WEIGHTS "$PTH_PATH""$ii"_Iter_00"$j".pth \
#  OUTPUT_DIR ./work_dirs/jmdc_V1000
#done

CUDA_VISIBLE_DEVICES=1 python ./train_net.py --num-gpus 1 \
--config-file ./configs/region_caption/jmdc/jmdc.yaml \
--eval-only  \
DECODE_STRATEGY.BEAM_SIZE 3  \
MODEL.WEIGHTS  ./work_dirs/daxiu_4K/model_Epoch_00111_Iter_0012431.pth \
MODEL.BILINEAR.ENCODE.LAYERS 4 \
OUTPUT_DIR ./work_dirs/jmdc_Vdebug \
INFERENCE.NAME 'RegionMPEvaler'
