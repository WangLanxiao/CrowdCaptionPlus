# XE
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_net.py \
--num-gpus 8 --dist-url tcp://127.0.0.1:18169 \
--resume \
--config-file ./configs/region_caption/jmdc/jmdc.yaml \
OUTPUT_DIR ./work_dirs/daxiu_4K


# RL
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_net.py --num-gpus 8 --dist-url tcp://127.0.0.1:21114 --config-file ./configs/region_caption/xlan/xlan_rl.yaml