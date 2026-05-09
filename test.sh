#!/bin/bash
# CUHK-PEDES ICFG-PEDES RSTPReid
CUDA_VISIBLE_DEVICES=0 python train.py \
--config ./configs/retrieval_icfg.yaml \
--output_dir output/test \
--batch_size_test 64 \
--k_test 32 \
--pretrained /path/checkpoint_best.pth \
--evaluate
