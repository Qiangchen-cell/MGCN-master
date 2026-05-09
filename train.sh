#!/bin/bash
# CUHK-PEDES ICFG-PEDES RSTPReid
CUDA_VISIBLE_DEVICES=0 python train.py \
--config ./configs/retrieval_cuhk.yaml \
--output_dir output/CUHK \
--max_epoch 30 \
--batch_size_train 24 \
--batch_size_test 64 \
--init_lr 1e-5  \
--k_test 16 \
--epoch_eval 1 \
