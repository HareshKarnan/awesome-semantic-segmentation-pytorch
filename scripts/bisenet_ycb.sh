#!/usr/bin/env bash

# train
CUDA_VISIBLE_DEVICES=0

python train_parallel.py --model bisenet \
    --backbone resnet18 --dataset ycb \
    --lr 0.0001 --epochs 10 --gpu-ids 0 \
    --batch-size 16 \
    --save-dir $SEG_HOME'/models'