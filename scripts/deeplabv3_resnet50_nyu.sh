#!/usr/bin/env bash

# train
CUDA_VISIBLE_DEVICES=0 python3.6 train.py --model deeplabv3 \
    --backbone resnet101 --dataset ade20k \
    --lr 0.0001 --epochs 100 --save-dir '/scratch/cluster/haresh92/awesome-semantic-segmentation-pytorch/models'