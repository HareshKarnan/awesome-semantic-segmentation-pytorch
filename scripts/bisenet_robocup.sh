#!/usr/bin/env bash

# train
CUDA_VISIBLE_DEVICES=0

python3.6 train_parallel.py --model bisenet \
    --backbone resnet18 --dataset robocup \
    --lr 0.0001 --epochs 100 --gpu-ids 0 \
    --batch-size 8 \
    --save-dir $SEG_HOME'/models'
#    --resume $SEG_HOME/models/bisenet_resnet18_ycb_best_model.pth