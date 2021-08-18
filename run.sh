#!/bin/bash

python main.py --batch-size 256 \
               --epochs 100 \
               --lr 0.001 \
               --decay 0.0005 \
               --warm-steps 10\
               --patience 10 \
               --loss logcosh \
               --num-layer 2 \
               --emb-dim 300 \
               --dropout-ratio 0.2 \
               --graph-pooling mean \
               --graph-dir data/graphs \
               --logging-dir GAT \
               --gnn-type gat \
               --split 10 \
               --seed 42 \
               --visualize
