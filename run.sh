#!/bin/bash

python main.py --batch_size 256 \
               --epochs 100 \
               --lr 0.001 \
               --decay 0.0005 \
               --warm_steps 10\
               --patience 10 \
               --loss logcosh \
               --num_layer 2 \
               --emb_dim 300 \
               --dropout_ratio 0.2 \
               --graph_pooling mean \
               --graph_dir data/graphs \
               --logging_dir GAT \
               --gnn_type gat \
               --split 10 \
               --seed 42 \
               --visualize
