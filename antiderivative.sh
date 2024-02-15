#!/bin/bash

python main.py \
  --problem 'antiderivative_unaligned' \
  --num_outputs 1 \
  --hidden_dim 40 \
  --data_dir './data/antiderivative_unaligned' \
  --branch_layers 40 \
  --n_sensors 100 \
  --branch_input_features 1 \
  --trunk_layers 128 \
  --trunk_input_features 1 \
  --seed 1337 \
  --lr 1e-3 \
  --epochs 10000 \
  --result_dir './result/antiderivative' \
  --log_iter 100