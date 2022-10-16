# !/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 1.0 --dataset eth --tag social_implicit_eth_test  --w_norm 0.00001 --w_cos 0.0001 && echo "eth Launched." &
P0=$!

CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 1.0 --dataset hotel --tag social_implicit_hotel_test  --w_norm 0.00001 --w_cos 0.00001 && echo "hotel Launched." &
P1=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 1.0 --dataset univ --tag social_implicit_univ_test  --w_norm 0.00001 --w_cos 0.0001 && echo "univ Launched." &
P2=$!

CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 1.0 --dataset zara1 --tag social_implicit_zara1_test  --w_norm 0.00001 --w_cos 0.0001 && echo "zara1 Launched." &
P3=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 1.0 --dataset zara1 --tag social_implicit_zara1_test  --w_norm 0.0001 --w_cos 0.0001 && echo "zara2 Launched." &
P4=$!

CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 1.0 --dataset sdd --tag social_implicit_sdd_test  --w_norm 0.0001 --w_cos 0.0001 && echo "sdd Launched." &
P5=$!


wait $P0 $P1 $P2 $P3 $P4 $P5