#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=1
NODE_RANK=0

BATCH_SIZE=8
ACCUM_STEP=1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

DATESTR=$(date +"%m-%d-%H-%M")
SAVE_PATH=output/stage1
mkdir -p ${SAVE_PATH}

set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    train.py \
    --data_path data/ \
    --train_file train_final.csv \
    --validation_file test.csv \
    --test_file real/CLEF.csv \
    --valid_file real/USPTO.csv \
    --vocab_file adaptmol/vocab/vocab_chars.json \
    --formats chartok_coords,edges \
    --coord_bins 64 --sep_xy \
    --input_size 384 \
    --encoder_lr 4e-6 \
    --decoder_lr 4e-6 \
    --save_path $SAVE_PATH --save_mode all \
    --label_smoothing 0.1 \
    --epochs 30 \
    --augment \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --warmup 0.00 \
    --print_freq 50 \
    --do_train --do_valid --do_test  \
    --fp16 --backend gloo 2>&1 \
    --molblock \
    # --resume \
    