#!/bin/bash

# Set any necessary environment variables here
export OMP_NUM_THREADS=1

NNODES=1
GPUS=4
CLASSES=1000
INPUT_SIZE=224  # 384/512
WEIGHT_DECAY=0.05
BATCH_SIZE=64
LAYER_SCALE_INIT_VALUE=0.1
LR=1e-4
UPDATE_FREQ=1
REPROB=0.25
EPOCHS=100
W_EPOCHS=5
LAYER_DECAY=0.80
MIN_LR=1e-6
WARMUP_LR=1e-6
DROP_PATH=0.1
MIXUP=0.8
CUTMIX=1.0
SMOOTHING=0.1
MODEL='fast_itpn_tiny_1112_patch16_224'
WEIGHT='../fast_itpn_tiny_1600e_1k.pt'
NODE_RANK=0
MASTER_ADDR=localhost

### Both are not necessary now
# --data_path /PATH/TO/IN1K/train \
# --eval_data_path /PATH/TO/IN1K/val \

torchrun \
    --nproc_per_node ${GPUS} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=6666 \
    run_class_finetuning.py  \
    --nb_classes ${CLASSES} \
    --data_set image_folder \
    --output_dir ./output \
    --input_size ${INPUT_SIZE} \
    --log_dir ./output \
    --model ${MODEL} \
    --weight_decay ${WEIGHT_DECAY}  \
    --finetune ${WEIGHT}  \
    --batch_size ${BATCH_SIZE}  \
    --layer_scale_init_value ${LAYER_SCALE_INIT_VALUE} \
    --lr ${LR} \
    --update_freq ${UPDATE_FREQ}  \
    --reprob ${REPROB} \
    --warmup_epochs ${W_EPOCHS} \
    --epochs ${EPOCHS}  \
    --layer_decay ${LAYER_DECAY} \
    --min_lr ${MIN_LR} \
    --warmup_lr ${WARMUP_LR} \
    --drop_path ${DROP_PATH}  \
    --mixup ${MIXUP} \
    --cutmix ${CUTMIX} \
    --smoothing ${SMOOTHING} \
    --imagenet_default_mean_and_std   \
    --dist_eval \
    --model_ema \
    --model_ema_eval \
    --save_ckpt_freq 20 \
    --second_input_size 224
