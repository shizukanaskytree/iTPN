# NODE_RANK=0
# MASTER_ADDR=localhost
# python -m torch.distributed.launch --nproc_per_node=4 --nnodes 1 --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR --master_port=6666  run_itpn_pretraining.py \
#     --world_size 4 \
#     --batch_size 8 \
#     --model clip_tpn_base_3324_patch16_224 \
#     --beta 0.98 \
#     --blr 1.5e-3 \
#     --clip_path ../ViT-B-16.pt \
#     --drop_path 0.1 \
#     --epochs 300 \
#     --input_size 224 \
#     --layer_scale_init_value 0.1 \
#     --opt_eps 1e-8 \
#     --second_input_size 224 \

NODE_RANK=0
MASTER_ADDR=localhost
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6666 run_itpn_pretraining.py \
    --world_size 4 \
    --batch_size 8 \
    --model clip_tpn_base_3324_patch16_224 \
    --opt_betas 0.98 \
    --lr 1.5e-3 \
    --clip_path ../ViT-B-16.pt \
    --drop_path 0.1 \
    --epochs 300 \
    --input_size 224 \
    --layer_scale_init_value 0.1 \
    --opt_eps 1e-8 \
    --second_input_size 224