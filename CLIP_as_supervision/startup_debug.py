import ast
import os
import argparse
import logging
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=8)
parser.add_argument('--model', type=str, default='clip_tpn_Giant_4452_patch16_224', metavar='MODEL',
                    help='the name of model to train : clip_tpn_base_3324_patch16_224/test_model/clip_tpn_large_2240_patch16_256 ')
parser.add_argument('--clip_path', type=str, default='../ViT-B-16.pt',
                    help='the path of the CLIP model: ViT-B-16.pt / ViT-L-14.pt')
parser.add_argument('--teacher_dim', default=512, type=int,
                        help='CLIP-B is 512, CLIP-L is 768')
parser.add_argument('--input_size', default=224, type=int,
                    help='images input size for backbone')
parser.add_argument('--second_input_size', default=224, type=int,
                    help='images input size for CLIP teacher -- note that CLIP-L is with 14x14 patch size')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')
parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
parser.add_argument('--update_freq', default=512, type=int)
parser.add_argument('--loss_type', default='smooth_l1', type=str,
                        help='Type of loss function: *cosine* or *smooth_l1* (default)')

parser.add_argument('--supervise_num', default=2, type=int)
parser.add_argument('--zero_stage', default=1, type=int,
                        help='ZeRO optimizer stage (default: 0)')

parser.add_argument('--num_mask_patches', default=75, type=int,
                        help='number of the visual tokens/patches need be masked: 75 for 224 input or 105 for 256 input')

parser.add_argument('--opt_eps', default=1e-6, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--beta', default=0.98, type=float,
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--blr', type=float, default=1.0e-3, metavar='LR',
                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
parser.add_argument('--layer_scale_init_value', default=0.0, type=float,
                    help="We use 0.1 for both base and large models -- which might not be the best setting")
parser.add_argument('--num_gpus', type=int, default=8, help='the number of gpus per device')
parser.add_argument('--rank', type=int, default=0, help='node rank')
parser.add_argument('--world_size', type=int, default=10, help='world size')
parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:6666')
args, unparsed = parser.parse_known_args()


# master_addr = args.init_method[:-5]
# master_port = args.init_method[-4:]
master_addr = 'localhost'
master_port = '12355'
os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = master_port

cmd_str = f"python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
            --node_rank={args.rank} --master_addr={master_addr} --master_port={master_port} \
            run_itpn_pretraining.py \
            --data_set=/home/TianYunjie/Workspace/dataset/ILSVRC2012_split_100/ \
            --data_path=/home/TianYunjie/Workspace/dataset/ILSVRC2012_split_100/train \
            --output_dir=./output \
            --log_dir=./output  \
            --model itpn_base_ConvMlp_xattn_fusedLN_NaiveSwiGLU_subln_RoPE_xavier_3324_patch16_224  \
            --num_mask_patches {args.num_mask_patches}  \
            --second_input_size {args.second_input_size}  \
            --second_interpolation 'bicubic'  \
            --batch_size {args.batch_size} \
            --input_size {args.input_size} \
            --lr {args.blr}  \
            --clip_grad 3.0  \
            --imagenet_default_mean_and_std  \
            --opt_betas 0.9 {args.beta}  \
            --opt_eps {args.opt_eps}   \
            --epochs {args.epochs} \
            --clip_path {args.clip_path} \
            --update_freq {args.update_freq} \
            --save_ckpt_freq 1  \
            --teacher_type openai_clip \
            --zero_stage {args.zero_stage} \
            --stop_grad_conv1 \
            --enable_deepspeed \
    "

print('The running command is: ' + cmd_str)
os.system(cmd_str)
