import ast
import os
# import moxing as mox
import argparse
import logging
import time
import utils

os.environ["NCCL_NET_GDR_LEVEL"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--train_url', type=str, default=' ', help='the output path')
# parser.add_argument('--s3_path', type=str, default='', help='the path of the config file')
parser.add_argument('--batch_size', type=int, default=64, help='the path of the config file')
parser.add_argument('--epochs', type=int, default=800, help='the path of the config file')
parser.add_argument('--warmup_epochs', type=int, default=10, help='the path of the config file')
parser.add_argument('--model', type=str,
                    default='itpn_base_ConvMlp_xattn_fusedLN_NaiveSwiGLU_subln_RoPE_xavier_3324_patch16_224',
                    help='the path of the config file')
parser.add_argument('--clip_path', type=str, default='', help='the path of the config file')
parser.add_argument('--clip_grad', type=float, default=3.0, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--input_size', default=224, type=int,
                    help='images input size for backbone')
parser.add_argument('--second_input_size', default=224, type=int,
                    help='images input size for discrete vae')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

parser.add_argument('--resume_path', default='', help='resume from checkpoint')
parser.add_argument('--resume', default='', help='resume from checkpoint')

parser.add_argument('--min_crop_scale', type=float, default=0.2, metavar='PCT',
                    help='min_crop_scale (default: 0.08)')
parser.add_argument('--rel_pos_bias', default=False, type=utils.bool_flag)
parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
parser.add_argument('--abs_pos_emb', default=True, type=utils.bool_flag)

parser.add_argument('--in1k', type=int, default=0, help='world size')
parser.add_argument('--opt_eps', default=1e-6, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--beta', default=0.98, type=float, help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--blr', type=float, default=1.5e-6, metavar='LR',
                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                    help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

# parser.add_argument('--num_gpus', type=int, default=8, help='the number of gpus')
parser.add_argument('--rank', type=int, default=0, help='node rank')
parser.add_argument('--world_size', type=int, default=2, help='world size')
parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:6666')

parser.add_argument('--num_mask_patches', type=int, default=75, help='the number of masked patches')

parser.add_argument('--update_freq', type=int, default=1, help='zero stage optimize')
parser.add_argument('--zero_stage', type=int, default=1, help='zero stage optimize')
parser.add_argument('--teacher_type', type=str, default='clip')
parser.add_argument('--teacher_dim', default=768, type=int,
                        help='CLIP-B is 512, CLIP-L is 768')
parser.add_argument('--cache_dir', type=str, default='/tmp/eva_clip_psz14.pt')

args, unparsed = parser.parse_known_args()

# os.system('cat /usr/local/cuda/version.txt')
# os.system('nvcc --version')
# print(args.train_url)
# mox.file.copy_parallel('./CLIP_as_supervision/', args.train_url + '/CLIP_as_supervision-code/')
# ############# preparation stage ####################
# print('Current path: ' + os.getcwd())
# print('Current dirs: ' + str(list(os.listdir())))

# os.chdir('./CLIP_as_supervision')
# print('Current path changed to: ' + os.getcwd())

# os.system("pip install torch==1.12.1 torchvision==0.13.1 submitit wandb")
# os.system("pip install regex timm==0.4.12 yacs diffdist termcolor lmdb tensorboard")

# os.system('pip install --ignore-installed PyYAML')
# os.system('pip install ftfy-6.0.1.tar.gz')
# os.system(
#     'pip install -q --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex/')

###################################################################################################
# print('Start copying dataset')
# if args.in1k == 1:
#     mox.file.copy_parallel('s3://bucket-3690/tianyunjie/datasets/imagenet-1000-tar/imagenet.tar',
#                            '/cache/imagenet.tar')
#     os.system('tar xf /cache/imagenet.tar -C /cache/')
# else:
#     mox.file.copy_parallel('s3://bucket-3690/tianyunjie/datasets/imagenet-10/', '/cache/imagenet')
# print('Finish copying dataset')
# #################################################################################################
#if args.resume_path:
#mox.file.copy_parallel(args.resume_path, '/cache/output/')
###################################################################################################
# if 'eva' in args.teacher_type:
#     mox.file.copy_parallel(args.clip_path, args.cache_dir)
# else:
#     if args.second_input_size == 196 or args.input_size > 224:
#         mox.file.copy_parallel(args.clip_path, '/cache/ViT-L-14.pt')
#         args.clip_path = '/cache/ViT-L-14.pt'
#     else:
#         mox.file.copy_parallel(args.clip_path, '/cache/ViT-B-16.pt')
#         args.clip_path = '/cache/ViT-B-16.pt'
# print('Finish copying dataset')
# os.system('nvcc --version')

# time.sleep(30)
###########################################################################################################
# master_host = os.environ['VC_WORKER_HOSTS'].split(',')[0]
# master_host = 'localhost'
# master_addr = master_host.split(':')[0]
# master_port = '8525'  # '8524'
# modelarts_rank = args.rank  # ModelArts receive FLAGS.rank means node_rank
# modelarts_world_size = args.world_size  # ModelArts receive FLAGS.worldsize means nodes_num
# os.environ['MASTER_ADDR'] = master_addr
# os.environ['MASTER_PORT'] = master_port
#######################################################################################################


nnodes = 1
node_rank = 0
master_addr = 'localhost'
master_port = '8888'
cmd_str = f"torchrun --nnodes={nnodes} --nproc_per_node=4 \
        --node_rank={node_rank} --master_addr={master_addr} --master_port={master_port} \
        run_itpn_pretraining.py \
        --output_dir=/tmp/output/ \
        --log_dir=/tmp/output  \
        --model {args.model}  \
        --teacher_type {args.teacher_type} \
        --clip_model EVA_CLIP_g_14_X \
        --cache_dir {args.cache_dir} \
        --input_size {args.input_size} \
        --second_input_size {args.second_input_size}  \
        --second_interpolation 'bicubic'  \
        --num_mask_patches {args.num_mask_patches} \
        --layer_scale_init_value {args.layer_scale_init_value}  \
        --batch_size {args.batch_size} \
        --lr {args.blr}  \
        --epochs {args.epochs} \
        --warmup_epochs {args.warmup_epochs}  \
        --teacher_dim {args.teacher_dim} \
        --clip_path {args.clip_path} \
        --clip_grad {args.clip_grad}  \
        --drop_path {args.drop_path}  \
        --imagenet_default_mean_and_std  \
        --opt_betas 0.9 {args.beta}  \
        --min_crop_scale {args.min_crop_scale} \
        --opt_eps {args.opt_eps}   \
        --save_ckpt_freq 10 \
        --zero_stage {args.zero_stage} \
        --update_freq {args.update_freq} \
        --stop_grad_conv1 \
        --enable_deepspeed "

print('The running command is: ' + cmd_str)
os.system(cmd_str)

