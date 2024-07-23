import ast
import os
import argparse
import logging
import time
import moxing as mox

parser = argparse.ArgumentParser()
parser.add_argument('--train_url', type=str, default=' ', help='the path of the pretrained checkpoints')
parser.add_argument('--s3_path', type=str, default=' ', help='the path of the pretrained checkpoints')
parser.add_argument('--pretrained', type=str, default=' ', help='the path of the pretrained checkpoints')
parser.add_argument('--batch_size', type=int, default=32, help='the batch size per GPU')
parser.add_argument('--epochs', type=int, default=100, help='total fine-tuning epochs')
parser.add_argument('--warmup_epochs', type=int, default=5, help='the path of the config file')
parser.add_argument('--model', type=str, default='itpn_base_3324_patch16_224', help='the path of the config file')
parser.add_argument('--dist_eval', action='store_true', default=True)
parser.add_argument('--weight', type=str, default='/cache/weight.pth', help='the checkpoint file')
parser.add_argument('--blr', type=float, default=5e-4, metavar='LR',
                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help='images input size')
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
parser.add_argument('--layer_decay', type=float, default=0.65,
                    help='layer-wise lr decay from ELECTRA/BEiT')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0.')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0.')
parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
parser.add_argument('--update_freq', default=1, type=int)
parser.add_argument('--clip_grad', type=float, default=5.0,
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--drop_path', type=float, default=.1,
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
parser.add_argument('--in1k', type=int, default=0, help='node rank')

parser.add_argument('--num_gpus', type=int, default=8, help='the number of gpus')
parser.add_argument('--rank', type=int, default=0, help='node rank')
parser.add_argument('--world_size', type=int, default=2, help='world size')
parser.add_argument('--data_path', default='/cache/imagenet/', type=str,
                    help='dataset path')
args, unparsed = parser.parse_known_args()

os.system('cat /usr/local/cuda/version.txt')
os.system('nvcc --version')
mox.file.copy_parallel('./CLIP_as_supervision/', args.train_url + '/CLIP_as_supervision-code/')
# ############# preparation stage ####################
print('Current path: ' + os.getcwd())
print('Current dirs: ' + str(list(os.listdir())))

os.chdir('./CLIP_as_supervision')
print('Current path changed to: ' + os.getcwd())

# os.system("pip install torch==1.7.1 torchvision==0.8.2 submitit")
os.system('pip install ftfy-6.0.1.tar.gz')
os.system("pip install regex timm==0.4.12 yacs diffdist termcolor lmdb tensorboard einops")

os.system('pip install --ignore-installed PyYAML')

###################################################################################################
print('Start copying dataset')
if args.in1k == 1:
    mox.file.copy_parallel('s3://bucket-3690/tianyunjie/datasets/imagenet-1000-tar/imagenet.tar',
                           '/cache/imagenet.tar')
    os.system('tar xf /cache/imagenet.tar -C /cache/')
else:
    mox.file.copy_parallel('s3://bucket-3690/tianyunjie/datasets/imagenet-10/', '/cache/imagenet')
print('Finish copying dataset')
###########################################################################################################

mox.file.copy_parallel(args.pretrained, '/cache/weight.pth')

time.sleep(30)
###########################################################################################################

master_host = os.environ['VC_WORKER_HOSTS'].split(',')[0]
master_addr = master_host.split(':')[0]
master_port = '8524'
# FLAGS.worldsize will be re-computed follow as FLAGS.ngpu*FLAGS.nodes_num
# FLAGS.rank will be re-computed in main_worker
modelarts_rank = args.rank  # ModelArts receive FLAGS.rank means node_rank
modelarts_world_size = args.world_size  # ModelArts receive FLAGS.worldsize means nodes_num
os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = master_port

print(f'IP: {master_addr},  Port: {master_port}')
print(f'modelarts rank {modelarts_rank}, world_size {modelarts_world_size}')

###################################################################################################

cmd_str = f"python -m torch.distributed.launch \
    --nproc_per_node {args.num_gpus} \
    --nnodes={args.world_size} \
    --node_rank={args.rank} \
    --master_addr={master_addr} \
    --master_port={master_port} \
    run_itpn_finetuning.py  \
    --data_path /cache/imagenet/train \
    --eval_data_path /cache/imagenet/val \
    --nb_classes {args.nb_classes} \
    --data_set 'image_folder' \
    --s3_path {args.s3_path} \
    --output_dir /cache/output \
    --input_size {args.input_size} \
    --log_dir /cache/output \
    --model {args.model} \
    --weight_decay {args.weight_decay}  \
    --finetune {args.weight}  \
    --batch_size {args.batch_size}  \
    --layer_scale_init_value {args.layer_scale_init_value} \
    --lr {args.blr} \
    --update_freq {args.update_freq}  \
    --nb_classes {args.nb_classes} \
    --warmup_epochs {args.warmup_epochs} \
    --epochs {args.epochs}  \
    --layer_decay {args.layer_decay} \
    --min_lr {args.min_lr} \
    --drop_path {args.drop_path}  \
    --mixup {args.mixup} \
    --cutmix {args.cutmix} \
    --imagenet_default_mean_and_std   \
    --dist_eval \
    --save_ckpt_freq 20 \
"

print('The running command is: ' + cmd_str)

os.system(cmd_str)
