# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# By Yunjie Tian
# Based on BEiT, BEiT-v2, timm, DeiT, EVA code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/baaivision/EVA
# --------------------------------------------------------'

# ################################################################################
# import datetime
# import os
# import sys
# import socket

# ### Get the absolute path of the current file
# current_file_path = os.path.abspath(__file__)
# ### Extract the file name without the extension
# file_name = os.path.splitext(os.path.basename(current_file_path))[0]
# ### Extract the file extension without the dot
# file_extension = os.path.splitext(os.path.basename(current_file_path))[1][1:]
# ### use different folders for a multiprocess program
# hostname = socket.gethostname()
# process_id = os.getpid()
# ### Create a folder path by joining the directory of the current file with a new folder name
# ### The new folder name includes 'logs-', the file name, and the file extension
# # log_folder = os.path.join(os.path.dirname(current_file_path), 'logs-' + file_name + '-' + file_extension)
# # log_folder = os.path.join(os.path.dirname(current_file_path), f'logs-{file_name}-pid_{process_id}-{file_extension}')
# log_folder = os.path.join(os.path.dirname(current_file_path), f'logs-{file_name}-host_{hostname}-pid_{process_id}-{file_extension}')
# ### Create the directory for the log folder if it doesn't already exist
# os.makedirs(log_folder, exist_ok=True)
# ### Generate a timestamp in the format YYYYMMDD_HHMMSS
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ### usage:
# ### @pysnooper.snoop(os.path.join(log_folder, f"funcname-{timestamp}.log"), color=False, max_variable_length=2000)
# ### def xxx:                 ...

# #-----------------------------------------------------------------------------#

# ### global overview

# ### pip install GitPython
# # import git

# ### Check if the code is within the desired directory or repository
# # repo = git.Repo('.', search_parent_directories=True)
# ### Get the repo path
# # repo_path = repo.git.rev_parse("--show-toplevel")
# # print(f"repo_path: {repo_path}")
# ### 建议手动写, 有时 git 获得 repo_path 会报错
# repo_path = "/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision"

# ### 你可以修改 tracefunc 函数以仅将输出写入文件而不打印在终端上。你只需要移除将消息写入 original_stdout 的部分
# def tracefunc(frame, event, arg, indent=[0], output_file=None, original_stdout=None):
#     """
#     tracefunc is defined to trace the execution of functions. It takes several parameters:
#         frame: The current stack frame.
#         event: The type of event that occurred (e.g., "call", "return").
#         arg: Additional argument (not used in this code).
#         indent: A list used to keep track of the indentation level for the output.
#         output_file: A file object where trace messages will be written.
#         original_stdout: The original standard output stream for console logging.
#     """
#     ### Get the file path and line number of the code being executed
#     file_path = frame.f_globals.get('__file__')
#     line_num = frame.f_lineno

#     ### If file_path is not None, it's converted to an absolute path.
#     if file_path:
#         file_path = os.path.abspath(file_path)
#         ### Check if the code is within the desired directory or repository
#         if file_path.startswith(repo_path):
#             if event == "call":
#                 ### Increases the indentation level.
#                 indent[0] += 2
#                 ### Constructs a message indicating the function call with the function name, file path, and line number.
#                 msg = f"{'-' * indent[0]}> call function {frame.f_code.co_name} in {file_path}:{line_num}\n"
#                 ### Writes the message to both output_file and original_stdout.
#                 output_file.write(msg)
#                 if original_stdout:
#                     original_stdout.write(msg)
#             elif event == "return":
#                 ### Constructs a message indicating the function exit with the function name, file path, and line number.
#                 msg = f"<{'-' * indent[0]} exit function {frame.f_code.co_name} in {file_path}:{line_num}\n"
#                 ### Writes the message to both output_file and original_stdout.
#                 output_file.write(msg)
#                 ### Decreases the indentation level.
#                 if original_stdout:
#                     original_stdout.write(msg)
#                 indent[0] -= 2
#     return tracefunc
# ################################################################################

import argparse
import datetime
import numpy as np
import time
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
from timm.models import create_model
from optim_factory import create_optimizer, get_parameter_groups

from itpn_hf_datasets import build_dataset

from engine_for_pretraining import train_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from clip import *
from clip_wrapper import *
import modeling_pretrain
# import moxing as mox

try:
    import xformers
except:
    print('Please install xFormer following the official instruction')
    #exit(0)

try:
    import apex
except:
    print('Please install apex following the official instruction')
    exit(0)


def get_args():
    parser = argparse.ArgumentParser('iTPN pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # parser.add_argument('--s3_path', type=str, default='', help='the path of the OBS')

    parser.add_argument('--clip_path', type=str, default='../ViT-B-16.pt',
                        help='the path of the CLIP model')
    parser.add_argument('--teacher_dim', default=512, type=int,
                        help='CLIP-B is 512, CLIP-L is 768')

    parser.add_argument('--teacher_type', type=str, default='evaclip')
    parser.add_argument('--teacher_model_path', type=str)

    parser.add_argument('--clip_model', type=str, default='EVA_CLIP_g_14_X')
    parser.add_argument('--cache_dir', type=str, default='/path/to/eva_clip_psz14.pt')

    # Model parameters
    parser.add_argument('--model', default='itpn_base_3324_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--grad_ckpt', action='store_true')
    parser.add_argument('--stop_grad_conv1', action='store_true')

    parser.add_argument('--rel_pos_bias', action='store_true')

    parser.add_argument('--decoupled_rel_pos_bias', action='store_true')
    parser.add_argument('--disable_decoupled_rel_pos_bias', action='store_false', dest='decoupled_rel_pos_bias')
    parser.set_defaults(decoupled_rel_pos_bias=False)

    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=True)

    parser.add_argument('--layer_scale_init_value', default=0, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")
    parser.add_argument('--load-tar', action='store_true',
                        help='Loading *.tar files for dataset')

    parser.add_argument('--num_mask_patches', default=105, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)

    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size for backbone')
    parser.add_argument('--second_input_size', default=224, type=int,
                        help='images input size for discrete vae')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-6, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.98], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=3.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD.
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1.5e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--second_interpolation', type=str, default='bicubic',
                        help='Interpolation for discrete vae (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--crop_scale', default=[0.2, 1.0], type=float, nargs='+',
                        help='crop_scale')
    parser.add_argument('--crop_ratio', default=[3. / 4., 4. / 3.], type=float, nargs='+',
                        help='crop_ratio')
    parser.add_argument('--min_crop_scale', type=float, default=0.2, metavar='PCT',
                        help='min_crop_scale (default: 0.08)')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default='IMNET', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='image_folder', type=str,
                        help='dataset path')

    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    #########################################################################################
    parser.add_argument('--bf16', default=False, action='store_true')
    parser.add_argument('--enable_deepspeed', default=False, action='store_true')
    parser.add_argument('--zero_stage', default=0, type=int, help='ZeRO optimizer stage (default: 0)')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print('Please install DeepSpeed')
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def get_model(args):
    print(f"Creating model: {args.model}")

    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        grad_ckpt=args.grad_ckpt,
        stop_grad_conv1=args.stop_grad_conv1,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_shared_decoupled_rel_pos_bias=args.decoupled_rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        teacher_dim=args.teacher_dim
    )
    return model


def get_clip(args):
    print(f"Creating clip teacher: ")
    if args.second_input_size == 196 or args.input_size > 224:
        model_name = 'ViT-L/14'
    else:
        model_name = 'ViT-B/16'
    print(f"Creating clip teacher: {model_name}") # ViT-B/16
    model = clip_distill(download_root='/tmp/', teacher_size=args.second_input_size, model_name=model_name).eval()
    return model


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    ###########################################################################
    if args.teacher_type == 'evaclip':
        teacher = EVACLIPWrapper(
            clip_model = args.clip_model,
            cache_dir = args.cache_dir
        )
    else:
        teacher = get_clip(args)
    teacher = teacher.to(device)
    print("teacher = %s" % str(teacher))
    ###########################################################################

    model = get_model(args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    ### get dataset
    dataset_train, nb_classes = build_dataset(is_train=True, args=args)

    # prepare teacher

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks // args.update_freq
        print(f"Debug: len(dataset_train): {len(dataset_train)}")
        print(f"Debug: args.batch_size: {args.batch_size}")
        print(f"Debug: num_tasks: {num_tasks}")
        print(f"Debug: args.update_freq: {args.update_freq}")
        print(f"Debug: num_training_steps_per_epoch: {num_training_steps_per_epoch}")

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # print("Tokenizer = %s" % str(vqkd))
    total_batch_size = args.batch_size * utils.get_world_size()
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))


    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, model.no_weight_decay()
        )
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" %
              model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        model.to(device)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        optimizer = create_optimizer(args, model_without_ddp)
        loss_scaler = NativeScaler()

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    print('Optimizer = %s' % str(optimizer))

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay

    print(f"args.weight_decay: {args.weight_decay}")
    print(f"args.weight_decay_end: {args.weight_decay_end}")
    print(f"args.epochs: {args.epochs}")
    print(f"num_training_steps_per_epoch: {num_training_steps_per_epoch}")
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)


    # Check and print the contents of wd_schedule_values
    print("wd_schedule_values:", wd_schedule_values)

    # # Proceed only if wd_schedule_values is not empty
    # if wd_schedule_values:
    #     print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    # else:
    #     print("wd_schedule_values is empty. Please check the code that populates this list.")
    #     # Optionally, raise an error or provide a default value
    #     raise ValueError("wd_schedule_values is empty")

    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if global_rank == 0:
        print("Check parameter scale !")
        for para_name, para_ver in model_without_ddp.named_parameters():
            mean = para_ver.mean()
            abs_mean = para_ver.abs().mean()
            delta = para_ver.max() - para_ver.min()
            print("{}: {} {:.8f}, {:.8f}, {:.8f}, require_grad = {}".format(
                para_name, para_ver.size(), mean, abs_mean, delta, para_ver.requires_grad))

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    print(f"args.start_epoch: {args.start_epoch}, args.epochs: {args.epochs}")
    for epoch in range(args.start_epoch, args.epochs):

        ### wxf: for profiling
        print(f"Epoch {epoch} starts...")
        if epoch == 1:
            print(f"break, profiling stops, Epoch {epoch} stops.")
            break

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch(
            model,
            teacher,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            max_norm=args.clip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            update_freq=args.update_freq,
            fp16=not args.bf16,
            args=args,
        )
        if args.output_dir and (epoch % 25 == 0 or epoch + 1 == args.epochs or epoch + 2 == args.epochs):
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, save_ckpt_freq=args.save_ckpt_freq)
            # mox.file.copy_parallel(args.output_dir, args.s3_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    # ############################################################################
    # ### Open the file to save the trace output

    # ### This constructs the path for the trace output file by joining log_folder
    # ### with a filename that includes a timestamp. The timestamp variable is
    # ### assumed to be a string representing the current time.
    # tracing_filename = os.path.join(log_folder, f"tracing-{file_name}-{timestamp}.log")

    # ### This opens the file in write mode ("w") and assigns the file object to
    # ### output_file. This file will be used to save the trace output.
    # output_file = open(tracing_filename, "w")

    # ### This line stores the original standard output stream (sys.stdout) in the
    # ### variable original_stdout. This allows you to write trace messages to both
    # ### the trace file and the console.
    # original_stdout = sys.stdout

    # ### Set the profile function with the output file
    # ### - sys.setprofile: This function sets the system's profiling function,
    # ###   which is called on every function call and return.
    # ### - lambda frame, event, arg: tracefunc(frame, event, arg,
    # ###   output_file=output_file, original_stdout=original_stdout): This is a
    # ###   lambda function that wraps the tracefunc with the additional arguments
    # ###   output_file and original_stdout.
    # ###   - frame: The current stack frame.
    # ###   - event: The type of event (e.g., "call", "return").
    # ###   - arg: Additional argument (not used in this code).
    # ###   This lambda function ensures that every function call and return event
    # ###   in the program is handled by tracefunc, which will log the event details
    # ###   to the output_file and the console (original_stdout).
    # sys.setprofile(lambda frame, event, arg: tracefunc(frame, event, arg, output_file=output_file, original_stdout=original_stdout))
    # ############################################################################

    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)

    ############################################################################
    ### Restore standard output
    # sys.stdout = sys.__stdout__
    ### Close the output file
    # output_file.close()
    ############################################################################
