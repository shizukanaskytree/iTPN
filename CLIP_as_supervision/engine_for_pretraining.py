# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Yunjie Tian
# Based on BEiT, BEiT-v2, timm, DeiT and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

# from cgitb import enable
import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils

import time

import torch.distributed as dist
import torch.nn.functional as F

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    loss_scale = None
    if hasattr(optimizer, 'loss_scale'):
        loss_scale = optimizer.loss_scale
    elif hasattr(optimizer, 'cur_scale'):
        loss_scale = optimizer.cur_scale
    return loss_scale, None  # optimizer._global_grad_norm


def compute_loss(output, label):
    loss_func = nn.CosineSimilarity(dim=-1)
    loss = loss_func(output.float(), label.float())
    return -loss.mean()

# def compute_loss(outputs, labels):
#     # Ensure both tensors have the same shape
#     if outputs.size(1) != labels.size(1):
#         # Example: if labels have size 768, reshape outputs to have size 768
#         outputs = outputs.view(-1, labels.size(1))

#     loss = F.cosine_similarity(outputs.float(), labels.float())
#     return loss

# def compute_loss(outputs, labels):
#     # Print tensor shapes for debugging
#     print(f"Outputs shape before reshape: {outputs.shape}")
#     print(f"Labels shape: {labels.shape}")

#     # Ensure both tensors have the same shape
#     if outputs.size(1) != labels.size(1):
#         try:
#             # Attempt to reshape outputs to match labels' dimensions
#             outputs = outputs.view(-1, labels.size(1))
#             print(f"Outputs shape after reshape: {outputs.shape}")
#         except RuntimeError as e:
#             print(f"Reshape error: {e}")

#     # Ensure tensors match in dimension 0
#     if outputs.size(0) != labels.size(0):
#         min_size = min(outputs.size(0), labels.size(0))
#         outputs = outputs[:min_size]
#         labels = labels[:min_size]
#         print(f"Outputs shape after size adjustment: {outputs.shape}")
#         print(f"Labels shape after size adjustment: {labels.shape}")

#     # Compute loss
#     loss = F.cosine_similarity(outputs.float(), labels.float())
#     return loss


def train_one_epoch(model: torch.nn.Module,
                    teacher: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    update_freq=1,
                    log_writer=None,
                    lr_scheduler=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    fp16=True,
                    args=None
                    ):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (batch, extra_info) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step

        ### wxf: for profiling
        print(f"Data iter step: {data_iter_step}...")
        if data_iter_step == 1:
            print('break, profiling stops...')
            break

        step = data_iter_step // update_freq

        it = start_steps + step  # global training iteration

        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples, images, bool_masked_pos = batch
        images = images.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)

        if loss_scaler is None:
            if fp16:
                samples = samples.half()
            else:
                samples = samples.bfloat16()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if 'eva' in args.teacher_type:
                    clip_features = teacher.infer_image({'image': [images]})
                else:
                    clip_features = teacher(images)
                bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
                labels = clip_features[bool_masked_pos]  # supervise all tokens (default) or only the masked tokens

        if loss_scaler is None:
            outputs = model(samples, bool_masked_pos=bool_masked_pos)
        else:
            with torch.cuda.amp.autocast():  # enabled=False
                outputs = model(samples, bool_masked_pos=bool_masked_pos)

        loss = compute_loss(outputs, labels)

        loss_value = loss.item()

        loss_list = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, loss)
        loss_list = torch.tensor(loss_list)

        all_loss_mean_value = loss_list.mean().item()
        metric_logger.update(all_loss_mean=all_loss_mean_value)

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at rank {utils.get_rank()}", force=True)
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
            loss_scale_value, grad_norm = get_loss_scale_for_deepspeed(model)
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(),
                                    create_graph=is_second_order,
                                    update_freq=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
            loss_scale_value = loss_scaler.state_dict()['scale']

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        #metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            #log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
