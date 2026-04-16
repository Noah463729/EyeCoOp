import math
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from utils.lr_sched import adjust_learning_rate
from utils.logger import MetricLogger, SmoothedValue
from utils.eval_single import compute_metrics, compute_classwise_metrics, print_result
import warnings


warnings.filterwarnings("ignore", category=FutureWarning, module="timm.layers")


def train_one_epoch(
    model: nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    tea_model: nn.Module,  
):
    model.train()
    if tea_model is not None:
        tea_model.eval()
        for p in tea_model.parameters():
            p.requires_grad = False

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce', SmoothedValue(window_size=20, fmt='{avg:.4f}'))
    metric_logger.add_meter('loss_sccm', SmoothedValue(window_size=20, fmt='{avg:.4f}'))
    metric_logger.add_meter('loss_kdsp', SmoothedValue(window_size=20, fmt='{avg:.4f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=20, fmt='{avg:.4f}'))

    header = f'Epoch: [{epoch}]'
    accum_iter = getattr(args, 'accum_iter', 1)

    scaler = torch.cuda.amp.GradScaler(enabled=getattr(args, 'amp', True) and torch.cuda.is_available())

    model.zero_grad(set_to_none=True)

    for data_iter_step, (samples, targets, _) in enumerate(
        metric_logger.log_every(data_loader, getattr(args, 'print_freq', 20), header)
    ):
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)


        (f_img) = samples
        (f_label) = targets

        
        f_img = f_img.to(device, non_blocking=True)
        f_label = f_label.to(device, non_blocking=True)

        teacher_img_size = int(getattr(args, "teacher_input_size", 224))
        if (f_img.shape[-2] != teacher_img_size) or (f_img.shape[-1] != teacher_img_size):
            try:
                teacher_img = F.interpolate(
                    f_img, size=(teacher_img_size, teacher_img_size),
                    mode="bilinear", align_corners=False, antialias=True
                )
            except TypeError:
                teacher_img = F.interpolate(
                    f_img, size=(teacher_img_size, teacher_img_size),
                    mode="bilinear", align_corners=False
                )
        else:
            teacher_img = f_img

        tau = getattr(args, 'tau', 0.5)
        lambda_sccm = getattr(args, 'lambda_sccm', 1.0)
        lambda_kdsp = getattr(args, 'lambda_kdsp', 1.0)
        temperature = getattr(args, 'temperature', 1.0)

        with torch.cuda.amp.autocast(enabled=getattr(args, 'amp', True) and torch.cuda.is_available()):
            
            logits, loss_ce, loss_sccm, loss_kdsp = model.forward_train(
                fundus_img=f_img,
                label=f_label,
                tea_img=teacher_img,   
                teacher=tea_model,     
                tau=tau,
                sccm_lambda=lambda_sccm,
                kdsp_lambda=lambda_kdsp,
                temperature=temperature,
            )
            loss = loss_ce + loss_sccm + loss_kdsp
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        scaler.scale(loss / accum_iter).backward()
        if (data_iter_step + 1) % accum_iter == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Logging
        metric_logger.update(loss=float(loss.item()))
        metric_logger.update(loss_ce=float(loss_ce.item()))
        metric_logger.update(loss_sccm=float(loss_sccm.item()))
        metric_logger.update(loss_kdsp=float(loss_kdsp.item()))

        max_lr = 0.
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group.get("lr", 0.))
        metric_logger.update(lr=max_lr)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, data_loader, model: nn.Module, device: torch.device):

   
    alldiseases = ['NOR', 'AMD', 'CSC', 'DR', 'GLC', 'MEM', 'MYO', 'RVO', 'WAMD']

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    prediction_decode_list = []
    prediction_prob_list = []
    true_label_decode_list = []
    img_name_list = []

    model.eval()

    for batch in metric_logger.log_every(data_loader, getattr(args, 'print_freq', 20), header):
        images, target, img_names = batch
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=getattr(args, 'amp', True) and torch.cuda.is_available()):
            logits, _ce, _sccm, _kdsp = model.forward_train(
                fundus_img=images,
                label=target,
                #
                teacher=None,
                tau=getattr(args, 'tau', 0.5),
                sccm_lambda=getattr(args, 'lambda_sccm', 1.0),
                kdsp_lambda=getattr(args, 'lambda_kdsp', 1.0),
                temperature=getattr(args, 'temperature', 1.0),
            )

        prediction_prob = torch.softmax(logits, dim=1)
        prediction_decode = torch.argmax(prediction_prob, dim=1)

        prediction_decode_list.extend(prediction_decode.cpu().tolist())
        true_label_decode_list.extend(target.cpu().tolist())
        prediction_prob_list.extend(prediction_prob.cpu().numpy().tolist())
        img_name_list.extend(list(img_names))

    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    prediction_prob_list = np.array(prediction_prob_list)

    # Metrics
    results = compute_metrics(true_label_decode_list, prediction_decode_list)
    class_wise_results = compute_classwise_metrics(true_label_decode_list, prediction_decode_list)

    print_result(class_wise_results, results, alldiseases)
    metric_logger.meters['kappa'].update(results['kappa'].item())
    metric_logger.meters['f1_pr'].update(results['f1'].item())
    metric_logger.meters['acc'].update(results['accuracy'].item())
    metric_logger.meters['precision'].update(results['precision'].item())
    metric_logger.meters['recall'].update(results['recall'].item())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, results['f1']


