# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import re
import torch
from .LARC import LARC

from .lr_scheduler import WarmupMultiStepLR
from .lr_scheduler import WarmupCosineAnnealingLR
from .optimization import AdamW
from .optimization import WarmupLinearSchedule


def make_optimizer(cfg, model, resume=False):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        
        for reg_lr in cfg.SOLVER.REGEXP_LR_FACTOR:
            regexp, lr_factor = reg_lr
            if re.match(regexp, key):
                if lr != cfg.SOLVER.BASE_LR:
                    print("WARNING: {} matched multiple "
                          "regular expressions!".format(key))
                lr *= lr_factor
        
        if "bias" in key:
            lr *= cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        if resume:
            params += [{"params": [value], "initial_lr": lr, "lr": lr, "weight_decay": weight_decay}]
        else:
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}] 

    if cfg.SOLVER.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(params)
    elif cfg.SOLVER.OPTIMIZER == 'adamw':
        # optimizer = torch.optim.AdamW(params)
        if hasattr(cfg, 'adam_epsilon'):
            optimizer = AdamW(params, eps=cfg.adam_epsilon)
        else:
            optimizer = AdamW(params)
    else:
        raise ValueError(
            'Optimizer "{}" is not supported'.format(cfg.SOLVER.OPTIMIZER)
        )
    if cfg.SOLVER.USE_LARC:
        optimizer = LARC(optimizer, clip=True, trust_coefficient=cfg.SOLVER.LARC_COEFFICIENT)
    return optimizer


def make_lr_scheduler(cfg, optimizer, last_iter=-1):
    lr_policy = cfg.SOLVER.LR_POLICY
    if lr_policy not in ("multistep", "cosine", 'linear'):
        raise ValueError(
            "Only 'multistep' or 'cosine' lr policy is accepted"
            "got {}".format(lr_policy)
        )
    if lr_policy == "multistep":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            last_epoch=last_iter
        )
    elif lr_policy == "cosine":
        return WarmupCosineAnnealingLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            cfg.SOLVER.MIN_LR,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            last_epoch=last_iter
        )
    elif lr_policy == "linear":
        return WarmupLinearSchedule(
            optimizer,
            warmup_steps=cfg.SOLVER.WARMUP_ITERS,
            t_total=cfg.SOLVER.MAX_ITER,
        )
