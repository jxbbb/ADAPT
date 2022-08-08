# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right

import torch
import math
from .LARC import LARC


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        if isinstance(optimizer, LARC):
            optimizer = optimizer.optim
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iter,
        min_lr=0,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [
                base_lr * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            return [
                self.min_lr + (base_lr - self.min_lr) *
                (1 + math.cos(math.pi * self.last_epoch / self.max_iter)) / 2
                for base_lr in self.base_lrs
            ]





class WarmupLinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iter,
        min_lr=1e-8,
        warmup_ratio=0.1,
        last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.warmup_ratio = warmup_ratio
        self.warmup_iters = int(warmup_ratio*max_iter)
        super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr_factor(self):
        tot_step = self.max_iter
        warmup_step = self.warmup_iters
        step = self.last_epoch
        if step < warmup_step:
            return max(0, step / warmup_step)
        return max(0, (tot_step-step)/(tot_step-warmup_step))

    def get_lr(self):
        warmup_factor = self.get_lr_factor()
        return [
            max(self.min_lr, base_lr * warmup_factor)
            for base_lr in self.base_lrs
        ]


"""
optimizer learning rate scheduling helpers

Copied from ClipBERT
supports linear/invsqrt/constant/multi_step
"""
def noam_schedule(step, warmup_step=4000):
    if step <= warmup_step:
        return step / warmup_step
    return (warmup_step ** 0.5) * (step ** -0.5)


def warmup_linear(step, warmup_step, tot_step):
    if step < warmup_step:
        return step / warmup_step
    return max(0, (tot_step-step)/(tot_step-warmup_step))


def multi_step_schedule(n_epoch, milestones, gamma=0.5):
    milestones = list(sorted(milestones))
    for i, m in enumerate(milestones):
        if n_epoch < m:
            return gamma**i
    return gamma**(len(milestones)+1)


def get_lr_sched(global_step, decay, learning_rate,
                 num_train_steps, warmup_ratio=0.1,
                 decay_epochs=[], multi_step_epoch=-1):
    warmup_steps = int(warmup_ratio*num_train_steps)
    if decay == 'linear':
        lr_this_step = learning_rate * warmup_linear(
            global_step, warmup_steps, num_train_steps)
    elif decay == 'invsqrt':
        lr_this_step = learning_rate * noam_schedule(
            global_step, warmup_steps)
    elif decay == 'constant':
        lr_this_step = learning_rate
    elif decay == "multi_step":
        assert multi_step_epoch >= 0
        lr_this_step = learning_rate * multi_step_schedule(
            multi_step_epoch, decay_epochs)
    if lr_this_step <= 0:
        # save guard for possible miscalculation of train steps
        lr_this_step = 1e-8
    return lr_this_step
