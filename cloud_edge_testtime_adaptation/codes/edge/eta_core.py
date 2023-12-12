import copy
import functools
from typing import Callable
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import functools
import logging
import types
from statistics import mode
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type


def prepare_model_for_tent_tta(model: nn.Module, is_tbr=False):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # m.eval()
            m.requires_grad_(True)
            # # force use of batch stats in train and eval modes
            if not is_tbr:
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.eval()
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    return model


def set_bn_training_mode(model: nn.Module, mode: bool):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train(mode=mode)
    return model


def tbr_bn_forward_impl(self: nn.BatchNorm2d, x: torch.Tensor):

    batch_var, batch_mean = torch.var_mean(x, dim=(0, 2, 3), keepdim=True)
    batch_std = torch.sqrt(batch_var+self.eps)

    if self.running_mean is None:
        self.running_mean, self.running_var = batch_mean.clone(
        ).detach(), batch_var.clone().detach()

    self.running_mean, self.running_var = self.running_mean.view(
        1, -1, 1, 1), self.running_var.view(1, -1, 1, 1)

    r = batch_std.detach() / torch.sqrt(self.running_var+self.eps)
    d = (batch_mean.detach() - self.running_mean) / \
        torch.sqrt(self.running_var+self.eps)
    x = ((x - batch_mean) / batch_std) * r + d

    if self.training:
        self.running_mean += self.momentum * \
            (batch_mean.detach() - self.running_mean)
        self.running_var += self.momentum * \
            (batch_var.detach() - self.running_var)
    else:
        pass

    x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)

    return x


BN_MODULE_TYPES: Tuple[Type[nn.Module]] = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)


def get_bn_from_model(model: nn.Module):
    return [
        (name, bn) for name, bn in model.named_modules() if isinstance(bn, BN_MODULE_TYPES)
    ]


def replace_bn_forward_with(model: nn.Module, fn):
    for name, bn in get_bn_from_model(model):
        bn: nn.BatchNorm2d
        bn.forward = types.MethodType(fn, bn)


def entropy(
    logits: torch.Tensor,
    reduction="none",
):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    losses = -p_log_p.sum(-1)
    if reduction == "none":
        return losses
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "mean":
        return losses.mean()
    else:
        raise ValueError(f"The parameter 'reduction' must be in ['none','mean','sum'], bot got {reduction}")


class EntropyCriterion:
    def __init__(self, ent_high_margin: float):
        self.ent_high_margin = ent_high_margin

        self.n_cnt_total = 0
        self.n_cnt_remain = 0
        self.n_cnt_filter = 0

        self.n_filter_samples = 0
        self.n_remain_samples = 0
        self.n_total_samples = 0

    def filter_out(self, logits: torch.Tensor):
        with torch.no_grad():
            batch_size, *_ = logits.shape
            ents = entropy(logits, reduction="none")
            remain_ids, *_ = torch.where(ents < self.ent_high_margin)

            self.n_cnt_total = batch_size
            self.n_cnt_remain = remain_ids.numel()
            self.n_cnt_filter = self.n_cnt_total - self.n_cnt_remain

            self.n_total_samples += self.n_cnt_total
            self.n_remain_samples += self.n_cnt_remain
            self.n_filter_samples += self.n_cnt_filter

        return remain_ids


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


class SimilarityCriterion:
    def __init__(self, history_model_probs: torch.Tensor = None, d_margin: float = None):
        self.history_model_probs = history_model_probs
        self.d_margin = d_margin

        self.n_cnt_total = 0
        self.n_cnt_remain = 0
        self.n_cnt_filter = 0

        self.n_filter_samples = 0
        self.n_remain_samples = 0
        self.n_total_samples = 0

    def filter_out(self, logits: torch.Tensor):
        with torch.no_grad():
            batch_size, *_ = logits.shape

            if self.history_model_probs is not None:
                cosine_similarities = F.cosine_similarity(self.history_model_probs.unsqueeze(dim=0), logits.softmax(1), dim=1)
                remain_ids, *_ = torch.where(torch.abs(cosine_similarities) < self.d_margin)
                self.history_model_probs = update_model_probs(self.history_model_probs, logits[remain_ids].softmax(1))
            else:
                self.history_model_probs = update_model_probs(self.history_model_probs, logits.softmax(1))
                remain_ids = torch.arange(batch_size, device=logits.device)

            if remain_ids is not None:
                self.n_cnt_total = batch_size
                self.n_cnt_remain = remain_ids.numel()
                self.n_cnt_filter = self.n_cnt_total - self.n_cnt_remain

                self.n_total_samples += self.n_cnt_total
                self.n_remain_samples += self.n_cnt_remain
                self.n_filter_samples += self.n_cnt_filter

            return remain_ids
