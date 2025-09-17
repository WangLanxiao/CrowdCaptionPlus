# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY
import torch.distributed as dist
from xmodaler.modeling.layers.point_generator import MlvlPointGenerator
from functools import partial
from torch.nn import functional as F
from xmodaler.losses.fcos_loss import GenTargets,LOSS,coords_fmap2orig

@LOSSES_REGISTRY.register()
class JMDCCrossEntropyRL(nn.Module):
    @configurable
    def __init__(self,
        *,
        max_seq_len: int,
        max_bbox_len: int,
        bbox_size: int,
        reg_per_img: int,
        seq_per_img: int,
        **kwargs):
        super(JMDCCrossEntropyRL, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.max_seq_len = max_seq_len
        self.max_bbox_len = max_bbox_len
        self.bbox_size = bbox_size
        self.center_sample_radius = 1.5
        self.reg_per_img = reg_per_img
        self.seq_per_img = seq_per_img
        self.h=12
        self.w=12
    @classmethod
    def from_config(cls, cfg):
        kwargs = {
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            "max_bbox_len": cfg.MODEL.MAX_BBOX_LEN,
            "bbox_size": cfg.MODEL.BBOX_SIZE,
            "reg_per_img": cfg.DATALOADER.REG_PER_SAMPLE,
            "seq_per_img": cfg.DATALOADER.SEQ_PER_SAMPLE,
        }
        return kwargs

    @classmethod
    def add_config(cls, cfg):
        pass

    def changeview_CE(self, result, target):
        logits = result.reshape(-1, result.shape[-1])
        targets = target.reshape(-1).long()
        loss = self.criterion(logits, targets)
        return loss

    def forward(self, outputs_dict):
        ret  = {}

        # pc_count_targets = outputs_dict[kfg.G_TARGET_COUNT_IDS]
        # pc_count_results=outputs_dict[kfg.PC_RESULTS]
        # loss_count = self.changeview_CE(pc_count_results, pc_count_targets)

        # cap_targets = outputs_dict[kfg.G_TARGET_CAP_IDS]
        # cap_results = outputs_dict[kfg.G_LOGITS_CAP][:,:-1]
        # loss_cap = self.changeview_CE(cap_results, cap_targets.reshape(cap_results.shape[:-1]))

        bbox_targets = outputs_dict[kfg.G_TARGET_BBOX_IDS]
        bbox_results = outputs_dict[kfg.G_LOGITS_BBOX]
        loss_bbox = -self.changeview_CE(bbox_results, bbox_targets)



        ret.update({'Bbox Loss': loss_bbox})
        return ret
