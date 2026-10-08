# encoding: utf-8
"""YOLOX-S 1088 on UAVSwarmV2 with a swarm-inferred position-prior map injected at stride 8 (Exp064).

A short fine-tune that starts from the Exp043 checkpoint. Data, split, input size, multi-scale range,
augmentation, optimiser and losses are those of yolox_s_uavswarmv2_1088.py; only the schedule length and the
learning rate are reduced. The control config (..._prior_control.py) differs only in `use_prior`.
"""
import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from yolox_s_uavswarmv2_1088 import Exp as BaseExp  # noqa: E402


class Exp(BaseExp):
    use_prior = True

    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.max_epoch = 15
        self.warmup_epochs = 1
        self.no_aug_epochs = 5
        self.basic_lr_per_img = 0.002 / 2.0  # lr 0.002 at batch size 2: half of the Exp043 rate
        self.eval_interval = 5

    def get_model(self):
        from yolox.models import PriorYOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)
            self.model = PriorYOLOX(backbone, head, channels=int(in_channels[0] * self.width), use_prior=self.use_prior)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        import torch.distributed as dist
        from yolox.data import DataLoader, InfiniteSampler, MosaicDetection, TrainTransform, YoloBatchSampler
        from yolox.data.datasets import SwarmPriorMOTDataset

        means, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        dataset = SwarmPriorMOTDataset(
            data_dir=self.data_root, json_file=self.train_ann, name="train", img_size=self.input_size,
            preproc=TrainTransform(rgb_means=means, std=std, max_labels=1000),
        )
        dataset = MosaicDetection(
            dataset, mosaic=not no_aug, img_size=self.input_size,
            preproc=TrainTransform(rgb_means=means, std=std, max_labels=2000),
            degrees=self.degrees, translate=self.translate, scale=self.scale, shear=self.shear,
            perspective=self.perspective, enable_mixup=self.enable_mixup,
        )
        self.dataset = dataset
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False,
                                         input_dimension=self.input_size, mosaic=not no_aug)
        return DataLoader(self.dataset, num_workers=self.data_num_workers, pin_memory=True, batch_sampler=batch_sampler)
