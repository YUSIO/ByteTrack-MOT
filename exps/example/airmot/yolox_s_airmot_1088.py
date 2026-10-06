# encoding: utf-8
"""YOLOX-S at 1088x1088 on the AIRMOT official-train temporal detector split (Exp053).

Same model and recipe as exps/example/uavswarm/yolox_s_uavswarmv2_1088.py (Exp043); only the
data differ. Paper-stated values (Chu et al. 2025, Sec. 4.3): YOLOX, 1088x1088, 50 epochs,
batch 2, SGD lr 0.004, momentum 0.9, weight decay 0.0005, COCO pre-training, NMS 0.7.
Not stated by the paper and chosen here: the S variant, warmup/no-aug epochs, multi-scale
range, EMA, mixed precision, validation confidence, checkpoint criterion and the
train/validation split inside the official training sequences.
"""
import os

import torch
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.seed = 0

        # <data_root>/annotations/{train,val}.json, <data_root>/train -> AIRMOT images/train
        self.data_root = os.path.join(get_yolox_datadir(), "airmot")
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.data_num_workers = 8

        self.input_size = (1088, 1088)
        self.test_size = (1088, 1088)
        self.random_size = (29, 39)  # 928..1248, +-5 strides of 32 around 1088

        self.max_epoch = 50
        self.warmup_epochs = 5
        self.no_aug_epochs = 15
        self.basic_lr_per_img = 0.004 / 2.0  # lr 0.004 at the paper's batch size 2
        self.weight_decay = 5e-4
        self.momentum = 0.9

        self.print_interval = 50
        self.eval_interval = 5
        self.test_conf = 0.001
        self.nmsthre = 0.7

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = MOTDataset(
            data_dir=self.data_root,
            json_file=self.train_ann,
            name="train",
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )
        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        return DataLoader(self.dataset, **dataloader_kwargs)

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        # Validation frames are the temporal tail of the official training sequences.
        valdataset = MOTDataset(
            data_dir=self.data_root,
            json_file=self.val_ann,
            name="train",
            img_size=self.test_size,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        return torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        return COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
