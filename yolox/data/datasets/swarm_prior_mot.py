"""MOTDataset that appends swarm-inferred position priors to each image's label rows (Exp064).

The object rows are untouched. Prior rows store their gap k >= 1 in the class column, so the
existing mosaic / affine / mixup / flip / resize code moves them together with the image. The trajectories come
from each sequence's gt/gt.txt; only frames before the current one are used.
"""
import os
import random

import numpy as np

from ..swarm_prior import TRAIN_CFG, load_mot_tracks, sample_training_priors
from .mot import MOTDataset


class SwarmPriorMOTDataset(MOTDataset):
    def __init__(self, *args, prior_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_cfg = dict(TRAIN_CFG if prior_cfg is None else prior_cfg)
        self.tracks = {}
        for _, info, file_name in self.annotations:
            seq = file_name.split("/")[0]
            if seq not in self.tracks:
                self.tracks[seq] = load_mot_tracks(os.path.join(self.data_dir, self.name, seq, "gt", "gt.txt"))

    def prior_rows(self, index):
        _, (height, width, frame_id, _, file_name), _ = self.annotations[index]
        return sample_training_priors(self.tracks[file_name.split("/")[0]], int(frame_id), width, height, random, self.prior_cfg)

    def pull_item(self, index):
        img, res, img_info, img_id = super().pull_item(index)
        return img, np.vstack((res, self.prior_rows(index))), img_info, img_id
