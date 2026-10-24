"""Exp162 common B0 prime: fit28-only X, fixed epoch80 EMA selection.

UAVSWARM_V1_ROOT must name the prepared fit28 view, not the original dataset.
Training-set monitoring uses the same fit28 annotation. No dev/test evaluation.
"""
import hashlib
import json
from pathlib import Path

from exps.example.uavswarm.yolox_x_uavswarmv1_full import Exp as FullTrainExp

FIT_SEQUENCES = [f'UAVSwarm-{n:02d}' for n in
                 (5, 7, 9, 11, 13, 15, 17, 19, 23, 25, 27, 29, 31, 33,
                  35, 37, 41, 43, 45, 47, 51, 53, 55, 57, 59, 61, 65, 67)]
EXPECTED_ANNOTATION_SHA256 = '8a6ba83e22e87cf75064dfca364ff12d704fa5faedbf45ec263fee3b5a1101d7'
COCO_SHA256 = '6ad4cc404c047ba882042debe946928207dc3c10692c1c7817780ca921625161'


def file_sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class Exp(FullTrainExp):
    def __init__(self):
        super().__init__()
        self.exp_name = 'yolox_x_exp162_b0_fit28'
        self.train_ann = self.val_ann = 'fit28.json'
        self.validate_partition()

    def validate_partition(self):
        root = Path(self.data_dir)
        marker = json.loads((root / 'split_identity.json').read_text())
        if marker['partition'] != 'fit28' or marker['sequences'] != FIT_SEQUENCES:
            raise ValueError('B0 prime requires the frozen fit28 partition')
        if marker['images'] != 4653 or marker['annotations'] != 36030:
            raise ValueError('Incorrect fit28 counts')
        if self.train_ann != 'fit28.json' or self.val_ann != 'fit28.json':
            raise ValueError('Training and monitoring must both use fit28')
        if marker['annotation_sha256'] != EXPECTED_ANNOTATION_SHA256:
            raise ValueError('Split marker annotation identity mismatch')
        if file_sha256(root / 'annotations/fit28.json') != EXPECTED_ANNOTATION_SHA256:
            raise ValueError('fit28 annotation checksum mismatch')
        if sorted(p.name for p in (root / 'train').iterdir()) != sorted(FIT_SEQUENCES):
            raise ValueError('Data view exposes sequences outside fit28')

    def merge(self, cfg_list):
        # Prevent CLI options from silently restoring full-train or dev monitoring.
        if cfg_list:
            if len(cfg_list) % 2 or any(k != 'output_dir' for k in cfg_list[::2]):
                raise ValueError('Only output_dir may be overridden for this frozen recipe')
            super().merge(cfg_list)

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        self.validate_partition()
        if batch_size != 4 or is_distributed:
            raise ValueError('Frozen B0 prime recipe is single-device batch4')
        return super().get_data_loader(batch_size, is_distributed, no_aug)

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        if testdev:
            raise ValueError('Test access is prohibited in this development protocol')
        self.validate_partition()
        return super().get_eval_loader(batch_size, is_distributed, testdev=False)

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        if testdev:
            raise ValueError('Test access is prohibited in this development protocol')
        return super().get_evaluator(batch_size, is_distributed, testdev=False)
