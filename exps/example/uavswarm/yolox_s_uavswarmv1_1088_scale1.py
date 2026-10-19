# encoding: utf-8
"""YOLOX-S at 1088x1088 on UAVSwarm V1 with the mosaic scale jitter switched off (Exp088, protocol v2).

Identical to yolox_s_uavswarmv1_1088.py except that the random scale of the mosaic affine is fixed
at 1 instead of YOLOX's default 0.1-2. Exp088 run_001 (default range) diverged in epoch 3; the
diagnosis groups (diag_001) trained stably with fixed scale 1 and with 0.5-1.5. Mosaic, mixup,
rotation, shear and translation are unchanged.
"""
import os

from yolox_s_uavswarmv1_1088 import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.scale = (1.0, 1.0)
