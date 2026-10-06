# encoding: utf-8
"""YOLOX-S at 1088x1088 on AIRMOT with the mosaic scale jitter switched off (Exp053, protocol v2).

Identical to yolox_s_airmot_1088.py except that the random scale of the mosaic affine is fixed
at 1 instead of YOLOX's default 0.1-2. AIRMOT targets are about 9 px at this input size; scaled
down, a large share falls under the 2 px label filter of random_perspective while the objects
stay in the image, and training with the default range did not converge (Exp053 run_001 and
its diagnosis). Mosaic, mixup, rotation, shear and translation are unchanged.
"""
import os

from yolox_s_airmot_1088 import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.scale = (1.0, 1.0)
