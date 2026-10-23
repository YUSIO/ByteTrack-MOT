"""YOLOX-X experiment config for SportsMOT.

Architecture of the MixSort model-zoo SportsMOT detectors
(github.com/MCG-NJU/MixSort): yolox_x_sports_train (train split, used for
all validation runs) and yolox_x_sports_mix (train+val, used for test).
Single class (person), test size 800x1440.
"""

import os

from yolox.exp import Exp as BaseExp


class Exp(BaseExp):
    """YOLOX-X SportsMOT experiment configuration."""

    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.test_conf = 0.001
        self.nmsthre = 0.7
