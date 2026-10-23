"""YOLOX-X experiment config for DanceTrack.

Matches OC-SORT's YOLOX-X trained on CrowdHuman + DanceTrack.
Single class (person), test size 800x1440.
"""

import os

from yolox.exp import Exp as BaseExp


class Exp(BaseExp):
    """YOLOX-X DanceTrack experiment configuration."""

    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.test_conf = 0.1
        self.nmsthre = 0.7
