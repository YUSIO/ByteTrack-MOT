# encoding: utf-8
"""YOLOX-S at 1088x1088 on the UAVSwarm V1 official-train temporal detector split (Exp088).

Identical model and recipe to yolox_s_uavswarmv2_1088.py (Exp043); only the data root differs.
The json files are converted from the frozen V1 YOLO26s split (Exp008): per official training
sequence of L frames, frames 1..floor(L/2)+1 are train and the rest validation, boxes clipped
to the image. <data_root>/train links to the V1 official `train` directory.
"""
import os

from yolox_s_uavswarmv2_1088 import Exp as BaseExp
from yolox.data import get_yolox_datadir


class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_root = os.path.join(get_yolox_datadir(), "uavswarmv1")
