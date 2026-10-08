# encoding: utf-8
"""Control for yolox_s_uavswarmv2_1088_prior.py (Exp064): same model, data stream and schedule, prior map kept at zero."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from yolox_s_uavswarmv2_1088_prior import Exp as PriorExp  # noqa: E402


class Exp(PriorExp):
    use_prior = False

    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
