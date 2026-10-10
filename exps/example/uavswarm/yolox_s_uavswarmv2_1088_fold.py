# encoding: utf-8
"""YOLOX-S 1088 on one sequence-disjoint fold of UAVSwarmV2 MOT-train (Exp066).

Same recipe as yolox_s_uavswarmv2_1088.py (Exp043). The only difference is the data: the detector is trained on every
frame of the sequences of one fold and never sees the other fold's videos, so that its outputs on those videos look
like outputs on unseen test videos. UAVSWARM_FOLD selects the fold: <data_root>/annotations/fold_<x>_train.json and
fold_<x>_val.json (a sparse sample of the other fold, for monitoring only; the last checkpoint is used).
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location("exp043_base", os.path.join(os.path.dirname(os.path.realpath(__file__)), "yolox_s_uavswarmv2_1088.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class Exp(_base.Exp):
    def __init__(self):
        super(Exp, self).__init__()
        fold = os.environ["UAVSWARM_FOLD"]
        self.exp_name = "yolox_s_uavswarmv2_1088_fold_" + fold
        self.train_ann = "fold_{}_train.json".format(fold)
        self.val_ann = "fold_{}_val.json".format(fold)
        self.eval_interval = 10
