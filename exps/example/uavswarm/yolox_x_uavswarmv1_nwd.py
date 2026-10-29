"""Exp198: frozen Exp141 recipe with pure NWD bbox regression."""
from yolox_x_uavswarmv1_full import Exp as BaseExp
from yolox.models.nwd_loss import NWDloss


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolox_x_uavswarmv1_nwd"
        self.nwd_constant = 12.8

    def get_model(self):
        model = super().get_model()
        model.head.iou_loss = NWDloss(self.nwd_constant, reduction="none")
        return model
