"""Exp199: standard DIoU regression, inherited Exp141 full-train recipe."""
from yolox_x_uavswarmv1_full import Exp as BaseExp

class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolox_x_uavswarmv1_diou"

    def get_model(self):
        model = super().get_model()
        model.head.iou_loss.loss_type = "diou"
        return model
