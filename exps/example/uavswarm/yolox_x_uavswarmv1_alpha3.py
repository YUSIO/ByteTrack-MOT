"""Exp197: Exp141 full-train recipe with unnormalized Alpha-IoU, alpha=3."""
from yolox_x_uavswarmv1_full import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolox_x_uavswarmv1_alpha3"
        self.iou_alpha = 3

    def get_model(self):
        model = super().get_model()
        model.head.iou_loss.iou_alpha = self.iou_alpha
        return model
