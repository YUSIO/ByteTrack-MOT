"""The YOLOX-S detector whose features the track head shares (Exp071).

Images are letterboxed exactly as `yolox.data.data_augment.preproc` does (resize by r = min(S / h, S / w), pad the
bottom and right with 114, BGR -> RGB, /255, ImageNet mean and std), but the normalisation runs on the GPU so that
loader workers only move uint8 arrays. `features` returns the stride-8 neck output P3 and the detector's own
stride-8 objectness map; `detect` additionally decodes boxes as tools/export_uavswarm_detections.py does.
"""
import cv2
import numpy as np
import torch

from yolox.exp import get_exp
from yolox.utils import postprocess

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SIZE = 1088
STRIDE = 8


def letterbox(img, size=SIZE):
    """uint8 BGR image -> (uint8 size x size x 3 padded image, ratio)."""
    r = min(size / img.shape[0], size / img.shape[1])
    out = np.full((size, size, 3), 114, np.uint8)
    rs = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR)
    out[: rs.shape[0], : rs.shape[1]] = rs
    return out, r


def normalise(batch_u8):
    """B x H x W x 3 uint8 BGR (tensor on the device) -> B x 3 x H x W float RGB, normalised."""
    x = batch_u8.permute(0, 3, 1, 2).flip(1).float() / 255.0
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def load_detector(exp_file, ckpt, device):
    """exp_file may start with "{code}/", the root of this code export."""
    from pathlib import Path
    exp_file = str(exp_file).replace("{code}", str(Path(__file__).resolve().parents[1]))
    exp = get_exp(exp_file, None)
    model = exp.get_model()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.head.decode_in_inference = False
    return exp, model.to(device).eval()


def objectness_level0(head, p3):
    """The detector's objectness logit at stride 8 (the same arithmetic as YOLOXHead on level 0)."""
    x = head.stems[0](p3)
    return head.obj_preds[0](head.reg_convs[0](x))


def decode(raw, hw, strides):
    """Same arithmetic as YOLOXHead.decode_outputs (copied from tools/export_uavswarm_detections.py)."""
    grids, expanded = [], []
    for (h, w), stride in zip(hw, strides):
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing="ij")
        grids.append(torch.stack((xv, yv), 2).view(1, -1, 2))
        expanded.append(torch.full((1, h * w, 1), float(stride)))
    grids = torch.cat(grids, dim=1).to(raw)
    expanded = torch.cat(expanded, dim=1).to(raw)
    out = raw.clone()
    out[..., :2] = (raw[..., :2] + grids) * expanded
    out[..., 2:4] = torch.exp(raw[..., 2:4]) * expanded
    return out


@torch.no_grad()
def features(model, x):
    """x: normalised batch. Returns (P3, objectness logit map at stride 8)."""
    fpn = model.backbone(x)
    return fpn[0], objectness_level0(model.head, fpn[0])


@torch.no_grad()
def detect(model, x, conf=0.001, nms=0.7):
    """Returns (P3, objectness logits, list of n x 5 arrays (x1, y1, x2, y2, score) in network-input pixels)."""
    fpn = model.backbone(x)
    raw = model.head(fpn).float()
    outs = postprocess(decode(raw, model.head.hw, model.head.strides), 1, conf, nms)
    dets = [np.zeros((0, 5), np.float32) if o is None else torch.cat((o[:, :4], (o[:, 4] * o[:, 5])[:, None]), 1).cpu().numpy() for o in outs]
    return fpn[0], objectness_level0(model.head, fpn[0]), dets
