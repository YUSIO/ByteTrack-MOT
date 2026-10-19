"""Detector wrappers for the in-loop runner: one class per detector family, all with the same three methods.

  det.begin_sequence();  d = det.detect(img_bgr)  -> n x 5 float32 (x1, y1, x2, y2, score) in image pixels, rounded like the cache text;
  det.update(rows, d)    -> feed the tracked frame back to the score adapter.
The adapter (fctrack.score_adapter.OnlineScoreAdapter) is attached to the final 1 x 1 scoring convolutions of the detector, so a
new YOLO-type detector only needs the list of those layers and the original-pixel -> input-pixel mapping of its preprocessing.
"""
import numpy as np
import torch

from fctrack.score_adapter import OnlineScoreAdapter

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _cache_round(b, sc):
    """Same text rounding as the exporters' det.txt (x y w h with 4 decimals, score with 6), then the runner's conversion."""
    x1, y1 = np.round(b[:, 0], 4), np.round(b[:, 1], 4)
    w, h = np.round(b[:, 2] - b[:, 0], 4), np.round(b[:, 3] - b[:, 1], 4)
    return np.column_stack((x1, y1, x1 + w, y1 + h, np.round(sc, 6))).astype(np.float32)


class YoloxDetector:
    """YOLOX (this repository's yolox package): hooks on head.obj_preds[k]; objectness is the adapted score."""

    def __init__(self, exp_file, ckpt, device, adapt, size=(1088, 1088), conf=0.001, nms=0.7):
        from yolox.exp import get_exp
        exp = get_exp(str(exp_file), None)
        self.model = exp.get_model()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.exp, self.device, self.size, self.conf, self.nms = exp, device, tuple(size), conf, nms
        self.adapter = OnlineScoreAdapter([self.model.head.obj_preds[k] for k in range(len(self.model.head.obj_preds))], adapt).attach()
        self.geo = None

    def begin_sequence(self):
        self.adapter.begin_sequence()

    @property
    def stats(self):
        return self.adapter.stats

    @torch.no_grad()
    def detect(self, img):
        from yolox.data.data_augment import preproc
        from yolox.utils import postprocess
        x, r = preproc(img, self.size, MEAN, STD)
        out = postprocess(self.model(torch.from_numpy(x).unsqueeze(0).to(self.device)), self.exp.num_classes, self.conf, self.nms)[0]
        self.geo = {"scale": r, "pad": (0.0, 0.0), "input_wh": (self.size[1], self.size[0])}
        if out is None:
            return np.zeros((0, 5), np.float32)
        o = out.cpu().numpy()
        return _cache_round(o[:, :4] / r, o[:, 4] * o[:, 5])

    def update(self, rows, dets):
        self.adapter.update(rows, dets, self.geo)


class UltralyticsDetector:
    """Ultralytics YOLO (v8 / 11 / 26, single class): hooks on the last conv of every classification branch (cv3[k][-1], or
    one2one_cv3 for the NMS-free YOLO26 head); predict() does the letterbox, the forward pass and the post-processing."""

    def __init__(self, weights, device, adapt, imgsz=1280, conf=0.001, iou=0.7, max_det=300):
        from ultralytics import YOLO
        self.yolo = YOLO(str(weights))
        self.net = self.yolo.model.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        det = self.net.model[-1]
        branch = getattr(det, "one2one_cv3", None) if getattr(det, "end2end", False) else det.cv3
        self.adapter = OnlineScoreAdapter([seq[-1] for seq in branch], adapt).attach()
        self.device, self.imgsz, self.conf, self.iou, self.max_det = device, imgsz, conf, iou, max_det
        self.geo = None

    def begin_sequence(self):
        self.adapter.begin_sequence()

    @property
    def stats(self):
        return self.adapter.stats

    @torch.no_grad()
    def detect(self, img):
        h, w = img.shape[:2]
        res = self.yolo.predict(img, imgsz=self.imgsz, conf=self.conf, iou=self.iou, max_det=self.max_det, verbose=False, device=str(self.device), half=False)[0]
        f0 = self.adapter.feat[0]
        in_h, in_w = f0.shape[-2] * 8, f0.shape[-1] * 8
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = round(w * r), round(h * r)
        self.geo = {"scale": r, "pad": ((in_w - nw) / 2.0, (in_h - nh) / 2.0), "input_wh": (in_w, in_h)}
        b = res.boxes
        if b is None or len(b) == 0:
            return np.zeros((0, 5), np.float32)
        return _cache_round(b.xyxy.cpu().numpy().astype(np.float64), b.conf.cpu().numpy().astype(np.float64))

    def update(self, rows, dets):
        self.adapter.update(rows, dets, self.geo)
