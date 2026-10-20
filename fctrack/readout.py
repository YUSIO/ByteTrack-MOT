"""Exp130: the Exp129 read-out head on a frozen FairMOT backbone, used as the fill-in decision of FCTrack.

The read-out answers the same question as the crop verifier (is the slot's member near the estimated position, and where is
its box) but on the detector's own features: the template is the track's last real detection (frame and box), the query is
the estimated box. FairMOT code (backbone, read-out head) is imported from a FairMOT checkout or export given by its path;
backbone outputs are cached per frame for the current sequence.
"""
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_FAIRMOT_ROOT = None  # FairMOT's modules are imported once per process: every read-out of a run must use the same checkout


class Readout:
    def __init__(self, fairmot_root, backbone, head, device, use_template=True, cache_frames=48, arch="v1"):
        """arch v1: Exp129 head; v2: Exp131 head (stride-2 base features, sub-cell offset), needs a FairMOT checkout with readout_head_v2."""
        here = Path(__file__).resolve().parent  # the fctrack directory is on sys.path when run.py is started as a script;
        sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]  # its tracker.py would shadow FairMOT's tracker package
        global _FAIRMOT_ROOT
        assert _FAIRMOT_ROOT in (None, Path(fairmot_root).resolve()), "all read-outs of one run must share one FairMOT checkout"
        _FAIRMOT_ROOT = Path(fairmot_root).resolve()
        tools = Path(fairmot_root) / "tools"
        if str(tools) not in sys.path:
            sys.path.insert(0, str(tools))
        import torch
        import exp128_dense_export as E          # puts FairMOT src/lib on sys.path; TF32 off
        from exp129_train_head import load_frame
        from models import readout_head as R
        self.V2 = None
        if arch == "v2":
            from models import readout_head_v2 as V2
            self.V2 = V2
        self.torch, self.R, self.load_frame = torch, R, load_frame
        self.device = torch.device(device)
        _, self.model, _, _ = E.load_model(Path(backbone), self.device)
        self.head = (self.V2.ReadoutHeadV2() if self.V2 else R.ReadoutHead()).to(self.device)
        self.head.load_state_dict(torch.load(head, map_location=self.device))
        self.head.eval()
        self.use_template, self.cache_frames = use_template, cache_frames
        self.seq, self.cache = None, OrderedDict()

    def begin(self, seq_dir):
        self.seq, self.cache = SimpleNamespace(dir=Path(seq_dir)), OrderedDict()

    def _frame(self, f):
        if f in self.cache:
            self.cache.move_to_end(f)
            return self.cache[f]
        img, lb = self.load_frame(self.seq, f)
        x = self.torch.from_numpy(img)[None].to(self.device)
        if self.V2:
            feat, idm, fine = self.V2.backbone_forward(self.model, x)
        else:
            (feat, idm), fine = self.R.backbone_forward(self.model, x), None
        self.cache[f] = (feat, idm, lb, fine)
        while len(self.cache) > self.cache_frames:
            self.cache.popitem(last=False)
        return self.cache[f]

    def __call__(self, frame, boxes, templates):
        """boxes: n x 4 estimated tlwh in `frame`; templates: n x (frame, tlwh) of each track's last real detection.
        Returns (p, regressed tlwh)."""
        torch, R = self.torch, self.R
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        if not len(boxes):
            return np.zeros(0), np.zeros((0, 4))
        with torch.no_grad():
            frames = sorted({int(frame)} | {int(f0) for f0, _ in templates})
            banks = [self._frame(f) for f in frames]
            index = {f: k for k, f in enumerate(frames)}
            feat = torch.cat([b[0] for b in banks])
            idm = torch.cat([b[1] for b in banks])
            lb = torch.tensor([b[2] for b in banks], dtype=torch.float32, device=self.device)
            dev = self.device
            idx_t = torch.full((len(boxes),), index[int(frame)], device=dev)
            idx_a = torch.tensor([index[int(f0)] for f0, _ in templates], device=dev)
            tb = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for _, b in templates], float)
            pred = boxes[:, :2] + boxes[:, 2:] / 2
            s0 = np.maximum(tb[:, 2] - tb[:, 0], tb[:, 3] - tb[:, 1])
            age = np.array([frame - f0 for f0, _ in templates], float)
            T = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
            pred_t, s0_t, lt, la = T(pred), T(s0), lb[idx_t], lb[idx_a]
            if self.V2:
                fine = torch.cat([b[3] for b in banks])
                inputs = self.V2.read_inputs(feat, idm, fine, idx_t, feat, idm, idx_a, T(tb), pred_t, s0_t,
                                             lt[:, 0], lt[:, 1], lt[:, 2], la[:, 0], la[:, 1], la[:, 2])
            else:
                inputs = R.read_inputs(feat, idm, idx_t, feat, idm, idx_a, T(tb), pred_t, s0_t,
                                       lt[:, 0], lt[:, 1], lt[:, 2], la[:, 0], la[:, 1], la[:, 2])
            out = self.head(*inputs, R.scalars(s0_t, T(age)), use_template=self.use_template)
            p, box = (self.V2 or R).decode(out, pred_t, s0_t)
        p, box = p.cpu().numpy().astype(float), box.cpu().numpy().astype(float)
        return p, np.column_stack((box[:, 0], box[:, 1], box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]))
