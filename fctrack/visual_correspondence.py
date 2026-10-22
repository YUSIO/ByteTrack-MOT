"""Causal endpoint proposals from the unmodified, frozen CoTracker3 online model.

A call at f reads only anchor..f, resets the model, and returns only f's coordinate.
Its visibility flag is logged, never treated as UAV presence. A's verifier remains
responsible for accepting an output. No detector execution or GT access occurs.
"""
from collections import Counter, OrderedDict
import configparser
import hashlib
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

WEIGHTS_SHA256 = '205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218'
SOURCE_SHA256 = {
    'cotracker/predictor.py': '783536d6c77790fa6c8e005f2df1d6bd4f0d8955c2c67d464bfb4e64d366375f',
    'cotracker/models/core/cotracker/cotracker3_online.py': '4c8bfce5546d85ba22c6b5292b870ad2aa0af1e0b6ba5a22e644c39bb496475e',
}
_MODELS = {}


def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_predictor(spec, device):
    """Explicit local files only; no torch.hub network fallback."""
    repo = Path(spec['repository']).expanduser().resolve()
    weights = Path(spec['checkpoint']).expanduser().resolve()
    if spec['weights_sha256'] != WEIGHTS_SHA256 or _sha(weights) != WEIGHTS_SHA256:
        raise ValueError('CoTracker checkpoint identity mismatch')
    for name, digest in SOURCE_SHA256.items():
        if _sha(repo / name) != digest:
            raise ValueError('CoTracker source identity mismatch: ' + name)
    key = (str(repo), str(weights), str(device))
    if key not in _MODELS:
        sys.path.insert(0, str(repo))
        import cotracker.predictor
        if Path(cotracker.predictor.__file__).resolve() != repo / 'cotracker/predictor.py':
            raise RuntimeError('a different CoTracker package is already imported')
        model = cotracker.predictor.CoTrackerOnlinePredictor(
            checkpoint=str(weights), window_len=16, v2=False).to(device).eval()
        if tuple(model.interp_shape) != (384, 512) or model.step != 8:
            raise ValueError('unexpected model dimensions')
        _MODELS[key] = model
    return _MODELS[key]


class VisualCorrespondence:
    """Proposal callback (frame, track) -> original-image xy or None.

    Required track fields are _visual_anchor_frame and _visual_anchor_box (tlwh),
    written together only by genuine detection updates, not verifier recovery.
    rawdet follows MOT [frame,id,x,y,w,h,score,...]. Predictor injection is for
    CPU unit tests and sequential sharing across sequences; the formal runner
    first constructs the default frozen model, then reuses its predictor.
    """
    def __init__(self, spec, seqpath, rawdet, device, predictor=None):
        self.spec = spec
        self.seqpath = Path(seqpath)
        self.device = torch.device(device)
        self.model = predictor if predictor is not None else load_predictor(spec, device)
        self.max_history = int(spec.get('max_history', 30))
        self.crop_wh = tuple(spec.get('crop_wh', [512, 384]))
        self.max_support = int(spec.get('max_support', 32))
        self.high_thresh = float(spec.get('support_threshold', .6))
        if self.max_history < 1 or self.max_support < 0 or min(self.crop_wh) < 2:
            raise ValueError('invalid visual correspondence limits')
        info = configparser.ConfigParser()
        if not info.read(self.seqpath / 'seqinfo.ini'):
            raise FileNotFoundError(self.seqpath / 'seqinfo.ini')
        section = info['Sequence']
        self.ext = section.get('imExt', '.jpg')
        self.image_dir = section.get('imDir', 'img1')
        self.det = np.asarray(rawdet, dtype=float).reshape(-1, rawdet.shape[1])
        if self.det.shape[1] < 7:
            raise ValueError('MOT detection rows require at least seven columns')
        self.cache = OrderedDict()
        self.stats, self.records = Counter(), []

    @property
    def predictor(self):
        """The stateful model, reusable by sequential per-sequence providers."""
        return self.model

    def _read_frame(self, frame, current):
        if frame < 1 or frame > current:
            raise ValueError('non-causal image request')
        if frame not in self.cache:
            path = self.seqpath / self.image_dir / f'{frame:06d}{self.ext}'
            im = cv2.imread(str(path))
            if im is None:
                raise FileNotFoundError(path)
            self.cache[frame] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            while len(self.cache) > self.max_history + 1:
                self.cache.popitem(last=False)
        self.cache.move_to_end(frame)
        return self.cache[frame]

    def _run(self, clip, queries):
        video = torch.from_numpy(clip).permute(0, 3, 1, 2)[None].float().to(self.device)
        q = np.column_stack((np.zeros(len(queries)), queries))
        q = torch.as_tensor(q, dtype=torch.float32, device=self.device)[None]
        total, step = video.shape[1], self.model.step
        chunks = [0] if total <= 2 * step else range(0, total - step, step)
        with torch.inference_mode():
            self.model(video_chunk=video[:, :2 * step], is_first_step=True,
                       queries=q, add_support_grid=True)
            tracks = visible = None
            for start in chunks:
                tracks, visible = self.model(video_chunk=video[:, start:start + 2 * step],
                                              queries=q, add_support_grid=True)
            if tracks.shape[1] != total:
                raise RuntimeError('unexpected CoTracker endpoint length')
            endpoint = tracks[0, total - 1, 0].detach().cpu().numpy().astype(float)
            visibility = bool(visible[0, total - 1, 0].detach().cpu())
        return endpoint, visibility

    def __call__(self, frame, track):
        start = time.perf_counter()
        frame = int(frame)
        anchor = getattr(track, '_visual_anchor_frame', None)
        box = getattr(track, '_visual_anchor_box', None)
        record = {'frame': frame, 'track_id_at_query': int(track.track_id),
                  'anchor_frame': int(anchor) if anchor is not None else None,
                  'latest_input_frame': None, 'fallback': None}
        self.stats['calls'] += 1
        try:
            if anchor is None or box is None:
                record['fallback'] = 'missing_anchor'
                return None
            anchor = int(anchor)
            if not 1 <= frame - anchor <= self.max_history:
                record['fallback'] = 'anchor_age'
                return None
            box = np.asarray(box, dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) <= 0:
                record['fallback'] = 'invalid_anchor'
                return None
            seed = box[:2] + box[2:] / 2
            first = self._read_frame(anchor, frame)
            h, w = first.shape[:2]
            if not (0 <= seed[0] < w and 0 <= seed[1] < h):
                record['fallback'] = 'anchor_outside'
                return None
            cw, ch = min(self.crop_wh[0], w), min(self.crop_wh[1], h)
            x0 = int(np.clip(round(seed[0] - cw / 2), 0, w - cw))
            y0 = int(np.clip(round(seed[1] - ch / 2), 0, h - ch))
            origin = np.array([x0, y0])
            # Support queries come only from frozen detections at the anchor.
            rows = self.det[(self.det[:, 0] == anchor) & (self.det[:, 6] > self.high_thresh)]
            centers = rows[:, 2:4] + rows[:, 4:6] / 2
            valid = (np.isfinite(centers).all(1) & (centers[:, 0] >= x0) &
                     (centers[:, 0] < x0 + cw) & (centers[:, 1] >= y0) &
                     (centers[:, 1] < y0 + ch))
            centers = centers[valid]
            distance = np.linalg.norm(centers - seed, axis=1)
            # Stable ordering, and exclude the seed's own detection within 1px.
            order = np.argsort(distance, kind='stable')
            order = order[distance[order] > 1.0][:self.max_support]
            queries = np.vstack([seed, centers[order]]) - origin
            indices = list(range(anchor, frame + 1))
            clip = np.stack([self._read_frame(i, frame)[y0:y0 + ch, x0:x0 + cw]
                             for i in indices])
            record.update(anchor_box=box.tolist(), seed_xy=seed.tolist(),
                          crop_xywh=[x0, y0, cw, ch], input_frames=indices,
                          latest_input_frame=indices[-1], support_detection_count=len(order),
                          support_grid_size=6, no_future_frames=max(indices) <= frame)
            endpoint, visible = self._run(clip, queries)
            point = endpoint + origin
            record.update(endpoint_xy=point.tolist(), visibility_confidence_flag=visible)
            self.stats['model_calls'] += 1
            self.stats['visibility_true'] += int(visible)
            # A lost point outside its fixed crop is not a new image observation.
            if not np.isfinite(point).all():
                record['fallback'] = 'nonfinite_endpoint'
                return None
            if not (x0 <= point[0] < x0 + cw and y0 <= point[1] < y0 + ch):
                record['fallback'] = 'escaped_crop'
                return None
            self.stats['valid_proposals'] += 1
            return point
        finally:
            elapsed = time.perf_counter() - start
            record['seconds'] = elapsed
            self.stats['seconds'] += elapsed
            if record['fallback'] is not None:
                self.stats['fallback_' + record['fallback']] += 1
            self.records.append(record)
