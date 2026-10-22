"""Read-only per-frame diagnostics. No GT, model calls, or tracker-state mutation."""
import gzip
import json
import math
from pathlib import Path

import numpy as np


def plain(value):
    """Snapshot values immediately so later in-place updates cannot change events."""
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class DecisionTrace:
    """One JSON object per frame; uid follows the Python track across ID relabels.

    Detection indices always refer to the frame-local input array, before score filtering.
    Only references held by this observer map objects to uid; no attribute is added to tracks.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = gzip.open(self.path, 'xt', encoding='utf-8')
        self.objects = {}
        self.frame = None
        self.events = []

    def uid(self, track):
        if track not in self.objects:
            self.objects[track] = len(self.objects) + 1
        return self.objects[track]

    def track(self, track, meta=None):
        state = int(track.state)
        row = dict(uid=self.uid(track), track_id=int(track.track_id), state=state,
                   state_name={0: 'New', 1: 'Tracked', 2: 'Lost', 3: 'Removed'}.get(state, str(state)),
                   is_activated=bool(track.is_activated), frame_id=int(track.frame_id),
                   start_frame=int(track.start_frame), end_frame=int(track.end_frame),
                   tlwh=track.tlwh, det_tlwh=getattr(track, '_det', None),
                   mean=track.mean, covariance_diag=None if track.covariance is None else np.diag(track.covariance),
                   score=track.score)
        if meta is not None:
            row['meta'] = meta.get(track.track_id)
        return plain(row)

    def pools(self, tracker, meta=None):
        return {name: [self.track(t, meta) for t in getattr(tracker, name + '_stracks')]
                for name in ('tracked', 'lost', 'removed')}

    def begin(self, frame):
        if self.frame is not None:
            raise RuntimeError('previous diagnostic frame not finished')
        self.frame, self.events = int(frame), []

    def record(self, stage, **payload):
        if self.frame is None:
            raise RuntimeError('diagnostic frame not started')
        self.events.append({'stage': stage, **plain(payload)})

    def association(self, stage, tracks, detections, indices, cost, matches, unmatched_tracks,
                    unmatched_detections, threshold):
        self.record(stage, tracks=[self.track(t) for t in tracks], detection_indices=indices,
                    detection_tlwh=[t.tlwh for t in detections], detection_scores=[t.score for t in detections],
                    cost=cost, matches=matches, unmatched_tracks=unmatched_tracks,
                    unmatched_detections=unmatched_detections, threshold=threshold)

    def finish(self):
        self.stream.write(json.dumps({'schema_version': 1, 'frame': self.frame, 'events': self.events},
                                     separators=(',', ':'), allow_nan=False) + '\n')
        self.frame, self.events = None, []

    def close(self):
        self.stream.close()
