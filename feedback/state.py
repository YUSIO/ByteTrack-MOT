"""Causal observation buffers. Inference has no GT or future-frame argument."""
from collections import deque
import numpy as np


class History:
    def __init__(self, window=8):
        if window < 2:
            raise ValueError('History window must be >= 2')
        self.window = window
        self.reset()

    def reset(self):
        self.frame = 0
        self.frames = deque(maxlen=self.window)

    def append(self, frame, observations):
        """observations: ID -> current matched detector xywh, including unconfirmed."""
        if frame != self.frame + 1:
            raise ValueError('Frames must be consecutive; reset at every sequence boundary')
        copy = {}
        for key, value in observations.items():
            box = np.asarray(value, dtype=np.float32).copy()
            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0):
                raise ValueError('Invalid observed box')
            copy[int(key)] = box
        self.frames.append(copy)
        self.frame = frame

    def reliable(self):
        if len(self.frames) < self.window:
            return np.empty((0, self.window, 2), np.float32), []
        ids = sorted(set.intersection(*(set(f) for f in self.frames)))
        paths = [[f[i][:2] + f[i][2:] / 2 for f in self.frames] for i in ids]
        return np.asarray(paths, np.float32).reshape(-1, self.window, 2), ids


def observe_detector_boxes():
    """Read-only observation hook, equivalent to Exp141/fctrack wrap_stracks.

    The underlying update, association, Kalman state and lifecycle are unaltered.
    """
    from yolox.tracker.byte_tracker import STrack
    if getattr(STrack, '_feedback_observed', False):
        return
    update, reactivate, activate = STrack.update, STrack.re_activate, STrack.activate

    def upd(self, new, frame):
        self.feedback_box = new._tlwh.copy()
        return update(self, new, frame)

    def rea(self, new, frame, new_id=False):
        self.feedback_box = new._tlwh.copy()
        return reactivate(self, new, frame, new_id)

    def act(self, kalman, frame):
        self.feedback_box = self._tlwh.copy()
        return activate(self, kalman, frame)

    STrack.update, STrack.re_activate, STrack.activate = upd, rea, act
    STrack._feedback_observed = True
