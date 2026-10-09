"""Birth policies for BYTETracker (Exp065): how an unconfirmed track is predicted, gated and confirmed.

Upstream behaviour: a track started from an unmatched box with score >= det_thresh keeps its start box as its
prediction (zero velocity), is matched in the next frame against the boxes left over by the first association with
IoU * score > 0.3, and is removed when that fails. `args.birth` (a dict) switches the pieces below; with it absent the
tracker runs the upstream code path unchanged.

  shift             'none' | 'own' | 'swarm' | 'swarm_local' | 'gmc'   where the unconfirmed track is predicted to be
                    own:         moved by the track's own last displacement (zero until it has two observations)
                    swarm:       start box moved by the median displacement of the tracks matched in both frames
                    swarm_local: moved by the mean displacement of the `k` nearest such tracks
                    gmc:         moved by the camera-motion transform of the frame (tracker.camera_motion[frame])
  inherit_velocity  the applied displacement also becomes the track's Kalman velocity (default True)
  buffer            b > 0 enlarges both boxes by b * (w, h) on every side before the IoU (buffered IoU)
  use_score         multiply the IoU by the box score, as upstream (default True)
  gate              the similarity must exceed this value (default 0.3)
  instant           a track started from a box with score >= det_thresh is output from its first frame
  weak_low          boxes with weak_low < score < det_thresh that no track took may start a hidden track ...
  weak_hits         ... which is output after this many consecutive matches (default 2) ...
  confirm_pool      'high' (default, as upstream) or 'all': whether unconfirmed tracks may also match those weak boxes
  weak_free_iou     ... unless the box overlaps a track updated in this frame by more than this IoU (default 0.3)

`tracker.birth_hook` (optional, supplied by an experiment, e.g. a ground-truth oracle) may edit the confirmation
assignment and decide individual births; see BirthHook.
"""
import numpy as np

from yolox.tracker import matching


class BirthHook:
    """Interface of tracker.birth_hook. Every method may be left as is."""

    def confirm(self, tracker, unconfirmed, pool, matches, u_unconfirmed, u_detection):
        return matches, u_unconfirmed, u_detection

    def birth(self, tracker, det, strong):
        """'default' applies the configured rule, 'now' starts a track that is output at once, 'skip' starts none."""
        return 'default'


def centre(tlwh):
    return np.asarray(tlwh[:2], float) + np.asarray(tlwh[2:4], float) / 2


def shifts(tracks, anchors, cfg, camera_motion):
    """Displacement applied to each unconfirmed track; anchors is a list of (previous centre, displacement)."""
    mode = cfg.get('shift', 'none')
    out = np.zeros((len(tracks), 2))
    if mode == 'none' or len(tracks) == 0:
        return out
    if mode == 'gmc':
        if camera_motion is None:
            return out
        a = np.asarray(camera_motion, float).reshape(2, 3)
        for i, t in enumerate(tracks):
            c = centre(t.tlwh)
            out[i] = a[:, :2] @ c + a[:, 2] - c
        return out
    if mode == 'own':
        for i, t in enumerate(tracks):
            out[i] = getattr(t, 'last_disp', 0.0)
        return out
    if len(anchors) < cfg.get('min_anchors', 1):
        return out
    pos = np.array([a[0] for a in anchors])
    disp = np.array([a[1] for a in anchors])
    if mode == 'swarm':
        out[:] = np.median(disp, axis=0)
    elif mode == 'swarm_local':
        k = cfg.get('k', 3)
        for i, t in enumerate(tracks):
            near = np.argsort(np.linalg.norm(pos - centre(t.tlwh), axis=1))[:k]
            out[i] = disp[near].mean(axis=0)
    else:
        raise ValueError('unknown birth shift: {}'.format(mode))
    return out


def similarity(tracks, dets, cfg):
    """IoU (optionally buffered, optionally times the box score) between unconfirmed tracks and candidate boxes."""
    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(tracks), len(dets)))
    a = np.array([t.tlbr for t in tracks], float)
    b = np.array([d.tlbr for d in dets], float)
    buf = cfg.get('buffer', 0.0)
    if buf > 0:
        for x in (a, b):
            w, h = x[:, 2] - x[:, 0], x[:, 3] - x[:, 1]
            x[:, 0] -= buf * w
            x[:, 2] += buf * w
            x[:, 1] -= buf * h
            x[:, 3] += buf * h
    sim = np.asarray(matching.ious(np.ascontiguousarray(a), np.ascontiguousarray(b)), float)
    if cfg.get('use_score', True):
        sim = sim * np.array([d.score for d in dets])[None, :]
    return sim
