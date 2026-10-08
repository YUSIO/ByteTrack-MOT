"""Swarm-inferred position priors for members of a UAV swarm (Exp064).

A prior says where a member that was last observed `k` frames ago should be in the current frame `t`.
It is causal: only positions up to frame t-1 are used.

    swarm   c_i(t-k) + D(t-k -> t-1) + D(t-2 -> t-1)
            D(a -> b) is the median displacement of the anchors (other members present at both a and b),
            i.e. the member is carried along by the swarm, then the swarm's last one-frame motion is extrapolated.
    cv      c_i(t-k) + k * v_i, v_i = the member's own mean velocity over its last (up to) 5 observed frames
    stale   c_i(t-k)

`tracks` maps frame -> {track id -> np.array([cx, cy, w, h])} in image pixels.
"""
import numpy as np

MAX_GAP = 30


def load_mot_tracks(gt_path):
    """Read a MOTChallenge gt.txt (frame, id, x, y, w, h, ...) into the `tracks` structure."""
    rows = np.loadtxt(gt_path, delimiter=",", ndmin=2)
    tracks = {}
    for frame, tid, x, y, w, h in rows[:, :6]:
        tracks.setdefault(int(frame), {})[int(tid)] = np.array([x + w / 2, y + h / 2, w, h], dtype=np.float64)
    return tracks


def swarm_shift(tracks, a, b, anchors):
    """Median displacement of `anchors` between frames a and b (zeros when a == b or nothing is shared)."""
    if a == b or a not in tracks or b not in tracks:
        return np.zeros(2)
    shared = [j for j in anchors if j in tracks[a] and j in tracks[b]]
    if not shared:
        return np.zeros(2)
    return np.median([tracks[b][j][:2] - tracks[a][j][:2] for j in shared], axis=0)


def own_velocity(tracks, tid, last, span=5):
    for back in range(span, 0, -1):
        if last - back in tracks and tid in tracks[last - back]:
            return (tracks[last][tid][:2] - tracks[last - back][tid][:2]) / back
    return np.zeros(2)


def predict(tracks, frame, tid, k, mode, anchors, noise=None):
    """Prior centre of member `tid` at `frame`, last observed at frame - k. `noise` is added to the last observation."""
    last = frame - k
    centre = tracks[last][tid][:2].copy()
    if noise is not None:
        centre = centre + noise
    if mode == "stale":
        return centre
    if mode == "cv":
        return centre + k * own_velocity(tracks, tid, last)
    if mode == "swarm":
        others = [j for j in anchors if j != tid]
        return centre + swarm_shift(tracks, last, frame - 1, others) + swarm_shift(tracks, frame - 2, frame - 1, others)
    raise ValueError(mode)


def sample_training_priors(tracks, frame, width, height, rng, cfg):
    """Prior rows [x1, y1, x2, y2, k, 0] for one training image.

    Object rows carry class 0 in the fifth column; a prior row carries its gap k >= 1 there, which both marks it
    as a prior and survives the trainer (the track-id column is dropped before the model is called).

    Members of the current frame get a swarm prior from a sampled gap (or none); members that were present earlier
    but are gone now may leave a ghost prior; a few false priors are placed near the swarm.
    """
    if rng.random() < cfg["drop_all"]:
        return np.zeros((0, 6))
    now = tracks.get(frame, {})
    rows = []

    def add(centre, size, k):
        w, h = size
        rows.append([centre[0] - w / 2, centre[1] - h / 2, centre[0] + w / 2, centre[1] + h / 2, float(k), 0.0])

    def gap():
        return 1 if rng.random() < cfg["p_gap1"] else rng.randint(2, MAX_GAP)

    for tid in now:
        if rng.random() < cfg["drop_member"]:
            continue
        k = gap()
        if frame - k not in tracks or tid not in tracks[frame - k]:
            k = 1
            if frame - 1 not in tracks or tid not in tracks[frame - 1]:
                continue
        anchors = [j for j in tracks.get(frame - 1, {}) if rng.random() < cfg["keep_anchor"]]
        size = tracks[frame - k][tid][2:]
        noise = np.array([rng.gauss(0, 1), rng.gauss(0, 1)]) * cfg["noise"] * np.sqrt(size[0] * size[1])
        add(predict(tracks, frame, tid, k, "swarm", anchors, noise), size, k)
    # ghosts: members seen k frames ago that are absent now
    k = gap()
    for tid, box in tracks.get(frame - k, {}).items():
        if tid not in now and rng.random() < cfg["ghost"]:
            add(predict(tracks, frame, tid, k, "swarm", list(tracks.get(frame - 1, {}))), box[2:], k)
    # false priors near the swarm
    if now and rng.random() < cfg["false_image"]:
        boxes = np.array(list(now.values()))
        lo, hi = boxes[:, :2].min(0), boxes[:, :2].max(0)
        pad = 0.25 * np.maximum(hi - lo, 4 * boxes[:, 2:].mean(0))
        for _ in range(rng.randint(1, cfg["false_max"])):
            centre = np.array([rng.uniform(lo[0] - pad[0], hi[0] + pad[0]), rng.uniform(lo[1] - pad[1], hi[1] + pad[1])])
            centre = np.clip(centre, 1, [width - 2, height - 2])
            add(centre, boxes[rng.randrange(len(boxes)), 2:], gap())
    return np.array(rows, dtype=np.float64).reshape(-1, 6)


TRAIN_CFG = {
    "drop_all": 0.1,      # image with no prior at all
    "drop_member": 0.25,  # member with no prior (never tracked)
    "p_gap1": 0.5,        # member tracked in the previous frame
    "keep_anchor": 0.8,   # each other member is an anchor with this probability
    "noise": 0.1,         # std of the last-observation jitter, in units of sqrt(w*h)
    "ghost": 0.5,
    "false_image": 0.3,
    "false_max": 2,
}
