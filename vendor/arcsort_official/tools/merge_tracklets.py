"""Offline tracklet association post-processing (GTATrack-style).

Operates on finished tracking results: optionally SPLITS tracks at
appearance discontinuities (identity contamination inside long tracks),
then MERGES tracklet pairs that are temporally disjoint, motion-
compatible, and appearance-similar. Appearance comes from a per-sequence
embedding cache (tools/cache_embeddings.py / convert_deep_eiou.py);
tracker output boxes are KF posteriors, so cache lookups use IoU-nearest
matching instead of exact coordinates.

Run BEFORE interpolation (merged gaps can then be filled by
tools/interpolate_tracks.py).

Usage:
    PYTHONPATH=. python tools/merge_tracklets.py \
        --input results/sportsmot_test_arcsort \
        --emb-cache emb_cache/sportsmot_test_osnet \
        --output results/sportsmot_test_arcsort_merged \
        --merge-thresh 0.6 --max-gap 150 --split
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger


@dataclass
class Tracklet:
    """A single track's per-frame boxes and appearance prototype."""

    track_id: int
    frames: list[int] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)   # xyxy
    embs: list[np.ndarray] = field(default_factory=list)    # per-frame or empty

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    def prototype(self) -> np.ndarray | None:
        """Mean of L2-normalized per-frame embeddings, renormalized."""
        if not self.embs:
            return None
        proto = np.mean(self.embs, axis=0)
        norm = np.linalg.norm(proto)
        return proto / norm if norm > 0 else None

    def terminal_velocity(self, k: int = 5) -> np.ndarray:
        """Mean per-frame center velocity over the last k observations."""
        if len(self.frames) < 2:
            return np.zeros(2)
        idx = max(0, len(self.frames) - k)
        c0 = _center(self.boxes[idx])
        c1 = _center(self.boxes[-1])
        dt = self.frames[-1] - self.frames[idx]
        return (c1 - c0) / max(dt, 1)


def _center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def _diag(box: np.ndarray) -> float:
    return float(np.hypot(box[2] - box[0], box[3] - box[1]))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def load_tracks(txt_file: Path) -> dict[int, Tracklet]:
    """Parse a MOT result file into Tracklets (embeddings not yet set)."""
    tracks: dict[int, Tracklet] = {}
    data = np.loadtxt(txt_file, delimiter=",", ndmin=2)
    if data.size == 0:
        return tracks
    order = np.lexsort((data[:, 0],))
    for row in data[order]:
        frame, tid = int(row[0]), int(row[1])
        x, y, w, h = row[2:6]
        box = np.array([x, y, x + w, y + h])
        tracks.setdefault(tid, Tracklet(tid))
        tracks[tid].frames.append(frame)
        tracks[tid].boxes.append(box)
    return tracks


def load_emb_cache(npz_file: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load {frame: (boxes, normalized embs)} from a cache npz."""
    data = np.load(str(npz_file))
    cache = {}
    for key in data.files:
        if not key.startswith("boxes_"):
            continue
        fid = int(key.split("_")[1])
        boxes = data[key].astype(np.float64)
        embs = data[f"embs_{fid}"].astype(np.float32)
        if embs.ndim > 2:
            embs = embs.reshape(embs.shape[0], -1)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        cache[fid] = (boxes, embs / np.maximum(norms, 1e-12))
    return cache


def attach_embeddings(
    tracks: dict[int, Tracklet],
    cache: dict[int, tuple[np.ndarray, np.ndarray]],
    min_iou: float = 0.5,
) -> None:
    """Attach per-frame embeddings by IoU-nearest detection match.

    Tracker outputs are KF posteriors (and possibly interpolated), so an
    exact-coordinate lookup would miss; instead take the cached detection
    with highest IoU >= min_iou for each output box.
    """
    for track in tracks.values():
        for frame, box in zip(track.frames, track.boxes):
            entry = cache.get(frame)
            if entry is None:
                continue
            det_boxes, det_embs = entry
            if det_boxes.shape[0] == 0:
                continue
            ious = np.array([_iou(box, db) for db in det_boxes])
            j = int(np.argmax(ious))
            if ious[j] >= min_iou:
                track.embs.append(det_embs[j])


def split_tracklets(
    tracks: dict[int, Tracklet],
    cache: dict[int, tuple[np.ndarray, np.ndarray]],
    split_thresh: float = 0.45,
    persist: int = 5,
    min_iou: float = 0.5,
) -> dict[int, Tracklet]:
    """Split tracks at sustained appearance discontinuities.

    Maintains an EMA appearance prototype along each track; when the
    per-frame embedding's cosine to the prototype stays below
    split_thresh for `persist` consecutive observed frames, the track is
    split at the first low-similarity frame.

    Returns a new dict with fresh sequential IDs.
    """
    out: dict[int, Tracklet] = {}
    next_id = 1

    for track in tracks.values():
        segments: list[tuple[list[int], list[np.ndarray]]] = []
        cur_frames: list[int] = []
        cur_boxes: list[np.ndarray] = []
        proto: np.ndarray | None = None
        low_run = 0
        split_at: int | None = None

        for i, (frame, box) in enumerate(zip(track.frames, track.boxes)):
            emb = None
            entry = cache.get(frame)
            if entry is not None and entry[0].shape[0] > 0:
                det_boxes, det_embs = entry
                ious = np.array([_iou(box, db) for db in det_boxes])
                j = int(np.argmax(ious))
                if ious[j] >= min_iou:
                    emb = det_embs[j]

            if emb is not None and proto is not None:
                cos = float(np.dot(proto, emb))
                if cos < split_thresh:
                    low_run += 1
                    if split_at is None:
                        split_at = len(cur_frames)
                else:
                    low_run = 0
                    split_at = None

                if low_run >= persist and split_at is not None and split_at > 0:
                    segments.append((cur_frames[:split_at], cur_boxes[:split_at]))
                    cur_frames = cur_frames[split_at:]
                    cur_boxes = cur_boxes[split_at:]
                    proto = None
                    low_run = 0
                    split_at = None

            cur_frames.append(frame)
            cur_boxes.append(box)
            if emb is not None:
                if proto is None:
                    proto = emb.copy()
                else:
                    proto = 0.9 * proto + 0.1 * emb
                    proto /= max(np.linalg.norm(proto), 1e-12)

        segments.append((cur_frames, cur_boxes))
        for frames, boxes in segments:
            if not frames:
                continue
            t = Tracklet(next_id, frames=frames, boxes=boxes)
            out[next_id] = t
            next_id += 1

    return out


def merge_tracklets(
    tracks: dict[int, Tracklet],
    merge_thresh: float = 0.6,
    max_gap: int = 150,
    motion_tol: float = 2.0,
    min_len: int = 3,
) -> dict[int, int]:
    """Greedily merge disjoint, motion-compatible, similar tracklets.

    A pair (A ends, then B starts within max_gap) is a candidate when:
    - both have appearance prototypes and cosine >= merge_thresh
    - A's terminal-velocity extrapolation to B's start frame lands
      within motion_tol * diag(B_start) of B's start center

    Candidates are merged best-similarity-first; each tracklet's tail
    and head can be used at most once, and chains resolve via union.

    Returns:
        Mapping old track_id -> merged (root) track_id.
    """
    ids = [t for t in tracks if len(tracks[t].frames) >= min_len]
    protos = {t: tracks[t].prototype() for t in ids}

    candidates = []
    for a in ids:
        if protos[a] is None:
            continue
        for b in ids:
            if a == b or protos[b] is None:
                continue
            ta, tb = tracks[a], tracks[b]
            gap = tb.start - ta.end
            if not (1 <= gap <= max_gap):
                continue
            cos = float(np.dot(protos[a], protos[b]))
            if cos < merge_thresh:
                continue
            pred = _center(ta.boxes[-1]) + ta.terminal_velocity() * gap
            dist = float(np.linalg.norm(pred - _center(tb.boxes[0])))
            if dist > motion_tol * _diag(tb.boxes[0]):
                continue
            candidates.append((cos, a, b))

    candidates.sort(reverse=True)
    parent = {t: t for t in tracks}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tail_used: set[int] = set()
    head_used: set[int] = set()
    for cos, a, b in candidates:
        if a in tail_used or b in head_used:
            continue
        if find(a) == find(b):
            continue
        parent[find(b)] = find(a)
        tail_used.add(a)
        head_used.add(b)

    return {t: find(t) for t in tracks}


def process_sequence(
    txt_file: Path,
    emb_file: Path | None,
    out_file: Path,
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    """Run split+merge on one sequence. Returns (in_ids, split_ids, out_ids)."""
    tracks = load_tracks(txt_file)
    n_in = len(tracks)

    cache = load_emb_cache(emb_file) if emb_file and emb_file.exists() else {}
    if not cache:
        logger.warning(f"{txt_file.stem}: no embedding cache; copying unchanged")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(txt_file.read_text())
        return n_in, n_in, n_in

    if args.split:
        tracks = split_tracklets(
            tracks, cache, split_thresh=args.split_thresh,
            persist=args.split_persist,
        )
    n_split = len(tracks)

    attach_embeddings(tracks, cache)
    mapping = merge_tracklets(
        tracks, merge_thresh=args.merge_thresh, max_gap=args.max_gap,
        motion_tol=args.motion_tol,
    )

    # Re-emit MOT lines with merged IDs (compact renumbering)
    roots = sorted(set(mapping.values()))
    renumber = {r: i + 1 for i, r in enumerate(roots)}
    lines = []
    for tid, track in tracks.items():
        new_id = renumber[mapping[tid]]
        for frame, box in zip(track.frames, track.boxes):
            w, h = box[2] - box[0], box[3] - box[1]
            lines.append(
                f"{frame},{new_id},{box[0]:.2f},{box[1]:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
            )
    lines.sort(key=lambda ln: (int(ln.split(",")[0]), int(ln.split(",")[1])))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + ("\n" if lines else ""))
    return n_in, n_split, len(roots)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline tracklet association")
    parser.add_argument("--input", required=True,
                        help="Directory of MOT result txt files")
    parser.add_argument("--emb-cache", required=True,
                        help="Directory of per-sequence embedding .npz caches")
    parser.add_argument("--output", required=True)
    parser.add_argument("--merge-thresh", type=float, default=0.6,
                        help="Min prototype cosine to merge (default 0.6)")
    parser.add_argument("--max-gap", type=int, default=150,
                        help="Max frame gap between tracklets (default 150)")
    parser.add_argument("--motion-tol", type=float, default=2.0,
                        help="Max extrapolation error in units of box diagonal")
    parser.add_argument("--split", action="store_true",
                        help="Split tracks at appearance discontinuities first")
    parser.add_argument("--split-thresh", type=float, default=0.45)
    parser.add_argument("--split-persist", type=int, default=5)
    args = parser.parse_args()

    in_dir = Path(args.input)
    emb_dir = Path(args.emb_cache)
    out_dir = Path(args.output)
    # Sequence results only -- skip TrackEval artifacts living alongside
    txt_files = [
        f for f in sorted(in_dir.glob("*.txt"))
        if not f.name.startswith("pedestrian_")
    ]
    if not txt_files:
        raise FileNotFoundError(f"No result txt files in {in_dir}")

    for txt_file in txt_files:
        seq = txt_file.stem
        emb_file = emb_dir / f"{seq}.npz"
        n_in, n_split, n_out = process_sequence(
            txt_file, emb_file, out_dir / txt_file.name, args
        )
        logger.info(f"{seq}: {n_in} ids -> split {n_split} -> merged {n_out}")

    logger.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
