"""Exp140 frozen observation probe; GT is used only by ``label_query``.

The target path is a local-group motion proxy, never a GT/owner trajectory.
All visual arms use the same 64-pixel input and frozen seed blocks 0/1.
``X_shuffled`` is a deterministic 90-degree residual-direction control, not
random temporal shuffling.  Fusion repeats the fixed-position historical crop:
this matches the maximum crop slots, not independent spatial coverage.
"""
import argparse
from collections import OrderedDict
import configparser
import hashlib
import json
from pathlib import Path
import time

import numpy as np

LAGS = (3, 6)
SIZE, ZOOM, POOL = 64, 3.0, 4
QUALITY_FIELDS = ("corner_count", "valid_count", "median_residual_px",
                  "p90_residual_px", "fb_median_px", "identity_fallback_ambiguous")
BASE_GEOMETRY = (
    "logit_p", "log_side", "log_proposal_aspect", "cx_fraction", "cy_fraction",
    "final_dx_side", "final_dy_side", "final_log_w_ratio", "final_log_h_ratio",
    "gap", "gap_available", "since_real", "since_real_available", "log_obs", "obs_available",
    "det_score", "det_score_available", "is_birth", "estimator_offset", "current_valid_fraction",
)
LAG_GEOMETRY = (
    "history_complete", "gmc_complete", "gmc_invertible", "anchor_pairs", "path_available",
    "residual_x_side", "residual_y_side", "residual_norm_side", "residual_spread_side",
    "background_dx_side", "background_dy_side", "target_dx_side", "target_dy_side",
    "path_separation_crop_px", "target_valid_fraction", "background_valid_fraction",
    "rotated_valid_fraction", "gmc_quality_available_fraction",
) + tuple(f"gmc_{name}_{stat}" for name in QUALITY_FIELDS for stat in ("mean", "max"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def apply_affine(matrix, points):
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def load_gmc(path):
    rows = np.loadtxt(path, ndmin=2)
    result = {}
    for row in rows:
        if len(row) != 7 or not np.isfinite(row).all():
            raise ValueError(f"Invalid GMC row in {path}: {row}")
        frame = int(row[0])
        if frame in result or frame != row[0]:
            raise ValueError(f"Duplicate/noninteger GMC frame: {path} {frame}")
        result[frame] = np.vstack((row[1:].reshape(2, 3), [0., 0., 1.]))
    return result


def compose_gmc(warps, start, end):
    """History-to-current composition; absent transforms are flagged identities."""
    if start < 1 or end < start:
        raise ValueError((start, end))
    matrix, complete = np.eye(3), True
    for frame in range(start + 1, end + 1):
        if frame not in warps:
            complete = False
        matrix = warps.get(frame, np.eye(3)) @ matrix
    return matrix, complete


def box_center(box):
    box = np.asarray(box, dtype=np.float64)
    return box[:2] + box[2:] / 2


def crop_grid(center, side):
    """Original-image coordinates of the current seed crop's output pixels."""
    yy, xx = np.mgrid[:SIZE, :SIZE]
    return np.stack((xx - SIZE / 2, yy - SIZE / 2), -1) * (ZOOM * side / SIZE) + center


def valid_grid(grid, width, height):
    return ((grid[..., 0] >= 0) & (grid[..., 0] <= width - 1) &
            (grid[..., 1] >= 0) & (grid[..., 1] <= height - 1))


def inverse_path_grid(grid, matrix, residual):
    """Map every current-grid point through G^-1(x-r), including scale/rotation."""
    return apply_affine(np.linalg.inv(matrix), grid - np.asarray(residual))


def _anchor_dict(row, exclude_handles, exclude_ids):
    out = {}
    for anchor in row["anchors"]:
        handle, tid = int(anchor["handle"]), int(anchor["track_id"])
        if handle in exclude_handles or tid in exclude_ids:
            continue
        if handle in out:
            raise ValueError(f"Duplicate anchor handle {handle}")
        box = np.asarray(anchor["box"], float)
        if len(box) != 4 or not np.isfinite(box).all() or (box[2:] <= 0).any():
            raise ValueError(f"Invalid anchor box {anchor}")
        out[handle] = anchor
    return out


def local_residual(query, current, history, matrix, max_anchors=5, min_anchors=2):
    """Median paired-real-anchor residual, excluding query lineage on both ends."""
    exclude_handles = set(int(x) for x in query.get("exclude_handles", []))
    exclude_ids = set(int(x) for x in query.get("exclude_track_ids", []))
    if query.get("query_handle") is not None:
        exclude_handles.add(int(query["query_handle"]))
    cur = _anchor_dict(current, exclude_handles, exclude_ids)
    past = _anchor_dict(history, exclude_handles, exclude_ids)
    proposal = np.asarray(query["proposal"], float)
    center, side = box_center(proposal), np.sqrt(proposal[2] * proposal[3])
    radius = max(8. * side, 64.)
    pairs = []
    for handle in sorted(cur.keys() & past.keys()):
        now, old = box_center(cur[handle]["box"]), box_center(past[handle]["box"])
        distance = float(np.linalg.norm(now - center))
        if distance <= radius:
            pairs.append((distance, handle, now - apply_affine(matrix, old)))
    pairs.sort(key=lambda item: (item[0], item[1]))
    pairs = pairs[:max_anchors]
    available = len(pairs) >= min_anchors
    if pairs:
        residuals = np.stack([item[2] for item in pairs])
        median = np.median(residuals, axis=0)
        spread = float(np.median(np.linalg.norm(residuals - median, axis=1)))
    else:
        median, spread = np.zeros(2), 0.
    return dict(residual=median if available else np.zeros(2), available=available,
                anchor_count=len(pairs), anchor_handles=[item[1] for item in pairs],
                spread=spread, neighbor_radius_px=float(radius))


def make_paths(query, anchors, warps, lag):
    frame = int(query["frame"])
    proposal = np.asarray(query["proposal"], float)
    center, side = box_center(proposal), float(np.sqrt(proposal[2] * proposal[3]))
    start = frame - lag
    if start < 1:
        return dict(history_complete=False, start=start, lag=lag, gmc_complete=False,
                    gmc_invertible=False, path_available=False, anchor_count=0,
                    anchor_handles=[], residual=np.zeros(2), spread=0., matrix=np.eye(3),
                    target_center=center.copy(), background_center=center.copy(),
                    rotated_center=center.copy(), neighbor_radius_px=max(8. * side, 64.))
    if frame not in anchors or start not in anchors:
        raise ValueError(f"Missing capture frame {query['sequence']} {start}/{frame}")
    matrix, complete = compose_gmc(warps, start, frame)
    invertible = bool(np.isfinite(matrix).all() and abs(np.linalg.det(matrix[:2, :2])) > 1e-8)
    if not invertible:
        matrix = np.eye(3)
    local = local_residual(query, anchors[frame], anchors[start], matrix)
    residual = local["residual"] if invertible else np.zeros(2)
    available = bool(local["available"] and complete and invertible)
    if not available:
        residual = np.zeros(2)
    inv = np.linalg.inv(matrix)
    rotated = np.array([-residual[1], residual[0]])
    return dict(history_complete=True, start=start, lag=lag, gmc_complete=complete,
                gmc_invertible=invertible, path_available=available,
                anchor_count=local["anchor_count"], anchor_handles=local["anchor_handles"],
                residual=residual, spread=local["spread"], matrix=matrix,
                target_center=apply_affine(inv, center - residual),
                background_center=apply_affine(inv, center),
                rotated_center=apply_affine(inv, center - rotated),
                neighbor_radius_px=local["neighbor_radius_px"])


def label_query(query, gt):
    """Canonical GT labels only; this function cannot influence paths/features."""
    gt = np.asarray(gt, float).reshape(-1, 4)
    final, proposal = np.asarray(query["final"], float), np.asarray(query["proposal"], float)
    if not len(gt):
        return dict(quality=0, presence=0, max_iou=0., nearest_center_in_sides=None)
    lt = np.maximum(final[:2], gt[:, :2])
    rb = np.minimum(final[:2] + final[2:], gt[:, :2] + gt[:, 2:])
    inter = np.maximum(rb - lt, 0.).prod(1)
    overlaps = inter / np.maximum(final[2:].prod() + gt[:, 2:].prod(1) - inter, 1e-12)
    side = np.sqrt(proposal[2:].prod())
    distance = np.linalg.norm(gt[:, :2] + gt[:, 2:] / 2 - box_center(proposal), axis=1).min() / side
    return dict(quality=int(overlaps.max() >= .5),
                presence=1 if distance <= 1. else 0 if distance > 1.5 else -1,
                max_iou=float(overlaps.max()), nearest_center_in_sides=float(distance))


def load_quality(path, sequence):
    if path is None:
        return {}, []
    path = Path(path)
    if path.is_dir():
        path = path / f"{sequence}.jsonl"
        if not path.exists():
            return {}, []
    rows = jsonl(path)
    output = {}
    for row in rows:
        if row.get("sequence", sequence) != sequence:
            continue
        frame = int(row["frame"])
        if frame in output:
            raise ValueError(f"Duplicate GMC quality frame {sequence} {frame}")
        output[frame] = row
    return output, [path]


def quality_summary(quality, start, end):
    frames = list(range(max(start + 1, 2), end + 1)) if start >= 1 else []
    rows = [quality[f] for f in frames if f in quality]
    values = []
    for key in QUALITY_FIELDS:
        raw = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
        values.extend((float(np.mean(raw)), float(np.max(raw))) if raw else (0., 0.))
    observed = sum(bool(row.get("quality_observed", all(row.get(key) is not None for key in
                   ("median_residual_px", "p90_residual_px", "fb_median_px")))) for row in rows)
    return observed / max(len(frames), 1), values


def base_geometry(query, width, height, current_valid):
    proposal, final = np.asarray(query["proposal"], float), np.asarray(query["final"], float)
    center, fc = box_center(proposal), box_center(final)
    side = np.sqrt(proposal[2:].prod())
    p = np.clip(float(query["p"]), 1e-6, 1 - 1e-6)
    values = [np.log(p / (1 - p)), np.log(side / 20.), np.log(proposal[2] / proposal[3]),
              center[0] / width, center[1] / height, *((fc - center) / side),
              *np.log(final[2:] / proposal[2:])]
    for key in ("gap", "since_real", "obs", "det_score"):
        available = query.get(key) is not None
        value = float(query[key]) if available else 0.
        values.extend((np.log1p(value) if key == "obs" else value, float(available)))
    values += [float(query["kind"] == "birth"), float(str(query.get("estimator", "")).startswith("offset")),
               float(current_valid)]
    assert len(values) == len(BASE_GEOMETRY)
    return values


def lag_geometry(query, path, fractions, quality):
    proposal = np.asarray(query["proposal"], float)
    center, side = box_center(proposal), np.sqrt(proposal[2:].prod())
    residual = path["residual"]
    separation = np.linalg.norm(path["target_center"] - path["background_center"]) * SIZE / (ZOOM * side)
    qfraction, qvalues = quality_summary(quality, path["start"], int(query["frame"]))
    values = [float(path["history_complete"]), float(path["gmc_complete"]), float(path["gmc_invertible"]),
              float(path["anchor_count"]), float(path["path_available"]), *(residual / side),
              float(np.linalg.norm(residual) / side), path["spread"] / side,
              *((path["background_center"] - center) / side), *((path["target_center"] - center) / side),
              float(separation), *fractions, qfraction, *qvalues]
    assert len(values) == len(LAG_GEOMETRY), (len(values), len(LAG_GEOMETRY))
    return values


def runtime_subgroups(query, paths, current_valid_fraction, fractions):
    """Reporting flags only; never remove queries or change feature vectors."""
    proposal = np.asarray(query["proposal"], float)
    side = np.sqrt(proposal[2:].prod())
    separations = [float(np.linalg.norm(path["target_center"] - path["background_center"]) *
                         SIZE / (ZOOM * side)) for path in paths]
    sampled = {str(path["lag"]): dict(zip(("target", "background", "rotated"), map(float, fraction)))
               for path, fraction in zip(paths, fractions)}
    valid = (all(path["history_complete"] for path in paths) and current_valid_fraction == 1. and
             all(value == 1. for fraction in fractions for value in fraction))
    separated = all(path["path_available"] and distance >= 1.
                    for path, distance in zip(paths, separations))
    return dict(all_path_crops_valid=bool(valid), paths_separated=bool(separated),
                sampled_valid_fractions=sampled,
                path_separation_crop_pixels={str(path["lag"]): distance for path, distance in zip(paths, separations)})


class ImageReader:
    def __init__(self, sequence_dir, info):
        self.root, self.info, self.cache, self.hashes = sequence_dir, info, OrderedDict(), {}

    def read(self, frame):
        import cv2
        if frame in self.cache:
            self.cache.move_to_end(frame)
            return self.cache[frame]
        path = self.root / self.info.get("imdir", "img1") / f"{frame:06d}{self.info.get('imext', '.jpg')}"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        expected = (int(self.info["imheight"]), int(self.info["imwidth"]))
        if image.shape[:2] != expected:
            raise ValueError(f"Image geometry mismatch {path}: {image.shape[:2]} != {expected}")
        if str(path) not in self.hashes:
            self.hashes[str(path)] = sha256(path)
        self.cache[frame] = image
        while len(self.cache) > 9:
            self.cache.popitem(last=False)
        return image


def remap_crop(image, grid):
    import cv2
    crop = cv2.remap(image, grid[..., 0].astype(np.float32), grid[..., 1].astype(np.float32),
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.ascontiguousarray(crop[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.


class FineFeatures:
    def __init__(self, weights, device):
        import torch
        from fctrack.verifier import SeedVerifier
        self.torch, self.device = torch, torch.device(device)
        self.model = SeedVerifier()
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, crops):
        torch = self.torch
        with torch.inference_mode():
            pixels = torch.from_numpy(np.stack(crops)).to(self.device)
            first = self.model.body[0](pixels)
            second = self.model.body[1](first)
            if tuple(first.shape[1:]) != (32, 64, 64) or tuple(second.shape[1:]) != (64, 32, 32):
                raise RuntimeError("Frozen seed fine-feature geometry changed")
            arrays = [x.cpu().numpy() for x in (pixels, first, second)]
        return [tuple(array[i] for array in arrays) for i in range(len(crops))]


def pool_map(values, mask=None, mode="mean"):
    """Pool after computing fine-grid evidence, never before registration distance."""
    values = np.asarray(values)
    if values.ndim == 2:
        values = values[None]
    channels, height, width = values.shape
    if height % POOL or width % POOL:
        raise ValueError(values.shape)
    tiles = values.reshape(channels, POOL, height // POOL, POOL, width // POOL).transpose(0, 1, 3, 2, 4)
    if mask is None:
        result = tiles.mean((-2, -1)) if mode == "mean" else tiles.max((-2, -1))
    else:
        valid = np.asarray(mask, bool).reshape(POOL, height // POOL, POOL, width // POOL).transpose(0, 2, 1, 3)
        if mode == "mean":
            result = (tiles * valid[None]).sum((-2, -1)) / np.maximum(valid.sum((-2, -1))[None], 1)
        else:
            result = np.where(valid[None], tiles, -np.inf).max((-2, -1))
            result = np.where(np.isfinite(result), result, 0.)
    return result.reshape(-1).astype(np.float32)


def resize_valid(mask, size):
    """Require all original crop pixels in the strided cell to be valid."""
    ratio = SIZE // size
    return mask.reshape(size, ratio, size, ratio).all((1, 3))


def spatial_descriptor(features):
    return np.concatenate([pool_map(array) for array in features])


def current_descriptor(features, query, mask):
    """Strong single-frame control: spatial appearance plus final-box ROI/ring."""
    proposal, final = np.asarray(query["proposal"], float), np.asarray(query["final"], float)
    center, side = box_center(proposal), np.sqrt(proposal[2:].prod())
    scale = SIZE / (ZOOM * side)
    lt = (final[:2] - center) * scale + SIZE / 2
    rb = (final[:2] + final[2:] - center) * scale + SIZE / 2
    midpoint, half = (lt + rb) / 2, (rb - lt) / 2
    values = [spatial_descriptor(features)]
    for array in features:
        channels, height, width = array.shape
        # Conv stride-2 features retain the input pixel-zero lattice (padding=1).
        yy, xx = np.mgrid[:height, :width] * (SIZE // height)
        valid = resize_valid(mask, height)
        roi = valid & (xx >= lt[0]) & (xx <= rb[0]) & (yy >= lt[1]) & (yy <= rb[1])
        outer = valid & (np.abs(xx - midpoint[0]) <= 1.5 * half[0]) & (np.abs(yy - midpoint[1]) <= 1.5 * half[1])
        ring = outer & ~roi
        def stats(selection):
            if not selection.any():
                return np.zeros(channels), np.zeros(channels), np.zeros(channels)
            samples = array[:, selection]
            return samples.mean(1), samples.std(1), samples.max(1)
        rm, rs, rx = stats(roi)
        bm, bs, _ = stats(ring)
        values.extend((rm, rs, rx, bm, bs, rm - bm, np.array([roi.mean(), ring.mean()])))
    return np.concatenate(values).astype(np.float32)


def pair_descriptor(current, historical, current_mask, history_mask):
    values = []
    for first, second in zip(current, historical):
        height = first.shape[-1]
        valid = resize_valid(current_mask & history_mask, height)
        energy = .5 * (np.square(first).sum(0) + np.square(second).sum(0))
        residual = np.square(first - second).sum(0) / (energy + 1e-6)
        denominator = np.sqrt(np.square(first).sum(0) * np.square(second).sum(0))
        cosine_distance = 1. - (first * second).sum(0) / np.maximum(denominator, 1e-6)
        cosine_distance = np.where(energy <= 1e-6, 0., cosine_distance)
        for evidence in (residual, cosine_distance):
            values.extend((pool_map(evidence, valid), pool_map(evidence, valid, "max")))
        values.append(pool_map(valid.astype(np.float32)))
    return np.concatenate(values).astype(np.float32)


def visual_vector(current_vector, current_features, current_mask, historical, masks, complete):
    spatial, pair, contrast = [], [], []
    sd = spatial_descriptor(current_features)
    pd = pair_descriptor(current_features, current_features, current_mask, current_mask)
    for k, available in enumerate(complete):
        pairs = []
        for slot in range(2):
            i = 2 * k + slot
            if available:
                spatial.append(spatial_descriptor(historical[i]))
                value = pair_descriptor(current_features, historical[i], current_mask, masks[i])
            else:
                spatial.append(np.zeros_like(sd))
                value = np.zeros_like(pd)
            pair.append(value)
            pairs.append(value)
        contrast.append(pairs[0] - pairs[1])
    return np.concatenate([current_vector, *spatial, *pair, *contrast]).astype(np.float32)


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True, help="probe_capture output containing queries/ and anchors/")
    parser.add_argument("--data-root", type=Path, required=True, help="sequence parent in the same reindexed view")
    parser.add_argument("--gmc", type=Path, required=True, help="old per-sequence GMC .txt cache")
    parser.add_argument("--gmc-quality", type=Path, help="per-sequence JSONL directory or combined JSONL")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="preflight only, first N queries")
    args = parser.parse_args()
    import cv2
    from fctrack.verifier import seed_crop
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    extractor = FineFeatures(args.weights, args.device)
    arrays = {f"X_{name}": [] for name in ("geometry", "single", "fusion", "dual", "shuffled")}
    rows, source_hashes, image_hashes = [], {}, {}
    source_hashes[str(args.weights.resolve())] = sha256(args.weights)
    capture_metadata = args.capture / "metadata.json"
    if capture_metadata.exists():
        source_hashes[str(capture_metadata.resolve())] = sha256(capture_metadata)
    for path in (Path(__file__), Path(__file__).with_name("verifier.py")):
        source_hashes[str(path.resolve())] = sha256(path)
    for query_file in sorted((args.capture / "queries").glob("*.jsonl")):
        sequence = query_file.stem
        queries = jsonl(query_file)
        if not queries:
            continue
        anchor_file = args.capture / "anchors" / f"{sequence}.jsonl"
        anchor_rows = jsonl(anchor_file)
        anchors = {int(row["frame"]): row for row in anchor_rows}
        if len(anchors) != len(anchor_rows):
            raise ValueError(f"Duplicate captured anchor frames: {sequence}")
        directory = args.data_root / sequence
        ini = configparser.ConfigParser()
        ini.read(directory / "seqinfo.ini")
        info = ini["Sequence"]
        length, width, height = int(info["seqlength"]), int(info["imwidth"]), int(info["imheight"])
        if set(anchors) != set(range(1, length + 1)):
            raise ValueError(f"Capture does not contain all frames: {sequence}")
        gt_file, gmc_file = directory / "gt/gt.txt", args.gmc / f"{sequence}.txt"
        raw_gt = np.loadtxt(gt_file, delimiter=",", ndmin=2)
        gt = raw_gt[raw_gt[:, 6] >= 1]
        warps = load_gmc(gmc_file)
        quality, quality_paths = load_quality(args.gmc_quality, sequence)
        for path in (query_file, anchor_file, gt_file, directory / "seqinfo.ini", gmc_file, *quality_paths):
            source_hashes[str(path.resolve())] = sha256(path)
        images = ImageReader(directory, info)
        addresses = set()
        for query in queries:
            if args.limit and len(rows) >= args.limit:
                break
            required = {"query_handle", "exclude_handles", "exclude_track_ids"}
            if not required.issubset(query):
                raise ValueError("Use probe_capture queries with self-exclusion metadata, not old Exp139 events")
            address = (query["kind"], int(query["frame"]), int(query["candidate_index"]), query["signature"])
            if address in addresses:
                raise ValueError(f"Duplicate query {sequence} {address}")
            addresses.add(address)
            signature_fields = {key: query[key] for key in
                                ("frame", "candidate_index", "proposal", "final", "p", "native_accept")}
            actual_signature = hashlib.sha256(json.dumps(signature_fields, sort_keys=True).encode()).hexdigest()
            if actual_signature != query["signature"]:
                raise ValueError(f"Query signature changed: {sequence} {address}")
            frame = int(query["frame"])
            if query["sequence"] != sequence or not 1 <= frame <= length:
                raise ValueError(f"Query/view mismatch: {query}")
            for key in ("proposal", "final"):
                box = np.asarray(query[key], float)
                if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= 0).any():
                    raise ValueError(f"Invalid {key}: {query}")
            proposal = np.asarray(query["proposal"], float)
            center, side = box_center(proposal), float(np.sqrt(proposal[2:].prod()))
            grid = crop_grid(center, side)
            current_mask = valid_grid(grid, width, height)
            image = images.read(frame)
            current_crop = seed_crop(image, *center, side)
            paths = [make_paths(query, anchors, warps, lag) for lag in LAGS]
            crops = [current_crop]
            arm_indices, arm_masks = {name: [] for name in ("fusion", "dual", "shuffled")}, {name: [] for name in ("fusion", "dual", "shuffled")}
            fractions = []
            for path in paths:
                if not path["history_complete"]:
                    for name in arm_indices:
                        arm_indices[name].extend((0, 0))
                        arm_masks[name].extend((np.zeros_like(current_mask), np.zeros_like(current_mask)))
                    fractions.append((0., 0., 0.))
                    continue
                historical = images.read(path["start"])
                residual = path["residual"]
                bg_grid = inverse_path_grid(grid, path["matrix"], np.zeros(2))
                target_grid = inverse_path_grid(grid, path["matrix"], residual)
                rotated_grid = inverse_path_grid(grid, path["matrix"], [-residual[1], residual[0]])
                sampled = {}
                for name, coordinates in (("fixed", grid), ("target", target_grid), ("background", bg_grid), ("rotated", rotated_grid)):
                    # Fixed-position fusion uses precisely the original seed interpolation.
                    patch = seed_crop(historical, *center, side) if name == "fixed" else remap_crop(historical, coordinates)
                    sampled[name] = (len(crops), valid_grid(coordinates, width, height))
                    crops.append(patch)
                for arm, names in (("fusion", ("fixed", "fixed")), ("dual", ("target", "background")),
                                   ("shuffled", ("rotated", "background"))):
                    for name in names:
                        index, mask = sampled[name]
                        arm_indices[arm].append(index)
                        arm_masks[arm].append(mask)
                fractions.append(tuple(float(sampled[name][1].mean()) for name in ("target", "background", "rotated")))
            features = extractor.encode(crops)
            current = features[0]
            descriptor = current_descriptor(current, query, current_mask)
            complete = [path["history_complete"] for path in paths]
            arrays["X_single"].append(visual_vector(descriptor, current, current_mask, [current] * 4,
                                                    [current_mask] * 4, complete))
            for arm in arm_indices:
                arrays[f"X_{arm}"].append(visual_vector(descriptor, current, current_mask,
                    [features[i] for i in arm_indices[arm]], arm_masks[arm], complete))
            geometry = base_geometry(query, width, height, current_mask.mean())
            for path, fraction in zip(paths, fractions):
                geometry.extend(lag_geometry(query, path, fraction, quality))
            arrays["X_geometry"].append(np.asarray(geometry, np.float32))
            # Annotation access is confined to this post-feature label operation.
            labels = label_query(query, gt[gt[:, 0] == frame, 2:6])
            if "any_hit" in query and labels["quality"] != int(query["any_hit"]):
                raise ValueError(f"GT/final-box quality differs from frozen event: {query['signature']}")
            rows.append(dict(sequence=sequence, frame=frame, kind=query["kind"], outcome=query["outcome"],
                             p=float(query["p"]), signature=query["signature"], candidate_index=int(query["candidate_index"]),
                             proposal=query["proposal"], final=query["final"], native_accept=bool(query["native_accept"]),
                             history_complete=bool(all(complete)), paths_available=bool(all(p["path_available"] for p in paths)),
                             current_valid_fraction=float(current_mask.mean()),
                             path_metadata=_plain(paths), labels_use_canonical_gt=True,
                             **runtime_subgroups(query, paths, float(current_mask.mean()), fractions), **labels))
        image_hashes.update(images.hashes)
        print(sequence, "queries", sum(row["sequence"] == sequence for row in rows), flush=True)
        if args.limit and len(rows) >= args.limit:
            break
    if not rows:
        raise ValueError("No captured queries")
    output = {key: np.stack(value) for key, value in arrays.items()}
    visual_shapes = {value.shape for key, value in output.items() if key != "X_geometry"}
    if len(visual_shapes) != 1 or not all(np.isfinite(value).all() for value in output.values()):
        raise ValueError("Visual dimensions differ or nonfinite feature")
    np.savez_compressed(args.out / "features.npz", **output)
    (args.out / "rows.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    (args.out / "image_sha256.json").write_text(json.dumps(image_hashes, indent=2) + "\n")
    metadata = dict(status="completed", source_hashes=source_hashes, image_hash_manifest="image_sha256.json",
                    arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    query_count=len(rows), sequence_count=len({row["sequence"] for row in rows}),
                    kind_counts={kind: sum(row["kind"] == kind for row in rows) for kind in ("fill", "birth")},
                    dimensions={key: list(value.shape) for key, value in output.items()},
                    geometry_names=list(BASE_GEOMETRY) + [f"lag{lag}_{name}" for lag in LAGS for name in LAG_GEOMETRY],
                    lags=list(LAGS), image_size=SIZE, seed_features=["body[0]:32x64x64", "body[1]:64x32x32"],
                    spatial_pool="4x4 means; pair distances computed on original fine grids then masked mean/max",
                    current_descriptor="spatial RGB/F1/F2 plus final-box ROI/ring mean/std/max and signed contrast",
                    target_path="local paired real-anchor group residual; >=2 of nearest <=5 within max(8side,64px); self excluded by handle and track_id",
                    background_path="whole current grid mapped by inverse composed historical-to-current affine",
                    shuffled_control="rotated_residual_control: fixed +90 degrees in current-image coordinates; same residual norm and anchors",
                    crop_budget="maximum one current + two crops per lag for each arm; fixed fusion slots repeat; not equal effective spatial coverage; shared crops cached",
                    capacity="visual arrays have equal raw dimension; evaluator fits train-fold StandardScaler/PCA32, then adds separately scaled common geometry to LR",
                    geometry_scope="available runtime fields only; no invented covariance, neighbor detector overlap, owner GT, or outcome feature; history geometry shared by single arm",
                    missing_history="zeros plus flags; never duplicate first frame; a missing expected image/capture frame is a hard error",
                    runtime_subgroups={
                        "all_path_crops_valid": "history complete and current plus target/background/rotated crop valid fractions at both lags equal 1; reporting subgroup only",
                        "paths_separated": "both lag paths available and each ||target_center-background_center||*64/(3*proposal_side)>=1; threshold is 1 pixel in the 64x64 crop, not original-image pixels or feature cells",
                        "sampled_valid_fractions": "per-lag target/background/rotated fractions of sample coordinates inside the original historical image; does not filter the main pool",
                    },
                    invalid_gmc="missing entries flagged and composed as identity; singular composition becomes identity; no query discarded",
                    labels="quality:any canonical GT IoU(final)>=.5; presence:nearest canonical GT center <=1side positive, >1.5side negative, otherwise ignored; GT confidence>=1",
                    seconds=time.time() - started)
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: metadata[key] for key in ("query_count", "sequence_count", "dimensions", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
