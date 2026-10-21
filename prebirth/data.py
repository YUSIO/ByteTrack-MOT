"""Native-resolution temporal patches from the two halves of MOT train sequences.

Ground truth determines labels and training patch sampling only. Every frame in a
sample is cropped at the same image coordinates: GT never aligns the history.
"""
from __future__ import annotations

import configparser
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Sequence:
    name: str
    root: Path
    frames: dict[int, Path]
    labels: dict[int, np.ndarray]
    start: int
    end: int
    width: int
    height: int


def resolve_train_root(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    if (root / "train").is_dir():
        root = root / "train"
    if "test" in {p.lower() for p in root.parts}:
        raise ValueError("Training/validation must not read a test directory")
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def read_sequences(root: str | Path, split: str, names: list[str] | None = None):
    """Read train GT with the existing V1 half protocol.

    Train uses 1..floor(seqLength/2)+1 and dev uses floor(seqLength/2)+2..N,
    matching the established detector training and Exp089 dev_view exactly.
    """
    if split not in {"train", "dev"}:
        raise ValueError("Only train/dev halves are valid for fitting and selection")
    root = resolve_train_root(root)
    selected = set(names) if names else None
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "seqinfo.ini").is_file())
    if selected is not None:
        missing = selected - {p.name for p in dirs}
        if missing:
            raise ValueError(f"Unknown sequences: {sorted(missing)}")
        dirs = [p for p in dirs if p.name in selected]
    if not dirs:
        raise ValueError(f"No MOT sequences in {root}")
    sequences, sources = [], []
    for directory in dirs:
        info_path, gt_path = directory / "seqinfo.ini", directory / "gt/gt.txt"
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {gt_path}")
        info = configparser.ConfigParser()
        info.read(info_path)
        section = info["Sequence"]
        n = section.getint("seqLength")
        width, height = section.getint("imWidth"), section.getint("imHeight")
        cut = n // 2 + 1
        start, end = (1, cut) if split == "train" else (cut + 1, n)
        if start > end:
            raise ValueError(f"Sequence {directory.name} cannot be split: length={n}")
        image_dir = directory / section.get("imDir", "img1")
        images = {int(p.stem): p for p in image_dir.iterdir()
                  if p.is_file() and p.stem.isdigit() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        absent = [f for f in range(start, end + 1) if f not in images]
        if absent:
            raise FileNotFoundError(f"{directory.name}: {len(absent)} missing {split} images, first={absent[:5]}")
        rows, valid_source_rows, outside_center_rows = [], 0, 0
        for line_no, line in enumerate(gt_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = np.asarray([float(x) for x in line.replace(",", " ").split()], np.float32)
            if len(row) < 6 or not np.isfinite(row).all():
                raise ValueError(f"Malformed GT {gt_path}:{line_no}")
            frame = int(row[0])
            if frame < 1 or frame > n or row[0] != frame:
                raise ValueError(f"Out-of-range/non-integer GT frame {gt_path}:{line_no}")
            if row[4] <= 0 or row[5] <= 0:
                raise ValueError(f"Non-positive GT size {gt_path}:{line_no}")
            valid_source = (len(row) < 7 or row[6] > 0) and (len(row) < 8 or row[7] == 1)
            center_inside = 0 <= row[2] + row[4] / 2 < width and 0 <= row[3] + row[5] / 2 < height
            valid = valid_source and center_inside
            if start <= frame <= end:
                valid_source_rows += int(valid_source)
                outside_center_rows += int(valid_source and not center_inside)
                # Preserve the source xywh coordinate convention, as other MOT loaders do.
                # An out-of-image center is ignored, including when crop padding could
                # otherwise make it look like a valid positive. Source GT is untouched.
                rows.append((frame, [row[2], row[3], row[4], row[5], float(valid)]))
        labels = {}
        for frame, row in rows:
            labels.setdefault(frame, []).append(row)
        labels = {f: np.asarray(v, np.float32) for f, v in labels.items()}
        sequences.append(Sequence(directory.name, directory, images, labels, start, end, width, height))
        sources.append({"sequence": directory.name, "source_root": str(directory),
                        "split_rule": "train=1..floor(N/2)+1; dev=floor(N/2)+2..N",
                        "source_seq_length": n, "split": split, "first_frame": start,
                        "last_frame": end, "frames": end - start + 1, "history_min_frame": start,
                        "width": width, "height": height, "gt_rows_in_split": len(rows),
                        "valid_gt_rows_in_split": valid_source_rows,
                        "supervised_gt_rows_in_split": sum(int(r[-1]) for _, r in rows),
                        "valid_gt_centers_outside_image": outside_center_rows,
                        "gt_sha256": sha256(gt_path), "seqinfo_sha256": sha256(info_path)})
    return sequences, sources


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> int:
    """CenterNet's three overlap constraints, in output-grid coordinates."""
    a1, b1, c1 = 1, height + width, width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(0.0, b1 * b1 - 4 * a1 * c1))) / 2
    a2, b2, c2 = 4, 2 * (height + width), (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(max(0.0, b2 * b2 - 4 * a2 * c2))) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (height + width), (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(max(0.0, b3 * b3 - 4 * a3 * c3))) / 2
    return max(0, int(min(r1, r2, r3)))


def draw_gaussian(heatmap: np.ndarray, x: int, y: int, radius: int):
    diameter = 2 * radius + 1
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * (diameter / 6) ** 2))
    h, w = heatmap.shape
    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)
    target = heatmap[y - top:y + bottom, x - left:x + right]
    source = kernel[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(target, source, out=target)


def dense_targets(boxes_xywh: np.ndarray, valid: np.ndarray, size: int, stride: int = 4):
    grid = math.ceil(size / stride)
    heatmap = np.zeros((1, grid, grid), np.float32)
    hm_valid = np.ones_like(heatmap)
    offset = np.zeros((2, grid, grid), np.float32)
    log_wh = np.zeros_like(offset)
    reg_mask = np.zeros((1, grid, grid), np.float32)
    areas = np.full((grid, grid), np.inf)
    visible_boxes, collisions = [], 0
    # Ignore truncated objects whose centers fall outside the crop, and marked GT.
    for box, is_valid in zip(boxes_xywh, valid):
        x, y, w, h = map(float, box)
        cx, cy = x + w / 2, y + h / 2
        if (not is_valid or not (0 <= cx < size and 0 <= cy < size)) and x < size and y < size and x + w > 0 and y + h > 0:
            x1, y1 = max(0, int(math.floor(x / stride))), max(0, int(math.floor(y / stride)))
            x2, y2 = min(grid, int(math.ceil((x + w) / stride))), min(grid, int(math.ceil((y + h) / stride)))
            hm_valid[:, y1:y2, x1:x2] = 0
    for box, is_valid in zip(boxes_xywh, valid):
        x, y, w, h = map(float, box)
        cx, cy = x + w / 2, y + h / 2
        if not is_valid or not (0 <= cx < size and 0 <= cy < size):
            continue
        gx, gy = int(cx / stride), int(cy / stride)
        draw_gaussian(heatmap[0], gx, gy, gaussian_radius(h / stride, w / stride))
        hm_valid[0, gy, gx] = 1
        visible_boxes.append([x, y, x + w, y + h])
        if reg_mask[0, gy, gx]:
            collisions += 1
        if w * h >= areas[gy, gx]:
            continue
        areas[gy, gx] = w * h
        offset[:, gy, gx] = (cx / stride - gx, cy / stride - gy)
        log_wh[:, gy, gx] = (math.log(w), math.log(h))
        reg_mask[0, gy, gx] = 1
    return {"heatmap": torch.from_numpy(heatmap), "hm_valid": torch.from_numpy(hm_valid),
            "offset": torch.from_numpy(offset), "log_wh": torch.from_numpy(log_wh),
            "reg_mask": torch.from_numpy(reg_mask),
            "boxes": torch.as_tensor(np.asarray(visible_boxes, np.float32).reshape(-1, 4)),
            "collisions": collisions}


class TemporalPatchDataset(Dataset):
    """Deterministic-per-index sampling, epoch-varying only for training.

    Half of patches are centered at a randomly offset target. The other half are
    random native-resolution windows; half of these attempt to find background
    within that same image. An optional train-only hard-negative manifest replaces
    the background-seeking quarter. All objects in every crop remain supervised.
    """
    def __init__(self, root, split="train", samples=3600, patch_size=320, history_gap=3,
                 seed=138, augment=True, sequences=None, hard_negative_json=None):
        if patch_size < 32 or patch_size % 4:
            raise ValueError("patch_size must be >=32 and divisible by stride 4")
        if history_gap < 1 or samples < 1:
            raise ValueError("history_gap and samples must be positive")
        self.sequences, self.source_info = read_sequences(root, split, sequences)
        self.split, self.samples, self.patch_size = split, samples, patch_size
        self.history_gap, self.seed = history_gap, seed
        self.augment = augment and split == "train"
        self.epoch = 0
        self.frames = [(s, frame) for s, seq in enumerate(self.sequences) for frame in range(seq.start, seq.end + 1)]
        self.positive_frames = [(s, frame) for s, frame in self.frames
                                if np.any(self.sequences[s].labels.get(frame, np.empty((0, 5)))[:, 4] > 0)]
        if not self.positive_frames:
            raise ValueError(f"No valid positive annotations in {split}")
        self.hard_negative_rows, self.hard_negative_info = [], None
        if hard_negative_json is not None:
            if split != "train":
                raise ValueError("Hard-negative manifests are forbidden for dev sampling")
            self._load_hard_negatives(Path(hard_negative_json))

    def _load_hard_negatives(self, path):
        payload = json.loads(path.read_text())
        metadata, rows = payload["metadata"], payload["rows"]
        if (metadata.get("status") != "completed" or metadata.get("split") != "train"
                or metadata.get("test_data_used") is not False
                or metadata.get("annotation_access") != "supervised_mining_train_half_only"):
            raise ValueError("Hard-negative metadata must identify completed train-only supervised mining")
        if not isinstance(rows, list) or not rows or metadata.get("rows") != len(rows):
            raise ValueError("Hard-negative manifest must contain a nonempty, correctly counted rows list")
        by_name = {seq.name: (index, seq) for index, seq in enumerate(self.sequences)}
        sources = {info["sequence"]: info for info in self.source_info}
        provenance = metadata.get("provenance", {})
        for name, record in provenance.items():
            if name not in by_name:
                raise ValueError(f"Hard-negative sequence absent from training selection: {name}")
            source = sources[name]
            if (record.get("sha256", {}).get("gt") != source["gt_sha256"]
                    or record.get("sha256", {}).get("seqinfo") != source["seqinfo_sha256"]):
                raise ValueError(f"Hard-negative GT/seqinfo hash mismatch: {name}")
            bounds = metadata.get("per_sequence", {}).get(name, {})
            if bounds.get("first_frame") != source["first_frame"] or bounds.get("last_frame") != source["last_frame"]:
                raise ValueError(f"Hard-negative train-half bounds mismatch: {name}")
        kinds = {}
        frames = set()
        for row_index, row in enumerate(rows):
            name, frame = row["sequence"], row["frame"]
            if name not in by_name or name not in provenance:
                raise ValueError(f"Unverified hard-negative sequence: {name}")
            sequence_id, seq = by_name[name]
            if type(frame) is not int or not seq.start <= frame <= seq.end or frame not in seq.frames:
                raise ValueError(f"Hard-negative frame outside training half: {name}:{frame}")
            values = np.asarray([row["center_x"], row["center_y"], row["score"], row["nearest_gt_iou"]], float)
            if (not np.isfinite(values).all() or not 0 <= values[0] < seq.width
                    or not 0 <= values[1] < seq.height or not 0 <= values[2] <= 1 or not 0 <= values[3] < 1):
                raise ValueError(f"Invalid hard-negative geometry/score at row {row_index}")
            box = np.asarray(row["box_xywh"], float)
            if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0)
                    or not np.allclose(box[:2] + box[2:] / 2, values[:2], atol=1e-4, rtol=0)):
                raise ValueError(f"Hard-negative box/center mismatch at row {row_index}")
            kind = row.get("kind", "unknown")
            if kind not in {"near_object", "far_background"}:
                raise ValueError(f"Unknown hard-negative category: {kind}")
            self.hard_negative_rows.append({**row, "sequence_id": sequence_id, "source_row": row_index})
            kinds[kind] = kinds.get(kind, 0) + 1
            frames.add((name, frame))
        self.hard_negative_info = {
            "path": str(path.resolve()), "sha256": sha256(path), "rows": len(rows),
            "sequences": len({name for name, _ in frames}), "frames": len(frames), "kinds": kinds,
            "metadata_split": "train", "mining_source_sha256": metadata.get("mining_source_sha256"),
            "source_checkpoint_sha256": metadata.get("source_checkpoint_sha256"),
            "expected_samples_per_epoch": self.samples // 4,
            "sampling_policy": "uniform manifest-row choice at index%4==3; random crop offset 0.12..0.88 of patch; retain all dense GT",
        }

    def __len__(self):
        return self.samples

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch) if self.split == "train" else 0

    def _origin(self, rng, seq, labels, target_sample, background_sample):
        size = self.patch_size
        max_x, max_y = max(0, seq.width - size), max(0, seq.height - size)
        valid = labels[labels[:, 4] > 0]
        if target_sample and len(valid):
            box = valid[rng.integers(len(valid))]
            # Random offset prevents the network from learning a center-only shortcut.
            x = round(box[0] + box[2] / 2 - rng.uniform(0.12, 0.88) * size)
            y = round(box[1] + box[3] / 2 - rng.uniform(0.12, 0.88) * size)
            return int(np.clip(x, 0, max_x)), int(np.clip(y, 0, max_y))
        for _ in range(12 if background_sample else 1):
            x, y = int(rng.integers(max_x + 1)), int(rng.integers(max_y + 1))
            if not background_sample or not len(valid):
                break
            intersects = ((valid[:, 0] < x + size) & (valid[:, 1] < y + size)
                          & (valid[:, 0] + valid[:, 2] > x) & (valid[:, 1] + valid[:, 3] > y))
            if not intersects.any():
                break
        return x, y

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        target_sample = index % 2 == 0
        hard_negative = None
        if self.hard_negative_rows and index % 4 == 3:
            hard_negative = self.hard_negative_rows[int(rng.integers(len(self.hard_negative_rows)))]
            sequence_id, frame = hard_negative["sequence_id"], hard_negative["frame"]
        else:
            pool = self.positive_frames if target_sample else self.frames
            sequence_id, frame = pool[int(rng.integers(len(pool)))]
        seq = self.sequences[sequence_id]
        labels = seq.labels.get(frame, np.empty((0, 5), np.float32)).copy()
        if hard_negative is None:
            x, y = self._origin(rng, seq, labels, target_sample, index % 4 == 3)
        else:
            size = self.patch_size
            x = round(hard_negative["center_x"] - rng.uniform(0.12, 0.88) * size)
            y = round(hard_negative["center_y"] - rng.uniform(0.12, 0.88) * size)
            x, y = int(np.clip(x, 0, max(0, seq.width - size))), int(np.clip(y, 0, max(0, seq.height - size)))
        history_frames = [max(seq.start, frame - k * self.history_gap) for k in range(3)]
        patches, cache = [], {}
        size = self.patch_size
        for f in history_frames:
            if f not in cache:
                im = cv2.imread(str(seq.frames[f]), cv2.IMREAD_COLOR)
                if im is None:
                    raise ValueError(f"Unreadable image: {seq.frames[f]}")
                if im.shape[:2] != (seq.height, seq.width):
                    raise ValueError(f"Image/seqinfo size mismatch: {seq.frames[f]} {im.shape[:2]} vs {(seq.height, seq.width)}")
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                patch = np.zeros((size, size, 3), dtype=np.uint8)
                crop = im[y:y + size, x:x + size]
                patch[:crop.shape[0], :crop.shape[1]] = crop
                cache[f] = patch
            patches.append(cache[f])
        images = np.stack(patches).astype(np.float32) / 255.0
        boxes = labels[:, :4].copy()
        boxes[:, :2] -= (x, y)
        flipped = self.augment and rng.random() < 0.5
        if flipped:
            images = images[:, :, ::-1].copy()
            boxes[:, 0] = size - boxes[:, 0] - boxes[:, 2]
        if self.augment:
            # One photometric transform for the whole clip preserves its temporal evidence.
            gain, bias = rng.uniform(0.8, 1.2), rng.uniform(-0.04, 0.04)
            images = np.clip(images * gain + bias, 0, 1)
        targets = dense_targets(boxes, labels[:, 4] > 0, size)
        # Padded image regions are not negative training evidence.
        real_w, real_h = min(size, seq.width - x), min(size, seq.height - y)
        valid_pixels = np.zeros((size, size), np.uint8)
        valid_pixels[:real_h, :real_w] = 1
        if flipped:
            valid_pixels = valid_pixels[:, ::-1].copy()
        images *= valid_pixels[None, :, :, None]
        grid_valid = cv2.resize(valid_pixels, (size // 4, size // 4), interpolation=cv2.INTER_NEAREST)
        targets["hm_valid"] *= torch.from_numpy(grid_valid[None].astype(np.float32))
        targets["images"] = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2)))
        targets["metadata"] = {"sequence": seq.name, "frame": frame,
                               "history_frames": history_frames, "crop_xy": [x, y],
                               "target_sample": target_sample, "flipped": bool(flipped)}
        if hard_negative is not None:
            targets["metadata"].update(hard_negative_source_row=hard_negative["source_row"],
                                       hard_negative_kind=hard_negative["kind"],
                                       hard_negative_source_score=hard_negative["score"])
        return targets


def collate_patches(batch):
    result = {key: torch.stack([sample[key] for sample in batch])
              for key in ("images", "heatmap", "hm_valid", "offset", "log_wh", "reg_mask")}
    result["boxes"] = [sample["boxes"] for sample in batch]
    result["metadata"] = [sample["metadata"] for sample in batch]
    result["collisions"] = sum(sample["collisions"] for sample in batch)
    return result


def seed_worker(worker_id):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
