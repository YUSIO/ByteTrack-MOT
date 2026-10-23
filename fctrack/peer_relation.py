"""Frozen-A RGB peer classification, with explicit extraction/training/readout stages.

Only train consumes externally prepared labels. Extraction and scoring never load
GT or an evaluator. Public IDs and geometric exclusions are observable identity
proxies, not a guarantee that two references depict different physical objects.
"""
import configparser
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from fctrack.output_verification import _parse_row, _decode_image, sha256


ARMS = ("query", "peer", "cross")
QUERY_COLUMNS = ("row_index", "frame", "id", "x", "y", "w", "h", "logit",
                 "scalar0", "scalar1", "prior_streak", "det_index", "det_iou",
                 "det_score", "anchor_eligible", "support_indices", "eligible")
SCORE_COLUMNS = QUERY_COLUMNS + ARMS + ("cross_refs",)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def names(numbers):
    return [f"UAVSwarm-{int(i):02d}" for i in numbers]


def config(path):
    cfg = json.loads(Path(path).read_bytes())
    p = cfg["peer_relation"]
    expected = {"kind": "seed", "query_coordinates": "saved_rounded_output",
                "crop_size": 64, "crop_zoom": 3.0, "feature_shape": [256, 4, 4],
                "storage_dtype": "float16", "feature_compute_dtype": "float32",
                "det_score_min": .8, "det_iou_min": .7, "prior_output_streak_min": 5,
                "detector_numeric_conversion": "float64_tlwh_to_xyxy_then_float32_xyxy_and_score",
                "same_public_id_excluded": True, "near_duplicate_iou_exclude": .1,
                "reference_crop_query_bbox_positive_intersection_excluded": True,
                "max_supports": 3, "regression_used": False, "state_feedback": False,
                "feature_projection_channels": 8, "projection_activation": "none",
                "normalization": "none", "query_hidden": 512, "relation_hidden": 128,
                "head_activation": "relu", "empty_support": "retain_A_all_arms"}
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported peer relation protocol")
    split_sets = [set(names(cfg["splits"][k])) for k in
                  ("train_numbers", "calibration_numbers", "dev_numbers")]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("split overlap")
    return cfg


class Evidence:
    """Hash every consumed file and reject mutation before completing a stage."""
    def __init__(self):
        self.hashes = {}

    def record(self, path):
        path = Path(path).resolve()
        digest = sha256(path)
        if str(path) in self.hashes and self.hashes[str(path)] != digest:
            raise RuntimeError(f"input mutated: {path}")
        self.hashes[str(path)] = digest
        return path

    def check(self):
        for path, digest in self.hashes.items():
            if sha256(path) != digest:
                raise RuntimeError(f"input mutated: {path}")


def start(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output, time.perf_counter()


def finish(output, began, evidence, stage, **fields):
    evidence.check()
    hashes = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
              if p.is_file()}
    value = dict(schema_version=1, status="completed", stage=stage,
                 gt_used=stage == "train", evaluator_used=False,
                 state_feedback=False, regression_used=False,
                 wall_seconds=time.perf_counter() - began,
                 input_hashes=evidence.hashes, output_hashes=hashes, **fields)
    dump(output / "summary.json", value)
    return value


def read_rows(path):
    rows, keys, last_frame = [], set(), 0
    for index, raw in enumerate(Path(path).read_bytes().splitlines(keepends=True)):
        row = _parse_row(raw, index, path)
        key = (row.frame, row.track_id)
        if row.frame < last_frame or key in keys:
            raise ValueError(f"noncausal or repeated frame/ID: {path} {key}")
        keys.add(key)
        last_frame = row.frame
        rows.append(row)
    return rows


def iou(box, others):
    a = np.asarray(box, dtype=float)
    b = np.asarray(others, dtype=float).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(a[:2] + a[2:], b[:, :2] + b[:, 2:])
                              - np.maximum(a[:2], b[:, :2])).prod(1)
    return intersection / np.maximum(a[2:].prod() + b[:, 2:].prod(1) - intersection, 1e-12)


def crop_contains_query(reference, query):
    r, q = np.asarray(reference, float), np.asarray(query, float)
    side = 3 * math.sqrt(r[2] * r[3])
    centre = r[:2] + r[2:] / 2
    return bool(np.all(np.minimum(centre + side / 2, q[:2] + q[2:])
                       - np.maximum(centre - side / 2, q[:2]) > 0))


def frame_support(rows, detections, history):
    """Read t-1 history, choose supports, then update history with current A rows.

    det_index is zero-based within this frame's ORIGINAL cache row order. Best
    match is selected among score >= .8 and IoU >= .7 candidates: max IoU,
    then score, then ORIGINAL cache index. Dedup chooses the A row closest to
    each selected detection.
    """
    if not rows:
        return []
    frame = rows[0].frame
    if any(row.frame != frame for row in rows):
        raise ValueError("frame_support requires one frame")
    detections = np.asarray(detections, float).reshape(-1, 5)
    result = []
    for row in rows:
        previous = history.get(row.track_id)
        streak = previous[1] if previous and previous[0] == frame - 1 else 0
        overlaps = iou(row.box, detections[:, :4])
        candidates = [j for j in range(len(detections)) if detections[j, 4] >= .8 and overlaps[j] >= .7]
        order = sorted(candidates, key=lambda j: (-overlaps[j], -detections[j, 4], j))
        j = order[0] if order else -1
        overlap = float(overlaps[j]) if j >= 0 else 0.0
        score = float(detections[j, 4]) if j >= 0 else 0.0
        result.append(dict(prior_streak=streak, det_index=j, det_iou=overlap, det_score=score,
                           anchor_eligible=bool(streak >= 5 and overlap >= .7 and score >= .8)))
    anchors = [i for i, r in enumerate(result) if r["anchor_eligible"]]
    unique = {}
    for i in sorted(anchors, key=lambda i: (-result[i]["det_iou"], rows[i].track_id)):
        unique.setdefault(result[i]["det_index"], i)
    anchors = sorted(unique.values(), key=lambda i: (-result[i]["det_score"],
                     -result[i]["det_iou"], rows[i].track_id))
    accepted = set(anchors)
    for i, row in enumerate(rows):
        result[i]["anchor_eligible"] = i in accepted
        peers = [j for j in anchors if rows[j].track_id != row.track_id
                 and iou(row.box, [rows[j].box])[0] < .1
                 and not crop_contains_query(rows[j].box, row.box)][:3]
        result[i]["support_indices"] = [rows[j].row_index for j in peers]
        result[i]["eligible"] = bool(peers)
    for row, info in zip(rows, result):
        history[row.track_id] = (frame, info["prior_streak"] + 1)
    return result


def detector_coordinates(cache):
    """Match A's xyxy/score FP32 cache loading without another tlwh rounding."""
    cache = np.asarray(cache, np.float64).reshape(-1, 10)
    xyxy = cache[:, 2:6].copy()
    xyxy[:, 2:] += xyxy[:, :2]
    xyxy = xyxy.astype(np.float32).astype(np.float64)
    xyxy[:, 2:] -= xyxy[:, :2]
    score = cache[:, 6].astype(np.float32).astype(np.float64)
    return np.column_stack((xyxy, score))


def load_encoder(weights, device):
    from fctrack.verifier import load, seed_crop, scalars
    model = load("seed", weights, torch.device(device)).eval()
    model.requires_grad_(False)

    @torch.no_grad()
    def encode(image, boxes):
        box = np.asarray(boxes, float).reshape(-1, 4)
        sides = np.sqrt(box[:, 2] * box[:, 3])
        centres = box[:, :2] + box[:, 2:] / 2
        crops = np.stack([seed_crop(image, *c, s) for c, s in zip(centres, sides)])
        scale = scalars(sides, np.full(len(box), image.shape[1], float))
        features = model.body(torch.from_numpy(crops).to(device))
        scalar_tensor = torch.from_numpy(scale).to(device)
        raw = model.head(torch.cat((features.mean((2, 3)), features.amax((2, 3)), scalar_tensor), 1))
        # Discard regression outputs without decoding or changing any coordinates.
        return features.cpu().numpy(), raw[:, 0].cpu().numpy(), scale
    return encode


def write_queries(path, queries, fields=QUERY_COLUMNS):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for q in queries:
            value = dict(q)
            for field in ("support_indices", "cross_refs"):
                if field in value:
                    value[field] = json.dumps(value[field], separators=(",", ":"))
            for field in ("eligible", "anchor_eligible"):
                value[field] = int(value[field])
            writer.writerow(value)


def bank_from_queries(queries, train_names):
    chosen = {}
    for sequence in sorted(train_names):
        for q in queries[sequence]:
            if not q["anchor_eligible"]:
                continue
            key = (sequence, q["id"])
            rank = (-q["det_score"], q["frame"], q["row_index"])
            if key not in chosen or rank < chosen[key][0]:
                chosen[key] = (rank, dict(sequence=sequence, row_index=q["row_index"],
                                         frame=q["frame"], id=q["id"], det_score=q["det_score"]))
    return [chosen[key][1] for key in sorted(chosen)]


def select_cross(bank, sequence, query):
    k = len(query["support_indices"])
    def key(b):
        identity = f"{sequence}|{query['frame']}|{query['id']}|{b['sequence']}|{b['frame']}|{b['id']}"
        return hashlib.sha256(identity.encode("ascii")).hexdigest()
    candidates = sorted((b for b in bank if b["sequence"] != sequence), key=key)
    if len(candidates) < k:
        raise ValueError("cross bank cannot match support count")
    result = candidates[:k]
    if len({(b["sequence"], b["id"]) for b in result}) != k:
        raise ValueError("duplicate bank object")
    return result


def extract(root, config_path, cell, output, device, *, encoder_loader=None, image_decoder=None):
    evidence = Evidence()
    config_path = evidence.record(config_path)
    cfg = config(config_path)
    root = Path(root).resolve()
    spec = cfg["cells"][cell]
    data = root / spec["dataset_root"] / spec["split"]
    weights = evidence.record(root / spec["verifiers"]["seed"]["weights"])
    encode = (encoder_loader or load_encoder)(weights, device)
    decode = image_decoder or _decode_image
    sequence_names = sorted(set(sum((names(cfg["splits"][k]) for k in
                                    ("train_numbers", "calibration_numbers", "dev_numbers")), [])))
    output, began = start(output)
    for directory in ("tracks/A", "queries", "features"):
        (output / directory).mkdir(parents=True)
    dump(output / "config.json", cfg)
    all_queries, counts, frame_count = {}, {}, 0
    for sequence in sequence_names:
        source = evidence.record(root / spec["reference_tracks"] / f"{sequence}.txt")
        rows = read_rows(source)
        ini = evidence.record(data / sequence / "seqinfo.ini")
        parser = configparser.ConfigParser()
        parser.read_string(ini.read_text())
        info = parser["Sequence"]
        length = info.getint("seqLength")
        im_dir, ext = info.get("imDir", "img1"), info.get("imExt", ".jpg")
        if Path(im_dir).is_absolute() or any(p in ("..", "gt") for p in Path(im_dir).parts) \
                or not ext.startswith(".") or "/" in ext or "\\" in ext:
            raise ValueError("invalid image location")
        if any(row.frame > length for row in rows):
            raise ValueError("A row outside sequence")
        cache_path = evidence.record(root / spec["detections"] / sequence / "det.txt")
        raw = cache_path.read_text().strip()
        cache = np.loadtxt(cache_path, delimiter=",", ndmin=2) if raw else np.empty((0, 10))
        if cache.shape[1] != 10 or not np.isfinite(cache).all() or \
                np.any(cache[:, 4:6] <= 0) or np.any(cache[:, 0] != cache[:, 0].astype(int)):
            raise ValueError("invalid cache")
        features = np.empty((len(rows), 256, 4, 4), np.float16)
        queries, history, position = [], {}, 0
        while position < len(rows):
            frame = rows[position].frame
            end = position + 1
            while end < len(rows) and rows[end].frame == frame:
                end += 1
            group = rows[position:end]
            image_path = evidence.record(data / sequence / im_dir / f"{frame:06d}{ext}")
            image = decode(image_path.read_bytes(), image_path)
            boxes = np.asarray([row.box for row in group], float)
            body, logits, scale = encode(image, boxes.copy())
            body, logits, scale = np.asarray(body), np.asarray(logits), np.asarray(scale)
            if body.shape != (len(group), 256, 4, 4) or logits.shape != (len(group),) \
                    or scale.shape != (len(group), 2) or not all(np.isfinite(v).all() for v in (body, logits, scale)):
                raise ValueError("invalid encoder output")
            features[position:end] = body.astype(np.float16)
            if not np.isfinite(features[position:end]).all():
                raise ValueError("float16 feature overflow")
            det = detector_coordinates(cache[cache[:, 0] == frame])
            support = frame_support(group, det, history)
            for offset, (row, runtime) in enumerate(zip(group, support)):
                queries.append(dict(row_index=row.row_index, frame=frame, id=row.track_id,
                                    **dict(zip(("x", "y", "w", "h"), row.box)),
                                    logit=float(logits[offset]), scalar0=float(scale[offset, 0]),
                                    scalar1=float(scale[offset, 1]), **runtime))
            position = end
        (output / "tracks/A" / source.name).write_bytes(source.read_bytes())
        write_queries(output / "queries" / f"{sequence}.csv", queries)
        np.savez_compressed(output / "features" / f"{sequence}.npz", body=features,
                            row_index=np.arange(len(rows), dtype=np.int64))
        all_queries[sequence] = queries
        counts[sequence] = dict(input_rows=len(rows), eligible_rows=sum(q["eligible"] for q in queries),
                                anchor_rows=sum(q["anchor_eligible"] for q in queries), frame_count=length)
        frame_count += length
    bank = bank_from_queries(all_queries, names(cfg["splits"]["train_numbers"]))
    dump(output / "bank.json", bank)
    return finish(output, began, evidence, "extract", device=device, protocol=cfg["peer_relation"],
                  sequence_names=sequence_names, sequence_count=len(sequence_names),
                  frame_count=frame_count, input_rows=sum(c["input_rows"] for c in counts.values()),
                  sequences=counts, bank_objects=len(bank))


def read_queries(path, rows, *, scoring=False):
    with Path(path).open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    if len(raw) != len(rows):
        raise ValueError("query/track row count mismatch")
    result = []
    for q, row in zip(raw, rows):
        for field in ("row_index", "frame", "id", "prior_streak", "det_index"):
            q[field] = int(q[field])
        for field in ("x", "y", "w", "h", "logit", "scalar0", "scalar1", "det_iou", "det_score"):
            q[field] = float(q[field])
            if not math.isfinite(q[field]):
                raise ValueError("non-finite query field")
        if (q["row_index"], q["frame"], q["id"]) != (row.row_index, row.frame, row.track_id) \
                or tuple(q[k] for k in ("x", "y", "w", "h")) != row.box:
            raise ValueError("query/track row identity mismatch")
        for field in ("eligible", "anchor_eligible"):
            if q[field] not in ("0", "1"):
                raise ValueError("invalid eligibility")
            q[field] = q[field] == "1"
        q["support_indices"] = json.loads(q["support_indices"])
        support = q["support_indices"]
        if len(support) > 3 or len(set(support)) != len(support) \
                or any(type(j) is not int or j < 0 or j >= len(rows) for j in support) \
                or q["eligible"] != bool(support):
            raise ValueError("invalid support indices")
        for j in support:
            if rows[j].frame != row.frame or rows[j].track_id == row.track_id \
                    or iou(row.box, [rows[j].box])[0] >= .1 or crop_contains_query(rows[j].box, row.box):
                raise ValueError("support violates causal/exclusion rules")
        if scoring:
            for arm in ARMS:
                q[arm] = float(q[arm]) if q[arm] != "" else None
                if q["eligible"] != (q[arm] is not None) or (q[arm] is not None and not math.isfinite(q[arm])):
                    raise ValueError("invalid score eligibility")
            q["cross_refs"] = json.loads(q["cross_refs"])
        result.append(q)
    for q in result:
        if any(not result[j]["anchor_eligible"] for j in q["support_indices"]):
            raise ValueError("support is not an admitted anchor")
    return result


def check_stage(directory, evidence, stage):
    directory = Path(directory).resolve()
    summary = json.loads(evidence.record(directory / "summary.json").read_bytes())
    if summary.get("status") != "completed" or summary.get("stage") != stage:
        raise ValueError("incomplete or incorrect input stage")
    return directory, summary


def recorded(path, directory, summary, evidence):
    path = evidence.record(path)
    expected = summary["output_hashes"].get(str(path.relative_to(directory)))
    if expected != evidence.hashes[str(path)]:
        raise ValueError(f"input artifact checksum mismatch: {path}")
    return path


def load_extraction(extraction, evidence, selected=None, features=True):
    directory, summary = check_stage(extraction, evidence, "extract")
    cfg = config(recorded(directory / "config.json", directory, summary, evidence))
    chosen = summary["sequence_names"] if selected is None else list(selected)
    if not set(chosen).issubset(summary["sequence_names"]):
        raise ValueError("missing requested sequences")
    all_rows, all_queries, bodies = {}, {}, {}
    for sequence in chosen:
        rows = read_rows(recorded(directory / "tracks/A" / f"{sequence}.txt", directory, summary, evidence))
        queries = read_queries(recorded(directory / "queries" / f"{sequence}.csv", directory, summary, evidence), rows)
        all_rows[sequence], all_queries[sequence] = rows, queries
        if features:
            path = recorded(directory / "features" / f"{sequence}.npz", directory, summary, evidence)
            with np.load(path, allow_pickle=False) as z:
                body = z["body"]
                if body.shape != (len(rows), 256, 4, 4) or body.dtype != np.float16 \
                        or not np.array_equal(z["row_index"], np.arange(len(rows))) or not np.isfinite(body).all():
                    raise ValueError("feature row identity/shape mismatch")
                bodies[sequence] = body
    return cfg, all_rows, all_queries, bodies, directory, summary


class QueryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(256, 8, 1, bias=True)
        self.head = nn.Sequential(nn.Linear(131, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, query, scalars):
        q = self.projection(query).flatten(1)
        return self.head(torch.cat((q, scalars), 1)).squeeze(1)


class RelationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(256, 8, 1, bias=True)
        self.head = nn.Sequential(nn.Linear(515, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, query, scalars, references, mask):
        batch, count = references.shape[:2]
        if torch.any(mask.sum(1) == 0):
            raise ValueError("empty support must bypass the model")
        q = self.projection(query).flatten(1)[:, None].expand(-1, count, -1)
        r = self.projection(references.reshape(-1, 256, 4, 4)).flatten(1).reshape(batch, count, -1)
        extras = scalars[:, None].expand(-1, count, -1)
        pairs = torch.cat((q, r, torch.abs(q - r), q * r, extras), 2)
        logits = self.head(pairs).squeeze(2)
        return (logits * mask).sum(1) / mask.sum(1)


def batch_data(keys, queries, bodies, bank=None):
    q, s, r, mask = [], [], [], []
    for sequence, index in keys:
        item = queries[sequence][index]
        refs = [(sequence, j) for j in item["support_indices"]] if bank is None else \
            [(b["sequence"], b["row_index"]) for b in select_cross(bank, sequence, item)]
        if not refs:
            raise ValueError("ineligible training/scoring query")
        q.append(bodies[sequence][index])
        s.append([item["logit"], item["scalar0"], item["scalar1"]])
        reference = np.zeros((3, 256, 4, 4), np.float32)
        for i, (seq, row) in enumerate(refs):
            reference[i] = bodies[seq][row]
        r.append(reference)
        mask.append([float(i < len(refs)) for i in range(3)])
    return tuple(torch.from_numpy(np.asarray(a, np.float32)) for a in (q, s, r, mask))


def sample_indices(labels, training):
    pools = {}
    for key, label in labels.items():
        pools.setdefault(label, {}).setdefault(key[0], []).append(key)
    if set(pools) != {0, 1}:
        raise ValueError("both classes required")
    negatives = sum(len(v) for v in pools[0].values())
    if negatives < training["minimum_negative_rows"] or len(pools[0]) < training["minimum_negative_sequences"]:
        raise ValueError("insufficient negative rows/sequences; do not train")
    rng = np.random.default_rng(training["seed"])
    result = []
    for _ in range(training["epochs"]):
        epoch = []
        for i in range(training["samples_per_epoch"]):
            label = i % 2
            sequences = sorted(pools[label])
            sequence = sequences[int(rng.integers(len(sequences)))]
            choices = sorted(pools[label][sequence])
            key = choices[int(rng.integers(len(choices)))]
            epoch.append((key[0], key[1], label))
        result.append(epoch)
    return result


def train(extraction, labels_path, config_path, output):
    evidence = Evidence()
    cfg = config(evidence.record(config_path))
    selected = names(cfg["splits"]["train_numbers"])
    old_cfg, rows, queries, bodies, directory, _ = load_extraction(extraction, evidence, selected)
    if old_cfg != cfg:
        raise ValueError("configuration changed since extraction")
    labels = {}
    with evidence.record(labels_path).open(newline="") as stream:
        for value in csv.DictReader(stream):
            sequence, index, label = value["sequence"], int(value["row_index"]), int(value["label"])
            key = (sequence, index)
            if sequence not in selected or index < 0 or index >= len(queries[sequence]) \
                    or label not in (0, 1) or key in labels or not queries[sequence][index]["eligible"]:
                raise ValueError("invalid, duplicate, nontrain or ineligible training label")
            labels[key] = label
    t = cfg["training"]
    samples = sample_indices(labels, t)
    output, began = start(output)
    dump(output / "config.json", cfg)
    with (output / "sample_indices.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "sample_index", "sequence", "row_index", "label"])
        for e, epoch in enumerate(samples):
            writer.writerows((e, i, *item) for i, item in enumerate(epoch))
    torch.manual_seed(t["seed"])
    models = {"query": QueryHead(), "peer": RelationHead()}
    history = []
    for arm, model in models.items():
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
        for epoch_index, epoch in enumerate(samples):
            total_loss, seen, steps = 0.0, 0, 0
            for start_index in range(0, len(epoch), t["batch_size"]):
                chunk = epoch[start_index:start_index + t["batch_size"]]
                x, s, r, m = batch_data([(seq, index) for seq, index, _ in chunk], queries, bodies)
                target = torch.tensor([label for _, _, label in chunk], dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x, s) if arm == "query" else model(x, s, r, m)
                loss = nn.functional.binary_cross_entropy_with_logits(prediction, target)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip_norm"], error_if_nonfinite=True)
                optimizer.step()
                total_loss += loss.item() * len(chunk)
                seen += len(chunk)
                steps += 1
            history.append(dict(arm=arm, epoch=epoch_index + 1, loss=total_loss / seen,
                                samples=seen, optimizer_steps=steps, last_gradient_norm=float(norm)))
        model.eval()
    dump(output / "history.json", history)
    torch.save(dict(schema_version=1, config=cfg,
                    heads={key: model.state_dict() for key, model in models.items()}), output / "head_final.pt")
    return finish(output, began, evidence, "train", device="cpu", protocol=cfg["peer_relation"],
                  sequence_names=selected, sequence_count=len(selected), input_rows=len(labels),
                  training_labels_used=True, dev_labels_used=False, labels_path=str(Path(labels_path).resolve()),
                  samples_per_head=sum(len(e) for e in samples),
                  parameters={k: sum(p.numel() for p in m.parameters()) for k, m in models.items()})


def score(extraction, trained, config_path, output):
    evidence = Evidence()
    cfg = config(evidence.record(config_path))
    old_cfg, rows, queries, bodies, directory, extraction_summary = load_extraction(extraction, evidence)
    if old_cfg != cfg:
        raise ValueError("configuration changed since extraction")
    train_dir, train_summary = check_stage(trained, evidence, "train")
    checkpoint_path = recorded(train_dir / "head_final.pt", train_dir, train_summary, evidence)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["config"] != cfg:
        raise ValueError("trained configuration mismatch")
    models = {"query": QueryHead(), "peer": RelationHead()}
    for key, model in models.items():
        model.load_state_dict(checkpoint["heads"][key])
        model.eval().requires_grad_(False)
    bank_path = recorded(directory / "bank.json", directory, extraction_summary, evidence)
    bank = json.loads(bank_path.read_bytes())
    if bank != bank_from_queries(queries, names(cfg["splits"]["train_numbers"])):
        raise ValueError("bank not equal to train-only GT-free anchors")
    output, began = start(output)
    (output / "tracks/A").mkdir(parents=True)
    (output / "scores").mkdir()
    dump(output / "config.json", cfg)
    dump(output / "bank.json", bank)
    count, eligible = 0, 0
    for sequence, qs in queries.items():
        keys = [(sequence, q["row_index"]) for q in qs if q["eligible"]]
        values = {}
        with torch.no_grad():
            for offset in range(0, len(keys), 64):
                chunk = keys[offset:offset + 64]
                x, s, r, m = batch_data(chunk, queries, bodies)
                _, _, cross, cm = batch_data(chunk, queries, bodies, bank)
                predictions = torch.stack((models["query"](x, s), models["peer"](x, s, r, m),
                                           models["peer"](x, s, cross, cm)), 1).numpy()
                if not np.isfinite(predictions).all():
                    raise ValueError("nonfinite head score")
                for (_, index), prediction in zip(chunk, predictions):
                    values[index] = prediction
        scored = []
        for q in qs:
            item = dict(q)
            prediction = values.get(q["row_index"], ["", "", ""])
            item.update(zip(ARMS, [float(v) if v != "" else "" for v in prediction]))
            item["cross_refs"] = select_cross(bank, sequence, q)
            scored.append(item)
        write_queries(output / "scores" / f"{sequence}.csv", scored, SCORE_COLUMNS)
        (output / "tracks/A" / f"{sequence}.txt").write_bytes(b"".join(row.raw for row in rows[sequence]))
        count += len(qs)
        eligible += len(keys)
    return finish(output, began, evidence, "score", device="cpu", protocol=cfg["peer_relation"],
                  sequence_names=list(queries), sequence_count=len(queries), input_rows=count,
                  eligible_rows=eligible, score_kind="raw_logits", bank_objects=len(bank))


def apply(scoring, thresholds_path, output):
    evidence = Evidence()
    directory, summary = check_stage(scoring, evidence, "score")
    threshold_file = evidence.record(thresholds_path)
    thresholds = json.loads(threshold_file.read_bytes())
    if set(thresholds) != set(ARMS) or any(type(v) not in (int, float) or not math.isfinite(v)
                                         for v in thresholds.values()):
        raise ValueError("thresholds must be finite query/peer/cross raw logits")
    output, began = start(output)
    for arm in ("A",) + tuple(f"A.{a}" for a in ARMS):
        (output / "tracks" / arm).mkdir(parents=True)
    dump(output / "thresholds.json", thresholds)
    count, retained = 0, dict.fromkeys(ARMS, 0)
    for sequence in summary["sequence_names"]:
        source = recorded(directory / "tracks/A" / f"{sequence}.txt", directory, summary, evidence)
        rows = read_rows(source)
        query_path = recorded(directory / "scores" / f"{sequence}.csv", directory, summary, evidence)
        qs = read_queries(query_path, rows, scoring=True)
        (output / "tracks/A" / source.name).write_bytes(source.read_bytes())
        for arm in ARMS:
            kept = [row.raw for row, q in zip(rows, qs)
                    if not q["eligible"] or not q[arm] < thresholds[arm]]
            (output / "tracks" / f"A.{arm}" / source.name).write_bytes(b"".join(kept))
            retained[arm] += len(kept)
        count += len(rows)
    return finish(output, began, evidence, "apply", sequence_names=summary["sequence_names"],
                  sequence_count=len(summary["sequence_names"]), input_rows=count,
                  thresholds=thresholds, retained_rows=retained, strict_less_rejection=True,
                  inference_used=False)
