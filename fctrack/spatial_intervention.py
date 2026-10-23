"""Fixed current-crop interventions on Seed, followed by exact-line filtering.

The central square is normalized central support, not a bbox footprint or the
Seed positive-label disk. Score differences measure perturbation sensitivity;
they are not semantic UAV probabilities. Neither stage reads GT, updates a
tracker, changes a box, or uses the verifier regression outputs.
"""
import configparser
import csv
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import time

import numpy as np

from fctrack.output_verification import _decode_image, _parse_row, sha256


VIEWS = ("original", "center", "tl", "tr", "bl", "br")
LOGIT_COLUMNS = ("logit0", "logit_center", "logit_tl", "logit_tr", "logit_bl", "logit_br")
SCORES = ("raw", "paired", "sham")
QUERY_COLUMNS = ("frame", "row_index", "id", "x", "y", "w", "h", *LOGIT_COLUMNS, *SCORES)
DECISION_COLUMNS = ("frame", "row_index", "id", *SCORES, "keep_raw", "keep_paired", "keep_sham")
FIXED_PROTOCOL = {
    "kind": "seed",
    "query_coordinates": "saved_rounded_output",
    "crop_size": 64,
    "central_support": [22, 43, 22, 43],
    "corners": {"tl": [0, 21, 0, 21], "tr": [0, 21, 43, 64],
                "bl": [43, 64, 0, 21], "br": [43, 64, 43, 64]},
    "fill": "original_crop_channel_median",
    "raw": "logit0",
    "paired": "mean_corner_logits_minus_center",
    "sham": "mean_top_logits_minus_mean_bottom_logits",
    "regression_used": False,
}


def intervention_masks():
    """Five disjoint boolean masks with identical 441-pixel area."""
    rectangles = {"center": FIXED_PROTOCOL["central_support"], **FIXED_PROTOCOL["corners"]}
    masks = {}
    for name, (y0, y1, x0, x1) in rectangles.items():
        mask = np.zeros((64, 64), dtype=bool)
        mask[y0:y1, x0:x1] = True
        masks[name] = mask
    return masks


def crop_views(crops):
    """Yield original then five masks, all filled from one original median.

    The input is not modified. Median is computed per row and channel over all
    4096 original pixels, including border replication from seed_crop.
    """
    crops = np.asarray(crops)
    if crops.ndim != 4 or crops.shape[1:] != (3, 64, 64):
        raise ValueError("expected N x 3 x 64 x 64 crops")
    if not np.isfinite(crops).all() or np.any((crops < 0) | (crops > 1)):
        raise ValueError("crop values must be finite RGB in [0, 1]")
    original = np.ascontiguousarray(crops, dtype=np.float32)
    medians = np.median(original, axis=(2, 3))
    yield "original", original
    for name, mask in intervention_masks().items():
        altered = original.copy()
        altered[:, :, mask] = medians[:, :, None]
        yield name, altered


def scores_from_logits(logits):
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or logits.shape[1] != 6 or not np.isfinite(logits).all():
        raise ValueError("expected finite N x 6 Seed logits")
    return np.column_stack((logits[:, 0], logits[:, 2:6].mean(axis=1) - logits[:, 1],
                            (logits[:, 2] + logits[:, 3] - logits[:, 4] - logits[:, 5]) / 2))


def _load_scorer(weights, device):
    import torch
    from fctrack.verifier import load, scalars, seed_crop

    target = torch.device(device)
    model = load("seed", weights, target)

    @torch.no_grad()
    def score(image, boxes):
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        side = np.sqrt(boxes[:, 2] * boxes[:, 3])
        centers = boxes[:, :2] + boxes[:, 2:] / 2
        crops = np.stack([seed_crop(image, x, y, s) for (x, y), s in zip(centers, side)])
        scale_input = torch.from_numpy(scalars(side, np.full(len(side), image.shape[1], float))).to(target)
        logits = []
        for _, view in crop_views(crops):
            # Same frame-row batch for all six forwards, including original.
            # Preserve the native logit: sigmoid + inverse-logit loses extremes.
            values = model(torch.from_numpy(view).to(target), scale_input)
            logits.append(values[:, 0].cpu().numpy().astype(float))
        return np.column_stack(logits)

    return score


def _save_summary(output, summary):
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def _now():
    return datetime.now(timezone.utc).isoformat()


def extract(root, config_path, cell_name, output, device, *, scorer_loader=None, image_decoder=None):
    """Score six current-crop views per saved row. GT is never an input."""
    started, started_utc = time.perf_counter(), _now()
    root, config_path, output = Path(root).resolve(), Path(config_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    config_bytes = config_path.read_bytes()
    cfg = json.loads(config_bytes)
    if cfg["spatial_intervention"] != FIXED_PROTOCOL:
        raise ValueError("spatial_intervention differs from the fixed protocol")
    cell = cfg["cells"][cell_name]
    spec = cell["verifiers"]["seed"]
    if spec.get("kind", "seed") != "seed":
        raise ValueError("only the original seed verifier is supported")
    weights = (root / spec["weights"]).resolve()
    data = root / cell["dataset_root"] / cell["split"]
    references = root / cell["reference_tracks"]
    sources = sorted(references.glob("UAVSwarm-*.txt"))
    if cell.get("sequences"):
        requested = set(cell["sequences"])
        missing = requested - {p.stem for p in sources}
        if missing:
            raise ValueError(f"missing reference sequences: {sorted(missing)}")
        sources = [p for p in sources if p.stem in requested]
    if not sources:
        raise ValueError(f"no reference MOT files: {references}")
    inputs, kinds = {}, {}

    def record(path, raw, kind):
        key = str(path.absolute())
        digest = hashlib.sha256(raw).hexdigest()
        if key in inputs and inputs[key] != digest:
            raise RuntimeError(f"input changed while reading: {path}")
        inputs[key], kinds[key] = digest, kind

    record(config_path, config_bytes, "config")
    inputs[str(weights)], kinds[str(weights)] = sha256(weights), "verifier_weights"
    immutable = {config_path, weights}
    load_start = time.perf_counter()
    score = (scorer_loader or _load_scorer)(weights, device)
    model_load_seconds = time.perf_counter() - load_start
    decode = image_decoder or _decode_image
    (output / "tracks/A").mkdir(parents=True, exist_ok=False)
    (output / "queries").mkdir()
    sequences, output_hashes, references_sha, a_sha = {}, {}, {}, {}

    for source in sources:
        name, sequence_start = source.stem, time.perf_counter()
        seq_dir = data / name
        info_path = seq_dir / "seqinfo.ini"
        info_bytes = info_path.read_bytes()
        record(info_path, info_bytes, "sequence_info")
        immutable.add(info_path)
        parser = configparser.ConfigParser()
        parser.read_string(info_bytes.decode("utf-8"))
        info = parser["Sequence"]
        length = info.getint("seqLength")
        im_dir, im_ext = info.get("imDir", "img1"), info.get("imExt", ".jpg")
        if (length < 1 or Path(im_dir).is_absolute() or ".." in Path(im_dir).parts
                or "gt" in Path(im_dir).parts or not im_ext.startswith(".")
                or "/" in im_ext or "\\" in im_ext):
            raise ValueError(f"invalid sequence metadata: {info_path}")
        inputs[str(source.absolute())], kinds[str(source.absolute())] = sha256(source), "reference_tracks"
        immutable.add(source)
        copied_path, query_path = output / "tracks/A" / source.name, output / "queries" / f"{name}.csv"
        counts = dict(frame_count=length, scored_frame_count=0, input_rows=0,
                      model_forward_calls=0, row_views=0, inference_seconds=0.0)
        digest = hashlib.sha256()
        with source.open("rb") as original, copied_path.open("wb") as copied, query_path.open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(QUERY_COLUMNS)

            def process_frame(rows):
                if not rows:
                    return
                image_path = seq_dir / im_dir / f"{rows[0].frame:06d}{im_ext}"
                raw = image_path.read_bytes()
                record(image_path, raw, "frame_image")
                image = decode(raw, image_path)
                boxes = np.asarray([r.box for r in rows], dtype=float)
                call_start = time.perf_counter()
                logits = np.asarray(score(image, boxes), dtype=float)
                if logits.shape != (len(rows), 6):
                    raise ValueError(f"invalid scorer row count: {name} frame {rows[0].frame}")
                values = scores_from_logits(logits)
                counts["inference_seconds"] += time.perf_counter() - call_start
                for row, logit, value in zip(rows, logits, values):
                    writer.writerow((row.frame, row.row_index, row.track_id, *row.box, *logit, *value))
                counts["scored_frame_count"] += 1
                counts["input_rows"] += len(rows)
                counts["model_forward_calls"] += 6
                counts["row_views"] += 6 * len(rows)

            rows, previous_frame = [], 0
            for index, raw in enumerate(original):
                digest.update(raw)
                copied.write(raw)
                row = _parse_row(raw, index, source)
                if row.frame < previous_frame or row.frame > length:
                    raise ValueError(f"noncausal or out-of-range frame: {source} row {index}")
                if row.frame != previous_frame:
                    process_frame(rows)
                    rows = []
                rows.append(row)
                previous_frame = row.frame
            process_frame(rows)
        references_sha[name] = digest.hexdigest()
        if references_sha[name] != inputs[str(source.absolute())]:
            raise RuntimeError(f"reference changed during extraction: {source}")
        for path in (copied_path, query_path):
            output_hashes[str(path.relative_to(output))] = sha256(path)
        a_sha[name] = output_hashes[str(copied_path.relative_to(output))]
        if a_sha[name] != references_sha[name]:
            raise RuntimeError(f"A copy differs from reference: {source}")
        counts["wall_seconds"] = time.perf_counter() - sequence_start
        sequences[name] = counts
    for path in sorted(immutable):
        if sha256(path) != inputs[str(path.absolute())]:
            raise RuntimeError(f"input changed during extraction: {path}")
    summed = ("frame_count", "scored_frame_count", "input_rows", "model_forward_calls", "row_views", "inference_seconds")
    return _save_summary(output, dict(
        schema_version=1, stage="extract", status="completed", cell=cell_name,
        sequence_names=list(sequences), sequence_count=len(sequences), per_sequence=sequences,
        **{k: sum(v[k] for v in sequences.values()) for k in summed},
        protocol=FIXED_PROTOCOL, device=str(device), started_utc=started_utc, finished_utc=_now(),
        model_load_seconds=model_load_seconds, wall_seconds=time.perf_counter() - started,
        input_sha256=inputs, input_kinds=kinds, output_sha256=output_hashes,
        reference_tracks_sha256=references_sha, A_track_sha256=a_sha,
        row_index_semantics="zero-based source MOT line index per sequence",
        image_hash_semantics="exact bytes decoded; only frames with A rows",
        output_semantics="A exact copy; unthresholded six-view logits and three scores",
        mask_semantics="central_support square, not bbox footprint or positive-label disk",
        gt_used=False, evaluator_called=False, tracker_called=False, state_feedback=False))


def apply(extraction, threshold_path, output):
    """Apply externally frozen thresholds without model, image, or GT access."""
    started, started_utc = time.perf_counter(), _now()
    extraction, threshold_path, output = Path(extraction).resolve(), Path(threshold_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    summary_path = extraction / "summary.json"
    summary_bytes, threshold_bytes = summary_path.read_bytes(), threshold_path.read_bytes()
    previous, threshold_record = json.loads(summary_bytes), json.loads(threshold_bytes)
    if (previous["status"] != "completed" or previous["stage"] != "extract"
            or previous["protocol"] != FIXED_PROTOCOL
            or any(previous[k] is not False for k in ("gt_used", "evaluator_called", "tracker_called", "state_feedback"))):
        raise ValueError("extraction does not satisfy the fixed GT-free protocol")
    thresholds = threshold_record["thresholds"]
    if (threshold_record.get("schema_version") != 1 or set(thresholds) != set(SCORES)
            or any(isinstance(t, bool) or not isinstance(t, (int, float)) or not np.isfinite(t)
                   for t in thresholds.values())):
        raise ValueError("thresholds must contain three finite raw/paired/sham values")
    names = previous["sequence_names"]
    if (not names or len(names) != len(set(names)) or previous["sequence_count"] != len(names)
            or any(not name.startswith("UAVSwarm-") or Path(name).name != name for name in names)):
        raise ValueError("invalid extraction sequence inventory")
    expected_paths = {f"{folder}/{name}.{suffix}" for name in names
                      for folder, suffix in (("tracks/A", "txt"), ("queries", "csv"))}
    if set(previous["output_sha256"]) != expected_paths:
        raise ValueError("extraction output inventory differs from sequence inventory")
    inputs = {str(summary_path): hashlib.sha256(summary_bytes).hexdigest(),
              str(threshold_path): hashlib.sha256(threshold_bytes).hexdigest()}
    for relative, digest in previous["output_sha256"].items():
        path = extraction / relative
        if sha256(path) != digest:
            raise RuntimeError(f"extraction input changed: {path}")
        inputs[str(path)] = digest
    for arm in ("A", *(f"A.{key}" for key in SCORES)):
        (output / "tracks" / arm).mkdir(parents=True, exist_ok=False)
    (output / "decisions").mkdir()
    sequences, output_hashes = {}, {}
    track_hashes = {arm: {} for arm in ("A", *(f"A.{key}" for key in SCORES))}
    for name in names:
        source, queries = extraction / "tracks/A" / f"{name}.txt", extraction / "queries" / f"{name}.csv"
        copied, decisions = output / "tracks/A" / source.name, output / "decisions" / f"{name}.csv"
        destinations = {k: output / "tracks" / f"A.{k}" / source.name for k in SCORES}
        counts = dict(input_rows=0, kept_rows={k: 0 for k in SCORES}, rejected_rows={k: 0 for k in SCORES})
        with source.open("rb") as original, queries.open(newline="") as query_stream, \
                copied.open("wb") as baseline, decisions.open("w", newline="") as decision_stream:
            reader = csv.DictReader(query_stream)
            if reader.fieldnames != list(QUERY_COLUMNS):
                raise ValueError(f"query columns changed: {queries}")
            writer = csv.writer(decision_stream, lineterminator="\n")
            writer.writerow(DECISION_COLUMNS)
            streams = {k: p.open("wb") for k, p in destinations.items()}
            try:
                previous_frame = 0
                for index, (raw, query) in enumerate(zip_longest(original, reader)):
                    if raw is None or query is None:
                        raise ValueError(f"query/reference row count mismatch: {name}")
                    row = _parse_row(raw, index, source)
                    if row.frame < previous_frame or row.frame > previous["per_sequence"][name]["frame_count"]:
                        raise ValueError(f"noncausal or out-of-range frame: {name} row {index}")
                    previous_frame = row.frame
                    if (int(query["row_index"]) != index or int(query["frame"]) != row.frame
                            or int(query["id"]) != row.track_id
                            or tuple(float(query[k]) for k in ("x", "y", "w", "h")) != row.box):
                        raise ValueError(f"query/reference identity mismatch: {name} row {index}")
                    logits = np.asarray([[float(query[k]) for k in LOGIT_COLUMNS]])
                    values = scores_from_logits(logits)[0]
                    if any(float(query[k]) != value for k, value in zip(SCORES, values)):
                        raise ValueError(f"query score formula mismatch: {name} row {index}")
                    keep = [not (value < thresholds[k]) for k, value in zip(SCORES, values)]
                    baseline.write(raw)
                    for k, retain in zip(SCORES, keep):
                        if retain:
                            streams[k].write(raw)
                        counts["kept_rows"][k] += int(retain)
                        counts["rejected_rows"][k] += int(not retain)
                    counts["input_rows"] += 1
                    writer.writerow((row.frame, index, row.track_id, *values, *(int(v) for v in keep)))
            finally:
                for stream in streams.values():
                    stream.close()
        if counts["input_rows"] != previous["per_sequence"][name]["input_rows"]:
            raise ValueError(f"query count differs from extraction summary: {name}")
        for path in (copied, decisions, *destinations.values()):
            output_hashes[str(path.relative_to(output))] = sha256(path)
        for arm in track_hashes:
            track_hashes[arm][name] = output_hashes[f"tracks/{arm}/{name}.txt"]
        if track_hashes["A"][name] != previous["A_track_sha256"][name]:
            raise RuntimeError(f"A copy differs from extraction: {name}")
        sequences[name] = counts
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise RuntimeError(f"input changed during apply: {path}")
    return _save_summary(output, dict(
        schema_version=1, stage="apply", status="completed", cell=previous["cell"],
        sequence_names=names, sequence_count=len(names), frame_count=previous["frame_count"],
        input_rows=sum(c["input_rows"] for c in sequences.values()),
        kept_rows={k: sum(c["kept_rows"][k] for c in sequences.values()) for k in SCORES},
        rejected_rows={k: sum(c["rejected_rows"][k] for c in sequences.values()) for k in SCORES},
        per_sequence=sequences, thresholds=thresholds, comparison="reject iff score < threshold; equality retained",
        protocol=FIXED_PROTOCOL, extraction=str(extraction), threshold_path=str(threshold_path),
        started_utc=started_utc, finished_utc=_now(), wall_seconds=time.perf_counter() - started,
        input_sha256=inputs, output_sha256=output_hashes, track_sha256=track_hashes,
        output_semantics="A exact copy; each other arm exact retained source-line subsequence",
        gt_used=False, evaluator_called=False, tracker_called=False, state_feedback=False,
        model_called=False, threshold_fitted=False))
