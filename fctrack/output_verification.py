"""Score every frozen A output with Seed and retain exact source MOT lines.

This is a current-frame output filter, not a tracker. Saved rounded boxes are
queries only; the verifier's regressed boxes are discarded. No GT, detector
cache, future image, tracker state, or evaluator is accessed.
"""
import configparser
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


QUERY_COLUMNS = ("frame", "row_index", "id", "x", "y", "w", "h", "p", "keep")
FIXED_PROTOCOL = {
    "threshold": 0.97,
    "kind": "seed",
    "query_coordinates": "saved_rounded_output",
    "regression_used": False,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SavedRow:
    frame: int
    row_index: int
    track_id: int
    box: tuple
    raw: bytes


def _parse_row(raw, row_index, path):
    fields = raw.decode("ascii").strip().split(",")
    if len(fields) != 10:
        raise ValueError(f"{path}: row {row_index}: expected 10 MOT fields")
    values = np.asarray([float(v) for v in fields], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: row {row_index}: non-finite MOT value")
    frame, track_id = values[:2]
    if frame < 1 or int(frame) != frame or track_id < 1 or int(track_id) != track_id:
        raise ValueError(f"{path}: row {row_index}: invalid frame or ID")
    if np.any(values[4:6] <= 0):
        raise ValueError(f"{path}: row {row_index}: non-positive box extent")
    return SavedRow(int(frame), row_index, int(track_id), tuple(values[2:6]), raw)


def _load_verifier(kind, weights, device):
    # Lazy imports keep the file/selection tests independent of model inference.
    import torch
    from fctrack.verifier import load, verify

    target = torch.device(device)
    model = load(kind, weights, target)
    return lambda image, boxes: verify(kind, model, image, boxes, target)


def _decode_image(raw, path):
    import cv2

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return image


def verify_output_tracks(root, config_path, cell_name, output, device,
                         *, verifier_loader=None, image_decoder=None):
    """Run the fixed Exp151 filter; optional callables support isolated tests.

    The verifier is called once per nonempty frame, for every saved row in its
    original order. A one-row file-parser lookahead only detects frame boundaries;
    scoring never receives a later row/image or any track history. Empty frames
    and empty sequences need no image inference. ``row_index`` is zero-based in
    the source sequence's MOT file. Output creation refuses existing directories.
    """
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    output = Path(output).resolve()
    config_bytes = config_path.read_bytes()
    cfg = json.loads(config_bytes)
    protocol = cfg["output_verification"]
    if protocol != FIXED_PROTOCOL or protocol["regression_used"] is not False:
        raise ValueError(f"output_verification must equal {FIXED_PROTOCOL!r}")
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
        found = {p.stem for p in sources}
        if requested - found:
            raise ValueError(f"missing reference sequences: {sorted(requested - found)}")
        sources = [p for p in sources if p.stem in requested]
    if not sources:
        raise ValueError(f"no reference MOT files: {references}")
    if output.exists():
        raise FileExistsError(output)

    input_hashes = {str(config_path): hashlib.sha256(config_bytes).hexdigest()}
    input_kinds = {str(config_path): "config"}
    immutable_inputs = {config_path, weights}

    def record_bytes(path, raw, kind):
        key = str(path.absolute())
        digest = hashlib.sha256(raw).hexdigest()
        if key in input_hashes and input_hashes[key] != digest:
            raise RuntimeError(f"input changed while reading: {path}")
        input_hashes[key] = digest
        input_kinds[key] = kind
        return digest

    input_hashes[str(weights)] = sha256(weights)
    input_kinds[str(weights)] = "verifier_weights"
    load_start = time.perf_counter()
    verify = (verifier_loader or _load_verifier)("seed", weights, device)
    model_load_seconds = time.perf_counter() - load_start
    decode = image_decoder or _decode_image
    (output / "tracks" / "A").mkdir(parents=True, exist_ok=False)
    (output / "tracks" / "A.verify").mkdir()
    (output / "queries").mkdir()
    sequences = {}
    output_hashes = {}
    reference_hashes = {}
    a_hashes, verified_hashes = {}, {}

    for source in sources:
        name = source.stem
        sequence_start = time.perf_counter()
        seq_dir = data / name
        info_path = seq_dir / "seqinfo.ini"
        info_bytes = info_path.read_bytes()
        record_bytes(info_path, info_bytes, "sequence_info")
        immutable_inputs.add(info_path)
        parser = configparser.ConfigParser()
        parser.read_string(info_bytes.decode("utf-8"))
        info = parser["Sequence"]
        length = info.getint("seqLength")
        if length < 1:
            raise ValueError(f"invalid sequence length: {info_path}")
        image_directory = info.get("imDir", "img1")
        image_extension = info.get("imExt", ".jpg")
        # The dataset uses image-only child directories. Do not allow metadata
        # to redirect the reader outside a sequence or into a GT directory.
        image_parts = Path(image_directory).parts
        if (Path(image_directory).is_absolute() or ".." in image_parts
                or "gt" in image_parts or not image_extension.startswith(".")
                or "/" in image_extension or "\\" in image_extension):
            raise ValueError(f"invalid image location: {info_path}")
        input_hashes[str(source.absolute())] = sha256(source)
        input_kinds[str(source.absolute())] = "reference_tracks"
        immutable_inputs.add(source)
        copied_path = output / "tracks" / "A" / source.name
        kept_path = output / "tracks" / "A.verify" / source.name
        query_path = output / "queries" / f"{name}.csv"
        counts = dict(frame_count=length, scored_frame_count=0, input_rows=0,
                      kept_rows=0, rejected_rows=0, verifier_calls=0,
                      verifier_seconds=0.0)
        source_digest = hashlib.sha256()

        with source.open("rb") as original, copied_path.open("wb") as copied, \
                kept_path.open("wb") as kept, query_path.open("w", newline="") as query:
            writer = csv.writer(query, lineterminator="\n")
            writer.writerow(QUERY_COLUMNS)

            def process_frame(rows):
                if not rows:
                    return
                frame = rows[0].frame
                image_path = seq_dir / image_directory / f"{frame:06d}{image_extension}"
                raw_image = image_path.read_bytes()
                record_bytes(image_path, raw_image, "frame_image")
                image = decode(raw_image, image_path)
                boxes = np.asarray([row.box for row in rows], dtype=float)
                call_start = time.perf_counter()
                probabilities, _unused_regressed_boxes = verify(image, boxes)
                counts["verifier_seconds"] += time.perf_counter() - call_start
                # Never read, validate, select, or write the regression output.
                probabilities = np.asarray(probabilities, dtype=float)
                if (probabilities.shape != (len(rows),)
                        or not np.isfinite(probabilities).all()
                        or np.any((probabilities < 0) | (probabilities > 1))):
                    raise ValueError(f"invalid verifier probabilities: {name} frame {frame}")
                counts["scored_frame_count"] += 1
                counts["verifier_calls"] += 1
                for row, probability in zip(rows, probabilities):
                    keep = bool(probability >= protocol["threshold"])
                    if keep:
                        kept.write(row.raw)
                    writer.writerow((row.frame, row.row_index, row.track_id,
                                     *row.box, float(probability), int(keep)))
                    counts["input_rows"] += 1
                    counts["kept_rows"] += int(keep)
                    counts["rejected_rows"] += int(not keep)

            frame_rows = []
            previous_frame = 0
            for row_index, raw_line in enumerate(original):
                source_digest.update(raw_line)
                copied.write(raw_line)
                row = _parse_row(raw_line, row_index, source)
                if row.frame < previous_frame or row.frame > length:
                    raise ValueError(f"noncausal or out-of-range frame: {source} row {row_index}")
                if row.frame != previous_frame:
                    process_frame(frame_rows)
                    frame_rows = []
                frame_rows.append(row)
                previous_frame = row.frame
            process_frame(frame_rows)

        digest = source_digest.hexdigest()
        if digest != input_hashes[str(source.absolute())]:
            raise RuntimeError(f"reference changed during filtering: {source}")
        reference_hashes[name] = digest
        for path in (copied_path, kept_path, query_path):
            output_hashes[str(path.relative_to(output))] = sha256(path)
        a_hashes[name] = output_hashes[str(copied_path.relative_to(output))]
        verified_hashes[name] = output_hashes[str(kept_path.relative_to(output))]
        if a_hashes[name] != digest:
            raise RuntimeError(f"copied A differs from reference: {source}")
        counts["wall_seconds"] = time.perf_counter() - sequence_start
        sequences[name] = counts

    # Images are hashed from the exact bytes decoded, not from a later reread.
    # Recheck compact control inputs and all original MOT files before success.
    for path in sorted(immutable_inputs):
        if sha256(path) != input_hashes[str(path.absolute())]:
            raise RuntimeError(f"input changed during execution: {path}")
    summed = ("frame_count", "scored_frame_count", "input_rows", "kept_rows",
              "rejected_rows", "verifier_calls", "verifier_seconds")
    summary = dict(schema_version=1, status="completed", cell=cell_name,
                   sequence_names=list(sequences), sequence_count=len(sequences),
                   **{key: sum(s[key] for s in sequences.values()) for key in summed},
                   per_sequence=sequences, protocol=protocol, device=str(device),
                   started_utc=started_utc, finished_utc=datetime.now(timezone.utc).isoformat(),
                   model_load_seconds=model_load_seconds,
                   wall_seconds=time.perf_counter() - started,
                   input_sha256=input_hashes, input_kinds=input_kinds,
                   output_sha256=output_hashes, reference_tracks_sha256=reference_hashes,
                   A_track_sha256=a_hashes, verified_track_sha256=verified_hashes,
                   row_index_semantics="zero-based source MOT line index per sequence",
                   image_hash_semantics="exact bytes decoded, only frames containing A outputs",
                   gt_used=False, evaluator_called=False, tracker_called=False,
                   state_feedback=False,
                   output_semantics="A exact copy; A.verify exact retained source-line subsequence")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
