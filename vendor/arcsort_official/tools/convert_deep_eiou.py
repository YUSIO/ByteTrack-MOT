"""Convert Deep-EIoU released SportsMOT assets to ARC-SORT cache formats.

Deep-EIoU (https://github.com/hsiangwei0903/Deep-EIoU) releases per-sequence
YOLOX detections and OSNet embeddings for the SportsMOT test split. Using
them directly makes comparisons against Deep-EIoU's published numbers
input-identical: any metric difference is attributable to association alone.

Input formats (one .npy per sequence, object array indexed by frame-1):
    detection/<seq>.npy: each element (N, 7) float32
        [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
        final detection score = obj_conf * cls_conf (ByteTrack convention)
    embedding/<seq>.npy: each element (N, D) float32, row-aligned with
        the detections of the same frame.

Outputs:
    detections cache: MOT txt (frame,-1,x,y,w,h,score,-1,-1,-1) compatible
        with tools/run_cached.py --detections
    embedding cache: per-sequence .npz (boxes_{fid}, embs_{fid}) compatible
        with tools/run_cached.py --emb-cache

Box round-trip note: the detection txt stores values with 2 decimals and
run_cached.load_detections reconstructs x2 = x1 + w. The npz boxes are
computed through the SAME quantization so the embedding-cache bbox keys
match the boxes the tracker actually sees.

Usage:
    PYTHONPATH=. python tools/convert_deep_eiou.py \
        --assets deep_eiou_assets \
        --out-dets detections/sportsmot_test_deepeiou \
        --out-embs emb_cache/sportsmot_test_deepeiou
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from loguru import logger


def quantize_box(x1: float, y1: float, x2: float, y2: float) -> tuple:
    """Apply the txt-format 2-decimal round trip to a box.

    Returns the box as it will appear after being written as
    (x1, y1, w, h) with %.2f and read back as (x1, y1, x1+w, y1+h).
    """
    x1q = round(float(x1), 2)
    y1q = round(float(y1), 2)
    wq = round(float(x2) - float(x1), 2)
    hq = round(float(y2) - float(y1), 2)
    return x1q, y1q, x1q + wq, y1q + hq


def read_image_size(seqinfo_file: Path) -> tuple[int, int]:
    """Read (imWidth, imHeight) from a MOT seqinfo.ini."""
    import configparser

    cp = configparser.ConfigParser()
    cp.read(seqinfo_file)
    seq = cp["Sequence"]
    return int(seq["imWidth"]), int(seq["imHeight"])


def convert_sequence(
    det_file: Path,
    emb_file: Path | None,
    out_det_file: Path,
    out_emb_file: Path | None,
    rescale: float = 1.0,
) -> tuple[int, int]:
    """Convert one sequence's detections (and optionally embeddings).

    Args:
        det_file: Deep-EIoU detection .npy.
        emb_file: Deep-EIoU embedding .npy, or None.
        out_det_file: Output MOT txt path.
        out_emb_file: Output .npz path, or None.
        rescale: Divisor applied to box coordinates. Deep-EIoU's npy
            boxes are in the 800x1440 letterboxed network-input space;
            pass min(1440/imW, 800/imH) to map back to image pixels.

    Returns:
        (num_detections, num_embeddings) written.
    """
    per_frame_dets = np.load(det_file, allow_pickle=True)
    per_frame_embs = None
    if emb_file is not None and emb_file.exists():
        per_frame_embs = np.load(emb_file, allow_pickle=True)
        if len(per_frame_embs) != len(per_frame_dets):
            logger.warning(
                f"{det_file.stem}: frame count mismatch "
                f"dets={len(per_frame_dets)} embs={len(per_frame_embs)}; "
                "skipping embeddings for this sequence"
            )
            per_frame_embs = None

    lines = []
    emb_arrays: dict[str, np.ndarray] = {}
    n_dets = 0
    n_embs = 0

    for frame_idx, dets in enumerate(per_frame_dets):
        frame_id = frame_idx + 1
        if dets is None or len(dets) == 0:
            continue
        dets = np.asarray(dets, dtype=np.float64)
        dets = dets.copy()
        dets[:, :4] /= rescale
        scores = dets[:, 4] * dets[:, 5]

        boxes_q = []
        for (x1, y1, x2, y2), score in zip(dets[:, :4], scores):
            x1q, y1q, x2q, y2q = quantize_box(x1, y1, x2, y2)
            lines.append(
                f"{frame_id},-1,{x1q:.2f},{y1q:.2f},"
                f"{x2q - x1q:.2f},{y2q - y1q:.2f},{score:.5f},-1,-1,-1"
            )
            boxes_q.append([x1q, y1q, x2q, y2q])
        n_dets += len(boxes_q)

        if per_frame_embs is not None:
            embs = np.asarray(per_frame_embs[frame_idx], dtype=np.float32)
            # Deep-EIoU stores (N, 1, D); flatten any extra dims to (N, D)
            if embs.ndim > 2:
                embs = embs.reshape(embs.shape[0], -1)
            if embs.shape[0] != len(boxes_q):
                logger.warning(
                    f"{det_file.stem} frame {frame_id}: row mismatch "
                    f"dets={len(boxes_q)} embs={embs.shape[0]}; skipping frame embs"
                )
            else:
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / np.maximum(norms, 1e-12)
                emb_arrays[f"boxes_{frame_id}"] = np.asarray(
                    boxes_q, dtype=np.float32
                )
                emb_arrays[f"embs_{frame_id}"] = embs.astype(np.float16)
                n_embs += embs.shape[0]

    out_det_file.parent.mkdir(parents=True, exist_ok=True)
    out_det_file.write_text("\n".join(lines) + ("\n" if lines else ""))

    if out_emb_file is not None and emb_arrays:
        out_emb_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_emb_file, **emb_arrays)

    return n_dets, n_embs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Deep-EIoU SportsMOT assets to ARC-SORT caches"
    )
    parser.add_argument(
        "--assets", type=str, required=True,
        help="Deep-EIoU assets dir containing detection/ and embedding/",
    )
    parser.add_argument(
        "--out-dets", type=str, required=True,
        help="Output directory for MOT-format detection txt files",
    )
    parser.add_argument(
        "--out-embs", type=str, default=None,
        help="Output directory for embedding .npz caches (omit to skip)",
    )
    parser.add_argument(
        "--seqinfo-root", type=str, default=None,
        help="Dataset split root containing <seq>/seqinfo.ini. When given, "
             "boxes are rescaled from the 800x1440 network-input space back "
             "to image pixels via min(1440/imW, 800/imH). REQUIRED for "
             "Deep-EIoU's released files (they are input-space).",
    )
    parser.add_argument(
        "--input-size", type=int, nargs=2, default=[800, 1440],
        help="Network input size (H W) used to compute the rescale factor",
    )
    args = parser.parse_args()

    assets = Path(args.assets)
    det_dir = assets / "detection"
    emb_dir = assets / "embedding"
    out_dets = Path(args.out_dets)
    out_embs = Path(args.out_embs) if args.out_embs else None

    det_files = sorted(det_dir.glob("*.npy"))
    if not det_files:
        raise FileNotFoundError(f"No detection .npy files in {det_dir}")

    total_dets = 0
    total_embs = 0
    for det_file in det_files:
        seq = det_file.stem
        emb_file = emb_dir / f"{seq}.npy" if emb_dir.exists() else None
        out_emb_file = (out_embs / f"{seq}.npz") if out_embs else None

        rescale = 1.0
        if args.seqinfo_root:
            seqinfo = Path(args.seqinfo_root) / seq / "seqinfo.ini"
            if not seqinfo.exists():
                raise FileNotFoundError(f"seqinfo.ini not found: {seqinfo}")
            img_w, img_h = read_image_size(seqinfo)
            in_h, in_w = args.input_size
            rescale = min(in_w / img_w, in_h / img_h)

        n_dets, n_embs = convert_sequence(
            det_file, emb_file, out_dets / f"{seq}.txt", out_emb_file,
            rescale=rescale,
        )
        total_dets += n_dets
        total_embs += n_embs
        logger.info(f"{seq}: {n_dets} dets, {n_embs} embs")

    logger.info(
        f"Converted {len(det_files)} sequences: "
        f"{total_dets} detections, {total_embs} embeddings"
    )


if __name__ == "__main__":
    main()
