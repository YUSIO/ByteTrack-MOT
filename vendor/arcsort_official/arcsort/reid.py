"""Re-identification (ReID) embedding computation for ARC-SORT.

Provides the EmbeddingComputer class which lazy-loads a TorchScript-traced
ReID model (FastReID SBS-S50 or sports OSNet) and computes L2-normalized
appearance embeddings from detection crops. Also provides IMM-gating helpers
(selective ReID activation based on IMM model probabilities).

The model is loaded as a standalone TorchScript file with zero external
dependencies. Weights must first be converted from FastReID format using
``tools/convert_reid_weights.py`` (one-time step).

Preprocessing matches the Deep-OC-SORT reference: BGR->RGB, resize to
128x384, float32 [0-255], FP16 inference, L2 normalization.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

# Embedding dimension for SBS ResNeSt-50
EMBEDDING_DIM = 2048


class EmbeddingComputer:
    """Computes appearance embeddings from detection crops.

    Loads a TorchScript-traced ReID model (no FastReID dependency at
    runtime). Crops are extracted from the frame, converted BGR->RGB,
    resized to 128x384, batched, and run through the model with FP16
    inference. Output embeddings are L2-normalized.

    Args:
        weights_path: Path to the TorchScript .pt model file.
        crop_size: (width, height) tuple for cv2.resize. Default
            (128, 384) = 128px wide, 384px tall (person ReID standard).
    """

    def __init__(
        self,
        weights_path: str = "weights/dance_sbs_S50_traced.pt",
        crop_size: tuple[int, int] = (128, 384),
    ) -> None:
        self.weights_path = weights_path
        self.crop_size = crop_size  # (W, H) for cv2.resize
        self.model = None

        # Optional pre-computed embedding cache (see tools/cache_embeddings.py)
        # Maps frame_id -> (boxes (N, 4) float64, embs (N, D) float32).
        # Lookups match by nearest box within CACHE_MATCH_TOL pixels --
        # immune to float32/float64 and formatting round-trip noise.
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
        self._cache_frame_id: int | None = None
        self._cache_miss_warned = False
        # Embedding dimension; overridden by the cache (e.g. 512 for OSNet)
        self._emb_dim = EMBEDDING_DIM

    # Max per-coordinate deviation (px) for a cache lookup to count as a hit
    CACHE_MATCH_TOL = 0.5

    def load_cache(self, cache_file: str | Path) -> None:
        """Load a per-sequence embedding cache produced by
        tools/cache_embeddings.py.

        The cache maps each frame to {bbox_key: embedding}. When active,
        compute_embeddings() serves lookups from the cache (no GPU, no
        frame images needed); boxes missing from the cache fall back to
        the model if a frame is available, else zero embeddings.

        Args:
            cache_file: Path to a .npz file with per-frame arrays
                "boxes_{frame_id}" (N, 4) and "embs_{frame_id}" (N, D).
        """
        data = np.load(str(cache_file))
        cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for key in data.files:
            if not key.startswith("boxes_"):
                continue
            frame_id = int(key.split("_")[1])
            boxes = data[key].astype(np.float64)
            embs = data[f"embs_{frame_id}"].astype(np.float32)
            if embs.ndim > 2:
                embs = embs.reshape(embs.shape[0], -1)
            # Re-normalize (fp16 storage slightly denormalizes)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / np.maximum(norms, 1e-12)
            cache[frame_id] = (boxes, embs)
            if embs.shape[0] > 0:
                self._emb_dim = embs.shape[1]
        self._cache = cache
        self._cache_frame_id = None
        self._cache_miss_warned = False
        logger.debug(f"Loaded embedding cache: {cache_file} ({len(cache)} frames)")

    def has_cache(self) -> bool:
        """Whether a pre-computed embedding cache is active."""
        return self._cache is not None

    def set_frame(self, frame_id: int) -> None:
        """Set the current frame id for cache lookups."""
        self._cache_frame_id = frame_id

    def _load_model(self) -> None:
        """Lazy-load the TorchScript ReID model."""
        import torch

        weights = self.weights_path
        if not os.path.isabs(weights):
            project_root = Path(__file__).resolve().parent.parent
            weights = str(project_root / weights)

        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"ReID TorchScript model not found at {weights}. "
                f"Run: PYTHONPATH=. python tools/convert_reid_weights.py "
                f"--input weights/dance_sbs_S50.pth "
                f"--output weights/dance_sbs_S50_traced.pt "
                f"--config dancetrack"
            )

        logger.info(f"Loading ReID TorchScript model from {weights}")
        self.model = torch.jit.load(weights, map_location="cuda")
        self.model.eval()
        logger.info("ReID model loaded successfully")

    def compute_embeddings(
        self, frame: np.ndarray, bboxes: np.ndarray
    ) -> np.ndarray:
        """Compute L2-normalized embeddings for detection bounding boxes.

        Args:
            frame: BGR image, shape (H, W, 3), dtype uint8.
            bboxes: Detection bounding boxes, shape (N, 4) in
                [x1, y1, x2, y2] format.

        Returns:
            L2-normalized embeddings, shape (N, 2048), dtype float32.
            Returns zeros if bboxes is empty, or if frame is None and no
            cache entry covers a box.
        """
        if bboxes.shape[0] == 0:
            return np.zeros((0, self._emb_dim), dtype=np.float32)

        # Serve from the pre-computed cache when active
        if self._cache is not None and self._cache_frame_id is not None:
            entry = self._cache.get(self._cache_frame_id)
            cached_boxes = entry[0] if entry is not None else np.empty((0, 4))
            cached_embs = entry[1] if entry is not None else None
            embs = np.zeros((bboxes.shape[0], self._emb_dim), dtype=np.float32)
            misses = []
            for i, bbox in enumerate(bboxes):
                hit = False
                if cached_boxes.shape[0] > 0:
                    diffs = np.abs(
                        cached_boxes - np.asarray(bbox[:4], dtype=np.float64)
                    ).max(axis=1)
                    j = int(np.argmin(diffs))
                    if diffs[j] <= self.CACHE_MATCH_TOL:
                        embs[i] = cached_embs[j]
                        hit = True
                if not hit:
                    misses.append(i)
            if misses:
                if frame is not None:
                    miss_boxes = bboxes[np.array(misses)]
                    embs[np.array(misses)] = self._compute_from_frame(
                        frame, miss_boxes
                    )
                elif not self._cache_miss_warned:
                    logger.warning(
                        f"{len(misses)} boxes missing from embedding cache "
                        f"at frame {self._cache_frame_id} and no frame "
                        "available; using zero embeddings"
                    )
                    self._cache_miss_warned = True
            return embs

        if frame is None:
            return np.zeros((bboxes.shape[0], self._emb_dim), dtype=np.float32)

        return self._compute_from_frame(frame, bboxes)

    def _compute_from_frame(
        self, frame: np.ndarray, bboxes: np.ndarray
    ) -> np.ndarray:
        """Run the ReID model on crops extracted from the frame."""
        import torch

        if self.model is None:
            self._load_model()

        h, w = frame.shape[:2]

        # Clip bboxes to image bounds
        coords = np.round(bboxes).astype(np.int32)
        coords[:, 0] = np.clip(coords[:, 0], 0, w)
        coords[:, 1] = np.clip(coords[:, 1], 0, h)
        coords[:, 2] = np.clip(coords[:, 2], 0, w)
        coords[:, 3] = np.clip(coords[:, 3], 0, h)

        # Extract and preprocess crops
        crops = []
        for p in coords:
            crop = frame[p[1]:p[3], p[0]:p[2]]
            if crop.size == 0:
                # Degenerate bbox -- black crop (H=crop_size[1], W=crop_size[0])
                crop = np.zeros(
                    (self.crop_size[1], self.crop_size[0], 3), dtype=np.uint8
                )
            else:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                # cv2.resize dsize is (width, height)
                crop = cv2.resize(
                    crop, self.crop_size, interpolation=cv2.INTER_LINEAR
                )
            # HWC -> CHW float32 tensor
            tensor = torch.as_tensor(
                crop.astype("float32").transpose(2, 0, 1)
            )
            crops.append(tensor.unsqueeze(0))

        batch = torch.cat(crops, dim=0)

        with torch.no_grad():
            batch = batch.cuda().half()
            embs = self.model(batch)

        embs = torch.nn.functional.normalize(embs)
        return embs.cpu().numpy()


def should_activate_reid(
    track,
    cv_thresh: float = 0.7,
    young_frames: int = 5,
) -> bool:
    """Decide whether ReID should be active for a given track (IMM gating).

    When a track's CV (constant velocity) model probability exceeds the
    threshold and the track is old enough, its motion is predictable and
    ReID is unnecessary. This reduces computation and prevents appearance
    noise from disrupting good motion-based matches.

    Args:
        track: STrack instance with IMM-KF.
        cv_thresh: If prob(CV) exceeds this, skip ReID. Default 0.7.
        young_frames: Always activate ReID for tracks younger than this
            many frames (they haven't established reliable motion yet).

    Returns:
        True if ReID should be activated for this track, False to skip it.
    """
    # Always use ReID for young tracks
    if track.age <= young_frames:
        return True

    # Only gate when using IMM (otherwise always activate)
    if not track.use_imm:
        return True

    # Check CV model probability (0.0 if CV is not configured, which
    # means motion is treated as non-linear and ReID stays active)
    cv_prob = track.kf.get_model_prob("cv")

    # High CV probability -> motion is predictable -> skip ReID
    return bool(cv_prob <= cv_thresh)


def compute_imm_adaptive_weights(
    tracks,
    base_weight: float = 1.25,
    cv_discount: float = 0.5,
    young_frames: int = 5,
) -> np.ndarray:
    """Compute per-track ReID weights adapted by IMM model probabilities.

    The key insight: tracks under non-linear motion (high CA/CT probability)
    have unreliable motion predictions, so appearance features should be
    weighted more heavily. Tracks with predictable CV motion can rely more
    on spatial matching, so ReID weight can be reduced.

    This is unique to ARC-SORT -- no other tracker has per-track motion
    model probabilities to condition appearance weight on.

    The weight for track i is:
        w_i = base_weight * (1 - cv_discount * cv_prob_i)

    When cv_prob=1.0 (pure CV): w = base * (1 - discount) = reduced
    When cv_prob=0.0 (pure CA/CT): w = base = full weight
    Young tracks always get full weight (motion not yet established).

    Args:
        tracks: List of STrack instances.
        base_weight: Maximum ReID weight (when motion is unpredictable).
        cv_discount: How much to reduce weight for high-CV tracks.
            Range [0, 1]. 0 = no adaptation, 1 = zero weight for pure CV.
        young_frames: Tracks younger than this always get full weight.

    Returns:
        Per-track weights, shape (M,).
    """
    if len(tracks) == 0:
        return np.empty(0)

    weights = np.full(len(tracks), base_weight)

    for i, track in enumerate(tracks):
        # Young tracks: full weight (haven't established motion model)
        if track.age <= young_frames:
            continue

        # Non-IMM tracks: full weight (no model probabilities available)
        if not track.use_imm:
            continue

        # Look up CV by name (0.0 if CV is not configured -> full weight)
        cv_prob = track.kf.get_model_prob("cv")

        # Scale weight inversely with CV probability
        weights[i] = base_weight * (1.0 - cv_discount * cv_prob)

    return weights
