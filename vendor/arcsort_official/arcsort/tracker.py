"""ARCTracker -- main tracking orchestration.

Entry point for the ARC-SORT multi-object tracker. Receives detections
per frame and maintains a set of tracked objects. Delegates to specialized
modules for Kalman filtering, association, and OC-SORT algorithms.

This module handles orchestration only: detection splitting, track
prediction, association dispatching, track lifecycle management, and
output formatting. All algorithmic logic lives in other modules.
"""

from __future__ import annotations

import numpy as np
import yaml
from loguru import logger

from arcsort.association import (
    associate,
    associate_byte_second,
    associate_ocr,
    swap_margins,
)
from arcsort.cmc import CameraMotionCompensator
from arcsort.track import STrack, TrackState, reset_track_id


DEFAULT_CONFIG = {
    "track_high_thresh": 0.6,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.7,
    "track_buffer": 30,
    "match_thresh": 0.3,
    "byte_match_thresh": 0.5,
    "min_hits": 3,
    # Box reported for each output track: "detection" (the box matched
    # this frame) or "kalman" (the filtered state estimate). The paper
    # results (arXiv:2610.22306) were produced with "kalman".
    "output_box": "detection",
    "aspect_ratio_thresh": 0.0,
    "min_box_area": 0.0,
    "use_byte": False,
    "use_cmc": False,
    "use_imm": False,
    "use_hybrid_distance": False,
    "use_reid": False,
    "use_gated_reid": False,
    "use_mahalanobis_gate": False,
    # Chi-square 0.95 quantile for 4 degrees of freedom
    "mahalanobis_thresh": 9.4877,
    # ExpansionIoU association (Deep-EIoU style). eiou.mode selects the
    # expansion mechanism: "fixed" (constant), "adaptive" (scalar from
    # the IMM innovation covariance: predictable tracks keep tight
    # boxes, uncertain tracks search wider), "velocity_switch" (binary
    # on normalized center speed), "aniso" (per-axis covariance
    # expansion), or "sigma" (expected DIoU at sigma points of the IMM
    # predictive distribution). mode: None derives fixed/adaptive from
    # the legacy `adaptive` flag so existing configs keep their exact
    # behavior. "adaptive", "aniso", and "sigma" require use_imm.
    "use_eiou": False,
    "eiou": {
        "expansion": 0.7,
        "adaptive": False,
        "adaptive_scale": 1.0,
        "min_expansion": 0.2,
        "max_expansion": 1.2,
        "mode": None,
        # velocity_switch parameters: threshold on normalized center
        # speed (MeMoSORT's SportsMOT percentile), binary expansions
        "velocity_thresh": 0.1172,
        "p_slow": 0.5,
        "p_fast": 0.6,
        # sigma mode: weight of the adaptive-EIoU backstop in the
        # blended score (1 - beta) * sigma-DIoU + beta * adaptive-EIoU.
        # beta = 1.0 reproduces mode "adaptive" exactly; beta = 0 is
        # pure sigma-DIoU. NOTE: the sigma score lives in (-1, 1], so
        # match_thresh must be re-tuned for this mode.
        "sigma_blend_beta": 0.3,
        # Accumulate per-frame plain-IoU saturation statistics in
        # tracker.sigma_saturation_stats (experiment instrumentation)
        "sigma_debug": False,
    },
    "cmc": {},
    # Sub-config defaults live in their modules (DEFAULT_IMM_CONFIG,
    # hybrid_score_batch signature); empty dicts mean "module defaults"
    "imm": {},
    "hybrid_distance": {},
    "reid": {
        "weights": "weights/dance_sbs_S50_traced.pt",
        "crop_size": [128, 384],
        "w_assoc_emb": 1.25,
        "alpha_fixed_emb": 0.95,
        "aw_off": False,
        "aw_param": 1.0,
        "imm_gate_cv_thresh": 0.7,
        "imm_gate_young_frames": 5,
        # Per-track ReID weight scaling from IMM model probabilities
        "imm_adaptive": False,
        "imm_cv_discount": 0.5,
        # Min embedding similarity for zero-IoU rescue inside the
        # Mahalanobis gate. None disables rescue. Requires
        # use_mahalanobis_gate (rescue must be motion-bounded).
        "emb_rescue_thresh": None,
    },
    "ocsort": {
        "use_oru": True,
        "use_ocm": True,
        "use_ocr": True,
        # CMC-corrected ORU: replay virtual observations through the
        # accumulated camera-motion affines (only fires when the camera
        # actually moved during the gap). Disable for plain OC-SORT ORU.
        "use_cmc_oru": True,
        "delta_t": 3,
        "inertia_weight": 0.2,
    },
}

# Valid eiou.mode values (see _compute_track_expansions and
# _compute_sigma_scores)
EIOU_MODES = ("fixed", "adaptive", "velocity_switch", "aniso", "sigma")


def load_config(config_path: str) -> dict:
    """Load tracker configuration from YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary with tracker parameters.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("tracker", config)


class ARCTracker:
    """Multi-object tracker with modular OC-SORT baseline and ARC-SORT extensions.

    Maintains a pool of active tracks and processes detections frame-by-frame.
    Each frame, the tracker predicts track positions, associates detections
    to tracks, and manages track lifecycle.

    The tracker is configurable via YAML config files. Module flags
    (use_byte, use_cmc, use_imm, etc.) enable/disable ARC-SORT extensions
    independently for ablation studies.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the tracker with configuration.

        Args:
            config: Configuration dictionary. Uses DEFAULT_CONFIG
                for any missing keys.
        """
        if config is None:
            config = {}

        self.cfg = {**DEFAULT_CONFIG, **config}
        # Merge every nested sub-config over its defaults so all
        # sub-configs share one provenance path (partial user dicts
        # keep the default values for unspecified keys)
        for sub_key in ("ocsort", "reid", "cmc", "imm", "hybrid_distance", "eiou"):
            self.cfg[sub_key] = {
                **DEFAULT_CONFIG.get(sub_key, {}),
                **(config.get(sub_key) or {}),
            }
        ocsort_cfg = self.cfg["ocsort"]

        self.track_high_thresh = self.cfg["track_high_thresh"]
        self.track_low_thresh = self.cfg["track_low_thresh"]
        self.new_track_thresh = self.cfg["new_track_thresh"]
        self.track_buffer = self.cfg["track_buffer"]
        self.match_thresh = self.cfg["match_thresh"]
        self.byte_match_thresh = self.cfg["byte_match_thresh"]
        self.min_hits = self.cfg["min_hits"]
        self.output_box = self.cfg["output_box"]
        if self.output_box not in ("detection", "kalman"):
            raise ValueError(f"output_box must be 'detection' or 'kalman', got {self.output_box!r}")
        self.aspect_ratio_thresh = self.cfg.get("aspect_ratio_thresh", 0.0)
        self.min_box_area = self.cfg.get("min_box_area", 0.0)
        self.use_byte = self.cfg["use_byte"]
        self.use_cmc = self.cfg["use_cmc"]
        self.use_imm = self.cfg.get("use_imm", False)
        self.imm_config = self.cfg["imm"]
        self.use_hybrid_distance = self.cfg.get("use_hybrid_distance", False)
        self.hybrid_distance_config = self.cfg["hybrid_distance"]

        # Surface the hybrid-fallback hazard: an unbounded fallback score
        # above match_thresh lets non-overlapping boxes match in stage 1.
        if self.use_hybrid_distance and not self.cfg.get(
            "use_mahalanobis_gate", False
        ):
            fallback_cap = self.hybrid_distance_config.get("fallback_cap", 0.5)
            if fallback_cap >= self.match_thresh:
                logger.warning(
                    f"hybrid fallback_cap ({fallback_cap}) >= match_thresh "
                    f"({self.match_thresh}) with no Mahalanobis gate: "
                    "non-overlapping boxes can match in stage 1. Set "
                    "hybrid_distance.fallback_cap below match_thresh or "
                    "enable use_mahalanobis_gate."
                )

        # Mahalanobis motion gate (requires IMM combined covariance)
        self.use_mahalanobis_gate = self.cfg.get("use_mahalanobis_gate", False)
        self.mahalanobis_thresh = self.cfg.get("mahalanobis_thresh", 9.4877)
        if self.use_mahalanobis_gate and not self.use_imm:
            logger.warning(
                "use_mahalanobis_gate requires use_imm; gate disabled"
            )
            self.use_mahalanobis_gate = False

        # ExpansionIoU association. eiou.mode: None derives the mode
        # from the legacy `adaptive` flag (backward compatibility).
        self.use_eiou = self.cfg.get("use_eiou", False)
        self.eiou_config = self.cfg["eiou"]
        eiou_mode = self.eiou_config.get("mode")
        if eiou_mode is None:
            eiou_mode = "adaptive" if self.eiou_config["adaptive"] else "fixed"
        elif eiou_mode not in EIOU_MODES:
            raise ValueError(
                f"unknown eiou.mode {eiou_mode!r}; "
                f"expected one of {EIOU_MODES}"
            )
        if (
            self.use_eiou
            and eiou_mode in ("adaptive", "aniso", "sigma")
            and not self.use_imm
        ):
            logger.warning(
                f"eiou.mode '{eiou_mode}' requires use_imm; "
                "using fixed expansion"
            )
            eiou_mode = "fixed"
        self.eiou_config = {
            **self.eiou_config,
            "mode": eiou_mode,
            "adaptive": eiou_mode == "adaptive",
        }

        # sigma-DIoU saturation instrumentation (eiou.sigma_debug)
        self.sigma_saturation_stats = self._new_sigma_stats()

        # ReID configuration
        self.use_reid = self.cfg.get("use_reid", False)
        self.use_gated_reid = self.cfg.get("use_gated_reid", False)
        reid_cfg = self.cfg["reid"]
        self.reid_config = reid_cfg

        if self.use_reid or self.use_gated_reid:
            from arcsort.reid import EmbeddingComputer
            self.embedding_computer = EmbeddingComputer(
                weights_path=reid_cfg["weights"],
                crop_size=tuple(reid_cfg["crop_size"]),
            )
        else:
            self.embedding_computer = None

        if self.use_cmc:
            self.cmc = CameraMotionCompensator(self.cfg["cmc"])
        else:
            self.cmc = None

        # Store affine transforms per frame for CMC-corrected ORU
        self.affine_transforms: dict[int, np.ndarray] = {}

        self.delta_t = ocsort_cfg["delta_t"]
        self.inertia_weight = ocsort_cfg["inertia_weight"]
        self.use_ocr = ocsort_cfg["use_ocr"]
        self.use_cmc_oru = ocsort_cfg.get("use_cmc_oru", True)

        # Ambiguity instrument: permutation margins between accepted
        # stage-1 matches (see association.swap_margins). Logging only —
        # never alters the assignment. Default off.
        amb_cfg = self.cfg.get("ambiguity") or {}
        self.ambiguity_enabled = amb_cfg.get("enabled", False)
        self.ambiguity_margin_thresh = amb_cfg.get("margin_thresh", 0.2)
        self.ambiguity_events: list[dict] = []

        # Velocity-continuity cue (crossing tie-breaker; default off)
        vel_cfg = self.cfg.get("vel_cue") or {}
        self.vel_cue_enabled = vel_cfg.get("enabled", False)
        self.vel_cue_weight = vel_cfg.get("weight", 0.2)
        self.vel_cue_r_cap = vel_cfg.get("r_cap", 0.5)

        # Deferred identity arbitration (default off): open an episode
        # when a match pair is entangled (low swap margin), snapshot the
        # pre-crossing appearance prototypes, integrate a permutation
        # LLR over post-crossing detections, and perform at most one
        # corrective ID swap per episode when the evidence says the
        # crossing decision was wrong.
        arb_cfg = self.cfg.get("arbitration") or {}
        self.arbitration_enabled = arb_cfg.get("enabled", False)
        self.arb_window = arb_cfg.get("window", 30)
        self.arb_min_frames = arb_cfg.get("min_frames", 4)
        self.arb_llr_thresh = arb_cfg.get("llr_thresh", 0.5)
        self.arb_episodes: dict[tuple[int, int], dict] = {}
        self.arbitration_swaps: list[dict] = []

        # Swap margins are computed when either consumer needs them
        self.margins_needed = self.ambiguity_enabled or self.arbitration_enabled

        self.tracks: list[STrack] = []
        # Per-tracker ID counter: independent trackers in one process
        # (e.g. several cameras) must not share or reset each other's IDs
        self._next_track_id = 0
        self.frame_count = 0

        logger.debug(
            f"ARCTracker initialized: match_thresh={self.match_thresh}, "
            f"buffer={self.track_buffer}, delta_t={self.delta_t}"
        )

    def _filter_detections(self, dets: np.ndarray) -> np.ndarray:
        """Filter detections by aspect ratio and minimum box area.

        Args:
            dets: Detection array of shape (N, 6) [x1, y1, x2, y2, conf, cls].

        Returns:
            Filtered detection array.
        """
        if dets.shape[0] == 0:
            return dets

        w = dets[:, 2] - dets[:, 0]
        h = dets[:, 3] - dets[:, 1]
        keep = np.ones(len(dets), dtype=bool)

        if self.aspect_ratio_thresh > 0:
            # Filter boxes wider than tall beyond threshold (non-pedestrian)
            # aspect_ratio = w/h; vertical pedestrians have w/h < 1.6
            keep &= (w / np.maximum(h, 1e-6)) <= self.aspect_ratio_thresh

        if self.min_box_area > 0:
            keep &= (w * h) >= self.min_box_area

        return dets[keep]

    @staticmethod
    def _new_sigma_stats() -> dict:
        """Fresh accumulator for sigma-DIoU saturation statistics."""
        return {
            "frames": 0,
            "total_tracks": 0,
            "saturated_tracks": 0,
            "per_frame": [],
        }

    @classmethod
    def from_preset(
        cls,
        name: str = "sportsmot",
        reid_weights: str | None = None,
        **overrides,
    ) -> ARCTracker:
        """Build a tracker from a shipped preset (see ``arcsort.configs``).

        Args:
            name: Preset name, e.g. "sportsmot" or "dancetrack".
            reid_weights: Path to a TorchScript ReID model (see README:
                Weights). When None, appearance features are disabled and
                the tracker runs motion-only on CPU.
            **overrides: Top-level config keys to override; dict values
                are merged into the preset's sub-config of the same name.

        Returns:
            A configured ARCTracker.
        """
        from pathlib import Path

        from arcsort.configs import load_preset

        config = load_preset(name)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value

        uses_reid = config.get("use_reid") or config.get("use_gated_reid")
        if reid_weights is not None:
            config["reid"] = {
                **config.get("reid", {}),
                "weights": str(Path(reid_weights).expanduser().resolve()),
            }
        elif uses_reid:
            logger.info(
                f"Preset '{name}' uses ReID but no reid_weights were given; "
                "running motion-only. Pass reid_weights=... to enable appearance."
            )
            config["use_reid"] = False
            config["use_gated_reid"] = False
        return cls(config)

    @property
    def camera_motion(self) -> np.ndarray | None:
        """2x3 affine from the previous frame to the current one, or None.

        Set by the last ``update()`` when camera-motion compensation ran.
        ``Visualizer(motion_compensated_trails=True)`` uses it to keep
        trails fixed to the scene while the camera moves.
        """
        return self.affine_transforms.get(self.frame_count)

    def reset(self) -> None:
        """Reset tracker state for a new sequence."""
        self.tracks = []
        self.frame_count = 0
        self.affine_transforms = {}
        if self.cmc is not None:
            self.cmc.reset()
        self.sigma_saturation_stats = self._new_sigma_stats()
        self.ambiguity_events = []
        self.arb_episodes = {}
        self.arbitration_swaps = []
        self._next_track_id = 0
        reset_track_id()

    def pop_ambiguity_events(self) -> list[dict]:
        """Return and clear recorded ambiguity events (see swap_margins)."""
        events = self.ambiguity_events
        self.ambiguity_events = []
        return events

    def _record_ambiguity(self, matched: np.ndarray, diag: dict) -> None:
        """Process low-margin assignment permutations among stage-1 matches.

        An entangled pair means: at this frame, swapping the identities
        of these two matched tracks would have cost the assignment less
        than margin_thresh in total score — the decision was
        near-ambiguous. Events are logged when the instrument is on;
        arbitration episodes are opened when arbitration is on.
        """
        for a, b, margin in swap_margins(
            matched, diag["score"], diag["spatial"]
        ):
            if margin >= self.ambiguity_margin_thresh:
                continue
            trk_a = self.tracks[matched[a, 1]]
            trk_b = self.tracks[matched[b, 1]]
            if self.ambiguity_enabled:
                self.ambiguity_events.append({
                    "frame": self.frame_count,
                    "track_a": trk_a.track_id,
                    "track_b": trk_b.track_id,
                    "margin": round(margin, 4),
                })
            if self.arbitration_enabled:
                self._arb_touch_episode(trk_a, trk_b)

    def _arb_touch_episode(self, trk_a, trk_b) -> None:
        """Open (or extend) an arbitration episode for an entangled pair.

        Snapshots are the tracks' EMA prototypes as of the frame the
        pair FIRST entangled — taken before this frame's embedding
        updates, i.e. the last pre-crossing appearance. A track can be
        in at most one episode at a time (first pair wins).
        """
        key = (min(trk_a.track_id, trk_b.track_id),
               max(trk_a.track_id, trk_b.track_id))
        ep = self.arb_episodes.get(key)
        if ep is not None:
            ep["last_entangled"] = self.frame_count
            return
        busy = {tid for k in self.arb_episodes for tid in k}
        if key[0] in busy or key[1] in busy:
            return
        if trk_a.emb is None or trk_b.emb is None:
            return
        self.arb_episodes[key] = {
            "open_frame": self.frame_count,
            "last_entangled": self.frame_count,
            "snap": {
                trk_a.track_id: trk_a.emb.copy(),
                trk_b.track_id: trk_b.emb.copy(),
            },
            "llr": 0.0,
            "n_evidence": 0,
        }

    def _arbitrate(self, frame_embs: dict[int, np.ndarray]) -> None:
        """Accumulate episode evidence and execute corrective swaps.

        Per frame with both members matched to embedded detections, the
        permutation LLR gains cos(f_a,s_a)+cos(f_b,s_b)−cos(f_a,s_b)
        −cos(f_b,s_a): positive when current detections match their own
        pre-crossing prototypes, negative when they match the partner's
        (the crossing decision was wrong). An episode closes window
        frames after its last entanglement, or when a member dies; it
        swaps the two live tracks' IDs at most once, when the evidence
        is decisively negative.
        """
        if not self.arb_episodes:
            return
        live = {t.track_id: t for t in self.tracks}

        for key in list(self.arb_episodes):
            ep = self.arb_episodes[key]
            id_a, id_b = key
            s_a, s_b = ep["snap"][id_a], ep["snap"][id_b]

            f_a, f_b = frame_embs.get(id_a), frame_embs.get(id_b)
            if f_a is not None and f_b is not None:
                ep["llr"] += float(
                    np.dot(f_a, s_a) + np.dot(f_b, s_b)
                    - np.dot(f_a, s_b) - np.dot(f_b, s_a)
                )
                ep["n_evidence"] += 1

            both_alive = id_a in live and id_b in live
            expired = (
                self.frame_count - ep["last_entangled"] >= self.arb_window
            )
            if not expired and both_alive:
                continue

            if (
                both_alive
                and ep["llr"] <= -self.arb_llr_thresh
                and ep["n_evidence"] >= self.arb_min_frames
            ):
                live[id_a].track_id, live[id_b].track_id = id_b, id_a
                self.arbitration_swaps.append({
                    "frame": self.frame_count,
                    "track_a": id_a,
                    "track_b": id_b,
                    "llr": round(ep["llr"], 4),
                    "n_evidence": ep["n_evidence"],
                })
            del self.arb_episodes[key]

    def update(
        self,
        detections: np.ndarray,
        frame: np.ndarray | None = None,
        cmc_affine: np.ndarray | None = None,
    ) -> np.ndarray:
        """Process one frame of detections and return tracked objects.

        Args:
            detections: Detection array of shape (N, 6) where each row
                is [x1, y1, x2, y2, confidence, class_id]. Can be empty
                (shape (0, 6)).
            frame: Raw BGR frame image, shape (H, W, 3). Used by CMC to
                compute affine transform. Ignored if cmc_affine is provided
                or use_cmc is disabled.
            cmc_affine: Pre-computed 2x3 CMC affine matrix. When provided,
                skips CMC computation from frame. Takes priority over frame.

        Returns:
            Tracked objects as array of shape (M, 5) where each row is
            [x1, y1, x2, y2, track_id]. Returns empty (0, 5) array
            if no tracks to output.
        """
        self.frame_count += 1

        # BYTE split: separate high-confidence and low-confidence detections
        if detections.shape[0] > 0:
            scores = detections[:, 4]
            high_mask = scores >= self.track_high_thresh
            det_high = detections[high_mask]

            if self.use_byte:
                low_mask = (scores < self.track_high_thresh) & (
                    scores >= self.track_low_thresh
                )
                det_low = detections[low_mask]
            else:
                det_low = np.empty((0, 6))
        else:
            det_high = np.empty((0, 6))
            det_low = np.empty((0, 6))

        # Filter by aspect ratio and min box area (for MOT17/MOT20 pedestrian filtering)
        if self.aspect_ratio_thresh > 0 or self.min_box_area > 0:
            det_high = self._filter_detections(det_high)
            det_low = self._filter_detections(det_low)

        bboxes_high = det_high[:, :4] if det_high.shape[0] > 0 else np.empty((0, 4))
        scores_high = det_high[:, 4] if det_high.shape[0] > 0 else np.empty(0)
        bboxes_low = det_low[:, :4] if det_low.shape[0] > 0 else np.empty((0, 4))
        scores_low = det_low[:, 4] if det_low.shape[0] > 0 else np.empty(0)

        # Predict all existing tracks
        track_predictions = np.empty((len(self.tracks), 4))
        for idx, track in enumerate(self.tracks):
            track_predictions[idx] = track.predict()

        # Remove tracks with invalid predictions (NaN)
        valid_mask = ~np.isnan(track_predictions).any(axis=1)
        if not valid_mask.all():
            self.tracks = [t for t, v in zip(self.tracks, valid_mask) if v]
            track_predictions = track_predictions[valid_mask]

        # Camera Motion Compensation: warp predictions into current frame
        # Priority: cached affine > compute from frame
        affine = None
        if self.use_cmc:
            if cmc_affine is not None:
                affine = cmc_affine
            elif frame is not None and self.cmc is not None:
                affine = self.cmc.apply(frame)

        if affine is not None:
            self.affine_transforms[self.frame_count] = affine
            # Prune affines older than any possible observation gap
            # (tracks are removed after track_buffer missed frames)
            oldest_needed = self.frame_count - self.track_buffer - 1
            for f in list(self.affine_transforms):
                if f < oldest_needed:
                    del self.affine_transforms[f]

            # Apply affine to all track KF states
            for track in self.tracks:
                track.apply_cmc(affine)

            # Re-read predictions after CMC warping
            for idx, track in enumerate(self.tracks):
                track_predictions[idx] = track.get_state()

        # Gather OCM data for association
        track_velocities = np.array([t.velocity for t in self.tracks])
        track_previous_obs = np.array([t.get_previous_obs() for t in self.tracks])

        if track_velocities.shape[0] == 0:
            track_velocities = np.empty((0, 2))
        if track_previous_obs.shape[0] == 0:
            track_previous_obs = np.empty((0, 5))

        # ReID: compute detection embeddings and gather track embeddings
        emb_cost = None
        det_embs = None
        det_alpha = None
        reid_active = (
            (self.use_reid or self.use_gated_reid)
            and self.embedding_computer is not None
            and (frame is not None or self.embedding_computer.has_cache())
            and bboxes_high.shape[0] > 0
            and len(self.tracks) > 0
        )

        if reid_active:
            if self.embedding_computer.has_cache():
                self.embedding_computer.set_frame(self.frame_count)
            det_embs = self.embedding_computer.compute_embeddings(
                frame, bboxes_high
            )

            # Confidence-weighted EMA alpha for embedding update
            af = self.reid_config["alpha_fixed_emb"]
            det_thresh = self.track_high_thresh
            trust = (scores_high - det_thresh) / (1.0 - det_thresh + 1e-6)
            trust = np.clip(trust, 0.0, 1.0)
            det_alpha = af + (1.0 - af) * (1.0 - trust)

            # Gather track embeddings
            from arcsort.matching import embedding_similarity_batch
            from arcsort.reid import (
                compute_imm_adaptive_weights,
                should_activate_reid,
            )

            trk_embs_list = []
            for track in self.tracks:
                emb = track.get_emb()
                if emb is not None:
                    trk_embs_list.append(emb)
                else:
                    # Track without embedding -- use zeros
                    trk_embs_list.append(
                        np.zeros(det_embs.shape[1], dtype=np.float32)
                    )
            trk_embs = np.array(trk_embs_list)

            # Compute cosine similarity matrix
            emb_cost = embedding_similarity_batch(det_embs, trk_embs)

            # IMM-adaptive per-track ReID weight (novel to ARC-SORT)
            # Scale each track's column by its IMM-derived weight
            if self.use_imm and self.reid_config["imm_adaptive"]:
                adaptive_weights = compute_imm_adaptive_weights(
                    self.tracks,
                    base_weight=1.0,  # Multiplicative scaling on emb_cost
                    cv_discount=self.reid_config["imm_cv_discount"],
                    young_frames=self.reid_config["imm_gate_young_frames"],
                )
                # Apply per-track weight as column-wise scaling
                emb_cost = emb_cost * adaptive_weights[np.newaxis, :]

            # IMM gating: zero out columns for tracks where ReID is skipped
            if self.use_gated_reid:
                cv_thresh = self.reid_config["imm_gate_cv_thresh"]
                young_frames = self.reid_config["imm_gate_young_frames"]
                for t_idx, track in enumerate(self.tracks):
                    if not should_activate_reid(
                        track, cv_thresh, young_frames
                    ):
                        emb_cost[:, t_idx] = 0.0

        # Per-track gating distributions, computed once per frame and
        # shared by the Mahalanobis gate and the covariance-based
        # expansion modes (adaptive/aniso/sigma)
        gating_means = None
        gating_covs = None
        needs_gating = self.use_mahalanobis_gate or (
            self.use_eiou
            and self.eiou_config["mode"] in ("adaptive", "aniso", "sigma")
        )
        if needs_gating and len(self.tracks) > 0:
            gating_means = np.empty((len(self.tracks), 4))
            gating_covs = np.empty((len(self.tracks), 4, 4))
            for t_idx, track in enumerate(self.tracks):
                mean, S = track.kf.get_gating_distribution()
                gating_means[t_idx] = mean
                gating_covs[t_idx] = S

        # Mahalanobis motion gate from IMM combined covariance
        gate_mask = None
        if (
            self.use_mahalanobis_gate
            and bboxes_high.shape[0] > 0
            and len(self.tracks) > 0
        ):
            from arcsort.matching import mahalanobis_distance_batch
            from arcsort.utils import xyxy_to_cxcywh

            det_measurements = xyxy_to_cxcywh(bboxes_high)
            d2 = mahalanobis_distance_batch(
                det_measurements, gating_means, gating_covs
            )
            gate_mask = d2 <= self.mahalanobis_thresh

        # ExpansionIoU: per-track expansion factors, or a full blended
        # sigma-DIoU score matrix for eiou.mode "sigma"
        track_expansions = None
        spatial_score = None
        if self.use_eiou and len(self.tracks) > 0:
            if self.eiou_config["mode"] == "sigma":
                if bboxes_high.shape[0] > 0:
                    spatial_score = self._compute_sigma_scores(
                        bboxes_high, track_predictions,
                        gating_means, gating_covs,
                    )
            else:
                track_expansions = self._compute_track_expansions(
                    gating_means=gating_means, gating_covs=gating_covs
                )

        # Velocity-continuity cue inputs (filter velocities are post-CMC
        # since CMC is applied to the states before association)
        vel_cue = None
        if self.vel_cue_enabled and len(self.tracks) > 0:
            filter_vels = np.full((len(self.tracks), 2), np.nan)
            for t_idx, track in enumerate(self.tracks):
                v = track.get_filter_velocity()
                if v is not None:
                    filter_vels[t_idx] = v
            vel_cue = {
                "filter_vels": filter_vels,
                "last_obs": np.array(
                    [t.last_observation[:4] for t in self.tracks]
                ),
                "dt": np.array(
                    [t.time_since_update for t in self.tracks]
                ),
                "weight": self.vel_cue_weight,
                "r_cap": self.vel_cue_r_cap,
            }

        # First-stage association: high-conf detections vs predicted tracks
        assoc_result = associate(
            bboxes_high,
            track_predictions,
            track_velocities,
            track_previous_obs,
            iou_threshold=self.match_thresh,
            inertia_weight=self.inertia_weight,
            use_hybrid_distance=self.use_hybrid_distance,
            hybrid_config=self.hybrid_distance_config,
            emb_cost=emb_cost,
            w_assoc_emb=self.reid_config["w_assoc_emb"],
            aw_off=self.reid_config["aw_off"],
            aw_param=self.reid_config["aw_param"],
            gate_mask=gate_mask,
            emb_rescue_thresh=self.reid_config.get("emb_rescue_thresh"),
            track_expansions=track_expansions,
            spatial_score=spatial_score,
            return_diagnostics=self.margins_needed,
            vel_cue=vel_cue,
        )
        if self.margins_needed:
            matched, unmatched_dets, unmatched_trks, assoc_diag = assoc_result
            self._record_ambiguity(matched, assoc_diag)
        else:
            matched, unmatched_dets, unmatched_trks = assoc_result

        # Raw det embeddings per matched track id (arbitration evidence)
        frame_embs: dict[int, np.ndarray] = {}

        # Update matched tracks (KF + embedding EMA)
        for det_idx, trk_idx in matched:
            gap_aff = self._gather_gap_affines(self.tracks[trk_idx])
            self.tracks[trk_idx].update(
                bboxes_high[det_idx], scores_high[det_idx],
                gap_affines=gap_aff,
            )
            if det_embs is not None and det_alpha is not None:
                if self.arbitration_enabled:
                    frame_embs[self.tracks[trk_idx].track_id] = (
                        det_embs[det_idx]
                    )
                self.tracks[trk_idx].update_emb(
                    det_embs[det_idx], alpha=det_alpha[det_idx]
                )

        # BYTE second stage: match unmatched TRACKED tracks vs low-conf dets
        if self.use_byte and bboxes_low.shape[0] > 0 and unmatched_trks.size > 0:
            # Only attempt second stage with actively tracked (not lost) tracks
            byte_trk_mask = np.array(
                [
                    self.tracks[t].state == TrackState.TRACKED
                    or self.tracks[t].state == TrackState.NEW
                    for t in unmatched_trks
                ]
            )
            byte_trk_indices = unmatched_trks[byte_trk_mask]
            non_byte_trk_indices = unmatched_trks[~byte_trk_mask]

            if byte_trk_indices.size > 0:
                byte_trk_preds = np.array(
                    [track_predictions[t] for t in byte_trk_indices]
                )

                byte_matched, byte_unmatched_dets, byte_unmatched_trks = (
                    associate_byte_second(
                        bboxes_low,
                        byte_trk_preds,
                        iou_threshold=self.byte_match_thresh,
                    )
                )

                for det_local, trk_local in byte_matched:
                    trk_orig = byte_trk_indices[trk_local]
                    gap_aff = self._gather_gap_affines(self.tracks[trk_orig])
                    self.tracks[trk_orig].update(
                        bboxes_low[det_local], scores_low[det_local],
                        gap_affines=gap_aff,
                    )

                # Remaining unmatched tracks = those not in BYTE + those BYTE missed
                unmatched_trks = np.concatenate([
                    non_byte_trk_indices,
                    byte_trk_indices[byte_unmatched_trks],
                ])

        # OCR: observation-centric recovery for still-unmatched
        if self.use_ocr and unmatched_dets.size > 0 and unmatched_trks.size > 0:
            ocr_dets = bboxes_high[unmatched_dets]
            ocr_trk_last_obs = np.array(
                [self.tracks[t].last_observation[:4] for t in unmatched_trks]
            )

            # Slice embedding cost for unmatched pairs (ReID-enhanced OCR)
            ocr_emb_cost = None
            if emb_cost is not None:
                ocr_emb_cost = emb_cost[np.ix_(unmatched_dets, unmatched_trks)]

            ocr_matched, ocr_unmatched_dets, ocr_unmatched_trks = associate_ocr(
                ocr_dets, ocr_trk_last_obs, iou_threshold=self.match_thresh,
                emb_cost=ocr_emb_cost,
            )

            for det_local, trk_local in ocr_matched:
                det_orig = unmatched_dets[det_local]
                trk_orig = unmatched_trks[trk_local]
                gap_aff = self._gather_gap_affines(self.tracks[trk_orig])
                self.tracks[trk_orig].update(
                    bboxes_high[det_orig], scores_high[det_orig],
                    gap_affines=gap_aff,
                )
                if det_embs is not None and det_alpha is not None:
                    if self.arbitration_enabled:
                        frame_embs[self.tracks[trk_orig].track_id] = (
                            det_embs[det_orig]
                        )
                    self.tracks[trk_orig].update_emb(
                        det_embs[det_orig], alpha=det_alpha[det_orig]
                    )

            # Remap local indices back to original
            unmatched_dets = unmatched_dets[ocr_unmatched_dets]
            unmatched_trks = unmatched_trks[ocr_unmatched_trks]

        # Update unmatched tracks with None (no observation)
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].update(None)

        # Create new tracks from unmatched high-confidence detections
        for det_idx in unmatched_dets:
            if scores_high[det_idx] >= self.new_track_thresh:
                new_track = STrack(
                    bboxes_high[det_idx],
                    scores_high[det_idx],
                    delta_t=self.delta_t,
                    use_imm=self.use_imm,
                    imm_config=self.imm_config,
                )
                new_track.track_id = self._next_track_id
                self._next_track_id += 1
                # Initialize embedding for new track
                if det_embs is not None:
                    new_track.update_emb(det_embs[det_idx], alpha=0.0)
                self.tracks.append(new_track)

        # Manage track lifecycle
        self._manage_tracks()

        # Deferred identity arbitration (may swap two live track IDs)
        if self.arbitration_enabled:
            self._arbitrate(frame_embs)

        # Generate output
        return self._get_output()

    def _compute_track_expansions(
        self,
        mode: str | None = None,
        gating_means: np.ndarray | None = None,
        gating_covs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute per-track ExpansionIoU factors.

        The mechanism is selected by ``eiou.mode``:

        - ``fixed``: the configured constant for every track (Deep-EIoU
          behavior).
        - ``adaptive``: each track's factor is derived from its IMM
          innovation covariance: the expansion covers ``adaptive_scale``
          standard deviations of predicted center error relative to the
          box size, clipped to [min_expansion, max_expansion].
        - ``velocity_switch``: binary factor on normalized center speed
          sqrt((v_cx/w)^2 + (v_cy/h)^2): ``p_slow`` for tracks at or
          below ``velocity_thresh``, ``p_fast`` above it.
        - ``aniso``: like ``adaptive`` but anisotropic: the width and
          height expansions are clipped separately from sigma_x and
          sigma_y.

        Mode ``sigma`` has no per-track factor (it scores pairs
        directly, see _compute_sigma_scores) and is rejected here.

        Args:
            mode: Expansion mode override. None uses the configured
                ``eiou.mode`` (used by mode "sigma" to obtain its
                adaptive-EIoU backstop term).
            gating_means: Optional cached gating means, shape (T, 4)
                [cx, cy, w, h]. Computed on demand when None.
            gating_covs: Optional cached innovation covariances,
                shape (T, 4, 4). Computed on demand when None.

        Returns:
            Expansion factors, shape (num_tracks,), or (num_tracks, 2)
            with [ex, ey] columns for mode ``aniso``.
        """
        cfg = self.eiou_config
        if mode is None:
            mode = cfg["mode"]
        num = len(self.tracks)

        if mode == "sigma":
            raise ValueError(
                "eiou.mode 'sigma' scores det/track pairs directly and "
                "has no per-track expansion; see _compute_sigma_scores"
            )

        if mode == "fixed":
            return np.full(num, cfg["expansion"])

        if mode == "velocity_switch":
            # Binary slow/fast expansion on normalized center speed
            # (MeMoSORT arXiv:2508.09796 Eq.16-style baseline)
            expansions = np.empty(num)
            for idx, track in enumerate(self.tracks):
                x1, y1, x2, y2 = track.get_state()
                w = max(float(x2 - x1), 1.0)
                h = max(float(y2 - y1), 1.0)
                v_cx, v_cy = track.kf.get_velocity()
                speed = float(np.hypot(v_cx / w, v_cy / h))
                expansions[idx] = (
                    cfg["p_slow"] if speed <= cfg["velocity_thresh"]
                    else cfg["p_fast"]
                )
            return expansions

        # Covariance-based modes share the per-frame gating cache
        if gating_means is None or gating_covs is None:
            gating_means = np.empty((num, 4))
            gating_covs = np.empty((num, 4, 4))
            for idx, track in enumerate(self.tracks):
                mean, S = track.kf.get_gating_distribution()
                gating_means[idx] = mean
                gating_covs[idx] = S

        if mode == "aniso":
            expansions = np.empty((num, 2))
            for idx in range(num):
                mean = gating_means[idx]
                S = gating_covs[idx]
                w = max(float(mean[2]), 1.0)
                h = max(float(mean[3]), 1.0)
                sigma_x = float(np.sqrt(max(S[0, 0], 0.0)))
                sigma_y = float(np.sqrt(max(S[1, 1], 0.0)))
                expansions[idx, 0] = np.clip(
                    cfg["adaptive_scale"] * (sigma_x / w),
                    cfg["min_expansion"],
                    cfg["max_expansion"],
                )
                expansions[idx, 1] = np.clip(
                    cfg["adaptive_scale"] * (sigma_y / h),
                    cfg["min_expansion"],
                    cfg["max_expansion"],
                )
            return expansions

        # mode == "adaptive"
        expansions = np.empty(num)
        for idx in range(num):
            mean = gating_means[idx]
            S = gating_covs[idx]
            w = max(float(mean[2]), 1.0)
            h = max(float(mean[3]), 1.0)
            sigma_x = float(np.sqrt(max(S[0, 0], 0.0)))
            sigma_y = float(np.sqrt(max(S[1, 1], 0.0)))
            rel_uncertainty = max(sigma_x / w, sigma_y / h)
            expansions[idx] = np.clip(
                cfg["adaptive_scale"] * rel_uncertainty,
                cfg["min_expansion"],
                cfg["max_expansion"],
            )
        return expansions

    def _compute_sigma_scores(
        self,
        detections: np.ndarray,
        track_predictions: np.ndarray,
        gating_means: np.ndarray,
        gating_covs: np.ndarray,
    ) -> np.ndarray:
        """Compute the blended sigma-DIoU spatial score matrix (W1c).

        score = (1 - beta) * sigma-DIoU + beta * adaptive-EIoU

        The sigma-DIoU term is the expected DIoU of each det/track pair
        under the track's positional uncertainty (5-point unscented sum
        over the Cholesky factor of the 2x2 positional block of the
        innovation covariance, see matching.sigma_diou_batch). The EIoU
        term is exactly the mode "adaptive" score, so
        ``sigma_blend_beta: 1.0`` reproduces mode "adaptive" bit-for-bit
        and ``0.0`` is pure sigma-DIoU.

        The blended score can be negative (DIoU range (-1, 1]); such
        pairs never become matches because stage-1 acceptance requires
        score >= match_thresh > 0, while their ranking gradient still
        guides the assignment solver.

        With ``eiou.sigma_debug`` enabled, accumulates the per-frame
        plain-IoU saturation rate in ``self.sigma_saturation_stats``.

        Args:
            detections: High-confidence detection boxes, shape (N, 4).
            track_predictions: Predicted track boxes, shape (T, 4).
            gating_means: Cached gating means, shape (T, 4) [cx,cy,w,h].
            gating_covs: Cached innovation covariances, shape (T, 4, 4).

        Returns:
            Blended score matrix, shape (N, T).
        """
        from arcsort.matching import eiou_batch, sigma_diou_batch

        cfg = self.eiou_config
        beta = cfg["sigma_blend_beta"]
        debug = cfg["sigma_debug"]

        result = sigma_diou_batch(
            detections,
            track_predictions,
            gating_covs[:, :2, :2],
            max_expansion=cfg["max_expansion"],
            return_saturation=debug,
        )
        if debug:
            sigma_scores, saturated = result
            stats = self.sigma_saturation_stats
            stats["frames"] += 1
            stats["total_tracks"] += int(saturated.size)
            stats["saturated_tracks"] += int(saturated.sum())
            stats["per_frame"].append(
                float(saturated.mean()) if saturated.size else 0.0
            )
        else:
            sigma_scores = result

        expansions = self._compute_track_expansions(
            mode="adaptive",
            gating_means=gating_means,
            gating_covs=gating_covs,
        )
        eiou_scores = eiou_batch(detections, track_predictions, expansions)
        return (1.0 - beta) * sigma_scores + beta * eiou_scores

    def _gather_gap_affines(self, track: STrack) -> list[np.ndarray] | None:
        """Gather stored CMC affine transforms for a track's observation gap.

        Returns the affines for the frames between the track's last update
        and the current frame, in chronological order. Used by CMC-corrected
        ORU to warp virtual observations into the correct coordinate frame.

        Only returns affines when significant camera motion occurred during
        the gap. For near-identity transforms (static camera), returns None
        to avoid introducing unnecessary numerical noise.

        Args:
            track: Track that is being re-matched after a gap.

        Returns:
            List of 2x3 affine matrices (one per gap frame), or None if
            CMC is disabled, no affines available, or camera was static.
        """
        if (
            not self.use_cmc
            or not self.use_cmc_oru
            or track.time_since_update <= 0
        ):
            return None

        gap_affines = []
        max_translation = 0.0
        for f in range(
            self.frame_count - track.time_since_update + 1,
            self.frame_count + 1,
        ):
            aff = self.affine_transforms.get(f)
            if aff is not None:
                gap_affines.append(aff)
                # Track maximum translation to detect real camera motion
                tx, ty = aff[0, 2], aff[1, 2]
                max_translation = max(max_translation, abs(tx), abs(ty))
            else:
                gap_affines.append(np.eye(2, 3))

        # Only use CMC-corrected ORU when camera actually moved
        # (> 1 pixel translation). For static cameras, original ORU
        # is more accurate (avoids floating-point noise from transforms).
        if max_translation < 1.0:
            return None

        return gap_affines if len(gap_affines) > 0 else None

    def _manage_tracks(self) -> None:
        """Update track states and remove expired tracks."""
        remaining = []
        for track in self.tracks:
            if track.time_since_update > self.track_buffer:
                track.mark_removed()
            elif track.time_since_update > 0 and track.state != TrackState.NEW:
                track.mark_lost()

            if track.state != TrackState.REMOVED:
                remaining.append(track)

        self.tracks = remaining

    def _get_output(self) -> np.ndarray:
        """Collect output from tracks that meet output criteria.

        Tracks are output if they have been recently observed and either
        have accumulated enough hits or the sequence is still warming up.

        Returns:
            Array of shape (M, 5) with [x1, y1, x2, y2, track_id] per row.
        """
        results = []
        for track in self.tracks:
            is_recent = track.time_since_update < 1
            has_min_hits = track.hit_streak >= self.min_hits
            is_warmup = self.frame_count <= self.min_hits

            if is_recent and (has_min_hits or is_warmup):
                # A recent track was matched this frame, so its last
                # observation is the detection it was matched to
                if self.output_box == "detection":
                    bbox = track.last_observation[:4]
                else:
                    bbox = track.get_state()
                results.append(np.concatenate([bbox, [track.track_id + 1]]))

        if len(results) == 0:
            return np.empty((0, 5))

        return np.array(results)
