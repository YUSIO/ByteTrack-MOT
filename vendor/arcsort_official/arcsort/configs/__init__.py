"""Tracker presets: the exact configurations behind the paper's results.

Each preset is a YAML file shipped with the package:

- ``sportsmot``: SportsMOT test row (81.82 HOTA); the recommended default for
  sports broadcast footage.
- ``dancetrack``: DanceTrack test row (62.1 HOTA).
- ``sportsmot_deepeiou_inputs``: SportsMOT row on Deep-EIoU's released
  detections and embeddings (80.31 HOTA); detector-calibration thresholds differ.
"""

from __future__ import annotations

from importlib import resources

import yaml


def available() -> list[str]:
    """Names of the shipped presets."""
    files = resources.files(__name__).iterdir()
    return sorted(f.name.removesuffix(".yaml") for f in files if f.name.endswith(".yaml"))


def load_preset(name: str) -> dict:
    """Load a preset's tracker config dict by name.

    Raises:
        ValueError: If no preset with that name exists.
    """
    path = resources.files(__name__) / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"Unknown preset '{name}'. Available: {available()}")
    config = yaml.safe_load(path.read_text())
    return config.get("tracker", config)
