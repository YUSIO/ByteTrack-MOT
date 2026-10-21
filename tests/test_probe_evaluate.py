"""Regression checks for threshold semantics and grouped evaluation leakage."""
import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fctrack.probe_evaluate import (FEATURE_KEYS, METHODS, calibrated_threshold,
                                  evaluate_probe, fit_predict, grouped_folds, main)


def fixture_data():
    rng = np.random.default_rng(19)
    rows = []
    for seq in range(8):
        for j in range(6):
            rows.append({"sequence": f"seq-{seq}", "frame": j + 1,
                         "kind": "fill" if j < 4 else "birth",
                         "outcome": "written" if j < 4 else "below_p",
                         "p": .1 + j * .15, "quality": j % 2,
                         "presence": -1 if j == 0 else j % 2,
                         "history_complete": j > 2})
    features = {key: rng.normal(size=(len(rows), 4 if key == "X_geometry" else 6))
                for key in FEATURE_KEYS}
    return features, rows


def test_strict_cutoff_budget_and_ties():
    quality = np.ones(20, int)
    scores = np.arange(20) / 20
    result = calibrated_threshold(scores, quality, np.ones(20, bool))
    assert result["threshold"] == .05
    assert result["allowed_false_rejects"] == 1
    assert np.sum(scores < result["threshold"]) == 1
    scores[:3] = 0
    result = calibrated_threshold(scores, quality, np.ones(20, bool))
    assert result["threshold"] == 0
    assert result["actual_false_rejects"] == 0
    # A lower-scored rejected query cannot enter the written-hit calibration pool.
    result = calibrated_threshold([0, .8], [1, 1], [False, True])
    assert result["threshold"] == .8


def test_missing_calibration_class_disables_veto():
    result = calibrated_threshold([.1, .2], [0, 0], [True, True])
    assert result["threshold"] is None and not result["enabled"]
    assert result["reason"] == "no_written_quality_positive_in_inner_oof"


def test_group_folds_are_disjoint_and_complete():
    names = np.asarray([f"seq{i}" for i in range(11) for _ in range(i + 1)])
    folds = grouped_folds(names, 5, 140)
    assert sorted(np.concatenate(folds)) == list(range(len(names)))
    for i, fold in enumerate(folds):
        for j, other in enumerate(folds):
            if i != j:
                assert not set(names[fold]) & set(names[other])
    assert all(np.array_equal(a, b) for a, b in zip(folds, grouped_folds(names, 5, 140)))


def test_constant_fallback_is_explicit_and_presence_ignore_excluded():
    x = np.arange(24).reshape(6, 4)
    scores, meta = fit_predict(x, np.asarray([-1, 1, 1, 0, 0, 0]), [0, 1, 2], [3, 4, 5])
    assert np.all(scores == 1)
    assert meta["fallback"] == "single_training_class" and meta["train_count"] == 2
    scores, meta = fit_predict(x, np.full(6, -1), [0, 1, 2], [3, 4, 5])
    assert np.all(scores == .5) and meta["fallback"] == "no_eligible_training_rows"


def test_outer_labels_and_features_cannot_leak_into_own_fit_or_cutoff():
    features, rows = fixture_data()
    _, before, folds = evaluate_probe(features, rows, outer_folds=4, bootstrap_repeats=0)
    target_sequences = folds["folds"][0]["test_sequences"]
    modified = copy.deepcopy(rows)
    for row in modified:
        if row["sequence"] in target_sequences:
            row["quality"] = 1 - row["quality"]
            row["presence"] = 1 - row["presence"] if row["presence"] >= 0 else -1
    _, after, changed_folds = evaluate_probe(features, modified, outer_folds=4, bootstrap_repeats=0)
    for old, new in zip(before, after):
        if old["outer_fold"] == 0:
            assert old["predictions"] == new["predictions"]
    assert folds["folds"][0]["inner_folds"] == changed_folds["folds"][0]["inner_folds"]
    # Held-out covariates may change their prediction, but not calibration cutoff.
    changed_x = {key: value.copy() for key, value in features.items()}
    test_mask = np.asarray([r["sequence"] in target_sequences for r in rows])
    for x in changed_x.values():
        x[test_mask] += 1000
    _, perturbed, _ = evaluate_probe(changed_x, rows, outer_folds=4, bootstrap_repeats=0)
    for old, new in zip(before, perturbed):
        if old["outer_fold"] == 0:
            for method in METHODS:
                assert old["predictions"][method]["threshold"] == new["predictions"][method]["threshold"]


def test_cli_outputs_are_finite_and_auxiliary_labels_preserved(tmp_path):
    features, rows = fixture_data()
    rows[1].update(paths_available=True, all_path_crops_valid=True, paths_separated=True)
    rows[3].update(paths_available=True, all_path_crops_valid=False, paths_separated=True)
    feature_path, row_path, output = tmp_path / "features.npz", tmp_path / "rows.jsonl", tmp_path / "out"
    np.savez(feature_path, **features)
    row_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    main(["--features", str(feature_path), "--rows", str(row_path), "--out-dir", str(output),
          "--outer-folds", "4", "--bootstrap-repeats", "3"])
    result = json.loads((output / "metrics.json").read_text())
    assert result["counts"]["rows"] == 48
    assert result["counts"]["presence"]["-1"] == 8
    assert result["heads"]["presence"]["methods"]["dual"]["auc"]["all"]["n"] == 40
    assert result["heads"]["quality"]["methods"]["dual"]["veto"]["all"]["written_count"] == 32
    quality_auc = result["heads"]["quality"]["methods"]["dual"]["auc"]
    assert quality_auc["paths_available"]["n"] == 2
    assert quality_auc["all_path_crops_valid"]["n"] == 1
    assert quality_auc["paths_separated"]["n"] == 2
    assert quality_auc["valid_separated"]["n"] == 1
    assert "veto" not in result["heads"]["presence"]["methods"]["dual"]
    assert "MOTA" not in result
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == len(rows)
    for prediction in predictions:
        if prediction["outcome"] != "written":
            assert not any(p["written_veto"] for p in prediction["predictions"].values())
    folds = json.loads((output / "folds.json").read_text())
    for fold in folds["folds"]:
        assert not set(fold["train_sequences"]) & set(fold["test_sequences"])
        for inner in fold["inner_folds"]:
            assert not set(inner["train_sequences"]) & set(inner["validation_sequences"])
            assert not set(fold["test_sequences"]) & set(inner["validation_sequences"])


if __name__ == "__main__":
    tests = (test_strict_cutoff_budget_and_ties,
             test_missing_calibration_class_disables_veto,
             test_group_folds_are_disjoint_and_complete,
             test_constant_fallback_is_explicit_and_presence_ignore_excluded,
             test_outer_labels_and_features_cannot_leak_into_own_fit_or_cutoff)
    for test in tests:
        test()
        print(f"PASS {test.__name__}", flush=True)
    with tempfile.TemporaryDirectory(prefix="exp140-probe-tests-") as directory:
        test_cli_outputs_are_finite_and_auxiliary_labels_preserved(Path(directory))
    print("PASS test_cli_outputs_are_finite_and_auxiliary_labels_preserved", flush=True)
    print("All 6 probe evaluation tests passed", flush=True)
