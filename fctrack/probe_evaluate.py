"""Sequence-grouped, nested OOF information probe; never runs or scores a tracker.

The quality target is fixed-box IoU hit, not physical target presence. A second
presence head excludes label -1 and is descriptive only. All methods share folds;
outer test labels are used only after predictions and thresholds are frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import warnings

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


FEATURE_KEYS = ("X_geometry", "X_single", "X_fusion", "X_dual", "X_shuffled")
METHODS = ("seed_p",) + tuple(k[2:] for k in FEATURE_KEYS)


def grouped_folds(sequences, n_splits=5, seed=140):
    """Label-blind, deterministic allocation of whole sequences to folds."""
    groups = np.asarray(sequences, dtype=str)
    names = np.unique(groups)
    if len(names) < 2:
        return []
    k = min(n_splits, len(names))
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    counts = {name: int(np.sum(groups == name)) for name in names}
    # Balance query volume without inspecting labels or features. The initial
    # permutation breaks equal-size ties; seed also breaks equal-load fold ties.
    ordered = sorted(names, key=lambda name: -counts[name])
    buckets, loads = [[] for _ in range(k)], np.zeros(k, int)
    for name in ordered:
        fold = int(rng.choice(np.flatnonzero(loads == loads.min())))
        buckets[fold].append(name)
        loads[fold] += counts[name]
    return [np.flatnonzero(np.isin(groups, bucket)) for bucket in buckets]


def calibrated_threshold(scores, quality, written, max_false_reject=.05):
    """Largest strict score<threshold veto cutoff under empirical hit-loss cap.

    Thresholds are restricted to the observed positive-score boundary. With ties,
    the boundary is retained. Missing calibration positives disables veto safely.
    """
    values = np.sort(np.asarray(scores)[np.asarray(written, bool) & (np.asarray(quality) == 1)])
    n = len(values)
    if n == 0:
        return {"threshold": None, "enabled": False,
                "reason": "no_written_quality_positive_in_inner_oof", "positive_count": 0,
                "allowed_false_rejects": 0, "actual_false_rejects": 0}
    allowed = int(math.floor(max_false_reject * n + 1e-12))
    threshold = float(values[min(allowed, n - 1)])
    return {"threshold": threshold, "enabled": True, "reason": "inner_oof_written_hit_boundary",
            "positive_count": n, "allowed_false_rejects": allowed,
            "actual_false_rejects": int(np.sum(values < threshold))}


def fit_predict(x, labels, train, test, seed=140, geometry=None):
    """Fit every preprocessing stage exclusively on eligible training rows."""
    labels = np.asarray(labels)
    train = np.asarray(train, int)
    train = train[labels[train] >= 0]
    y = labels[train]
    meta = {"train_count": int(len(train)), "class_counts":
            {str(v): int(np.sum(y == v)) for v in (0, 1)}, "fallback": None,
            "components": 0, "geometry_dimensions": int(x.shape[1] if geometry is None else geometry.shape[1]),
            "warnings": []}
    if len(np.unique(y)) < 2:
        value = float(np.mean(y)) if len(y) else .5
        meta.update(fallback="single_training_class" if len(y) else "no_eligible_training_rows",
                    constant_score=value)
        return np.full(len(test), value), meta
    components = min(32, x.shape[1], len(train) - 1) if geometry is not None else 0
    model = LogisticRegression(C=1., max_iter=2000, solver="lbfgs", random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaler = StandardScaler().fit(x[train])
        x_train, x_test = scaler.transform(x[train]), scaler.transform(x[test])
        if geometry is not None:
            pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
            x_train, x_test = pca.fit_transform(x_train), pca.transform(x_test)
            geometry_scaler = StandardScaler().fit(geometry[train])
            x_train = np.concatenate((geometry_scaler.transform(geometry[train]), x_train), axis=1)
            x_test = np.concatenate((geometry_scaler.transform(geometry[test]), x_test), axis=1)
        model.fit(x_train, y)
        prediction = model.predict_proba(x_test)[:, 1]
    meta["components"] = int(components)
    meta["warnings"] = sorted({str(w.message) for w in caught})
    return prediction, meta


def auc_record(labels, scores, mask):
    mask = np.asarray(mask, bool) & (labels >= 0)
    y, s = labels[mask], scores[mask]
    counts = {"n": int(len(y)), "positive": int(np.sum(y == 1)), "negative": int(np.sum(y == 0))}
    if len(np.unique(y)) < 2:
        return dict(counts, auc=None, reason="empty_or_single_class")
    return dict(counts, auc=float(roc_auc_score(y, s)), reason=None)


def veto_record(quality, veto, mask):
    y, v = quality[mask], veto[mask]
    pos, neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    lost, removed = int(np.sum(v & (y == 1))), int(np.sum(v & (y == 0)))
    return {"written_count": int(len(y)), "true_hits": pos, "iou_misses": neg,
            "true_hits_rejected": lost, "iou_misses_rejected": removed,
            "true_hit_reject_rate": lost / pos if pos else None,
            "iou_miss_reject_rate": removed / neg if neg else None,
            "retained_true_hits": pos - lost, "retained_iou_misses": neg - removed}


def subgroup_masks(rows):
    n = len(rows)
    groups = {"all": np.ones(n, dtype=bool)}
    for field, values in (("kind", ("fill", "birth")),
                          ("outcome", ("written", "below_p", "overlap"))):
        for value in values:
            groups[value] = np.asarray([r[field] == value for r in rows])
    groups["history_complete"] = np.asarray([r["history_complete"] for r in rows], bool)
    groups["history_incomplete"] = ~groups["history_complete"]
    for name in ("paths_available", "all_path_crops_valid", "paths_separated"):
        groups[name] = np.asarray([bool(r.get(name, False)) for r in rows])
    groups["valid_separated"] = (groups["paths_available"] & groups["all_path_crops_valid"] &
                                 groups["paths_separated"])
    for value in (-1, 0, 1):
        groups[f"presence_{value}"] = np.asarray([r["presence"] == value for r in rows])
    return groups


def scalar_metrics(labels, quality, scores, veto, written):
    result = {"auc_all": auc_record(labels, scores, np.ones(len(labels), bool))["auc"],
              "auc_written": auc_record(labels, scores, written)["auc"]}
    if veto is not None:
        v = veto_record(quality, veto, written)
        result.update(true_hit_reject_rate=v["true_hit_reject_rate"],
                      iou_miss_reject_rate=v["iou_miss_reject_rate"])
    return result


def conditional_bootstrap(sequences, labels, quality, predictions, vetoes, written,
                          repeats, seed):
    """Paired cluster bootstrap of frozen OOF predictions, without model refit."""
    names = sorted(set(sequences))
    members = [np.flatnonzero(sequences == name) for name in names]
    rng = np.random.default_rng(seed)
    samples = {m: {} for m in METHODS}
    pairs = [("dual", b) for b in ("seed_p", "geometry", "single", "fusion", "shuffled")]
    differences = {f"{a}_minus_{b}": {} for a, b in pairs}
    for _ in range(repeats):
        take = np.concatenate([members[i] for i in rng.integers(0, len(names), size=len(names))])
        rep = {}
        for method in METHODS:
            v = None if vetoes is None else vetoes[method][take]
            rep[method] = scalar_metrics(labels[take], quality[take], predictions[method][take],
                                        v, written[take])
            for key, value in rep[method].items():
                if value is not None:
                    samples[method].setdefault(key, []).append(value)
        for a, b in pairs:
            for key in rep[a]:
                if rep[a][key] is not None and rep[b][key] is not None:
                    differences[f"{a}_minus_{b}"].setdefault(key, []).append(rep[a][key] - rep[b][key])
    def summarize(data):
        return {name: {key: {"low": float(np.quantile(values, .025)),
                            "high": float(np.quantile(values, .975)), "valid_repeats": len(values)}
                       for key, values in metrics.items()} for name, metrics in data.items()}
    return {"unit": "sequence", "sequence_count": len(names), "repeats": repeats,
            "seed": seed, "confidence": .95,
            "interpretation": "Conditional CI for frozen OOF predictions and selected thresholds; no refit, no fold/selection uncertainty, not a policy-effect CI.",
            "methods": summarize(samples), "paired_differences": summarize(differences)}


def validate_inputs(features, rows):
    if not rows:
        raise ValueError("No rows")
    for i, row in enumerate(rows):
        for key in ("sequence", "frame", "kind", "outcome", "p", "quality", "presence", "history_complete"):
            if key not in row:
                raise ValueError(f"Row {i} lacks {key}")
        if row["quality"] not in (0, 1) or row["presence"] not in (-1, 0, 1):
            raise ValueError(f"Invalid labels at row {i}")
        if not isinstance(row["sequence"], str) or not row["sequence"]:
            raise ValueError(f"Invalid sequence at row {i}")
        if row["kind"] not in ("fill", "birth") or not isinstance(row["history_complete"], bool):
            raise ValueError(f"Invalid kind/history flag at row {i}")
        if not math.isfinite(float(row["p"])) or not 0 <= float(row["p"]) <= 1:
            raise ValueError(f"Invalid p at row {i}")
    for key in FEATURE_KEYS:
        if key not in features:
            raise ValueError(f"Missing feature array {key}")
        x = np.asarray(features[key])
        if x.ndim != 2 or x.shape[0] != len(rows) or x.shape[1] == 0 or not np.isfinite(x).all():
            raise ValueError(f"Invalid/nonfinite feature array {key}: {x.shape}")
    if len({np.shape(features[key])[1] for key in FEATURE_KEYS if key != "X_geometry"}) != 1:
        raise ValueError("Visual methods must have equal feature dimensions")
    if len({r["sequence"] for r in rows}) < 3:
        raise ValueError("Nested grouped OOF requires at least three distinct sequences")


def evaluate_probe(features, rows, outer_folds=5, seed=140, bootstrap_repeats=1000):
    validate_inputs(features, rows)
    if outer_folds not in (4, 5):
        raise ValueError("Use four or five outer folds")
    if bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats must be nonnegative")
    n = len(rows)
    all_indices = np.arange(n)
    sequence = np.asarray([r["sequence"] for r in rows])
    quality = np.asarray([r["quality"] for r in rows], int)
    presence = np.asarray([r["presence"] for r in rows], int)
    raw_p = np.asarray([r["p"] for r in rows], float)
    geometry = np.asarray(features["X_geometry"], float)
    masks = subgroup_masks(rows)
    outer = grouped_folds(sequence, outer_folds, seed)
    folds, fold_ids = [], np.full(n, -1, int)
    predictions = {head: {m: np.full(n, np.nan) for m in METHODS} for head in ("quality", "presence")}
    vetoes = {m: np.zeros(n, bool) for m in METHODS}
    thresholds = {m: np.full(n, np.nan) for m in METHODS}
    for fold_id, test in enumerate(outer):
        train = np.setdiff1d(all_indices, test)
        fold_ids[test] = fold_id
        inner = grouped_folds(sequence[train], 3, seed + 1000 + fold_id)
        info = {"fold": fold_id, "train_sequences": sorted(set(sequence[train])),
                "test_sequences": sorted(set(sequence[test])), "train_rows": len(train),
                "test_rows": len(test), "inner_folds": [], "heads": {"quality": {}, "presence": {}}}
        for inner_id, valid_local in enumerate(inner):
            valid = train[valid_local]
            info["inner_folds"].append({"fold": inner_id, "validation_sequences": sorted(set(sequence[valid])),
                "train_sequences": sorted(set(sequence[np.setdiff1d(train, valid)]))})
        for head, labels in (("quality", quality), ("presence", presence)):
            for method in METHODS:
                if method == "seed_p":
                    test_scores, fit_meta = raw_p[test], {"fallback": None, "fit": "raw_p_no_fit"}
                else:
                    x = np.asarray(features["X_" + method], float)
                    common = None if method == "geometry" else geometry
                    test_scores, fit_meta = fit_predict(x, labels, train, test, seed, common)
                predictions[head][method][test] = test_scores
                record = {"outer_fit": fit_meta,
                          "test_class_counts": {str(c): int(np.sum(labels[test] == c)) for c in (-1, 0, 1)},
                          "test_auc_all": auc_record(labels[test], test_scores, np.ones(len(test), bool)),
                          "test_auc_written": auc_record(labels[test], test_scores, masks["written"][test])}
                if head == "quality":
                    inner_scores, fit_records = np.full(len(train), np.nan), []
                    for inner_id, valid_local in enumerate(inner):
                        valid = train[valid_local]
                        inner_train = np.setdiff1d(train, valid)
                        if method == "seed_p":
                            score, fit = raw_p[valid], {"fallback": None, "fit": "raw_p_no_fit"}
                        else:
                            score, fit = fit_predict(x, labels, inner_train, valid, seed, common)
                        inner_scores[valid_local] = score
                        fit_records.append({"fold": inner_id, **fit})
                    if not np.isfinite(inner_scores).all():
                        raise ValueError("Incomplete inner OOF prediction")
                    calibration = calibrated_threshold(inner_scores, quality[train], masks["written"][train])
                    if calibration["enabled"]:
                        cutoff = calibration["threshold"]
                        thresholds[method][test] = cutoff
                        vetoes[method][test] = test_scores < cutoff
                    record.update(calibration=calibration, inner_fits=fit_records)
                    record["test_veto"] = veto_record(quality[test], vetoes[method][test], masks["written"][test])
                info["heads"][head][method] = record
        folds.append(info)
    if np.any(fold_ids < 0) or any(not np.isfinite(x).all() for h in predictions.values() for x in h.values()):
        raise ValueError("Incomplete outer OOF prediction")
    result = {"protocol": {"seed": seed, "requested_outer_folds": outer_folds,
              "actual_outer_folds": len(outer), "inner_folds": 3,
              "fold_allocation": "whole sequences descending query count, greedily assigned to least-loaded fold; numpy default_rng seed breaks ties; no labels/features",
              "quality_target": "fixed final-box IoU hit; miss is not a background label",
              "presence_target": "separate auxiliary head; -1 ignored during fitting and metrics",
              "preprocessing": "train-only geometry StandardScaler retaining every scalar; visual StandardScaler + randomized PCA min(32, feature_dim, train_n-1), concatenated with scaled geometry before LR",
              "shuffled_control": "X_shuffled is a fixed same-query 90-degree residual-path rotation, not shuffled examples or time",
              "common_information_limit": "Single also receives historical runtime geometry; contrasts measure added historical visual information conditional on the same geometry",
              "padding_limit": "Spatial descriptors include border replication; pair masks do not exclude full CNN receptive-field padding. All-path-valid subgroups diagnose coverage effects, not a proof of pure motion evidence",
              "direction_control_limit": "The rotated path is not guaranteed incorrect in a dense swarm; unavailable or zero-residual paths may coincide. Report availability and separation subgroups",
              "capacity_limit": "Equal visual dimensions and PCA cap do not imply equal effective rank or PCA weighting; repeated single/fusion slots change covariance weighting",
              "mechanism_limit": "No target-only/background-only ablation; a dual advantage does not establish necessary dual-path interaction",
              "label_limit": "Presence is annotated-centre geometry, not visibility or identity; fixed-box quality does not imply incremental tracking value",
              "classifier": "LogisticRegression C=1 max_iter=2000 lbfgs; no class reweighting or tuning",
              "threshold": "highest observed positive boundary with strict score<threshold and at most floor(.05*N) inner-OOF written true hits rejected",
              "threshold_transfer": "inner OOF cutoff applied unchanged to outer refit; outer 5% constraint is not guaranteed",
              "interpretation": "Frozen-query information probe, not closed-loop MOT or an unbiased untouched-test claim; below_p subgroup is descriptive only"},
              "counts": {"rows": n, "nonempty_sequences": len(set(sequence)),
                         "quality": {str(c): int(np.sum(quality == c)) for c in (0, 1)},
                         "presence": {str(c): int(np.sum(presence == c)) for c in (-1, 0, 1)}},
              "feature_dimensions": {k: int(np.shape(features[k])[1]) for k in FEATURE_KEYS}, "heads": {}}
    for head, labels in (("quality", quality), ("presence", presence)):
        head_result = {"methods": {}}
        for method in METHODS:
            scores = predictions[head][method]
            entry = {"auc": {name: auc_record(labels, scores, mask) for name, mask in masks.items()},
                     "per_sequence": {}}
            if head == "quality":
                entry["veto"] = {name: veto_record(quality, vetoes[method], mask & masks["written"])
                                 for name, mask in masks.items()}
            sequence_aucs = []
            for seq in sorted(set(sequence)):
                mask = sequence == seq
                per_seq = {"auc": auc_record(labels, scores, mask)}
                if per_seq["auc"]["auc"] is not None:
                    sequence_aucs.append(per_seq["auc"]["auc"])
                if head == "quality":
                    per_seq["veto"] = veto_record(quality, vetoes[method], mask & masks["written"])
                entry["per_sequence"][seq] = per_seq
            entry["macro_sequence_auc"] = {"value": float(np.mean(sequence_aucs)) if sequence_aucs else None,
                                           "eligible_sequences": len(sequence_aucs)}
            fold_aucs = [f["heads"][head][method]["test_auc_all"]["auc"] for f in folds]
            fold_aucs = [a for a in fold_aucs if a is not None]
            entry["macro_fold_auc"] = {"value": float(np.mean(fold_aucs)) if fold_aucs else None,
                                       "eligible_folds": len(fold_aucs)}
            head_result["methods"][method] = entry
        head_result["conditional_bootstrap"] = conditional_bootstrap(sequence, labels, quality,
            predictions[head], vetoes if head == "quality" else None, masks["written"], bootstrap_repeats, seed + 2000)
        result["heads"][head] = head_result
    prediction_rows = []
    for i, row in enumerate(rows):
        item = {"row_index": i, **row, "outer_fold": int(fold_ids[i]), "predictions": {}}
        for method in METHODS:
            item["predictions"][method] = {"quality": float(predictions["quality"][method][i]),
                "presence": float(predictions["presence"][method][i]),
                "threshold": float(thresholds[method][i]) if np.isfinite(thresholds[method][i]) else None,
                "veto_if_written": bool(vetoes[method][i]),
                "written_veto": bool(vetoes[method][i] and masks["written"][i])}
        prediction_rows.append(item)
    return result, prediction_rows, {"seed": seed, "folds": folds}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, choices=(4, 5), default=5)
    parser.add_argument("--seed", type=int, default=140)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = ("metrics.json", "predictions.jsonl", "folds.json", "summary.csv")
    if any((args.out_dir / name).exists() for name in names):
        raise FileExistsError("Output artifacts already exist; use a new run/output directory")
    with np.load(args.features, allow_pickle=False) as data:
        features = {key: data[key] for key in FEATURE_KEYS}
    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    result, predictions, folds = evaluate_probe(features, rows, args.outer_folds, args.seed, args.bootstrap_repeats)
    result["provenance"] = {"inputs": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                             for path in (args.features, args.rows)},
                            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}
    def write_json(name, value):
        (args.out_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_json("metrics.json", result)
    write_json("folds.json", folds)
    (args.out_dir / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in predictions))
    with (args.out_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("head", "method", "subgroup", "n", "positive", "negative", "auc", "true_hits_rejected", "iou_misses_rejected", "true_hit_reject_rate", "iou_miss_reject_rate"))
        writer.writeheader()
        for head, content in result["heads"].items():
            for method, metrics in content["methods"].items():
                for subgroup, auc in metrics["auc"].items():
                    row = {"head": head, "method": method, "subgroup": subgroup,
                           **{k: auc[k] for k in ("n", "positive", "negative", "auc")}}
                    if head == "quality":
                        row.update({k: metrics["veto"][subgroup][k] for k in ("true_hits_rejected", "iou_misses_rejected", "true_hit_reject_rate", "iou_miss_reject_rate")})
                    writer.writerow(row)
    print(json.dumps({"status": "completed", "rows": len(rows), "outputs": [str(args.out_dir / name) for name in names]}))


if __name__ == "__main__":
    main()
