"""Hit-probability model of a fill-in attempt (Exp078).

An attempt is "a lost member's estimated box, verified on the image". Its hit probability is the probability that the
box, had it been written, is matched by the evaluator (IoU >= 0.5 with an annotated row that nothing else covers).
The model is a logistic regression on a few quantities that are known when the decision is made; the same `featurize`
is used for training (from the logged attempts) and inside the tracker, so the two cannot drift apart.
"""
import json

import numpy as np

FEATURES = ("logit_p", "shift", "dlog_side", "gap", "since_real", "log_obs", "log_cov", "offset",
            "n_nb", "spread", "det_iou", "det_score", "cur_iou")


def featurize(r):
    """r: mapping with the logged attempt fields (see ATTEMPT_COLUMNS in fctrack/run.py). Returns the feature vector."""
    pw, ph, fw, fh = r["prop_w"], r["prop_h"], r["final_w"], r["final_h"]
    side = float(np.sqrt(max(pw * ph, 1.0)))
    pc = np.array([r["prop_x"] + pw / 2, r["prop_y"] + ph / 2])
    fc = np.array([r["final_x"] + fw / 2, r["final_y"] + fh / 2])
    p = min(max(float(r["p"]), 1e-6), 1 - 1e-6)
    return np.array([np.log(p / (1 - p)), np.linalg.norm(fc - pc) / side, 0.5 * np.log(max(fw * fh, 1.0) / max(pw * ph, 1.0)),
                     r["gap"], r["since_real"], np.log1p(r["obs"]), np.log(max(r["cov_x"], 1e-6)),
                     1.0 if str(r["estimator"]).startswith("offset") else 0.0, r["n_nb"], r["spread"] / side,
                     r["det_iou"], r["det_score"], r["cur_iou"]], float)


def load(path):
    m = json.loads(open(path).read())
    assert tuple(m["features"]) == FEATURES, "model was trained with other features"
    return {k: np.asarray(m[k], float) if k in ("mean", "std", "coef") else m[k] for k in m}


def predict(model, x):
    z = (np.asarray(x, float) - model["mean"]) / model["std"]
    return float(1.0 / (1.0 + np.exp(-(z @ model["coef"] + model["intercept"]))))
