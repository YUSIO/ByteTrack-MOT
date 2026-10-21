"""Pure NumPy boundary tests; encoder/image smoke runs require torch/OpenCV."""
import unittest

import numpy as np

from fctrack.probe_features import (BASE_GEOMETRY, LAG_GEOMETRY, apply_affine, base_geometry,
    compose_gmc, crop_grid, current_descriptor, inverse_path_grid, label_query, lag_geometry,
    local_residual, make_paths, pair_descriptor, quality_summary, runtime_subgroups,
    spatial_descriptor, valid_grid, visual_vector)


def anchor(handle, x, y, tid=None):
    return dict(handle=handle, track_id=handle if tid is None else tid, box=[x - 1, y - 1, 2, 2], score=.9)


def query(frame=7):
    return dict(sequence="synthetic", frame=frame, proposal=[40., 40., 20., 20.], final=[40., 40., 20., 20.],
                p=.7, kind="fill", exclude_handles=[1], exclude_track_ids=[1], query_handle=1,
                gap=1, since_real=1, obs=3, estimator="offset")


class ProbeFeaturesTests(unittest.TestCase):
    def test_full_affine_inverse_and_composition(self):
        first = np.array([[0., -2., 30.], [2., 0., 4.], [0., 0., 1.]])
        second = np.array([[1., 0., 5.], [0., 1., -3.], [0., 0., 1.]])
        matrix, complete = compose_gmc({2: first, 3: second}, 1, 3)
        np.testing.assert_allclose(matrix, second @ first)
        self.assertTrue(complete)
        grid = crop_grid(np.array([50., 60.]), 20.)
        residual = np.array([3., -7.])
        inverse = inverse_path_grid(grid, matrix, residual)
        np.testing.assert_allclose(apply_affine(matrix, inverse) + residual, grid, atol=1e-12)
        self.assertFalse(compose_gmc({2: first}, 1, 3)[1])

    def test_handles_pair_but_self_and_reused_track_id_excluded(self):
        q = query()
        old = dict(anchors=[anchor(1, 20, 20), anchor(2, 40, 40), anchor(3, 60, 50), anchor(4, 45, 45, tid=1)])
        now = dict(anchors=[anchor(1, 500, 500), anchor(2, 44, 43, tid=20), anchor(3, 64, 53), anchor(4, 500, 500, tid=4)])
        result = local_residual(q, now, old, np.eye(3))
        self.assertEqual(result["anchor_handles"], [2, 3])
        np.testing.assert_allclose(result["residual"], [4., 3.])
        self.assertTrue(result["available"])

    def test_missing_history_is_not_clamped(self):
        result = make_paths(query(frame=3), {}, {}, lag=3)
        self.assertFalse(result["history_complete"])
        self.assertEqual(result["start"], 0)
        with self.assertRaises(ValueError):
            make_paths(query(), {}, {}, lag=3)

    def test_path_direction_and_rotated_control(self):
        q = query()
        old = dict(anchors=[anchor(2, 40, 40), anchor(3, 60, 50)])
        now = dict(anchors=[anchor(2, 46, 43), anchor(3, 66, 53)])
        warps = {frame: np.eye(3) for frame in range(5, 8)}
        warps[7][:2, 2] = [2., 0.]
        result = make_paths(q, {4: old, 7: now}, warps, 3)
        np.testing.assert_allclose(result["background_center"], [48., 50.])
        np.testing.assert_allclose(result["target_center"], [44., 47.])
        np.testing.assert_allclose(result["rotated_center"], [51., 46.])
        values = lag_geometry(q, result, (1., 1., 1.), {})
        self.assertEqual(len(values), len(LAG_GEOMETRY))

    def test_presence_and_quality_have_different_labels(self):
        q = query()
        q["final"] = [0., 0., 1., 1.]
        labels = label_query(q, [[45., 45., 10., 10.]])
        self.assertEqual((labels["quality"], labels["presence"]), (0, 1))
        self.assertEqual(label_query(query(), [[70., 45., 10., 10.]])["presence"], -1)
        self.assertEqual(label_query(query(), np.empty((0, 4)))["presence"], 0)

    def test_presence_radius_boundaries(self):
        # Proposal centre=(50,50), side=20: target centres at 70/80/80.01.
        self.assertEqual(label_query(query(), [[65., 45., 10., 10.]])["presence"], 1)
        self.assertEqual(label_query(query(), [[75., 45., 10., 10.]])["presence"], -1)
        self.assertEqual(label_query(query(), [[75.01, 45., 10., 10.]])["presence"], 0)

    def test_quality_rows_do_not_imply_observed_registration(self):
        missing = {"corner_count": 0, "valid_count": 0, "median_residual_px": None,
                   "p90_residual_px": None, "fb_median_px": None, "identity_fallback_ambiguous": True,
                   "quality_observed": False}
        fraction, values = quality_summary({2: missing, 3: missing}, 1, 3)
        self.assertEqual(fraction, 0.)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values[-2:], [1., 1.])

    def test_runtime_subgroup_boundaries(self):
        # side=20 means one crop pixel is 60/64 original-image pixels.
        paths = [dict(lag=lag, history_complete=True, path_available=True,
                      target_center=np.array([60. / 64., 0.]), background_center=np.zeros(2))
                 for lag in (3, 6)]
        fractions = [(1., 1., 1.), (1., 1., 1.)]
        flags = runtime_subgroups(query(), paths, 1., fractions)
        self.assertTrue(flags["all_path_crops_valid"])
        self.assertTrue(flags["paths_separated"])
        self.assertEqual(flags["path_separation_crop_pixels"], {"3": 1., "6": 1.})
        self.assertEqual(flags["sampled_valid_fractions"]["6"]["rotated"], 1.)
        self.assertFalse(runtime_subgroups(query(), paths, .999, fractions)["all_path_crops_valid"])
        self.assertFalse(runtime_subgroups(query(), paths, 1., [(1., 1., .99), fractions[1]])["all_path_crops_valid"])
        paths[1]["target_center"][0] -= 1e-6
        self.assertFalse(runtime_subgroups(query(), paths, 1., fractions)["paths_separated"])
        paths[1]["target_center"][0] = 60. / 64.
        paths[1]["path_available"] = False
        self.assertFalse(runtime_subgroups(query(), paths, 1., fractions)["paths_separated"])
        paths[1]["history_complete"] = False
        self.assertFalse(runtime_subgroups(query(), paths, 1., fractions)["all_path_crops_valid"])

    def test_equal_visual_dimensions_and_zero_missing_history(self):
        rng = np.random.default_rng(140)
        features = tuple(rng.random(shape).astype(np.float32) for shape in ((3, 64, 64), (32, 64, 64), (64, 32, 32)))
        mask = np.ones((64, 64), bool)
        current = current_descriptor(features, query(), mask)
        self.assertEqual(len(spatial_descriptor(features)), 1584)
        self.assertEqual(len(current), 2184)
        distance = pair_descriptor(features, features, mask, mask)
        self.assertEqual(len(distance), 240)
        full = visual_vector(current, features, mask, [features] * 4, [mask] * 4, [True, True])
        absent = visual_vector(current, features, mask, [features] * 4, [mask] * 4, [False, False])
        self.assertEqual(full.shape, absent.shape)
        self.assertEqual(full.size, 9960)
        np.testing.assert_allclose(absent[len(current):], 0.)
        self.assertEqual(len(base_geometry(query(), 100, 100, 1.)), len(BASE_GEOMETRY))
        self.assertEqual(valid_grid(crop_grid(np.array([-100., -100.]), 10.), 100, 100).sum(), 0)


if __name__ == "__main__":
    unittest.main()
