"""Run: py -3.10 -B heat_map/test_heatmaps.py (no writes outside heat_map)."""

import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent


class HeatmapTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.find_spec("gradcam_analysis")
        self.assertIsNotNone(spec, "Grad-CAM implementation is not present yet")
        import gradcam_analysis

        return gradcam_analysis

    def test_cam_preserves_positive_spatial_contribution_and_target(self):
        # Changing target, averaging over channels or removing ReLU breaks this.
        m = self.module()
        spatial = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[4.0, 3.0], [2.0, 1.0]]]])
        head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[2.0, -1.0], [-2.0, 1.0]]))
        result = m.gradcam_from_spatial(spatial, head, 0)
        np.testing.assert_allclose(result["cam"], [[0.0, 0.25], [1.0, 1.75]])
        np.testing.assert_allclose(result["logits"], [2.5, -2.5])
        alternate = m.gradcam_from_spatial(spatial, head, 1)
        np.testing.assert_allclose(alternate["cam"], [[0.5, 0.0], [0.0, 0.0]])
        self.assertFalse(spatial.requires_grad)

    def test_zero_cam_is_not_painted_as_positive_and_shared_scale_is_common(self):
        m = self.module()
        maps = [np.zeros((2, 2)), np.array([[0.0, 2.0], [1.0, 0.0]])]
        normalized, maximum = m.normalize_maps(maps)
        self.assertEqual(maximum, 2.0)
        np.testing.assert_array_equal(normalized[0], np.zeros((2, 2)))
        np.testing.assert_allclose(normalized[1], [[0.0, 1.0], [0.5, 0.0]])

    def test_prediction_mismatch_fails_closed(self):
        m = self.module()
        m.check_prediction([0.8, 0.2], [0.80001, 0.19999], tolerance=1e-4)
        with self.assertRaises(ValueError):
            m.check_prediction([0.8, 0.2], [0.7, 0.3], tolerance=1e-4)
        with self.assertRaises(ValueError):
            m.check_prediction([0.50001, 0.49999], [0.49999, 0.50001], tolerance=1e-4)

    def test_source_refuses_traversal_and_links(self):
        m = self.module()
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "unsafe.tar.gz"
            with tarfile.open(path, "w:gz") as tar:
                member = tarfile.TarInfo("../escape.txt")
                member.size = 3
                tar.addfile(member, io.BytesIO(b"bad"))
            with self.assertRaises(ValueError):
                with m.Snapshot(path):
                    pass
            with tarfile.open(path, "w:gz") as tar:
                member = tarfile.TarInfo("linked")
                member.type = tarfile.SYMTYPE
                member.linkname = "../outside"
                tar.addfile(member)
            with self.assertRaises(ValueError):
                with m.Snapshot(path):
                    pass

    def test_selection_is_stable_balanced_and_disagreement_is_explicit(self):
        m = self.module()
        import pandas as pd

        frame = pd.DataFrame(
            {
                "objid": ["5", "4", "3", "2", "1", "6"],
                "true_class": ["a", "a", "b", "b", "c", "c"],
                "pred_B2": ["a", "a", "b", "b", "c", "c"],
                "pred_Q": ["a", "b", "b", "b", "c", "c"],
                "pred_QF": ["a", "a", "b", "b", "c", "c"],
            }
        )
        a = m.select_examples(frame, ["a", "b", "c"], 1, 123, False)
        b = m.select_examples(frame.sample(frac=1, random_state=5), ["a", "b", "c"], 1, 123, False)
        self.assertEqual(a.objid.tolist(), b.objid.tolist())
        self.assertEqual(a.true_class.tolist(), ["a", "b", "c"])
        self.assertTrue(a.selection_reason.eq("hash_sample_by_true_class").all())

    def test_b2_and_quantum_heads_use_checkpoint_without_optimizing(self):
        m = self.module()
        config = {
            "model": "B2",
            "n_qubits": 4,
            "n_layers": 2,
            "class_names": ["a", "b"],
            "training_config": {"feature_bank": {"feature_dimension": 512}},
        }
        head = m.make_head(config, "tanh_pi")
        self.assertEqual(tuple(head(torch.zeros(1, 512)).shape), (1, 2))
        for family in ["Q", "QF"]:
            config["model"] = family
            head = m.make_head(config, "tanh_pi")
            before = {k: v.detach().clone() for k, v in head.state_dict().items()}
            result = m.gradcam_from_spatial(torch.randn(1, 512, 2, 2), head, 0)
            self.assertTrue(np.isfinite(result["cam"]).all())
            self.assertGreater(np.abs(result["gradient"]).max(), 0.0)
            for key, value in head.state_dict().items():
                self.assertTrue(torch.equal(before[key], value))


if __name__ == "__main__":
    unittest.main(verbosity=2)
