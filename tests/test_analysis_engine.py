import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis_engine import DeepfakeAnalyzer, confidence_from_probability, risk_level_from_probability, verdict_from_probability


class ProbabilityLogicTests(unittest.TestCase):
    def test_fake_verdict(self):
        self.assertEqual(verdict_from_probability(0.81, threshold=0.5), "FAKE")

    def test_real_verdict(self):
        self.assertEqual(verdict_from_probability(0.31, threshold=0.5), "REAL")

    def test_confidence_for_fake(self):
        self.assertAlmostEqual(confidence_from_probability(0.9, "FAKE"), 0.9)

    def test_confidence_for_real(self):
        self.assertAlmostEqual(confidence_from_probability(0.2, "REAL"), 0.8)

    def test_risk_level_boundaries(self):
        self.assertEqual(risk_level_from_probability(0.85), "High")
        self.assertEqual(risk_level_from_probability(0.65), "Medium")
        self.assertEqual(risk_level_from_probability(0.2), "Low")


class HeuristicAdjustmentTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.analyzer = DeepfakeAnalyzer([], Path(self.tempdir.name))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_large_smooth_portrait_gets_fake_boost(self):
        frame = np.full((849, 849, 3), 170, dtype=np.uint8)
        adjustment, notes = self.analyzer._image_heuristic_adjustment(  # pylint: disable=protected-access
            frame,
            face_count=1,
            rejected_face_regions=0,
            max_face_area_ratio=0.72,
        )
        self.assertGreater(adjustment, 0.0)
        self.assertTrue(any("portrait" in note.lower() for note in notes))

    def test_high_resolution_small_face_photo_gets_real_bias(self):
        frame = np.full((4000, 3000, 3), 120, dtype=np.uint8)
        adjustment, notes = self.analyzer._image_heuristic_adjustment(  # pylint: disable=protected-access
            frame,
            face_count=1,
            rejected_face_regions=0,
            max_face_area_ratio=0.08,
        )
        self.assertLess(adjustment, 0.0)
        self.assertTrue(any("high-resolution camera framing" in note.lower() for note in notes))

    def test_low_motion_video_gets_fake_boost(self):
        adjustment, threshold_offset, notes = self.analyzer._video_heuristic_adjustment(  # pylint: disable=protected-access
            face_frame_ratio=0.0,
            avg_face_probability=None,
            avg_background_probability=0.515,
            motion_mean=1.8,
            motion_std=0.5,
            score_std=0.0004,
            sampled_frames=64,
        )
        self.assertGreater(adjustment, 0.0)
        self.assertLess(threshold_offset, 0.0)
        self.assertTrue(any("low-motion" in note.lower() for note in notes))

    def test_high_motion_video_gets_real_bias(self):
        adjustment, threshold_offset, notes = self.analyzer._video_heuristic_adjustment(  # pylint: disable=protected-access
            face_frame_ratio=0.03,
            avg_face_probability=None,
            avg_background_probability=0.516,
            motion_mean=9.0,
            motion_std=4.5,
            score_std=0.0008,
            sampled_frames=72,
        )
        self.assertLess(adjustment, 0.0)
        self.assertGreater(threshold_offset, 0.0)
        self.assertTrue(any("motion" in note.lower() for note in notes))


if __name__ == "__main__":
    unittest.main()
