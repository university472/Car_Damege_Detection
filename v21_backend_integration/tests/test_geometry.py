import sys
import types

try:
    import ultralytics  # noqa: F401
except ModuleNotFoundError:  # Allows pure helper tests where the model package is absent.
    stub = types.ModuleType("ultralytics")
    stub.YOLO = object
    sys.modules["ultralytics"] = stub

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from app.pipeline import (
    bbox_iou,
    mask_iou,
    associate_damage_with_parts,
    _canonical_part_instance_id,
    _intersection_coverage,
    _serialize_damage_track,
)


class GeometryTests(unittest.TestCase):
    def test_singleton_parts_rejoin_track_fragments_but_wheels_do_not(self):
        self.assertEqual(
            _canonical_part_instance_id("front_bumper", "part_track_1"),
            _canonical_part_instance_id("front_bumper", "part_track_99"),
        )
        self.assertNotEqual(
            _canonical_part_instance_id("wheel", "part_track_1"),
            _canonical_part_instance_id("wheel", "part_track_99"),
        )

    def test_vehicle_scopes_singleton_part_ids(self):
        self.assertEqual(
            _canonical_part_instance_id("front_bumper", "part_track_1", "vehicle_A"),
            "vehicle_A:part:front_bumper",
        )
        self.assertNotEqual(
            _canonical_part_instance_id("front_bumper", "part_track_1", "vehicle_A"),
            _canonical_part_instance_id("front_bumper", "part_track_1", "vehicle_B"),
        )

    def test_repeated_parts_keep_separate_vehicle_scoped_tracks(self):
        self.assertNotEqual(
            _canonical_part_instance_id("wheel", "part_track_1", "vehicle_A"),
            _canonical_part_instance_id("wheel", "part_track_2", "vehicle_A"),
        )

    def test_bbox_iou_identical(self):
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_bbox_iou_disjoint(self):
        self.assertEqual(bbox_iou([0, 0, 10, 10], [11, 11, 20, 20]), 0.0)

    def test_mask_iou_identical(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:10, 5:10] = True
        self.assertAlmostEqual(mask_iou(mask, mask.copy()), 1.0)

    def test_mask_iou_mismatch_is_not_resized(self):
        self.assertEqual(mask_iou(np.ones((10, 10), dtype=bool), np.ones((20, 20), dtype=bool)), 0.0)

    def test_damage_associates_by_mask_containment(self):
        part_mask = np.zeros((20, 20), dtype=bool)
        part_mask[2:18, 2:18] = True
        damage_mask = np.zeros((20, 20), dtype=bool)
        damage_mask[8:10, 8:10] = True
        parts = [{"detection_id": "p1", "part_name": "hood", "confidence": 0.9,
                  "bbox": [2, 2, 18, 18], "mask": part_mask, "track_id": "part_track_1"}]
        damages = [{"detection_id": "d1", "confidence": 0.8, "bbox": [8, 8, 10, 10],
                    "mask": damage_mask, "part_track_id": None, "part_name": None,
                    "association_status": "unassigned", "candidate_parts": []}]
        associate_damage_with_parts(damages, parts, min_damage_coverage=0.25, ambiguity_margin=0.1)
        self.assertEqual(damages[0]["association_status"], "matched")
        self.assertEqual(damages[0]["part_name"], "hood")
        self.assertEqual(damages[0]["part_track_id"], "part_track_1")

    def test_bbox_only_legacy_candidate_is_review_not_auto_match(self):
        parts = [{
            "detection_id": "p1", "part_name": "hood", "confidence": 0.95,
            "bbox": [0, 0, 20, 20], "mask": None, "track_id": "part_track_1",
        }]
        damages = [{
            "detection_id": "d1", "confidence": 0.90, "bbox": [5, 5, 10, 10],
            "mask": None, "part_track_id": None, "part_name": None,
            "association_status": "unassigned", "candidate_parts": [],
        }]
        associate_damage_with_parts(damages, parts)
        self.assertEqual(damages[0]["association_status"], "manual_review")
        self.assertIsNone(damages[0]["part_name"])
        self.assertIsNone(damages[0]["part_track_id"])
        self.assertEqual(damages[0]["candidate_parts"][0]["method"], "bbox_fallback")

    def test_ambiguous_overlap_requires_review(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:12, 8:12] = True
        p1 = np.zeros((20, 20), dtype=bool); p1[2:15, 2:15] = True
        p2 = np.zeros((20, 20), dtype=bool); p2[5:18, 5:18] = True
        parts = [
            {"detection_id": "p1", "part_name": "hood", "confidence": 0.9,
             "bbox": [2, 2, 15, 15], "mask": p1, "track_id": "part_track_1"},
            {"detection_id": "p2", "part_name": "front_bumper", "confidence": 0.9,
             "bbox": [5, 5, 18, 18], "mask": p2, "track_id": "part_track_2"},
        ]
        damages = [{"detection_id": "d1", "confidence": 0.8, "bbox": [8, 8, 12, 12],
                    "mask": mask, "part_track_id": None, "part_name": None,
                    "association_status": "unassigned", "candidate_parts": []}]
        associate_damage_with_parts(damages, parts, min_damage_coverage=0.25, ambiguity_margin=0.10)
        self.assertEqual(damages[0]["association_status"], "ambiguous")
        self.assertIsNone(damages[0]["part_track_id"])

    def test_intersection_reports_fraction_of_part_covered(self):
        part_mask = np.zeros((20, 20), dtype=bool)
        part_mask[2:18, 2:18] = True
        damage_mask = np.zeros((20, 20), dtype=bool)
        damage_mask[2:18, 2:18] = True
        damage = {"mask": damage_mask, "bbox": [2, 2, 18, 18]}
        part = {"mask": part_mask, "bbox": [2, 2, 18, 18]}
        damage_coverage, part_coverage, iou, method = _intersection_coverage(damage, part)
        self.assertAlmostEqual(damage_coverage, 1.0)
        self.assertAlmostEqual(part_coverage, 1.0)
        self.assertAlmostEqual(iou, 1.0)
        self.assertEqual(method, "mask_overlap")

    def test_intersection_mismatched_masks_stays_unusable(self):
        damage = {"mask": np.ones((20, 20), dtype=bool), "bbox": [0, 0, 20, 20]}
        part = {"mask": np.ones((10, 10), dtype=bool), "bbox": [0, 0, 20, 20]}
        self.assertEqual(_intersection_coverage(damage, part), (0.0, 0.0, 0.0, "mask_shape_mismatch"))

    def test_large_part_coverage_requires_review_not_auto_confirmation(self):
        observations = [
            {"frame_number": 0, "timestamp_seconds": 0.0, "sample_index": 0,
             "confidence": 0.85, "bbox": [0, 0, 100, 100], "frame_path": None},
            {"frame_number": 15, "timestamp_seconds": 0.5, "sample_index": 1,
             "confidence": 0.83, "bbox": [0, 0, 100, 100], "frame_path": None},
        ]
        track = {
            "observations": observations,
            "best_confidence": 0.85,
            "best_association_status": "matched",
            "part_name": "front_glass",
            "best_part_confidence": 0.95,
            "best_bbox": [0, 0, 100, 100],
            "best_frame_number": 0,
            "best_timestamp_seconds": 0.0,
            "best_frame_path": None,
            "best_candidate_parts": [{
                "part_name": "front_glass", "damage_coverage": 0.98,
                "part_coverage": 0.92, "part_confidence": 0.95,
                "method": "mask_overlap",
            }],
        }
        config = SimpleNamespace(
            min_repeat_observations=2,
            review_confidence=0.25,
            max_damage_part_coverage_auto_confirm=0.70,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            region = _serialize_damage_track(track, "damage_test", Path(temp_dir), config)
        self.assertEqual(region["damage_status"], "uncertain")
        self.assertIn("damage_mask_covers_large_fraction_of_part", region["review_reasons"])


if __name__ == "__main__":
    unittest.main()
