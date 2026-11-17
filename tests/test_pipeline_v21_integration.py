import sys
import types

try:
    import ultralytics  # noqa: F401
except ModuleNotFoundError:  # Pure pipeline tests do not load or run YOLO.
    stub = types.ModuleType("ultralytics")
    stub.YOLO = object
    sys.modules["ultralytics"] = stub

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.association_v2 import AssociationConfig, confirm_damage_track
from app.config import PipelineConfig
from app.pipeline import (
    FrameSample,
    VideoMetadata,
    _add_damage_tracks,
    _pair_score,
    aggregate_video_results,
)


TARGET = "vehicle_demo"
PART_ID = "vehicle_demo:part:front_bumper"


def mask_box(x1=10, y1=10, x2=30, y2=30, shape=(48, 64)):
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def sample(index):
    return FrameSample(
        frame=np.zeros((48, 64, 3), dtype=np.uint8),
        frame_number=index * 15,
        timestamp_seconds=float(index) * 0.5,
        sample_index=index,
    )


def make_assessment(frame_number, *, status="candidate", candidate_part_id=PART_ID):
    candidate = {
        "part_detection_id": f"part_det_{frame_number}",
        "part_instance_id": candidate_part_id,
        "stable_part_instance_id": True,
        "part_name": "front_bumper",
        "vehicle_track_id": TARGET,
        "part_confidence": 0.90,
        "damage_coverage": 0.80,
        "part_coverage": 0.12,
        "mask_or_box_iou": 0.10,
        "intersection_pixels": 64,
        "method": "mask_overlap",
        "review_only": False,
    }
    return {
        "damage_detection_id": f"damage_det_{frame_number}",
        "damage_track_id": None,
        "vehicle_track_id": TARGET,
        "frame_number": frame_number * 15,
        "damage_confidence": 0.90,
        "association_status": status,
        "candidate_parts": [candidate],
        "review_reasons": [],
    }


def make_detection(frame_index, *, old_part_track_id=None):
    bbox = [12.0, 12.0, 20.0, 20.0]
    return {
        "detection_id": f"damage_det_{frame_index}",
        "confidence": 0.90,
        "bbox": bbox,
        "mask": mask_box(12, 12, 20, 20),
        # Deliberately different old part-track values; V2.1 damage tracking
        # must not depend on part-track identity.
        "part_track_id": old_part_track_id,
        "association_assessment": make_assessment(frame_index),
    }


class PipelineV21IntegrationTests(unittest.TestCase):
    def test_pair_score_does_not_use_bbox_to_override_disjoint_masks(self):
        detection = {
            "mask": mask_box(2, 2, 8, 8),
            "bbox": [0, 0, 40, 40],
        }
        track = {
            "last_mask": mask_box(20, 20, 28, 28),
            "last_bbox": [0, 0, 40, 40],
        }
        self.assertEqual(_pair_score(detection, track), 0.0)

    def test_damage_track_is_independent_of_part_track_id(self):
        tracks = []
        first = make_detection(0, old_part_track_id="part_track_1")
        second = make_detection(1, old_part_track_id="part_track_99")
        _add_damage_tracks([first], tracks, sample(0), None, target_vehicle_id=TARGET)
        _add_damage_tracks([second], tracks, sample(1), None, target_vehicle_id=TARGET)
        self.assertEqual(first["damage_track_id"], second["damage_track_id"])
        self.assertEqual(len(tracks), 1)

    def test_short_missed_detection_is_in_temporal_denominator(self):
        tracks = []
        first = make_detection(0)
        _add_damage_tracks([first], tracks, sample(0), None, target_vehicle_id=TARGET)
        _add_damage_tracks([], tracks, sample(1), None, target_vehicle_id=TARGET)
        third = make_detection(2)
        _add_damage_tracks([third], tracks, sample(2), None, target_vehicle_id=TARGET)
        fourth = make_detection(3)
        _add_damage_tracks([fourth], tracks, sample(3), None, target_vehicle_id=TARGET)

        self.assertEqual(len(tracks), 1)
        assessments = tracks[0]["assessments"]
        self.assertEqual(len(assessments), 4)
        self.assertEqual(assessments[1]["association_status"], "unassigned")
        self.assertEqual(assessments[1]["frame_number"], 15)
        final = confirm_damage_track(
            assessments,
            target_vehicle_id=TARGET,
            config=AssociationConfig(),
        )
        self.assertEqual(final["association_status"], "matched")
        self.assertEqual(final["damage_status"], "detected")
        self.assertAlmostEqual(final["association_consistency"], 0.75)

    def test_review_candidate_is_not_exposed_as_confirmed_part_name(self):
        tracks = []
        first = make_detection(0)
        first["association_assessment"] = make_assessment(0, status="manual_review")
        first["association_assessment"]["review_reasons"] = ["test_requires_review"]
        _add_damage_tracks([first], tracks, sample(0), None, target_vehicle_id=TARGET)
        second = make_detection(1)
        second["association_assessment"] = make_assessment(1, status="manual_review")
        second["association_assessment"]["review_reasons"] = ["test_requires_review"]
        _add_damage_tracks([second], tracks, sample(1), None, target_vehicle_id=TARGET)

        with tempfile.TemporaryDirectory() as tmp:
            report = aggregate_video_results(
                [], tracks,
                config=PipelineConfig(),
                base_dir=tmp,
                video_id="demo",
                video_filename="demo.mp4",
                video_metadata=VideoMetadata(30.0, False, 100, 64, 48, 3.33),
                processed_frames=2,
                target_vehicle_id=TARGET,
                association_config=AssociationConfig(),
            )

        self.assertEqual(report["summary"]["confirmed_part_instances"], 0)
        self.assertEqual(report["summary"]["review_part_instances"], 1)
        self.assertEqual(report["confirmed_damaged_parts"], [])
        self.assertEqual(len(report["review_candidates"]), 1)
        region = report["review_candidates"][0]["damage_region"]
        self.assertIsNone(region["part_name"])
        self.assertEqual(region["candidate_part_names"], ["front_bumper"])

    def test_report_has_v21_collections_and_vehicle_assumption(self):
        tracks = []
        det = make_detection(0)
        det["association_assessment"] = make_assessment(0, status="manual_review")
        _add_damage_tracks([det], tracks, sample(0), None, target_vehicle_id=TARGET)
        with tempfile.TemporaryDirectory() as tmp:
            report = aggregate_video_results(
                [], tracks,
                config=PipelineConfig(),
                base_dir=tmp,
                video_id="demo",
                video_filename="demo.mp4",
                video_metadata=VideoMetadata(30.0, False, 100, 64, 48, 3.33),
                processed_frames=1,
                target_vehicle_id=TARGET,
                association_config=AssociationConfig(),
            )
        self.assertEqual(report["schema_version"], "2.1")
        self.assertEqual(report["vehicle_identity"]["mode"], "single_target_vehicle_per_video_assumption")
        for key in (
            "confirmed_damaged_parts",
            "review_candidates",
            "unassigned_damage_regions",
            "excluded_detections",
        ):
            self.assertIn(key, report)


if __name__ == "__main__":
    unittest.main()
