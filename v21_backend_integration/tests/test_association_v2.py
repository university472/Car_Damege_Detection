import unittest

import numpy as np

from app.association_v2 import (
    AssociationConfig,
    confirm_damage_track,
    deduplicate_part_detections,
    score_damage_against_parts,
    summarize_part_instances,
)


TARGET = "vehicle_1"
DAMAGE_TRACK = "damage_track_1"


def rect_mask(shape=(48, 64), x1=10, y1=10, x2=30, y2=30):
    result = np.zeros(shape, dtype=bool)
    result[y1:y2, x1:x2] = True
    return result


def part(
    name="front_bumper",
    *,
    detection_id="part_det_1",
    instance_id="part_front_bumper_1",
    vehicle_id=TARGET,
    confidence=0.90,
    mask=None,
    bbox=(10, 10, 30, 30),
):
    return {
        "detection_id": detection_id,
        "part_instance_id": instance_id,
        "part_name": name,
        "vehicle_track_id": vehicle_id,
        "confidence": confidence,
        "mask": rect_mask() if mask is None else mask,
        "bbox": list(bbox) if bbox is not None else None,
    }


def damage(
    *,
    detection_id="damage_det_1",
    damage_track_id=DAMAGE_TRACK,
    vehicle_id=TARGET,
    frame=1,
    confidence=0.85,
    mask=None,
    bbox=(12, 12, 20, 20),
):
    return {
        "detection_id": detection_id,
        "damage_track_id": damage_track_id,
        "vehicle_track_id": vehicle_id,
        "frame_number": frame,
        "confidence": confidence,
        "mask": rect_mask(x1=12, y1=12, x2=20, y2=20) if mask is None else mask,
        "bbox": list(bbox) if bbox is not None else None,
    }


def make_good_assessment(
    part_id="part_front_bumper_1",
    part_name="front_bumper",
    frame=1,
    *,
    damage_track_id=DAMAGE_TRACK,
    damage_confidence=0.80,
    damage_coverage=0.75,
    part_coverage=0.10,
    part_confidence=0.90,
):
    candidate = {
        "part_detection_id": f"p-{frame}",
        "part_instance_id": part_id,
        "stable_part_instance_id": True,
        "part_name": part_name,
        "vehicle_track_id": TARGET,
        "part_confidence": part_confidence,
        "damage_coverage": damage_coverage,
        "part_coverage": part_coverage,
        "mask_or_box_iou": 0.08,
        "intersection_pixels": 48,
        "method": "mask_overlap",
        "review_only": False,
    }
    return {
        "damage_detection_id": f"d-{frame}",
        "damage_track_id": damage_track_id,
        "vehicle_track_id": TARGET,
        "frame_number": frame,
        "damage_confidence": damage_confidence,
        "association_status": "candidate",
        "candidate_parts": [candidate],
        "review_reasons": [],
    }


class AssociationV2Tests(unittest.TestCase):
    def test_config_rejects_auto_threshold_below_candidate_threshold(self):
        cfg = AssociationConfig(
            candidate_min_damage_coverage=0.50,
            auto_min_damage_coverage=0.40,
        )
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_config_requires_at_least_two_repeat_observations(self):
        cfg = AssociationConfig(min_repeat_observations=1)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_config_rejects_non_finite_threshold(self):
        cfg = AssociationConfig(ambiguity_margin=float("nan"))
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_mask_overlap_produces_candidate_not_instant_confirmation(self):
        assessment = score_damage_against_parts(
            damage(), [part()], target_vehicle_id=TARGET
        )
        self.assertEqual(assessment["association_status"], "candidate")
        self.assertIsNone(assessment["part_name"])
        self.assertEqual(assessment["candidate_parts"][0]["method"], "mask_overlap")

    def test_bbox_overlap_with_disjoint_masks_does_not_create_match(self):
        d_mask = rect_mask(x1=40, y1=30, x2=50, y2=40)
        p_mask = rect_mask(x1=10, y1=10, x2=25, y2=25)
        assessment = score_damage_against_parts(
            damage(mask=d_mask, bbox=(10, 10, 50, 40)),
            [part(mask=p_mask, bbox=(10, 10, 50, 40))],
            target_vehicle_id=TARGET,
        )
        self.assertEqual(assessment["association_status"], "unassigned")
        self.assertEqual(assessment["candidate_parts"], [])

    def test_missing_masks_produce_bbox_review_only(self):
        d = damage(mask=None)
        d["mask"] = None
        p = part(mask=None)
        p["mask"] = None
        assessment = score_damage_against_parts(d, [p], target_vehicle_id=TARGET)
        self.assertEqual(assessment["association_status"], "manual_review")
        self.assertEqual(
            assessment["candidate_parts"][0]["method"],
            "bbox_fallback_review_only",
        )
        finalized = confirm_damage_track(
            [assessment, {**assessment, "frame_number": 2}],
            target_vehicle_id=TARGET,
        )
        self.assertEqual(finalized["damage_status"], "uncertain")
        self.assertIsNone(finalized["part_name"])

    def test_bbox_fallback_does_not_displace_valid_mask_candidate(self):
        d_mask = rect_mask(x1=12, y1=12, x2=20, y2=20)
        valid = part(
            "front_bumper", detection_id="mask_part", instance_id="bumper_1",
            mask=rect_mask(x1=10, y1=10, x2=30, y2=30),
            bbox=(10, 10, 30, 30), confidence=0.90,
        )
        box_only = part(
            "front_fender", detection_id="bbox_part", instance_id="fender_1",
            mask=None, bbox=(0, 0, 60, 45), confidence=0.99,
        )
        box_only["mask"] = None
        assessment = score_damage_against_parts(
            damage(mask=d_mask), [box_only, valid], target_vehicle_id=TARGET
        )
        self.assertEqual(assessment["association_status"], "candidate")
        self.assertEqual(assessment["candidate_parts"][0]["part_name"], "front_bumper")
        self.assertEqual(assessment["candidate_parts"][0]["method"], "mask_overlap")

    def test_mask_shape_mismatch_is_reviewed_not_resized(self):
        assessment = score_damage_against_parts(
            damage(mask=np.ones((48, 64), dtype=bool)),
            [part(mask=np.ones((96, 128), dtype=bool))],
            target_vehicle_id=TARGET,
        )
        self.assertEqual(assessment["association_status"], "manual_review")
        self.assertIn(
            "mask_shape_mismatch_original_frame_coordinates",
            assessment["review_reasons"],
        )

    def test_track_with_only_other_vehicle_observations_is_excluded(self):
        result = confirm_damage_track(
            [{
                "damage_track_id": "other_damage_track",
                "vehicle_track_id": "vehicle_2",
                "frame_number": 1,
                "association_status": "excluded_not_target_vehicle",
            }],
            target_vehicle_id=TARGET,
        )
        self.assertEqual(result["association_status"], "excluded_not_target_vehicle")
        self.assertEqual(result["target_observation_count"], 0)

    def test_other_vehicle_damage_is_excluded(self):
        assessment = score_damage_against_parts(
            damage(vehicle_id="vehicle_2"),
            [part(vehicle_id="vehicle_2")],
            target_vehicle_id=TARGET,
        )
        self.assertEqual(assessment["association_status"], "excluded_not_target_vehicle")
        self.assertEqual(assessment["candidate_parts"], [])

    def test_parts_from_other_vehicles_are_not_considered(self):
        assessment = score_damage_against_parts(
            damage(), [part(vehicle_id="vehicle_2")], target_vehicle_id=TARGET
        )
        self.assertEqual(assessment["association_status"], "unassigned")
        self.assertEqual(assessment["candidate_parts"], [])

    def test_missing_target_vehicle_identity_prevents_association(self):
        assessment = score_damage_against_parts(
            damage(), [part()], target_vehicle_id=None
        )
        self.assertEqual(assessment["association_status"], "manual_review")
        self.assertIn("target_vehicle_not_identified", assessment["review_reasons"])

    def test_missing_damage_vehicle_identity_prevents_association(self):
        assessment = score_damage_against_parts(
            damage(vehicle_id=None), [part()], target_vehicle_id=TARGET
        )
        self.assertEqual(assessment["association_status"], "manual_review")
        self.assertIn("damage_vehicle_identity_missing", assessment["review_reasons"])

    def test_candidate_threshold_is_separate_from_auto_threshold(self):
        p_mask = rect_mask(x1=10, y1=10, x2=30, y2=30)
        d_mask = np.zeros((48, 64), dtype=bool)
        d_mask[10:30, 10:20] = True
        d_mask[10:30, 30:50] = True  # 1/3 of damage mask overlaps part
        assessment = score_damage_against_parts(
            damage(mask=d_mask), [part(mask=p_mask)], target_vehicle_id=TARGET
        )
        self.assertEqual(assessment["association_status"], "candidate")
        finalized = confirm_damage_track(
            [
                {**assessment, "frame_number": 1},
                {**assessment, "frame_number": 2},
            ],
            target_vehicle_id=TARGET,
        )
        self.assertEqual(finalized["damage_status"], "uncertain")
        self.assertIn("damage_coverage_below_auto_threshold", finalized["review_reasons"])

    def test_near_tied_candidates_are_ambiguous(self):
        d_mask = rect_mask(x1=18, y1=18, x2=26, y2=26)
        p1 = part(
            "front_bumper", detection_id="p1", instance_id="bumper1",
            mask=rect_mask(x1=10, y1=10, x2=28, y2=28),
        )
        p2 = part(
            "front_fender", detection_id="p2", instance_id="fender1",
            mask=rect_mask(x1=15, y1=15, x2=35, y2=35),
        )
        assessment = score_damage_against_parts(
            damage(mask=d_mask), [p1, p2], target_vehicle_id=TARGET
        )
        self.assertEqual(assessment["association_status"], "ambiguous")

    def test_clear_best_candidate_is_ranked_first(self):
        d_mask = rect_mask(x1=12, y1=12, x2=20, y2=20)
        best = part(
            "front_bumper", detection_id="p1", instance_id="bumper1",
            mask=rect_mask(x1=10, y1=10, x2=30, y2=30),
        )
        weak = part(
            "front_fender", detection_id="p2", instance_id="fender1",
            mask=rect_mask(x1=25, y1=10, x2=45, y2=30),
        )
        assessment = score_damage_against_parts(
            damage(mask=d_mask), [best, weak], target_vehicle_id=TARGET
        )
        self.assertEqual(assessment["association_status"], "candidate")
        self.assertEqual(assessment["best_candidate_part_name"], "front_bumper")

    def test_large_part_coverage_is_not_auto_confirmed(self):
        p_mask = rect_mask(x1=10, y1=10, x2=30, y2=30)
        frame_assessments = []
        for frame in (1, 2, 3):
            a = score_damage_against_parts(
                damage(mask=p_mask, frame=frame),
                [part(mask=p_mask)],
                target_vehicle_id=TARGET,
            )
            frame_assessments.append(a)
        result = confirm_damage_track(frame_assessments, target_vehicle_id=TARGET)
        self.assertEqual(result["damage_status"], "uncertain")
        self.assertIn("damage_mask_covers_large_fraction_of_part", result["review_reasons"])

    def test_stable_repeated_association_can_be_confirmed(self):
        result = confirm_damage_track(
            [make_good_assessment(frame=i) for i in (1, 2, 3)],
            target_vehicle_id=TARGET,
        )
        self.assertEqual(result["association_status"], "matched")
        self.assertEqual(result["damage_status"], "detected")
        self.assertEqual(result["part_name"], "front_bumper")
        self.assertEqual(result["part_instance_id"], "part_front_bumper_1")
        self.assertEqual(result["association_consistency"], 1.0)

    def test_single_frame_detection_remains_review(self):
        result = confirm_damage_track(
            [make_good_assessment(frame=1)], target_vehicle_id=TARGET
        )
        self.assertEqual(result["damage_status"], "uncertain")
        self.assertIn("insufficient_distinct_frame_observations", result["review_reasons"])

    def test_same_frame_cannot_be_counted_twice(self):
        same = make_good_assessment(frame=10)
        result = confirm_damage_track(
            [same, dict(same)], target_vehicle_id=TARGET
        )
        self.assertEqual(result["damage_status"], "uncertain")
        self.assertEqual(result["target_observation_count"], 1)
        self.assertIn("multiple_assessments_for_same_frame_review_required", result["review_reasons"])
        self.assertIn("insufficient_distinct_frame_observations", result["review_reasons"])

    def test_unassigned_target_frame_reduces_association_consistency(self):
        good_1 = make_good_assessment(frame=1)
        good_2 = make_good_assessment(frame=2)
        no_match = {
            "damage_track_id": DAMAGE_TRACK,
            "vehicle_track_id": TARGET,
            "frame_number": 3,
            "association_status": "unassigned",
            "candidate_parts": [],
            "review_reasons": ["no_overlapping_target_vehicle_part_geometry"],
        }
        result = confirm_damage_track(
            [good_1, good_2, no_match], target_vehicle_id=TARGET
        )
        self.assertEqual(result["target_observation_count"], 3)
        self.assertAlmostEqual(result["association_consistency"], 2 / 3)
        self.assertEqual(result["damage_status"], "uncertain")
        self.assertIn(
            "part_association_not_consistent_across_target_frames",
            result["review_reasons"],
        )

    def test_three_consistent_matches_out_of_four_target_frames_can_confirm(self):
        good = [make_good_assessment(frame=i) for i in (1, 2, 3)]
        no_match = {
            "damage_track_id": DAMAGE_TRACK,
            "vehicle_track_id": TARGET,
            "frame_number": 4,
            "association_status": "unassigned",
            "candidate_parts": [],
            "review_reasons": ["no_overlapping_target_vehicle_part_geometry"],
        }
        result = confirm_damage_track(good + [no_match], target_vehicle_id=TARGET)
        self.assertEqual(result["target_observation_count"], 4)
        self.assertEqual(result["qualified_observation_count"], 3)
        self.assertEqual(result["damage_status"], "detected")
        self.assertAlmostEqual(result["association_consistency"], 0.75)
        self.assertIn(
            "no_overlapping_target_vehicle_part_geometry",
            result["observation_warnings"],
        )

    def test_non_target_vehicle_observations_do_not_dilute_target_consistency(self):
        target_assessments = [make_good_assessment(frame=i) for i in (1, 2)]
        other = {
            "damage_track_id": "other_damage_track",
            "vehicle_track_id": "vehicle_2",
            "frame_number": 3,
            "association_status": "excluded_not_target_vehicle",
        }
        result = confirm_damage_track(
            target_assessments + [other], target_vehicle_id=TARGET
        )
        self.assertEqual(result["damage_status"], "detected")
        self.assertEqual(result["other_vehicle_observations_excluded"], 1)

    def test_inconsistent_part_votes_remain_review(self):
        result = confirm_damage_track(
            [
                make_good_assessment("bumper1", "front_bumper", 1),
                make_good_assessment("fender1", "front_fender", 2),
                make_good_assessment("bumper1", "front_bumper", 3),
            ],
            target_vehicle_id=TARGET,
        )
        self.assertEqual(result["damage_status"], "uncertain")
        self.assertIn(
            "part_association_not_consistent_across_target_frames",
            result["review_reasons"],
        )

    def test_mixed_damage_track_ids_block_confirmation(self):
        result = confirm_damage_track(
            [
                make_good_assessment(frame=1, damage_track_id="track_A"),
                make_good_assessment(frame=2, damage_track_id="track_B"),
            ],
            target_vehicle_id=TARGET,
        )
        self.assertEqual(result["damage_status"], "uncertain")
        self.assertIn("damage_track_identity_missing_or_inconsistent", result["review_reasons"])

    def test_missing_damage_track_id_blocks_confirmation(self):
        first = make_good_assessment(frame=1)
        first["damage_track_id"] = None
        result = confirm_damage_track(
            [first, make_good_assessment(frame=2)], target_vehicle_id=TARGET
        )
        self.assertEqual(result["damage_status"], "uncertain")
        self.assertIn("damage_track_identity_missing_or_inconsistent", result["review_reasons"])

    def test_fractional_or_negative_frame_number_is_rejected(self):
        fractional = score_damage_against_parts(
            damage(frame=1.5), [part()], target_vehicle_id=TARGET
        )
        negative = score_damage_against_parts(
            damage(frame=-1), [part()], target_vehicle_id=TARGET
        )
        self.assertEqual(fractional["association_status"], "manual_review")
        self.assertEqual(negative["association_status"], "manual_review")
        self.assertIn("original_frame_number_missing_or_invalid", fractional["review_reasons"])
        self.assertIn("original_frame_number_missing_or_invalid", negative["review_reasons"])

    def test_missing_frame_identity_blocks_confirmation(self):
        first = make_good_assessment(frame=1)
        second = make_good_assessment(frame=2)
        second["frame_number"] = None
        result = confirm_damage_track([first, second], target_vehicle_id=TARGET)
        self.assertEqual(result["damage_status"], "uncertain")
        self.assertIn("observation_missing_target_vehicle_or_frame_identity", result["review_reasons"])

    def test_part_without_stable_instance_id_cannot_be_auto_confirmed(self):
        p = part()
        p["part_instance_id"] = None
        p["track_id"] = "temporary_frame_track_1"
        a1 = score_damage_against_parts(damage(frame=1), [p], target_vehicle_id=TARGET)
        a2 = score_damage_against_parts(damage(frame=2), [p], target_vehicle_id=TARGET)
        result = confirm_damage_track([a1, a2], target_vehicle_id=TARGET)
        self.assertEqual(a1["association_status"], "manual_review")
        self.assertEqual(result["damage_status"], "uncertain")

    def test_review_candidates_keep_part_names_without_stable_instance_ids(self):
        p = part()
        p["part_instance_id"] = None
        a1 = score_damage_against_parts(damage(frame=1), [p], target_vehicle_id=TARGET)
        a2 = score_damage_against_parts(damage(frame=2), [p], target_vehicle_id=TARGET)
        result = confirm_damage_track([a1, a2], target_vehicle_id=TARGET)
        self.assertEqual(result["damage_status"], "uncertain")
        self.assertEqual(result["candidate_part_names"], ["front_bumper"])

    def test_duplicate_same_class_near_identical_masks_are_collapsed(self):
        p1 = part("back_door", detection_id="a", instance_id="door_a", confidence=0.90)
        p2 = part("back_door", detection_id="b", instance_id="door_b", confidence=0.80)
        kept, audit = deduplicate_part_detections([p1, p2], duplicate_mask_iou=0.90)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["detection_id"], "a")
        self.assertEqual(len(audit), 1)

    def test_distinct_wheels_are_not_merged_by_class_name(self):
        p1 = part("wheel", detection_id="a", instance_id="wheel_a",
                  mask=rect_mask(x1=3, y1=5, x2=15, y2=17))
        p2 = part("wheel", detection_id="b", instance_id="wheel_b",
                  mask=rect_mask(x1=40, y1=20, x2=55, y2=35))
        kept, audit = deduplicate_part_detections([p1, p2])
        self.assertEqual(len(kept), 2)
        self.assertEqual(audit, [])

    def test_missing_vehicle_identity_prevents_part_deduplication(self):
        p1 = part("back_door", detection_id="a", instance_id="door_a", vehicle_id=None)
        p2 = part("back_door", detection_id="b", instance_id="door_b", vehicle_id=None)
        kept, audit = deduplicate_part_detections([p1, p2])
        self.assertEqual(len(kept), 2)
        self.assertEqual(audit, [])

    def test_string_array_is_not_accepted_as_segmentation_mask(self):
        bad_mask = np.full((48, 64), "yes", dtype=object)
        with self.assertRaises(ValueError):
            score_damage_against_parts(
                damage(mask=bad_mask), [part()], target_vehicle_id=TARGET
            )

    def test_blank_target_vehicle_id_is_not_accepted(self):
        assessment = score_damage_against_parts(
            damage(), [part()], target_vehicle_id="   "
        )
        self.assertEqual(assessment["association_status"], "manual_review")
        self.assertIn("target_vehicle_not_identified", assessment["review_reasons"])

    def test_different_vehicle_instances_are_not_deduplicated(self):
        p1 = part("headlight", detection_id="a", instance_id="h1", vehicle_id="vehicle_1")
        p2 = part("headlight", detection_id="b", instance_id="h2", vehicle_id="vehicle_2")
        kept, audit = deduplicate_part_detections([p1, p2])
        self.assertEqual(len(kept), 2)
        self.assertEqual(audit, [])

    def test_empty_damage_mask_stays_review(self):
        a = score_damage_against_parts(
            damage(mask=np.zeros((48, 64), dtype=bool)), [part()], target_vehicle_id=TARGET
        )
        self.assertEqual(a["association_status"], "manual_review")
        self.assertIn("empty_damage_mask", a["review_reasons"])

    def test_nan_mask_is_rejected(self):
        bad_mask = np.zeros((48, 64), dtype=float)
        bad_mask[0, 0] = np.nan
        with self.assertRaises(ValueError):
            score_damage_against_parts(
                damage(mask=bad_mask), [part()], target_vehicle_id=TARGET
            )

    def test_invalid_confidence_is_rejected(self):
        with self.assertRaises(ValueError):
            score_damage_against_parts(
                damage(confidence=1.5), [part()], target_vehicle_id=TARGET
            )

    def test_duplicate_same_frame_count_does_not_confirm_even_with_two_copies(self):
        assessment = make_good_assessment(frame=4)
        result = confirm_damage_track(
            [assessment, dict(assessment), dict(assessment)], target_vehicle_id=TARGET
        )
        self.assertEqual(result["qualified_observation_count"], 1)
        self.assertEqual(result["target_observation_count"], 1)
        self.assertEqual(result["damage_status"], "uncertain")

    def test_summary_rejects_conflicting_names_for_same_instance_id(self):
        with self.assertRaises(ValueError):
            summarize_part_instances(
                confirmed_parts=[
                    {"part_instance_id": "part_1", "part_name": "front_bumper"},
                    {"part_instance_id": "part_1", "part_name": "front_fender"},
                ],
                review_candidates=[],
            )

    def test_summary_separates_confirmed_and_review_counts(self):
        summary = summarize_part_instances(
            confirmed_parts=[
                {"part_instance_id": "bumper1", "part_name": "front_bumper"},
            ],
            review_candidates=[
                {"candidate_part_instance_id": "bumper2", "candidate_part_name": "front_bumper"},
                {"candidate_part_instance_id": "door1", "candidate_part_name": "back_door"},
                {"candidate_part_instance_id": "bumper1", "candidate_part_name": "front_bumper"},
            ],
        )
        self.assertEqual(summary["confirmed_part_instances"], 1)
        self.assertEqual(summary["review_part_instances"], 2)
        self.assertEqual(summary["confirmed_parts_by_name"], [
            {"part_name": "front_bumper", "count": 1}
        ])
        self.assertEqual(summary["review_candidates_by_name"], [
            {"part_name": "back_door", "count": 1},
            {"part_name": "front_bumper", "count": 1},
        ])


if __name__ == "__main__":
    unittest.main()
