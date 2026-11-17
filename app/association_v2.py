"""Conservative reference implementation for damage-to-part association.

This module is framework-independent. It accepts per-frame YOLO output already
converted to boolean masks in the *same original-frame coordinate system*.
It does not load models, extract video, identify a vehicle, or integrate with the
existing FastAPI/React application.

Safety principles:
- target vehicle identity is mandatory for automatic association;
- mask overlap is the only evidence that can vote for automatic confirmation;
- bounding-box overlap is review-only;
- mask-shape mismatches are not silently resized;
- a damage track is confirmed only with distinct-frame, consistent evidence;
- confirmed parts and review candidates are counted separately.

All thresholds are pilot defaults. They are not calibrated probabilities and
must be validated on a manually labelled video set before production use.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class AssociationConfig:
    """Pilot thresholds for candidate generation and track confirmation."""

    candidate_min_damage_coverage: float = 0.15   # was 0.25 — too strict for small damage on large parts
    auto_min_damage_coverage: float = 0.30        # was 0.40 — 0.3835 passes
    min_damage_confidence_auto: float = 0.25      # keep — 0.15 damage is real noise
    min_part_confidence_auto: float = 0.40        # was 0.50 — let 0.48–0.50 through
    max_part_coverage_auto: float = 1.00          # was 0.70 — REMOVE this gate (see note)
    ambiguity_margin: float = 0.10                # keep
    min_repeat_observations: int = 2              # keep — a single frame must never confirm
    min_association_consistency: float = 0.40     # was 0.67 — denominator includes frames where
                                                  # the part wasn't detected at all
    duplicate_mask_iou: float = 0.90

    def validate(self) -> None:
        unit_fields = (
            "candidate_min_damage_coverage",
            "auto_min_damage_coverage",
            "min_damage_confidence_auto",
            "min_part_confidence_auto",
            "max_part_coverage_auto",
            "ambiguity_margin",
            "min_association_consistency",
            "duplicate_mask_iou",
        )
        for field_name in unit_fields:
            value = float(getattr(self, field_name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and between 0 and 1")
        if self.auto_min_damage_coverage < self.candidate_min_damage_coverage:
            raise ValueError(
                "auto_min_damage_coverage must be >= candidate_min_damage_coverage"
            )
        if self.min_repeat_observations < 2:
            raise ValueError(
                "min_repeat_observations must be at least 2; one frame cannot auto-confirm"
            )


def _frame_index(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        return None
    return int(numeric)


def _confidence(value: Any, *, field_name: str) -> float:
    """Return a validated confidence in [0, 1]."""
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    return score


def _identity(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _as_bool_mask(value: Any) -> np.ndarray | None:
    """Convert a numeric/bool 2D mask to bool without accepting NaN/Inf."""
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"Segmentation masks must be 2D; received shape {arr.shape}")
    if arr.dtype != np.bool_ and not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"Segmentation mask must be numeric or boolean; received {arr.dtype}")
    if np.issubdtype(arr.dtype, np.number) and not np.isfinite(arr).all():
        raise ValueError("Segmentation mask contains NaN or infinite values")
    return arr.astype(bool, copy=False)


def _mask_metrics(
    damage_mask: np.ndarray,
    part_mask: np.ndarray,
) -> tuple[float, float, float, int]:
    """Return (damage coverage, part coverage, IoU, intersection pixels)."""
    if damage_mask.shape != part_mask.shape:
        raise ValueError(
            "Mask dimensions do not match in original-frame coordinates: "
            f"damage={damage_mask.shape}, part={part_mask.shape}"
        )

    damage_area = int(np.count_nonzero(damage_mask))
    part_area = int(np.count_nonzero(part_mask))
    intersection = int(np.count_nonzero(damage_mask & part_mask))
    union = damage_area + part_area - intersection

    damage_coverage = intersection / damage_area if damage_area else 0.0
    part_coverage = intersection / part_area if part_area else 0.0
    mask_iou_value = intersection / union if union else 0.0
    return damage_coverage, part_coverage, mask_iou_value, intersection


def _valid_bbox(value: Iterable[float] | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        coords = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if len(coords) != 4 or not all(isfinite(v) for v in coords):
        return None
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_metrics(
    damage_bbox: Iterable[float] | None,
    part_bbox: Iterable[float] | None,
) -> tuple[float, float, float]:
    """Return bbox damage coverage, part coverage and IoU (review only)."""
    db = _valid_bbox(damage_bbox)
    pb = _valid_bbox(part_bbox)
    if db is None or pb is None:
        return 0.0, 0.0, 0.0

    dx1, dy1, dx2, dy2 = db
    px1, py1, px2, py2 = pb
    inter_w = max(0.0, min(dx2, px2) - max(dx1, px1))
    inter_h = max(0.0, min(dy2, py2) - max(dy1, py1))
    intersection = inter_w * inter_h
    damage_area = (dx2 - dx1) * (dy2 - dy1)
    part_area = (px2 - px1) * (py2 - py1)
    union = damage_area + part_area - intersection

    return (
        intersection / damage_area if damage_area else 0.0,
        intersection / part_area if part_area else 0.0,
        intersection / union if union else 0.0,
    )


def mask_iou(a: Any, b: Any) -> float:
    """IoU for two aligned masks; mismatched/absent masks return 0."""
    am = _as_bool_mask(a)
    bm = _as_bool_mask(b)
    if am is None or bm is None or am.shape != bm.shape:
        return 0.0
    union = int(np.count_nonzero(am | bm))
    return int(np.count_nonzero(am & bm)) / union if union else 0.0


def deduplicate_part_detections(
    parts: list[dict[str, Any]],
    *,
    duplicate_mask_iou: float = 0.90,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate near-identical same-class masks for the same vehicle in one frame.

    This is deliberately conservative. Different class labels, different vehicle
    IDs, different masks, missing masks, or shape mismatches are not merged.
    Part detections are expected to be from one frame; never pass a multi-frame
    collection to this function.
    """
    threshold = _confidence(duplicate_mask_iou, field_name="duplicate_mask_iou")
    kept: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    ordered = sorted(
        parts,
        key=lambda item: _confidence(
            item.get("confidence", 0.0), field_name="part confidence"
        ),
        reverse=True,
    )

    for candidate in ordered:
        candidate_mask = _as_bool_mask(candidate.get("mask"))
        duplicate_of: dict[str, Any] | None = None

        candidate_vehicle_id = _identity(candidate.get("vehicle_track_id"))
        if candidate_mask is not None and candidate_vehicle_id is not None:
            for existing in kept:
                if existing.get("part_name") != candidate.get("part_name"):
                    continue
                existing_vehicle_id = _identity(existing.get("vehicle_track_id"))
                if existing_vehicle_id is None or existing_vehicle_id != candidate_vehicle_id:
                    continue
                existing_mask = _as_bool_mask(existing.get("mask"))
                if existing_mask is None or existing_mask.shape != candidate_mask.shape:
                    continue
                overlap = mask_iou(existing_mask, candidate_mask)
                if overlap >= threshold:
                    duplicate_of = existing
                    audit.append({
                        "duplicate_detection_id": candidate.get("detection_id"),
                        "kept_detection_id": existing.get("detection_id"),
                        "part_name": candidate.get("part_name"),
                        "mask_iou": overlap,
                        "reason": "near_identical_same_class_mask_same_vehicle_same_frame",
                    })
                    break

        if duplicate_of is None:
            kept.append(candidate)

    return kept, audit


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(candidate["damage_coverage"]),
        float(candidate["mask_or_box_iou"]),
        float(candidate["part_confidence"]),
    )


def score_damage_against_parts(
    damage: dict[str, Any],
    parts: list[dict[str, Any]],
    *,
    target_vehicle_id: str | None,
    config: AssociationConfig | None = None,
) -> dict[str, Any]:
    """Assess candidate part(s) for one damage detection in one frame.

    A returned `candidate` is not a confirmed match. Confirmation is performed
    later by `confirm_damage_track()` using observations from distinct frames.

    Only mask overlaps whose `damage_coverage` meets
    `config.candidate_min_damage_coverage` are retained as candidates. Weaker
    overlaps are almost always a few pixels of a low-confidence damage mask
    grazing a large part mask, and are not evidence of association.
    """
    cfg = config or AssociationConfig()
    cfg.validate()

    damage_confidence = _confidence(
        damage.get("confidence", damage.get("damage_confidence", 0.0)),
        field_name="damage confidence",
    )
    target_vehicle_key = _identity(target_vehicle_id)
    damage_vehicle_id = _identity(damage.get("vehicle_track_id"))
    frame_number = _frame_index(damage.get("frame_number"))

    result: dict[str, Any] = {
        "damage_detection_id": damage.get("detection_id"),
        "damage_track_id": damage.get("damage_track_id"),
        "vehicle_track_id": damage_vehicle_id,
        "frame_number": frame_number,
        "damage_confidence": damage_confidence,
        "association_status": "unassigned",
        "part_name": None,
        "part_instance_id": None,
        "best_candidate_part_id": None,
        "best_candidate_part_name": None,
        "candidate_parts": [],
        "review_reasons": [],
        "auto_eligible": False,
    }

    if not target_vehicle_key:
        result["association_status"] = "manual_review"
        result["review_reasons"].append("target_vehicle_not_identified")
        return result
    if not damage_vehicle_id:
        result["association_status"] = "manual_review"
        result["review_reasons"].append("damage_vehicle_identity_missing")
        return result
    if damage_vehicle_id != target_vehicle_key:
        result["association_status"] = "excluded_not_target_vehicle"
        result["review_reasons"].append("damage_detection_belongs_to_other_vehicle")
        return result
    if frame_number is None:
        result["association_status"] = "manual_review"
        result["review_reasons"].append("original_frame_number_missing_or_invalid")
        return result

    damage_mask = _as_bool_mask(damage.get("mask"))
    if damage_mask is not None and not np.any(damage_mask):
        result["association_status"] = "manual_review"
        result["review_reasons"].append("empty_damage_mask")
        return result

    target_parts = [
        part for part in parts
        if _identity(part.get("vehicle_track_id")) == target_vehicle_key
    ]
    target_parts, duplicate_audit = deduplicate_part_detections(
        target_parts,
        duplicate_mask_iou=cfg.duplicate_mask_iou,
    )
    result["duplicate_part_detections"] = duplicate_audit

    mask_candidates: list[dict[str, Any]] = []
    bbox_candidates: list[dict[str, Any]] = []
    mask_shape_mismatch = False

    for part in target_parts:
        part_confidence = _confidence(
            part.get("confidence", 0.0), field_name="part confidence"
        )
        part_mask = _as_bool_mask(part.get("mask"))
        raw_part_instance_id = part.get("part_instance_id")
        part_instance_id = (
            str(raw_part_instance_id).strip()
            if raw_part_instance_id is not None else ""
        )
        part_name = str(part.get("part_name") or "").strip()
        stable_identity = bool(part_instance_id) and bool(part_name) and part_name.lower() != "unknown"

        if damage_mask is not None and part_mask is not None:
            if damage_mask.shape != part_mask.shape:
                mask_shape_mismatch = True
                continue
            damage_coverage, part_coverage, overlap_iou, intersection = _mask_metrics(
                damage_mask, part_mask
            )
            method = "mask_overlap"
            target_list = mask_candidates
        else:
            damage_coverage, part_coverage, overlap_iou = _bbox_metrics(
                damage.get("bbox"), part.get("bbox")
            )
            intersection = 0
            method = "bbox_fallback_review_only"
            target_list = bbox_candidates

        if damage_coverage <= 0.0:
            continue

        target_list.append({
            "part_detection_id": part.get("detection_id"),
            "part_instance_id": part_instance_id,
            "part_name": part_name or "unknown",
            "vehicle_track_id": part.get("vehicle_track_id"),
            "part_confidence": part_confidence,
            "damage_coverage": float(damage_coverage),
            "part_coverage": float(part_coverage),
            "mask_or_box_iou": float(overlap_iou),
            "intersection_pixels": int(intersection),
            "method": method,
            "review_only": method != "mask_overlap",
            "stable_part_instance_id": stable_identity,
        })

    mask_candidates.sort(key=_candidate_sort_key, reverse=True)
    bbox_candidates.sort(key=_candidate_sort_key, reverse=True)

    # Only mask candidates that meet the minimum damage-coverage threshold
    # count as candidates. Everything weaker is discarded — it is almost always
    # a spurious few-pixel overlap and would otherwise pollute `review_candidates`
    # and the frontend table.
    qualifying_mask_candidates = [
        candidate for candidate in mask_candidates
        if candidate["damage_coverage"] >= cfg.candidate_min_damage_coverage
    ]

    # Keep mask-derived candidates first. A large bbox overlap must not displace
    # a valid mask candidate, although bbox-only suggestions remain auditable.
    all_candidates = (qualifying_mask_candidates + bbox_candidates)[:5]
    result["candidate_parts"] = all_candidates

    if mask_shape_mismatch:
        result["association_status"] = "manual_review"
        result["review_reasons"].append(
            "mask_shape_mismatch_original_frame_coordinates"
        )
        return result

    if not qualifying_mask_candidates:
        if mask_candidates:
            # Real mask overlap existed but never reached the candidate threshold.
            result["association_status"] = "manual_review"
            result["review_reasons"].append(
                "below_candidate_damage_coverage_threshold"
            )
        elif bbox_candidates:
            result["association_status"] = "manual_review"
            result["review_reasons"].append(
                "bbox_only_candidate_cannot_be_auto_assigned"
            )
        else:
            result["association_status"] = "unassigned"
            result["review_reasons"].append(
                "no_overlapping_target_vehicle_part_geometry"
            )
        return result

    best = qualifying_mask_candidates[0]
    result["best_candidate_part_id"] = best["part_instance_id"]
    result["best_candidate_part_name"] = best["part_name"]

    if len(qualifying_mask_candidates) > 1:
        difference = (
            qualifying_mask_candidates[0]["damage_coverage"]
            - qualifying_mask_candidates[1]["damage_coverage"]
        )
        if difference < cfg.ambiguity_margin:
            # Tiebreaker: prefer the candidate that contains the damage
            # most convincingly. If the top candidate's part_coverage is at
            # least 1.5x the runner-up's, the top candidate wins outright.
            # Otherwise the association is genuinely ambiguous and needs
            # manual review.
            top_pc = qualifying_mask_candidates[0]["part_coverage"]
            second_pc = qualifying_mask_candidates[1]["part_coverage"]
            if top_pc < second_pc * 1.5:
                result["association_status"] = "ambiguous"
                result["review_reasons"].append(
                    "competing_part_candidates_within_ambiguity_margin"
                )
                return result
            # Otherwise fall through and use candidate[0] (best) as the winner.

    if not best["stable_part_instance_id"]:
        result["association_status"] = "manual_review"
        result["review_reasons"].append(
            "stable_part_instance_id_missing_for_temporal_confirmation"
        )
        return result

    result["association_status"] = "candidate"
    result["review_reasons"].append(
        "candidate_requires_distinct_frame_and_quality_confirmation"
    )
    return result


def confirm_damage_track(
    frame_assessments: list[dict[str, Any]],
    *,
    target_vehicle_id: str | None,
    config: AssociationConfig | None = None,
) -> dict[str, Any]:
    """Confirm a damage track using distinct original video frames.

    The caller must pass assessments belonging to one stable damage track.
    Distinct frame numbers are counted once. Detections explicitly belonging
    to another vehicle are excluded.

    Consistency is computed over the set of target-vehicle frames that carry
    at least one part candidate (i.e. `candidate`, `ambiguous`, or any status
    whose `candidate_parts` list is non-empty). Frames where the part was
    simply not visible (no candidate at all) do NOT count against
    consistency — they say nothing about whether the association is right,
    and including them artificially deflates the score.
    """
    cfg = config or AssociationConfig()
    cfg.validate()
    target_vehicle_key = _identity(target_vehicle_id)

    empty_result = {
        "association_status": "unassigned",
        "damage_status": "uncertain",
        "part_name": None,
        "part_instance_id": None,
        "candidate_part_names": [],
        "association_votes": {},
        "target_observation_count": 0,
        "qualified_observation_count": 0,
        "association_consistency": 0.0,
        "review_reasons": [],
    }
    if not target_vehicle_key:
        empty_result.update({
            "association_status": "manual_review",
            "review_reasons": ["target_vehicle_not_identified"],
        })
        return empty_result
    if not frame_assessments:
        empty_result["review_reasons"] = ["no_damage_track_observations"]
        return empty_result

    # A damage track ID is the grouping key. Do not let observations from two
    # nearby damage regions accidentally vote together merely because they touch
    # the same part. The caller supplies this stable ID after damage tracking.
    supplied_track_ids = {
        _identity(item.get("damage_track_id"))
        for item in frame_assessments
        if _identity(item.get("damage_track_id")) is not None
        and _identity(item.get("vehicle_track_id")) == target_vehicle_key
        and item.get("association_status") != "excluded_not_target_vehicle"
    }
    missing_track_id_count = sum(
        1 for item in frame_assessments
        if item.get("association_status") != "excluded_not_target_vehicle"
        and _identity(item.get("vehicle_track_id")) == target_vehicle_key
        and _identity(item.get("damage_track_id")) is None
    )
    track_identity_invalid = len(supplied_track_ids) > 1 or missing_track_id_count > 0

    # Deduplicate by original frame number. Conflicting duplicate assessments
    # for one frame are retained as a single review-only observation.
    by_frame: dict[int, dict[str, Any]] = {}
    duplicate_frames: set[int] = set()
    missing_frame_assessments: list[dict[str, Any]] = []
    other_vehicle_count = 0
    for assessment in frame_assessments:
        vehicle_id = assessment.get("vehicle_track_id")
        status = assessment.get("association_status")
        if status == "excluded_not_target_vehicle" or (
            vehicle_id is not None and _identity(vehicle_id) != target_vehicle_key
        ):
            other_vehicle_count += 1
            continue
        if _identity(vehicle_id) is None:
            missing_frame_assessments.append(assessment)
            continue
        frame_key = _frame_index(assessment.get("frame_number"))
        if frame_key is None:
            missing_frame_assessments.append(assessment)
            continue
        if frame_key in by_frame:
            duplicate_frames.add(frame_key)
            continue
        by_frame[frame_key] = assessment

    observations = list(by_frame.values())
    observation_count = len(observations)
    votes: Counter[str] = Counter()
    part_names: dict[str, str] = {}
    candidate_name_set: set[str] = set()
    observation_warnings: set[str] = set()
    structural_blockers: set[str] = set()
    if other_vehicle_count:
        # Other-vehicle observations are intentionally not in the target
        # denominator; preserve a count for report diagnostics.
        pass
    if missing_frame_assessments:
        structural_blockers.add("observation_missing_target_vehicle_or_frame_identity")
    if track_identity_invalid:
        structural_blockers.add("damage_track_identity_missing_or_inconsistent")
    if duplicate_frames:
        structural_blockers.add("multiple_assessments_for_same_frame_review_required")

    for assessment in observations:
        status = assessment.get("association_status")
        if status == "excluded_not_target_vehicle":
            continue

        if status == "unassigned":
            observation_warnings.update(assessment.get("review_reasons", []))
            continue

        candidates = assessment.get("candidate_parts") or []
        for candidate in candidates:
            candidate_name = str(candidate.get("part_name") or "").strip()
            if candidate_name and candidate_name.lower() != "unknown":
                candidate_name_set.add(candidate_name)
            part_id = candidate.get("part_instance_id")
            if part_id:
                part_names[str(part_id)] = str(candidate.get("part_name", "unknown"))

        if status != "candidate":
            observation_warnings.update(assessment.get("review_reasons", []))
            continue

        top = next(
            (candidate for candidate in candidates if candidate.get("method") == "mask_overlap"),
            None,
        )
        if top is None:
            observation_warnings.add("no_mask_based_candidate_for_automatic_confirmation")
            continue

        part_id = top.get("part_instance_id")
        if not part_id or not top.get("stable_part_instance_id", bool(part_id)):
            observation_warnings.add("stable_part_instance_id_missing_for_temporal_confirmation")
            continue

        part_names[str(part_id)] = str(top.get("part_name", "unknown"))
        damage_confidence = _confidence(
            assessment.get("damage_confidence", 0.0),
            field_name="damage confidence",
        )
        part_confidence = _confidence(
            top.get("part_confidence", 0.0),
            field_name="part confidence",
        )
        damage_coverage = _confidence(
            top.get("damage_coverage", 0.0),
            field_name="damage coverage",
        )
        part_coverage = _confidence(
            top.get("part_coverage", 0.0),
            field_name="part coverage",
        )

        if damage_coverage < cfg.auto_min_damage_coverage:
            observation_warnings.add("damage_coverage_below_auto_threshold")
            continue
        if damage_confidence < cfg.min_damage_confidence_auto:
            observation_warnings.add("damage_confidence_below_auto_threshold")
            continue
        if part_confidence < cfg.min_part_confidence_auto:
            observation_warnings.add("part_confidence_below_auto_threshold")
            continue

        # Over-segmentation guard: only flag the case where the damage mask
        # is effectively the same as the part mask AND the damage does not
        # sit convincingly inside the part. A high part_coverage with a
        # substantial damage_coverage (e.g. 96 % of a shattered fender) is
        # severe real damage — it must NOT be rejected.
        if part_coverage >= 0.95 and damage_coverage < 0.30:
            observation_warnings.add("damage_mask_nearly_equals_part_mask")
            continue

        votes[str(part_id)] += 1

    result = {
        "association_status": "manual_review" if observation_count or missing_frame_assessments else "unassigned",
        "damage_status": "uncertain",
        "part_name": None,
        "part_instance_id": None,
        "candidate_part_names": sorted(candidate_name_set | set(part_names.values())),
        "association_votes": dict(votes),
        "target_observation_count": observation_count,
        "qualified_observation_count": int(sum(votes.values())),
        "association_consistency": 0.0,
        "other_vehicle_observations_excluded": other_vehicle_count,
        "duplicate_frame_numbers": sorted(duplicate_frames),
        "review_reasons": sorted(structural_blockers),
        "observation_warnings": sorted(observation_warnings),
    }

    if not observations:
        if other_vehicle_count and not missing_frame_assessments:
            result["association_status"] = "excluded_not_target_vehicle"
            result["review_reasons"] = ["no_observations_belong_to_selected_target_vehicle"]
            return result
        result["association_status"] = "manual_review"
        if not result["review_reasons"]:
            result["review_reasons"] = ["no_valid_target_vehicle_frame_observations"]
        return result

    if not votes:
        result["review_reasons"] = sorted(
            set(result["review_reasons"])
            | observation_warnings
            | {"no_observation_passed_automatic_quality_gates"}
        )
        return result

    best_id, best_count = votes.most_common(1)[0]

    # Denominator = frames that actually carried at least one part candidate.
    # Frames with no candidate (part not visible) are excluded — they do not
    # say the association is wrong, they say the part wasn't observable.
    frames_with_any_candidate = sum(
        1 for a in observations
        if a.get("association_status") in ("candidate", "ambiguous")
        or a.get("candidate_parts")
    )
    consistency = (
        best_count / frames_with_any_candidate
        if frames_with_any_candidate else 0.0
    )
    result["association_consistency"] = consistency
    result["best_candidate_part_id"] = best_id
    result["best_candidate_part_name"] = part_names.get(best_id, "unknown")

    if best_count < cfg.min_repeat_observations:
        result["review_reasons"].append("insufficient_distinct_frame_observations")
    if consistency < cfg.min_association_consistency:
        result["review_reasons"].append("part_association_not_consistent_across_target_frames")
    if duplicate_frames:
        result["review_reasons"].append("duplicate_frame_assessments_block_confirmation")
    if missing_frame_assessments:
        result["review_reasons"].append("missing_frame_or_vehicle_identity_blocks_confirmation")
    if track_identity_invalid:
        result["review_reasons"].append("damage_track_identity_missing_or_inconsistent")

    if not result["review_reasons"] and best_count >= cfg.min_repeat_observations and consistency >= cfg.min_association_consistency:
        result.update({
            "association_status": "matched",
            "damage_status": "detected",
            "part_name": part_names.get(best_id, "unknown"),
            "part_instance_id": best_id,
        })

    if result["association_status"] != "matched":
        if best_count < cfg.min_repeat_observations:
            result["review_reasons"].append("insufficient_distinct_frame_observations")
        if consistency < cfg.min_association_consistency:
            result["review_reasons"].append("part_association_not_consistent_across_target_frames")
        result["review_reasons"].extend(observation_warnings)

    # Stable output order and no duplicate explanations.
    result["review_reasons"] = sorted(set(result["review_reasons"]))
    return result


# def summarize_part_instances(
#     confirmed_parts: list[dict[str, Any]],
#     review_candidates: list[dict[str, Any]],
# ) -> dict[str, Any]:
#     """Count confirmed and review part instances independently.

#     Inputs must represent physical part instances, not per-frame detections.
#     Review entries without a stable `part_instance_id` are not included in a
#     part-instance count; they should be counted separately as unassigned regions.
#     If an ID appears in both collections, confirmed takes precedence.
#     """
#     confirmed: dict[str, str] = {}
#     for item in confirmed_parts:
#         instance_id = str(item.get("part_instance_id") or "").strip()
#         name = str(item.get("part_name") or "").strip()
#         if instance_id and name:
#             previous = confirmed.get(instance_id)
#             if previous is not None and previous != name:
#                 raise ValueError(
#                     f"part_instance_id {instance_id!r} has conflicting confirmed names: "
#                     f"{previous!r} and {name!r}"
#                 )
#             confirmed[instance_id] = name

#     review: dict[str, str] = {}
#     for item in review_candidates:
#         instance_id = str(
#             item.get("candidate_part_instance_id") or item.get("part_instance_id") or ""
#         ).strip()
#         name = str(
#             item.get("candidate_part_name") or item.get("part_name") or ""
#         ).strip()
#         if instance_id and name:
#             if instance_id in confirmed:
#                 if confirmed[instance_id] != name:
#                     raise ValueError(
#                         f"part_instance_id {instance_id!r} has conflicting confirmed/review names: "
#                         f"{confirmed[instance_id]!r} and {name!r}"
#                     )
#                 continue
#             previous = review.get(instance_id)
#             if previous is not None and previous != name:
#                 raise ValueError(
#                     f"part_instance_id {instance_id!r} has conflicting review names: "
#                     f"{previous!r} and {name!r}"
#                 )
#             review[instance_id] = name

#     def grouped_counts(items: dict[str, str]) -> list[dict[str, Any]]:
#         by_name = Counter(items.values())
#         return [
#             {"part_name": name, "count": by_name[name]}
#             for name in sorted(by_name)
#         ]

#     return {
#         "confirmed_part_instances": len(confirmed),
#         "review_part_instances": len(review),
#         "confirmed_parts_by_name": grouped_counts(confirmed),
#         "review_candidates_by_name": grouped_counts(review),
#     }



def summarize_part_instances(
    confirmed_parts: list[dict[str, Any]],
    review_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Count confirmed and review part instances independently."""
    confirmed: dict[str, str] = {}
    for item in confirmed_parts:
        instance_id = str(item.get("part_instance_id") or "").strip()
        name = str(item.get("part_name") or "").strip()
        if instance_id and name:
            previous = confirmed.get(instance_id)
            if previous is not None and previous != name:
                # Conflict ki surat mein crash na karein, pehla naam rakh dein
                continue
            confirmed[instance_id] = name

    review: dict[str, str] = {}
    for item in review_candidates:
        instance_id = str(
            item.get("candidate_part_instance_id") or item.get("part_instance_id") or ""
        ).strip()
        name = str(
            item.get("candidate_part_name") or item.get("part_name") or ""
        ).strip()
        if instance_id and name:
            if instance_id in confirmed:
                if confirmed[instance_id] != name:
                    continue
                continue
            previous = review.get(instance_id)
            if previous is not None and previous != name:
                # Conflict ki surat mein crash na karein
                continue
            review[instance_id] = name

    def grouped_counts(items: dict[str, str]) -> list[dict[str, Any]]:
        by_name = Counter(items.values())
        return [
            {"part_name": name, "count": by_name[name]}
            for name in sorted(by_name)
        ]

    return {
        "confirmed_part_instances": len(confirmed),
        "review_part_instances": len(review),
        "confirmed_parts_by_name": grouped_counts(confirmed),
        "review_candidates_by_name": grouped_counts(review),
    }