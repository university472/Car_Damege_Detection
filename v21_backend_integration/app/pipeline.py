from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import json
import logging
import math
import shutil

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import PipelineConfig
from .association_v2 import (
    AssociationConfig,
    confirm_damage_track,
    score_damage_against_parts,
    summarize_part_instances,
)

logger = logging.getLogger(__name__)


@dataclass
class FrameSample:
    frame: np.ndarray  # OpenCV BGR image
    frame_number: int  # zero-based original video frame index
    timestamp_seconds: float
    sample_index: int  # zero-based index among sampled frames


@dataclass
class VideoMetadata:
    fps: float
    fps_was_fallback: bool
    frame_count: int
    width: int
    height: int
    duration_seconds: float


def inspect_video(video_path: str | Path) -> VideoMetadata:
    """Read basic video metadata, raising a clear error for unsupported files."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"OpenCV could not open the video: {video_path}")
    fps_value = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_was_fallback = not math.isfinite(fps_value) or fps_value <= 0.0
    fps = 25.0 if fps_was_fallback else fps_value
    frame_count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    width = max(0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
    height = max(0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
    cap.release()
    if width <= 0 or height <= 0:
        raise ValueError("The video has invalid or unreadable frame dimensions.")
    duration = frame_count / fps if frame_count > 0 else 0.0
    return VideoMetadata(fps, fps_was_fallback, frame_count, width, height, duration)


def extract_video_frames(
    video_path: str | Path,
    sample_fps: float = 2.0,
    *,
    source_fps: float | None = None,
) -> Iterator[FrameSample]:
    """Yield regularly sampled frames with original frame numbers and timestamps.

    The frame is not resized here; model masks and boxes are normalized back to the
    original frame's pixel coordinate system by the inference helpers below.
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than 0")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"OpenCV could not decode video: {video_path}")
    actual_fps = float(source_fps or cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(actual_fps) or actual_fps <= 0:
        actual_fps = 25.0
    step = max(1, int(round(actual_fps / sample_fps)))
    frame_number = 0
    sample_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_number % step == 0:
                yield FrameSample(
                    frame=frame,
                    frame_number=frame_number,
                    timestamp_seconds=frame_number / actual_fps,
                    sample_index=sample_index,
                )
                sample_index += 1
            frame_number += 1
    finally:
        cap.release()


def _names_dict(model: YOLO) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def _mask_for_detection(result: Any, detection_index: int, width: int, height: int) -> np.ndarray | None:
    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        return None
    if detection_index >= len(masks.data):
        return None
    raw = masks.data[detection_index].detach().cpu().numpy()
    raw = (raw > 0).astype(np.uint8)
    if raw.shape[:2] != (height, width):
        raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
    return raw.astype(bool)


def _bbox_from_xyxy(values: Any) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in values]
    return [x1, y1, x2, y2]


def _model_predict(model: YOLO, frame: np.ndarray, *, conf: float, iou: float,
                   image_size: int, device: str | int) -> Any:
    result_list = model.predict(
        source=frame,
        conf=conf,
        iou=iou,
        imgsz=image_size,
        device=device,
        retina_masks=True,
        verbose=False,
        max_det=300,
    )
    if not result_list:
        return None
    return result_list[0]


def predict_car_parts(
    model: YOLO,
    frame: np.ndarray,
    *,
    conf: float = 0.25,
    iou: float = 0.60,
    image_size: int = 640,
    device: str | int = "cpu",
) -> list[dict[str, Any]]:
    """Run the trained part-segmentation model on a full original frame."""
    h, w = frame.shape[:2]
    result = _model_predict(model, frame, conf=conf, iou=iou, image_size=image_size, device=device)
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []
    names = _names_dict(model)
    detections: list[dict[str, Any]] = []
    for idx, box in enumerate(result.boxes):
        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        bbox = _bbox_from_xyxy(box.xyxy[0].detach().cpu().tolist())
        detections.append({
            "detection_id": f"part_det_{idx:03d}",
            "part_name": names.get(cls_id, f"class_{cls_id}"),
            "class_id": cls_id,
            "confidence": confidence,
            "bbox": bbox,
            "mask": _mask_for_detection(result, idx, w, h),
            "track_id": None,
        })
    return detections


def predict_damage(
    model: YOLO,
    frame: np.ndarray,
    *,
    conf: float = 0.10,
    iou: float = 0.50,
    image_size: int = 640,
    device: str | int = "cpu",
) -> list[dict[str, Any]]:
    """Run the trained generic damage-segmentation model on the full frame.

    This preserves the notebook's generic `damage` class. It does not infer a
    scratch/dent/crack subtype because the inspected notebook trains one class.
    """
    h, w = frame.shape[:2]
    result = _model_predict(model, frame, conf=conf, iou=iou, image_size=image_size, device=device)
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []
    names = _names_dict(model)
    detections: list[dict[str, Any]] = []
    for idx, box in enumerate(result.boxes):
        cls_id = int(box.cls[0].item())
        bbox = _bbox_from_xyxy(box.xyxy[0].detach().cpu().tolist())
        detections.append({
            "detection_id": f"damage_det_{idx:03d}",
            "class_id": cls_id,
            "model_label": names.get(cls_id, "damage"),
            "damage_type": "unknown",
            "confidence": float(box.conf[0].item()),
            "bbox": bbox,
            "mask": _mask_for_detection(result, idx, w, h),
            "part_track_id": None,
            "part_name": None,
            "part_confidence": None,
            "association_status": "unassigned",
            "candidate_parts": [],
        })
    return detections


def bbox_iou(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Return mask IoU only when masks share the same coordinate dimensions."""
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def _intersection_coverage(
    damage: dict[str, Any], part: dict[str, Any]
) -> tuple[float, float, float, str]:
    """Return (fraction of damage inside part, fraction of part marked damaged, IoU, method).

    The two coverage ratios answer different questions. The first helps assign a
    damage detection to a part; the second identifies suspicious masks that paint
    most of a panel/glass area as damage.
    """
    dm, pm = damage.get("mask"), part.get("mask")
    if dm is not None and pm is not None:
        if dm.shape != pm.shape:
            return 0.0, 0.0, 0.0, "mask_shape_mismatch"
        inter = int(np.logical_and(dm, pm).sum())
        d_area = int(dm.sum())
        p_area = int(pm.sum())
        union = int(np.logical_or(dm, pm).sum())
        return (
            inter / d_area if d_area else 0.0,
            inter / p_area if p_area else 0.0,
            inter / union if union else 0.0,
            "mask_overlap",
        )
    # Conservative fallback when an output lacks a mask; mark method for review.
    db, pb = damage.get("bbox"), part.get("bbox")
    if db is None or pb is None:
        return 0.0, 0.0, 0.0, "no_geometry"
    x1, y1 = max(db[0], pb[0]), max(db[1], pb[1])
    x2, y2 = min(db[2], pb[2]), min(db[3], pb[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    d_area = max(0.0, db[2] - db[0]) * max(0.0, db[3] - db[1])
    p_area = max(0.0, pb[2] - pb[0]) * max(0.0, pb[3] - pb[1])
    union = d_area + p_area - inter
    return (
        inter / d_area if d_area else 0.0,
        inter / p_area if p_area else 0.0,
        inter / union if union else 0.0,
        "bbox_fallback",
    )


def associate_damage_with_parts(
    damage_detections: list[dict[str, Any]],
    part_detections: list[dict[str, Any]],
    *,
    min_damage_coverage: float = 0.25,
    ambiguity_margin: float = 0.10,
) -> list[dict[str, Any]]:
    """Associate masks by fraction of damage inside a part, not box intersection alone.

    A close second candidate makes the association ambiguous instead of confidently
    attaching the damage to whichever part won by a tiny margin.
    """
    for damage in damage_detections:
        candidates: list[dict[str, Any]] = []
        for part in part_detections:
            coverage, part_coverage, iou, method = _intersection_coverage(damage, part)
            if coverage <= 0:
                continue
            candidates.append({
                "part_detection_id": part["detection_id"],
                "part_track_id": part.get("track_id"),
                "part_name": part["part_name"],
                "part_confidence": float(part["confidence"]),
                "damage_coverage": float(coverage),
                "part_coverage": float(part_coverage),
                "mask_or_box_iou": float(iou),
                "method": method,
            })
        candidates.sort(
            key=lambda c: (
                c["method"] == "mask_overlap",
                c["damage_coverage"],
                c["mask_or_box_iou"],
                c["part_confidence"],
            ),
            reverse=True,
        )
        damage["candidate_parts"] = candidates[:3]
        if not candidates or candidates[0]["damage_coverage"] < min_damage_coverage:
            damage["association_status"] = "unassigned"
            continue
        best = candidates[0]
        if best["method"] != "mask_overlap":
            # Retain bbox-only suggestions for review; never auto-assign them.
            damage["association_status"] = "manual_review"
            damage["part_name"] = None
            damage["part_track_id"] = None
            damage["review_reasons"] = ["bbox_only_candidate_cannot_be_auto_assigned"]
            continue
        second_coverage = candidates[1]["damage_coverage"] if len(candidates) > 1 else 0.0
        is_close = len(candidates) > 1 and second_coverage >= min_damage_coverage and (best["damage_coverage"] - second_coverage) < ambiguity_margin
        if is_close:
            damage["association_status"] = "ambiguous"
            # Do not count an uncertain match as a damaged part. Preserve candidates for review.
            continue
        damage["association_status"] = "matched"
        damage["part_track_id"] = best.get("part_track_id")
        damage["part_name"] = best["part_name"]
        damage["part_confidence"] = best["part_confidence"]
        damage["association_method"] = best["method"]
    return damage_detections


def _pair_score(detection: dict[str, Any], track: dict[str, Any]) -> float:
    """Conservative short-term tracking score.

    When both masks exist they must match in shape and mask IoU is authoritative.
    Bounding boxes are used only when at least one mask is unavailable.
    """
    detection_mask = detection.get("mask")
    track_mask = track.get("last_mask")
    if detection_mask is not None and track_mask is not None:
        if detection_mask.shape != track_mask.shape:
            return 0.0
        return mask_iou(detection_mask, track_mask)
    return bbox_iou(detection.get("bbox"), track.get("last_bbox"))


def _track_observation(sample: FrameSample, detection: dict[str, Any], frame_path: str | None = None) -> dict[str, Any]:
    return {
        "frame_number": int(sample.frame_number),
        "timestamp_seconds": round(float(sample.timestamp_seconds), 3),
        "sample_index": int(sample.sample_index),
        "confidence": float(detection["confidence"]),
        "bbox": [round(float(v), 2) for v in detection["bbox"]],
        "frame_path": frame_path,
    }


def assign_part_tracks(
    detections: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    sample: FrameSample,
    *,
    min_iou: float = 0.10,
    max_gap: int = 2,
) -> None:
    """Greedy short-term association; deliberately does not claim cross-view re-ID."""
    candidates: list[tuple[float, int, int]] = []
    for di, det in enumerate(detections):
        for ti, track in enumerate(tracks):
            if det["part_name"] != track["part_name"]:
                continue
            if sample.sample_index - track["last_sample_index"] > max_gap + 1:
                continue
            score = _pair_score(det, track)
            if score >= min_iou:
                candidates.append((score, di, ti))
    candidates.sort(reverse=True)
    used_dets: set[int] = set()
    used_tracks: set[int] = set()
    for score, di, ti in candidates:
        if di in used_dets or ti in used_tracks:
            continue
        det, track = detections[di], tracks[ti]
        det["track_id"] = track["track_id"]
        track["last_sample_index"] = sample.sample_index
        track["last_frame_number"] = sample.frame_number
        track["last_bbox"] = det["bbox"]
        track["last_mask"] = det.get("mask")
        track["max_confidence"] = max(track["max_confidence"], float(det["confidence"]))
        track["observations"].append(_track_observation(sample, det))
        used_dets.add(di)
        used_tracks.add(ti)
    for di, det in enumerate(detections):
        if di in used_dets:
            continue
        track_id = f"part_track_{len(tracks) + 1:06d}"
        det["track_id"] = track_id
        tracks.append({
            "track_id": track_id,
            "part_name": det["part_name"],
            "max_confidence": float(det["confidence"]),
            "last_sample_index": sample.sample_index,
            "last_frame_number": sample.frame_number,
            "last_bbox": det["bbox"],
            "last_mask": det.get("mask"),
            "observations": [_track_observation(sample, det)],
        })


def _add_damage_tracks(
    detections: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    sample: FrameSample,
    frame_path: str | None,
    *,
    target_vehicle_id: str,
    min_iou: float = 0.10,
    max_gap: int = 2,
) -> None:
    """Assign independent damage-track IDs and record V2.1 frame assessments.

    Damage tracking deliberately does not depend on part_track_id. This prevents
    an uncertain/changed part association from splitting or merging damage tracks.
    A short run of missed damage detections is recorded as an unassigned target
    observation so temporal consistency is not inflated by intermittent matches.
    """
    candidates: list[tuple[float, int, int]] = []
    for di, det in enumerate(detections):
        for ti, track in enumerate(tracks):
            if sample.sample_index - track["last_detected_sample_index"] > max_gap + 1:
                continue
            score = _pair_score(det, track)
            if score >= min_iou:
                candidates.append((score, di, ti))

    candidates.sort(reverse=True)
    used_dets: set[int] = set()
    used_tracks: set[int] = set()
    for _score, di, ti in candidates:
        if di in used_dets or ti in used_tracks:
            continue
        det, track = detections[di], tracks[ti]
        det["damage_track_id"] = track["track_id"]
        used_dets.add(di)
        used_tracks.add(ti)

    for di, det in enumerate(detections):
        if di in used_dets:
            continue
        track_id = f"damage_track_{len(tracks) + 1:06d}"
        det["damage_track_id"] = track_id
        tracks.append({
            "track_id": track_id,
            "vehicle_track_id": target_vehicle_id,
            "last_detected_sample_index": sample.sample_index,
            "last_sample_index": sample.sample_index,
            "last_frame_number": sample.frame_number,
            "last_bbox": det.get("bbox"),
            "last_mask": det.get("mask"),
            "best_confidence": -1.0,
            "best_mask": None,
            "best_bbox": None,
            "best_frame_number": sample.frame_number,
            "best_timestamp_seconds": sample.timestamp_seconds,
            "best_frame_path": None,
            "observations": [],
            "assessments": [],
        })

    tracks_by_id = {track["track_id"]: track for track in tracks}
    observed_track_ids: set[str] = set()
    for det in detections:
        track_id = str(det["damage_track_id"])
        track = tracks_by_id[track_id]
        observed_track_ids.add(track_id)
        track["last_detected_sample_index"] = sample.sample_index
        track["last_sample_index"] = sample.sample_index
        track["last_frame_number"] = sample.frame_number
        track["last_bbox"] = det.get("bbox")
        track["last_mask"] = det.get("mask")

        assessment = dict(det.get("association_assessment") or {})
        assessment["damage_track_id"] = track_id
        assessment["vehicle_track_id"] = target_vehicle_id
        assessment["frame_number"] = int(sample.frame_number)
        track["assessments"].append(assessment)

        observation = _track_observation(sample, det, frame_path)
        observation["damage_track_id"] = track_id
        observation["association_status"] = assessment.get("association_status", "unassigned")
        observation["candidate_parts"] = assessment.get("candidate_parts", [])
        track["observations"].append(observation)

        confidence = float(det.get("confidence", 0.0))
        if confidence > track["best_confidence"]:
            track["best_confidence"] = confidence
            track["best_mask"] = det.get("mask")
            track["best_bbox"] = det.get("bbox")
            track["best_frame_number"] = int(sample.frame_number)
            track["best_timestamp_seconds"] = float(sample.timestamp_seconds)
            track["best_frame_path"] = frame_path
            track["best_assessment"] = assessment

    # Count short gaps as target-vehicle observations, but do not keep a track
    # alive indefinitely after the damage region disappears or leaves view.
    for track in tracks:
        track_id = track["track_id"]
        if track_id in observed_track_ids:
            continue
        gap = sample.sample_index - track["last_detected_sample_index"]
        if 1 <= gap <= max_gap:
            last_assessment_frame = (
                track["assessments"][-1].get("frame_number")
                if track.get("assessments") else None
            )
            if last_assessment_frame == sample.frame_number:
                continue
            track["assessments"].append({
                "damage_detection_id": None,
                "damage_track_id": track_id,
                "vehicle_track_id": target_vehicle_id,
                "frame_number": int(sample.frame_number),
                "damage_confidence": 0.0,
                "association_status": "unassigned",
                "candidate_parts": [],
                "review_reasons": ["damage_not_detected_in_intervening_target_frame"],
            })


def _draw_evidence_frame(
    frame: np.ndarray,
    parts: list[dict[str, Any]],
    damages: list[dict[str, Any]],
) -> np.ndarray:
    """Draw the predicted part boundaries and damage masks for review images."""
    out = frame.copy()
    overlay = frame.copy()
    for det in damages:
        mask = det.get("mask")
        if mask is not None and mask.shape == frame.shape[:2]:
            overlay[mask] = (0, 0, 255)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    for det in parts:
        mask = det.get("mask")
        if mask is not None and mask.shape == frame.shape[:2]:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(out, contours, -1, (0, 220, 0), 1)
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 180, 0), 1)
        cv2.putText(out, f'{det["part_name"]} {det["confidence"]:.2f}', (x1, max(16, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1, cv2.LINE_AA)
    for det in damages:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        label = f'damage {det["confidence"]:.2f}'
        assessment = det.get("association_assessment") or {}
        candidate_name = assessment.get("best_candidate_part_name")
        if candidate_name:
            label += f' candidate -> {candidate_name}'
        else:
            label += f' [{assessment.get("association_status", "unassigned")}]'
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, label, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2, cv2.LINE_AA)
    return out


def _relative_path(path: str | Path | None, base_dir: Path) -> str | None:
    if not path:
        return None
    try:
        return Path(path).resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _serialize_damage_track(
    track: dict[str, Any], damage_id: str, base_dir: Path, config: PipelineConfig,
) -> dict[str, Any]:
    mask_path: str | None = None
    if track.get("best_mask") is not None:
        mask_dir = base_dir / "damage_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_file = mask_dir / f"{damage_id}_mask.png"
        mask = (track["best_mask"].astype(np.uint8) * 255)
        if not cv2.imwrite(str(mask_file), mask):
            logger.warning("Could not write mask image: %s", mask_file)
        else:
            mask_path = _relative_path(mask_file, base_dir)
    observations = track.get("observations", [])
    observations_sorted = sorted(observations, key=lambda item: item["timestamp_seconds"])
    seen: set[int] = set()
    evidence_frames: list[dict[str, Any]] = []
    for obs in observations_sorted:
        frame_no = int(obs["frame_number"])
        if frame_no in seen:
            continue
        seen.add(frame_no)
        evidence_frames.append({
            "frame_number": frame_no,
            "timestamp_seconds": float(obs["timestamp_seconds"]),
            "damage_confidence": round(float(obs["confidence"]), 4),
            "frame_path": _relative_path(obs.get("frame_path"), base_dir),
        })
    best_conf = float(track["best_confidence"])
    association = track.get("best_association_status", "unassigned")
    repeat_count = len(evidence_frames)
    reasons: list[str] = []
    if repeat_count < config.min_repeat_observations:
        reasons.append("observed_in_too_few_sampled_frames")
    if best_conf < config.review_confidence:
        reasons.append("model_score_below_review_threshold")
    if association == "ambiguous":
        reasons.append("overlaps_multiple_candidate_parts")
    elif association != "matched":
        reasons.append("could_not_confidently_assign_to_a_part")
    candidates = track.get("best_candidate_parts", [])
    assigned_candidate = next(
        (c for c in candidates if c.get("part_name") == track.get("part_name")),
        candidates[0] if candidates else None,
    )
    if (
        association == "matched"
        and assigned_candidate is not None
        and float(assigned_candidate.get("part_coverage", 0.0))
        >= config.max_damage_part_coverage_auto_confirm
    ):
        # Broad masks are retained for review rather than discarded. This catches
        # cases such as a reflective windshield being painted almost entirely red.
        reasons.append("damage_mask_covers_large_fraction_of_part")
    status = "detected" if not reasons else "uncertain"
    return {
        "damage_id": damage_id,
        "damage_type": "unknown",
        "model_label": "damage",
        "damage_confidence": round(best_conf, 4),
        "confidence_interpretation": "raw model score; not a calibrated probability",
        "damage_status": status,
        "association_status": association,
        "part_confidence_at_representative_frame": (
            round(float(track["best_part_confidence"]), 4)
            if track.get("best_part_confidence") is not None else None
        ),
        "bbox_xyxy_original_frame": [round(float(v), 2) for v in track.get("best_bbox", [])],
        "first_seen_timestamp_seconds": min((o["timestamp_seconds"] for o in observations_sorted), default=None),
        "last_seen_timestamp_seconds": max((o["timestamp_seconds"] for o in observations_sorted), default=None),
        "representative_frame": int(track["best_frame_number"]),
        "representative_timestamp_seconds": round(float(track["best_timestamp_seconds"]), 3),
        "representative_frame_path": _relative_path(track.get("best_frame_path"), base_dir),
        "mask_available": mask_path is not None,
        "mask_path": mask_path,
        "temporal_support_frames": repeat_count,
        "evidence_frames": evidence_frames,
        "candidate_parts": [
            {
                "part_name": c["part_name"],
                "damage_coverage": round(float(c["damage_coverage"]), 4),
                "part_coverage": round(float(c.get("part_coverage", 0.0)), 4),
                "part_confidence": round(float(c["part_confidence"]), 4),
                "method": c["method"],
            }
            for c in candidates
        ],
        "review_reasons": reasons,
    }


# Parts that the trained taxonomy represents as one component per vehicle.
# These can be rejoined across distant short-term track fragments by class;
# repeated classes such as wheel, front_door, front_fender, or quarter_panel are
# deliberately excluded because a vehicle can contain multiple such instances.
_SINGLETON_PART_CLASSES = {
    "front_bumper", "back_bumper", "hood", "front_glass", "back_glass",
    "trunk", "tailgate", "left_mirror", "right_mirror",
    "front_left_door", "front_right_door", "back_left_door", "back_right_door",
    "front_left_light", "front_right_light", "back_left_light", "back_right_light",
}

def _canonical_part_instance_id(
    part_name: str,
    track_id: str,
    vehicle_track_id: str | None = None,
) -> str:
    """Create a video/vehicle-scoped physical part ID.

    Singleton taxonomy classes can be rejoined across short track fragments.
    Repeated classes retain their short-term track ID so two wheels/doors are not
    collapsed just because their class names match. This is not cross-view re-ID.
    """
    if vehicle_track_id:
        if part_name in _SINGLETON_PART_CLASSES:
            return f"{vehicle_track_id}:part:{part_name}"
        return f"{vehicle_track_id}:part:{track_id}"
    if part_name in _SINGLETON_PART_CLASSES:
        return f"part_instance_{part_name}"
    return track_id


def _serialize_v21_damage_track(
    track: dict[str, Any],
    final_association: dict[str, Any],
    damage_id: str,
    base_dir: Path,
) -> dict[str, Any]:
    """Serialize one independently tracked damage region and its V2.1 outcome."""
    mask_path: str | None = None
    if track.get("best_mask") is not None:
        mask_dir = base_dir / "damage_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_file = mask_dir / f"{damage_id}_mask.png"
        mask = np.asarray(track["best_mask"], dtype=np.uint8) * 255
        if cv2.imwrite(str(mask_file), mask):
            mask_path = _relative_path(mask_file, base_dir)
        else:
            logger.warning("Could not write mask image: %s", mask_file)

    observations_sorted = sorted(
        track.get("observations", []),
        key=lambda item: (int(item["frame_number"]), float(item["timestamp_seconds"])),
    )
    evidence_frames: list[dict[str, Any]] = []
    seen_frames: set[int] = set()
    for obs in observations_sorted:
        frame_no = int(obs["frame_number"])
        if frame_no in seen_frames:
            continue
        seen_frames.add(frame_no)
        evidence_frames.append({
            "frame_number": frame_no,
            "timestamp_seconds": float(obs["timestamp_seconds"]),
            "damage_confidence": round(float(obs["confidence"]), 4),
            "frame_path": _relative_path(obs.get("frame_path"), base_dir),
            "association_status": obs.get("association_status", "unassigned"),
        })

    # Select a diagnostic candidate from mask-based suggestions first. This is
    # review information only; it is not exposed as a confirmed part_name.
    candidate_pool: dict[str, dict[str, Any]] = {}
    candidate_counts: dict[str, int] = {}
    for assessment in track.get("assessments", []):
        for candidate in assessment.get("candidate_parts", []) or []:
            name = str(candidate.get("part_name") or "").strip()
            if not name or name.lower() == "unknown":
                continue
            instance_id = str(candidate.get("part_instance_id") or "").strip()
            key = instance_id or f"name:{name}"
            candidate_counts[key] = candidate_counts.get(key, 0) + 1
            previous = candidate_pool.get(key)
            rank = (
                1 if candidate.get("method") == "mask_overlap" else 0,
                float(candidate.get("damage_coverage", 0.0)),
                float(candidate.get("part_confidence", 0.0)),
            )
            previous_rank = (
                (1 if previous.get("method") == "mask_overlap" else 0,
                 float(previous.get("damage_coverage", 0.0)),
                 float(previous.get("part_confidence", 0.0)))
                if previous else (-1, -1.0, -1.0)
            )
            if previous is None or rank > previous_rank:
                candidate_pool[key] = candidate

    selected_candidates = sorted(
        candidate_pool.items(),
        key=lambda item: (
            candidate_counts.get(item[0], 0),
            1 if item[1].get("method") == "mask_overlap" else 0,
            float(item[1].get("damage_coverage", 0.0)),
        ),
        reverse=True,
    )
    serialized_candidates = [
        {
            "candidate_part_name": str(candidate.get("part_name") or "unknown"),
            "candidate_part_instance_id": candidate.get("part_instance_id") or None,
            "damage_coverage": round(float(candidate.get("damage_coverage", 0.0)), 4),
            "part_coverage": round(float(candidate.get("part_coverage", 0.0)), 4),
            "part_confidence": round(float(candidate.get("part_confidence", 0.0)), 4),
            "method": candidate.get("method", "unknown"),
            "observations_with_candidate": candidate_counts.get(
                str(candidate.get("part_instance_id") or "").strip()
                or f"name:{candidate.get('part_name')}", 0
            ),
        }
        for _key, candidate in selected_candidates[:5]
    ]

    final_status = final_association.get("association_status", "manual_review")
    is_confirmed = (
        final_status == "matched"
        and final_association.get("damage_status") == "detected"
        and bool(final_association.get("part_instance_id"))
        and bool(final_association.get("part_name"))
    )
    candidate_names = sorted({
        str(item["candidate_part_name"])
        for item in serialized_candidates
        if item.get("candidate_part_name") and item["candidate_part_name"] != "unknown"
    })
    review_reasons = sorted(set(
        (final_association.get("review_reasons") or [])
        + (final_association.get("observation_warnings") or [])
    ))
    best_confidence = max(0.0, float(track.get("best_confidence", 0.0)))

    region = {
        "damage_id": damage_id,
        "damage_track_id": track.get("track_id"),
        "vehicle_track_id": track.get("vehicle_track_id"),
        "damage_type": "unknown",
        "model_label": "damage",
        "damage_confidence": round(best_confidence, 4),
        "confidence_interpretation": "raw model score; not a calibrated probability",
        "damage_status": "detected" if is_confirmed else "uncertain",
        "association_status": "matched" if is_confirmed else final_status,
        # A final part_name exists only after V2.1 confirmation.
        "part_name": final_association.get("part_name") if is_confirmed else None,
        "part_instance_id": final_association.get("part_instance_id") if is_confirmed else None,
        "candidate_part_names": candidate_names,
        "candidate_parts": serialized_candidates,
        "association_consistency": round(
            float(final_association.get("association_consistency", 0.0)), 4
        ),
        "target_observation_count": int(final_association.get("target_observation_count", 0)),
        "qualified_observation_count": int(final_association.get("qualified_observation_count", 0)),
        "association_votes": final_association.get("association_votes", {}),
        "bbox_xyxy_original_frame": [
            round(float(v), 2) for v in (track.get("best_bbox") or [])
        ],
        "first_seen_timestamp_seconds": min(
            (float(o["timestamp_seconds"]) for o in observations_sorted), default=None
        ),
        "last_seen_timestamp_seconds": max(
            (float(o["timestamp_seconds"]) for o in observations_sorted), default=None
        ),
        "representative_frame": int(track.get("best_frame_number", 0)),
        "representative_timestamp_seconds": round(
            float(track.get("best_timestamp_seconds", 0.0)), 3
        ),
        "representative_frame_path": _relative_path(track.get("best_frame_path"), base_dir),
        "mask_available": mask_path is not None,
        "mask_path": mask_path,
        "temporal_support_frames": len(evidence_frames),
        "evidence_frames": evidence_frames,
        "review_reasons": review_reasons,
    }
    return region


def aggregate_video_results(
    part_tracks: list[dict[str, Any]],
    damage_tracks: list[dict[str, Any]],
    *,
    config: PipelineConfig,
    base_dir: str | Path,
    video_id: str,
    video_filename: str,
    video_metadata: VideoMetadata,
    processed_frames: int,
    target_vehicle_id: str,
    association_config: AssociationConfig,
) -> dict[str, Any]:
    """Build the V2.1 report from independent damage tracks."""
    base = Path(base_dir)
    confirmed_by_instance: dict[str, dict[str, Any]] = {}
    review_candidates: list[dict[str, Any]] = []
    unassigned_regions: list[dict[str, Any]] = []
    excluded_detections: list[dict[str, Any]] = []
    all_regions: list[dict[str, Any]] = []

    ordered_tracks = sorted(
        damage_tracks,
        key=lambda item: (
            item.get("observations", [{}])[0].get("timestamp_seconds", 0.0)
            if item.get("observations") else 0.0
        ),
    )
    for index, track in enumerate(ordered_tracks, start=1):
        assessments = track.get("assessments", [])
        final_association = confirm_damage_track(
            assessments,
            target_vehicle_id=target_vehicle_id,
            config=association_config,
        )
        region = _serialize_v21_damage_track(
            track,
            final_association,
            f"damage_{index:04d}",
            base,
        )
        all_regions.append(region)

        if region["association_status"] == "excluded_not_target_vehicle":
            excluded_detections.append(region)
            continue

        if region["damage_status"] == "detected":
            part_instance_id = str(region["part_instance_id"])
            part_name = str(region["part_name"])
            group = confirmed_by_instance.setdefault(part_instance_id, {
                "part_instance_id": part_instance_id,
                "part_name": part_name,
                "damage_status": "detected",
                "damage_regions": [],
                "source_damage_track_ids": [],
            })
            if group["part_name"] != part_name:
                raise ValueError(
                    f"part_instance_id {part_instance_id!r} has conflicting part names"
                )
            group["damage_regions"].append(region)
            group["source_damage_track_ids"].append(region["damage_track_id"])
            # The stable part instance ID is the association key. Per-frame
            # detection IDs are intentionally not promoted to long-lived IDs.
            continue

        candidates = region.get("candidate_parts") or []
        best_candidate = candidates[0] if candidates else None
        if best_candidate or region.get("candidate_part_names"):
            review_candidates.append({
                "damage_id": region["damage_id"],
                "damage_track_id": region["damage_track_id"],
                "candidate_part_name": (
                    best_candidate.get("candidate_part_name") if best_candidate else
                    region["candidate_part_names"][0]
                ),
                "candidate_part_instance_id": (
                    best_candidate.get("candidate_part_instance_id") if best_candidate else None
                ),
                "candidate_part_names": region.get("candidate_part_names", []),
                "damage_status": "uncertain",
                "association_status": region["association_status"],
                "association_consistency": region["association_consistency"],
                "review_reasons": region["review_reasons"],
                "damage_region": region,
            })
        else:
            unassigned_regions.append(region)

    confirmed_parts = sorted(
        confirmed_by_instance.values(), key=lambda item: (item["part_name"], item["part_instance_id"])
    )
    counts = summarize_part_instances(confirmed_parts, review_candidates)
    # Preserve a deprecated alias for older consumers, but it contains only
    # confirmed parts. Review candidates have their own explicit collection.
    confirmed_regions = [region for region in all_regions if region["damage_status"] == "detected"]
    review_region_count = sum(
        1 for region in all_regions
        if region["damage_status"] != "detected"
        and region["association_status"] != "excluded_not_target_vehicle"
    )

    report = {
        "schema_version": "2.1",
        "video_id": video_id,
        "processing_status": "completed",
        "vehicle_identity": {
            "mode": "single_target_vehicle_per_video_assumption",
            "target_vehicle_id": target_vehicle_id,
            "note": (
                "All model detections are scoped to the one inspected vehicle supplied by the user. "
                "This does not detect or reject incidental background vehicles; footage must be framed "
                "so the inspected vehicle is the only vehicle being analyzed."
            ),
        },
        "video": {
            "filename": video_filename,
            "fps": round(video_metadata.fps, 3),
            "fps_was_fallback": video_metadata.fps_was_fallback,
            "duration_seconds": round(video_metadata.duration_seconds, 3),
            "total_frames": video_metadata.frame_count,
            "width": video_metadata.width,
            "height": video_metadata.height,
            "coordinate_system": "pixel coordinates in the original decoded frame; frame numbers are zero-based",
            "sampling_fps_target": config.sample_fps,
            "processed_frames": processed_frames,
        },
        "confidence_note": (
            "Confidence values are raw Ultralytics model scores, not calibrated probabilities. "
            "V2.1 automatic association requires mask evidence, confidence/coverage gates, stable IDs, "
            "distinct-frame support, and temporal consistency. Pilot thresholds are not production calibrated."
        ),
        "summary": {
            "confirmed_part_instances": counts["confirmed_part_instances"],
            "review_part_instances": counts["review_part_instances"],
            "confirmed_parts_by_name": counts["confirmed_parts_by_name"],
            "review_candidates_by_name": counts["review_candidates_by_name"],
            "total_damage_regions": len(all_regions),
            "confirmed_damage_regions": len(confirmed_regions),
            "uncertain_damage_regions": review_region_count,
            "unassigned_damage_regions": len(unassigned_regions),
            "excluded_detections": len(excluded_detections),
        },
        "confirmed_damaged_parts": confirmed_parts,
        "review_candidates": review_candidates,
        "unassigned_damage_regions": unassigned_regions,
        "excluded_detections": excluded_detections,
        # Backward-compatible field: confirmed parts only; frontend update is a
        # separate step and should read the new collections directly.
        "damaged_parts": confirmed_parts,
        "review_required": [
            {
                "damage_id": item["damage_id"],
                "candidate_part_names": item.get("candidate_part_names", []),
                "reasons": item.get("review_reasons", []),
            }
            for item in review_candidates
        ] + [
            {
                "damage_id": item["damage_id"],
                "candidate_part_names": [],
                "reasons": item.get("review_reasons", []),
            }
            for item in unassigned_regions
        ],
        "tracking_note": (
            "Part observations use short-term same-class mask tracking. Singleton taxonomy classes are "
            "vehicle-scoped; repeated classes retain distinct short-term track identities. Damage tracking "
            "is independent of part-track IDs. Brief missed detections inside the configured track gap add "
            "unassigned target observations to the temporal denominator. Cross-view part/damage re-identification "
            "and explicit background-vehicle rejection are not implemented."
        ),
    }
    return report


def generate_damage_report(
    part_tracks: list[dict[str, Any]],
    damage_tracks: list[dict[str, Any]],
    *,
    config: PipelineConfig,
    output_dir: str | Path,
    video_id: str,
    video_filename: str,
    video_metadata: VideoMetadata,
    processed_frames: int,
    target_vehicle_id: str,
    association_config: AssociationConfig,
) -> dict[str, Any]:
    report = aggregate_video_results(
        part_tracks,
        damage_tracks,
        config=config,
        base_dir=output_dir,
        video_id=video_id,
        video_filename=video_filename,
        video_metadata=video_metadata,
        processed_frames=processed_frames,
        target_vehicle_id=target_vehicle_id,
        association_config=association_config,
    )
    output_path = Path(output_dir) / "damage_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    report["report_path"] = "damage_report.json"
    # Keep the saved report in sync with the response's report_path field.
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    return report


class VideoDamagePipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.config.validate()
        self.parts_model: YOLO | None = None
        self.damage_model: YOLO | None = None
        self.device: str | int = "cpu"

    def load_models(self) -> None:
        if not self.config.parts_weights.is_file():
            raise FileNotFoundError(
                f"Car-parts checkpoint missing: {self.config.parts_weights}. Copy the notebook's FINAL_best.pt there."
            )
        if not self.config.damage_weights.is_file():
            raise FileNotFoundError(
                f"Damage checkpoint missing: {self.config.damage_weights}. Copy best_damage_model.pt or a validated checkpoints/best.pt there."
            )
        if self.config.device.lower() == "auto":
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = int(self.config.device) if self.config.device.isdigit() else self.config.device
        self.parts_model = YOLO(str(self.config.parts_weights))
        self.damage_model = YOLO(str(self.config.damage_weights))
        if getattr(self.parts_model, "task", None) != "segment":
            raise ValueError(f"Expected a segment model for parts, got task={getattr(self.parts_model, 'task', None)}")
        if getattr(self.damage_model, "task", None) != "segment":
            raise ValueError(f"Expected a segment model for damage, got task={getattr(self.damage_model, 'task', None)}")
        logger.info("Loaded parts model (%d classes) and damage model (%d classes) on %s",
                    len(self.parts_model.names), len(self.damage_model.names), self.device)

    def analyze_video(
        self,
        video_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        video_id: str | None = None,
    ) -> dict[str, Any]:
        if self.parts_model is None or self.damage_model is None:
            self.load_models()
        assert self.parts_model is not None and self.damage_model is not None
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        video_id_value = video_id or video_path.stem
        # The user confirmed one inspected vehicle per video. This stable ID is
        # scoped to this video/job; it is not a visual vehicle detector/tracker.
        target_vehicle_id = f"vehicle_{video_id_value}"
        association_config = AssociationConfig(
            candidate_min_damage_coverage=self.config.min_damage_coverage,
            auto_min_damage_coverage=self.config.auto_min_damage_coverage,
            min_damage_confidence_auto=self.config.min_damage_confidence_auto,
            min_part_confidence_auto=self.config.min_part_confidence_auto,
            max_part_coverage_auto=self.config.max_damage_part_coverage_auto,
            ambiguity_margin=self.config.ambiguity_margin,
            min_repeat_observations=max(2, self.config.min_repeat_observations),
            min_association_consistency=self.config.min_association_consistency,
            duplicate_mask_iou=self.config.duplicate_mask_iou,
        )
        association_config.validate()

        out_dir = Path(output_dir or self.config.output_dir / video_id_value)
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata = inspect_video(video_path)
        part_tracks: list[dict[str, Any]] = []
        damage_tracks: list[dict[str, Any]] = []
        processed_frames = 0
        evidence_dir = out_dir / "evidence_frames"
        evidence_dir.mkdir(parents=True, exist_ok=True)

        for sample in extract_video_frames(
            video_path,
            self.config.sample_fps,
            source_fps=metadata.fps,
        ):
            processed_frames += 1
            frame = sample.frame
            parts = predict_car_parts(
                self.parts_model,
                frame,
                conf=self.config.parts_conf,
                iou=self.config.parts_iou,
                image_size=self.config.image_size,
                device=self.device,
            )
            assign_part_tracks(
                parts,
                part_tracks,
                sample,
                min_iou=self.config.part_track_iou,
                max_gap=self.config.max_track_gap,
            )

            # Add the video-scoped target identity and a stable physical part ID
            # after short-term tracking has assigned the local track IDs.
            for part in parts:
                part["vehicle_track_id"] = target_vehicle_id
                part["part_instance_id"] = _canonical_part_instance_id(
                    str(part.get("part_name") or "unknown"),
                    str(part.get("track_id") or part.get("detection_id") or "unknown"),
                    target_vehicle_id,
                )

            damages = predict_damage(
                self.damage_model,
                frame,
                conf=self.config.damage_conf,
                iou=self.config.damage_iou,
                image_size=self.config.image_size,
                device=self.device,
            )
            for damage in damages:
                damage["vehicle_track_id"] = target_vehicle_id
                damage["frame_number"] = int(sample.frame_number)
                damage["association_assessment"] = score_damage_against_parts(
                    damage,
                    parts,
                    target_vehicle_id=target_vehicle_id,
                    config=association_config,
                )

            frame_path: str | None = None
            if damages:
                annotated = _draw_evidence_frame(frame, parts, damages)
                filename = (
                    f"frame_{sample.frame_number:06d}_"
                    f"t_{sample.timestamp_seconds:08.3f}s.jpg"
                )
                file_path = evidence_dir / filename
                if cv2.imwrite(str(file_path), annotated):
                    frame_path = str(file_path)

            _add_damage_tracks(
                damages,
                damage_tracks,
                sample,
                frame_path,
                target_vehicle_id=target_vehicle_id,
                min_iou=self.config.damage_track_iou,
                max_gap=self.config.max_track_gap,
            )
            if processed_frames % 20 == 0:
                logger.info(
                    "Processed %d sampled frames (%.2f s video time)",
                    processed_frames,
                    sample.timestamp_seconds,
                )

        report = generate_damage_report(
            part_tracks,
            damage_tracks,
            config=self.config,
            output_dir=out_dir,
            video_id=video_id_value,
            video_filename=video_path.name,
            video_metadata=metadata,
            processed_frames=processed_frames,
            target_vehicle_id=target_vehicle_id,
            association_config=association_config,
        )
        report["models"] = {
            "car_parts": {
                "task": self.parts_model.task,
                "class_count": len(self.parts_model.names),
                "weights_file": self.config.parts_weights.name,
            },
            "car_damage": {
                "task": self.damage_model.task,
                "class_count": len(self.damage_model.names),
                "class_names": list(_names_dict(self.damage_model).values()),
                "weights_file": self.config.damage_weights.name,
            },
        }
        with (out_dir / "damage_report.json").open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
        return report
