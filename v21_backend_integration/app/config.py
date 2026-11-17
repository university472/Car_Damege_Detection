"""Configuration for the car damage video pipeline and V2.1 association."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_path(name: str, default: str) -> Path:
    value = os.getenv(name, default).strip()
    return Path(value).expanduser()


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, str(default)).strip()
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number; received {value!r}") from exc


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip()
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; received {value!r}") from exc


@dataclass(frozen=True)
class PipelineConfig:
    """Settings used by app/pipeline.py. Thresholds are pilot defaults."""

    parts_weights: Path = _env_path("PARTS_WEIGHTS", "models/car_parts.pt")
    damage_weights: Path = _env_path("DAMAGE_WEIGHTS", "models/car_damage.pt")
    output_dir: Path = _env_path("OUTPUT_DIR", "data/artifacts")
    upload_dir: Path = _env_path("UPLOAD_DIR", "data/uploads")
    device: str = os.getenv("DEVICE", "auto").strip()

    sample_fps: float = _env_float("SAMPLE_FPS", 2.0)
    image_size: int = _env_int("IMAGE_SIZE", 640)
    parts_conf: float = _env_float("PARTS_CONF", 0.25)
    damage_conf: float = _env_float("DAMAGE_CONF", 0.10)
    parts_iou: float = _env_float("PARTS_IOU", 0.60)
    damage_iou: float = _env_float("DAMAGE_IOU", 0.50)

    # V2.1 candidate and temporal-association settings.
    min_damage_coverage: float = _env_float("MIN_DAMAGE_COVERAGE", 0.25)
    auto_min_damage_coverage: float = _env_float("AUTO_MIN_DAMAGE_COVERAGE", 0.40)
    min_damage_confidence_auto: float = _env_float("MIN_DAMAGE_CONFIDENCE_AUTO", 0.25)
    min_part_confidence_auto: float = _env_float("MIN_PART_CONFIDENCE_AUTO", 0.50)
    min_association_consistency: float = _env_float("MIN_ASSOCIATION_CONSISTENCY", 0.67)
    duplicate_mask_iou: float = _env_float("DUPLICATE_MASK_IOU", 0.90)
    ambiguity_margin: float = _env_float("AMBIGUITY_MARGIN", 0.10)

    # Short-term trackers are not cross-view re-identification systems.
    part_track_iou: float = _env_float("PART_TRACK_IOU", 0.10)
    damage_track_iou: float = _env_float("DAMAGE_TRACK_IOU", 0.10)
    max_track_gap: int = _env_int("MAX_TRACK_GAP", 2)
    min_repeat_observations: int = _env_int("MIN_REPEAT_OBSERVATIONS", 2)
    review_confidence: float = _env_float("REVIEW_CONFIDENCE", 0.25)
    max_damage_part_coverage_auto_confirm: float = _env_float(
        "MAX_DAMAGE_PART_COVERAGE_AUTO_CONFIRM", 0.70
    )
    max_upload_mb: int = _env_int("MAX_UPLOAD_MB", 500)

    def validate(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError("SAMPLE_FPS must be greater than 0")
        if self.image_size <= 0:
            raise ValueError("IMAGE_SIZE must be greater than 0")
        if self.max_track_gap < 0:
            raise ValueError("MAX_TRACK_GAP must be 0 or greater")
        if self.min_repeat_observations < 2:
            raise ValueError("MIN_REPEAT_OBSERVATIONS must be at least 2 for V2.1")
        if self.max_upload_mb < 1:
            raise ValueError("MAX_UPLOAD_MB must be at least 1")

        threshold_fields = (
            "parts_conf", "damage_conf", "parts_iou", "damage_iou",
            "min_damage_coverage", "auto_min_damage_coverage",
            "min_damage_confidence_auto", "min_part_confidence_auto",
            "min_association_consistency", "duplicate_mask_iou",
            "ambiguity_margin", "part_track_iou", "damage_track_iou",
            "review_confidence", "max_damage_part_coverage_auto_confirm",
        )
        for field_name in threshold_fields:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name.upper()} must be between 0 and 1; received {value}"
                )
        if self.auto_min_damage_coverage < self.min_damage_coverage:
            raise ValueError(
                "AUTO_MIN_DAMAGE_COVERAGE must be >= MIN_DAMAGE_COVERAGE"
            )
