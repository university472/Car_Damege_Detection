from __future__ import annotations
import argparse
from pathlib import Path
import sys

import cv2
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the two car-damage model checkpoints load.")
    parser.add_argument("--parts", default="models/car_parts.pt")
    parser.add_argument("--damage", default="models/car_damage.pt")
    parser.add_argument("--image", help="Optional test image path for one-frame inference")
    args = parser.parse_args()

    for path in (Path(args.parts), Path(args.damage)):
        if not path.is_file():
            print(f"MISSING: {path}", file=sys.stderr)
            return 2

    parts = YOLO(args.parts)
    damage = YOLO(args.damage)
    print("Parts model:", args.parts, "task=", parts.task, "classes=", parts.names)
    print("Damage model:", args.damage, "task=", damage.task, "classes=", damage.names)
    if parts.task != "segment" or damage.task != "segment":
        print("ERROR: expected both checkpoints to be segmentation models", file=sys.stderr)
        return 3
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"Cannot read test image: {args.image}", file=sys.stderr)
            return 4
        for label, model in (("parts", parts), ("damage", damage)):
            result = model.predict(image, imgsz=640, conf=0.10 if label == "damage" else 0.25,
                                   retina_masks=True, verbose=False)[0]
            boxes = 0 if result.boxes is None else len(result.boxes)
            masks = 0 if result.masks is None else len(result.masks.data)
            print(f"{label}: boxes={boxes}, masks={masks}, original_shape={image.shape[:2]}")
    print("Checkpoint smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
