# Backend integration: Revised Damage-to-Part Association V2.1

This patch connects the existing two-model FastAPI video pipeline to the standalone V2.1 association module. It does not retrain or replace either YOLO checkpoint.

## What changed

- `VideoDamagePipeline.analyze_video()` now calls `score_damage_against_parts()` for each damage detection and calls `confirm_damage_track()` after the video has been processed.
- Every observation is assigned a video-scoped target ID (`vehicle_<video_id>`) because this inspection workflow assumes one target vehicle per video.
- Part detections receive a vehicle-scoped `part_instance_id` after the existing short-term part tracker assigns `track_id`.
- Damage tracks are now matched by damage geometry independently of `part_track_id`.
- Short gaps of up to `MAX_TRACK_GAP` sampled frames are entered as unassigned target observations for an existing damage track. This prevents a short run of intermittent detections from making association consistency look stronger than it is.
- Valid mask overlap is required for an automatic vote. Bbox-only matching remains review-only.
- Mismatched masks are not resized for association or tracking.
- The report now exposes `confirmed_damaged_parts`, `review_candidates`, `unassigned_damage_regions`, and `excluded_detections` as separate collections.
- The old `damaged_parts` key remains as a compatibility alias containing **confirmed parts only**. It does not contain review candidates.

## Important vehicle assumption

The user confirmed that each inspection video has one vehicle being inspected. The patch creates a stable ID for that video; it does **not** add a visual vehicle detector, a vehicle tracker, or foreground/background vehicle filtering. The whole frame is still processed by both YOLO models. If a second vehicle is visible and the models detect its parts or damage, those detections will also receive the same video-scoped ID. For this first version, frame the video so the inspected car is the only vehicle being analyzed. If background vehicles must be handled robustly, a vehicle detector/tracker and target crop stage are a separate required phase.

## Known tracking limits

- The existing part tracker is a short-term same-class mask-IoU tracker, not cross-view re-identification.
- Singleton taxonomy classes use a video-scoped class ID (for example, the one `front_bumper`). Repeated classes use the short-term part track ID so different wheels/doors are not merged by class name.
- If a repeated physical part is lost and reappears after a large viewpoint change, it may receive a new instance ID and remain a review candidate.
- Damage tracks are independent of part tracks, but use short-term damage-mask IoU. A large viewpoint change can split one real damage region into separate tracks; this is safer than guessing that two regions are the same damage.
- The pilot thresholds are not calibrated probabilities. Real-video validation against manually labelled frames is still required.

## Files to copy into the existing project

From the extracted `v21_backend_integration` folder, copy:

1. `app/pipeline.py` → replace the project's `app/pipeline.py`.
2. `app/config.py` → replace the project's `app/config.py`.
3. `app/association_v2.py` → replace the project's file with the matching V2.1 version in this package (the user already has this file; this ensures the module matches the integration).
4. `tests/test_pipeline_v21_integration.py` → add this new test file.
5. `tests/test_geometry.py` → optionally replace the existing file with this extended version. It preserves the supplied geometry tests and adds checks for vehicle-scoped IDs, mask-dimension mismatch, and bbox-only review behavior.

Do not overwrite `app/api.py`, `frontend/src/App.jsx`, either model checkpoint, `requirements.txt`, or `.env` as part of this backend patch. `api.py` can continue to call `analyze_video()` and return its report.

## Backup first (PowerShell)

Run these from the actual `car_damage_video_pipeline` project root:

```powershell
Copy-Item app\pipeline.py app\pipeline.py.pre_v21.bak
Copy-Item app\config.py app\config.py.pre_v21.bak
Copy-Item app\association_v2.py app\association_v2.py.pre_v21.bak
Copy-Item tests\test_geometry.py tests\test_geometry.py.pre_v21.bak
```

Then copy the files listed above into place. Do not copy the whole `app` or `tests` directory over the project.

## Configuration

No requirements change is needed. The existing dependencies already include NumPy, OpenCV, FastAPI, Ultralytics, and python-dotenv.

The following optional `.env` values are newly supported; omit them to use the defaults:

```dotenv
AUTO_MIN_DAMAGE_COVERAGE=0.40
MIN_DAMAGE_CONFIDENCE_AUTO=0.25
MIN_PART_CONFIDENCE_AUTO=0.50
MIN_ASSOCIATION_CONSISTENCY=0.67
DUPLICATE_MASK_IOU=0.90
```

Existing settings remain supported, including `MIN_DAMAGE_COVERAGE`, `AMBIGUITY_MARGIN`, `MAX_DAMAGE_PART_COVERAGE_AUTO_CONFIRM`, `MIN_REPEAT_OBSERVATIONS`, `PART_TRACK_IOU`, `DAMAGE_TRACK_IOU`, and `MAX_TRACK_GAP`. V2.1 requires `MIN_REPEAT_OBSERVATIONS` to be at least 2.

## Verify after copying

Activate the existing project environment, from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m compileall -q app tests
python -m unittest discover -s tests -v
```

If you copy the extended `test_geometry.py` and the new integration test, the included test set is expected to report 60 passing tests. If you keep your original 8-test geometry file, it will report 55 tests (42 association tests + 8 geometry tests + 5 integration tests).

## Start the backend

```powershell
uvicorn app.api:app --host 127.0.0.1 --port 8000 --env-file .env
```

Check `http://127.0.0.1:8000/health`, then upload a short original vehicle video through the existing frontend. The API job endpoints do not need to change for this backend integration.

## Frontend status

The React component has not been modified in this backend patch. It still reads `report.damaged_parts`, which is now a confirmed-only compatibility alias. It will therefore show confirmed results but will not yet display the new review candidates or the new review counts. The frontend should be updated in the next phase to read `confirmed_damaged_parts`, `review_candidates`, `unassigned_damage_regions`, and their summary counts. Do not interpret the old UI's review count as complete until that frontend update is made.

## Verification scope

The patch was syntax-compiled and its included synthetic/unit suite passed 60/60 tests in the build environment. That suite verifies algorithm integration, independent damage tracking, missed-frame denominator handling, report separation, and the retained geometry behavior. It does not run your actual `.pt` checkpoints or your original video. Real inference, visual tracking quality, and threshold calibration remain to be validated in your Windows project environment.
