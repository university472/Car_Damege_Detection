from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from threading import Lock
from uuid import uuid4
import logging
import os
import re
import shutil
import zipfile

import cv2
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import PipelineConfig
from .pipeline import VideoDamagePipeline, inspect_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
_base_config = PipelineConfig()


def _absolute_from_project(path: Path) -> Path:
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


CONFIG = replace(
    _base_config,
    parts_weights=_absolute_from_project(_base_config.parts_weights),
    damage_weights=_absolute_from_project(_base_config.damage_weights),
    output_dir=_absolute_from_project(_base_config.output_dir),
    upload_dir=_absolute_from_project(_base_config.upload_dir),
)
CONFIG.validate()
CONFIG.upload_dir.mkdir(parents=True, exist_ok=True)
CONFIG.output_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Video Car Damage Analysis API", version="1.0.0")
_default_origins = "http://127.0.0.1:5173,http://localhost:5173"
_allowed_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", _default_origins).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.mount("/artifacts", StaticFiles(directory=str(CONFIG.output_dir)), name="artifacts")

# Prototype storage. Move job metadata to a database and use a durable queue for multi-worker production.
jobs: dict[str, dict] = {}
_pipeline: VideoDamagePipeline | None = None
_pipeline_lock = Lock()
_inference_lock = Lock()  # One model pair / one GPU: run only one job at a time.
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _get_pipeline() -> VideoDamagePipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = VideoDamagePipeline(CONFIG)
            _pipeline.load_models()
    return _pipeline


def _run_job(job_id: str, video_path: str) -> None:
    jobs[job_id]["status"] = "processing"
    try:
        artifact_dir = CONFIG.output_dir / job_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with _inference_lock:
            pipeline = _get_pipeline()
            report = pipeline.analyze_video(video_path, output_dir=artifact_dir, video_id=job_id)
        jobs[job_id].update({
            "status": "completed",
            "report": report,
            "report_url": f"/artifacts/{job_id}/damage_report.json",
            "error": None,
        })
    except Exception as exc:
        logger.exception("Video analysis failed for job %s", job_id)
        jobs[job_id].update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        })
    finally:
        # Keep uploaded video until results are retrieved; user can configure a cleanup policy.
        pass


@app.get("/health")
def health() -> dict:
    parts_ok = CONFIG.parts_weights.is_file()
    damage_ok = CONFIG.damage_weights.is_file()
    return {
        "status": "ok" if parts_ok and damage_ok else "missing_model_checkpoints",
        "car_parts_checkpoint_found": parts_ok,
        "damage_checkpoint_found": damage_ok,
        "device_setting": CONFIG.device,
    }


@app.post("/api/jobs", status_code=202)
async def create_analysis_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    filename = Path(video.filename or "upload.mp4").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported video extension: {suffix or '(none)'}")

    job_id = uuid4().hex
    upload_job_dir = CONFIG.upload_dir / job_id
    upload_job_dir.mkdir(parents=True, exist_ok=False)
    destination = upload_job_dir / f"input{suffix}"
    max_bytes = CONFIG.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with destination.open("wb") as output:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(status_code=413, detail=f"Video exceeds the {CONFIG.max_upload_mb} MB limit")
                output.write(chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        # Validate decode before accepting an expensive model job.
        inspect_video(destination)
    except HTTPException:
        shutil.rmtree(upload_job_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(upload_job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Uploaded file is not a readable video: {exc}") from exc
    finally:
        await video.close()

    jobs[job_id] = {
        "job_id": job_id,
        "filename": filename,
        "status": "queued",
        "report": None,
        "report_url": None,
        "error": None,
    }
    background_tasks.add_task(_run_job, job_id, str(destination))
    return JSONResponse(status_code=202, content={
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/api/jobs/{job_id}",
        "report_url": f"/api/jobs/{job_id}/report",
        "artifacts_base_url": f"/artifacts/{job_id}/",
    })


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found in this API process")
    return {
        "job_id": job_id,
        "filename": job["filename"],
        "status": job["status"],
        "error": job.get("error"),
        "report_url": job.get("report_url"),
    }


@app.get("/api/jobs/{job_id}/report")
def get_job_report(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found in this API process")
    if job["status"] == "failed":
        raise HTTPException(status_code=409, detail=job.get("error") or "Analysis failed")
    if job["status"] != "completed" or job.get("report") is None:
        raise HTTPException(status_code=409, detail=f"Report is not ready; current status is {job['status']}")
    return job["report"]


def _completed_job_or_error(job_id: str) -> dict:
    """Return a completed job, with consistent API errors for evidence endpoints."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found in this API process")
    if job.get("status") != "completed":
        if job.get("status") == "failed":
            raise HTTPException(status_code=409, detail=job.get("error") or "Analysis failed")
        raise HTTPException(status_code=409, detail=f"Evidence is not ready; current status is {job.get('status', 'unknown')}")
    return job


def _list_evidence_image_paths(job_id: str) -> list[Path]:
    """List image files in this completed job's evidence_frames folder."""
    _completed_job_or_error(job_id)
    evidence_dir = (CONFIG.output_dir / job_id / "evidence_frames").resolve()
    # Job IDs are generated by the API, but keep the directory boundary explicit.
    if not evidence_dir.is_relative_to(CONFIG.output_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid job evidence path")
    if not evidence_dir.is_dir():
        return []
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        (path for path in evidence_dir.iterdir() if path.is_file() and path.suffix.lower() in allowed_extensions),
        key=lambda path: path.name.lower(),
    )


@app.get("/api/jobs/{job_id}/evidence-frames")
def get_evidence_frames(job_id: str) -> dict:
    """List generated annotated evidence images for rendering in the frontend gallery."""
    image_paths = _list_evidence_image_paths(job_id)
    frames = []
    filename_pattern = re.compile(r"frame_(\d+)_t_([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)
    for path in image_paths:
        match = filename_pattern.match(path.stem)
        frames.append({
            "filename": path.name,
            "url": f"/artifacts/{job_id}/evidence_frames/{path.name}",
            "frame_number": int(match.group(1)) if match else None,
            "timestamp_seconds": float(match.group(2)) if match else None,
            "size_bytes": path.stat().st_size,
        })
    return {"job_id": job_id, "count": len(frames), "evidence_frames": frames}


@app.get("/api/jobs/{job_id}/evidence-frames/download")
def download_evidence_frames(job_id: str) -> FileResponse:
    """Create and download a ZIP containing every image in the evidence_frames folder."""
    image_paths = _list_evidence_image_paths(job_id)
    if not image_paths:
        raise HTTPException(status_code=404, detail="No evidence frame images were generated for this video")

    job_dir = (CONFIG.output_dir / job_id).resolve()
    if not job_dir.is_relative_to(CONFIG.output_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid job evidence path")
    archive_path = job_dir / "evidence_frames.zip"
    try:
        with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for image_path in image_paths:
                archive.write(image_path, arcname=f"evidence_frames/{image_path.name}")
    except OSError as exc:
        logger.exception("Could not package evidence frames for job %s", job_id)
        raise HTTPException(status_code=500, detail="Could not package evidence frame images") from exc

    return FileResponse(
        path=str(archive_path),
        media_type="application/zip",
        filename=f"{job_id}_evidence_frames.zip",
    )


# Serve the built React app (frontend/dist) from the same domain when it exists,
# e.g. in the Railway Docker image. Mounted last so /api, /health and /artifacts win.
_FRONTEND_DIST = ROOT / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
