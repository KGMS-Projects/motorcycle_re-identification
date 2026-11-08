"""FastAPI service: video registry, analysis jobs, evidence results, exports."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import DATA_DIR, SAMPLES_DIR, Params
from .metrics import evaluate
from .pipeline import analyze, cache_dir, export_csv, export_json, parse_start_ts, to_builtin

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("reid.api")

UPLOADS = Path(DATA_DIR) / "uploads"
JOBS_DIR = Path(DATA_DIR) / "jobs"
for d in (UPLOADS, JOBS_DIR, Path(cache_dir())):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Motorcycle Re-ID & Evidence Review")
app.mount("/media/cache", StaticFiles(directory=cache_dir()), name="cache")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

JOBS: dict[str, dict] = {}
RESULTS: dict[str, dict] = {}
LOCK = threading.Lock()  # one heavy analysis at a time


# ------------------------------------------------------------------ videos
def _meta(vid: str) -> dict:
    m = json.loads((UPLOADS / vid / "meta.json").read_text())
    m.setdefault("start_ts", parse_start_ts(m.get("name")))
    return m


def _register(src: Path, name: str) -> dict:
    h = hashlib.sha1()
    with open(src, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    vid = h.hexdigest()[:12]
    d = UPLOADS / vid
    if (d / "meta.json").exists():
        return _meta(vid)
    d.mkdir(parents=True, exist_ok=True)
    proxy = d / "proxy.mp4"
    # Browser-friendly H.264 proxy; detection runs on this file so overlay coordinates match the player.
    cmd = ["ffmpeg", "-y", "-i", str(src), "-an", "-vf", "scale=-2:'min(720,ih)'", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(proxy)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not proxy.exists():
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, f"Could not read this video: {r.stderr[-300:]}")
    cap = cv2.VideoCapture(str(proxy))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = {"id": vid, "name": name, "fps": fps, "width": int(cap.get(3)), "height": int(cap.get(4)),
            "duration": round(n / fps, 1), "url": f"/media/video/{vid}.mp4"}
    cap.release()
    (d / "meta.json").write_text(json.dumps(meta))
    return meta


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/videos")
def list_videos():
    vids = [_meta(p.name) for p in sorted(UPLOADS.iterdir()) if (p / "meta.json").exists()]
    samples = sorted(p.name for p in Path(SAMPLES_DIR).glob("*") if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"))
    return {"videos": vids, "samples": samples}


@app.post("/api/videos")
def upload_video(file: UploadFile = File(...)):
    tmp = UPLOADS / f"_tmp_{uuid.uuid4().hex}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        return _register(tmp, file.filename or "upload")
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/api/videos/sample/{name}")
def use_sample(name: str):
    src = Path(SAMPLES_DIR) / Path(name).name
    if not src.exists():
        raise HTTPException(404, "Sample not found")
    return _register(src, src.name)


@app.get("/media/video/{vid}.mp4")
def video(vid: str, request: Request):
    path = UPLOADS / Path(vid).name / "proxy.mp4"
    if not path.exists():
        raise HTTPException(404)
    size = path.stat().st_size
    rng = request.headers.get("range")
    if not rng:
        return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    start_s, _, end_s = rng.replace("bytes=", "").partition("-")
    start = int(start_s or 0)
    end = min(int(end_s) if end_s else size - 1, size - 1)
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start + 1)
    return Response(data, 206, media_type="video/mp4", headers={
        "Content-Range": f"bytes {start}-{end}/{size}", "Accept-Ranges": "bytes", "Content-Length": str(len(data))})


# ------------------------------------------------------------------ jobs
def _run(job_id: str, body: dict):
    job = JOBS[job_id]

    def progress(stage, frac, msg=""):
        job.update(stage=stage, progress=round(float(frac), 3), message=msg)

    try:
        va, vb = body["video_a"], body["video_b"]
        info = {}
        for k, vid in (("a", va), ("b", vb)):
            m = _meta(vid)
            info[k] = {"id": vid, "name": m["name"], "url": m["url"], "path": str(UPLOADS / vid / "proxy.mp4"), "start_ts": m.get("start_ts")}
        with LOCK:
            result = analyze(job_id, info["a"], info["b"], Params.from_dict(body.get("params")),
                             redetect=bool(body.get("redetect")), progress=progress,
                             ground_truth=body.get("ground_truth"))
        RESULTS[job_id] = result
        (JOBS_DIR / f"{job_id}.json").write_text(json.dumps(result, default=to_builtin))
        job.update(status="done", progress=1.0, stage="done", message="")
    except Exception as e:  # surfaced in the UI
        log.error(traceback.format_exc())
        job.update(status="error", error=str(e))


@app.post("/api/analyze")
def start(body: dict):
    for k in ("video_a", "video_b"):
        if not body.get(k) or not (UPLOADS / body[k] / "meta.json").exists():
            raise HTTPException(400, f"Select a valid video for {k[-1].upper()}")
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"id": job_id, "status": "running", "stage": "queued", "progress": 0.0, "message": "", "started": time.time()}
    threading.Thread(target=_run, args=(job_id, body), daemon=True).start()
    return JOBS[job_id]


def _result(job_id: str) -> dict:
    if job_id not in RESULTS:
        f = JOBS_DIR / f"{job_id}.json"
        if not f.exists():
            raise HTTPException(404, "Result not ready")
        RESULTS[job_id] = json.loads(f.read_text())
    return RESULTS[job_id]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job")
    return JOBS[job_id]


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    return JSONResponse(_result(job_id))


@app.post("/api/jobs/{job_id}/evaluate")
def job_evaluate(job_id: str, body: dict):
    res = _result(job_id)
    gt = body.get("ground_truth") or []
    # ranks/ctx are needed, so evaluate on the stored candidate list (top-5 per track)
    res["metrics"] = evaluate(res["summary_a"], res["candidates"], gt)
    return res["metrics"]


@app.get("/api/jobs/{job_id}/export.json")
def export_j(job_id: str):
    return Response(json.dumps(export_json(_result(job_id)), indent=2), media_type="application/json",
                    headers={"Content-Disposition": f"attachment; filename=reid_{job_id}.json"})


@app.get("/api/jobs/{job_id}/export.csv")
def export_c(job_id: str):
    return Response(export_csv(_result(job_id)), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=reid_{job_id}.csv"})


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()
