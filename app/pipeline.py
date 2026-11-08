"""Detection -> tracking -> track representation -> matching -> review -> export."""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
import hashlib
import heapq
import io
import itertools
import logging
import os
import pickle
import time

import cv2
import numpy as np

from .config import DATA_DIR, Params
from .features import get_embedder, ocr_state, sharpness_score
from .matcher import calibration, rank_all
from .metrics import evaluate
from .reviewer import review_pair
from .stitch import stitch_raw
from .tracks import build_track

log = logging.getLogger("reid.pipeline")
Progress = lambda stage, frac, msg="": None  # noqa: E731


def to_builtin(o):
    """json `default=` hook: NumPy scalars/arrays -> plain Python."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


_TS = re.compile(r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[ T_.\-]?(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})")


def parse_start_ts(name: str | None):
    """Recording start time from a file name such as 20261106_141615.mp4 (None if absent)."""
    m = _TS.search(name or "")
    if not m:
        return None
    try:
        return datetime(*map(int, m.groups())).timestamp()
    except ValueError:
        return None


FEATURE_VERSION = "v4"  # bump when track features change so stale caches are not reused


def cache_dir() -> str:
    d = os.path.join(DATA_DIR, "cache")
    os.makedirs(d, exist_ok=True)
    return d


def _inter(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _area(b) -> float:
    return max(1.0, (b[2] - b[0]) * (b[3] - b[1]))


def detect_tracks(path: str, p: Params, progress=Progress) -> dict:
    """Run YOLO + ByteTrack. Returns raw per-track observations and the best crops per track."""
    from ultralytics import YOLO  # lazy: keeps the rest of the package importable without torch

    model = YOLO(p.yolo_weights)  # fresh instance => fresh tracker state per video
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw: dict[int, dict] = {}
    counter = itertools.count()
    idx, processed, t_start = 0, 0, time.time()
    while True:
        if idx % p.stride:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0, 3],
                          conf=p.det_conf, imgsz=p.imgsz, verbose=False)[0]
        processed += 1
        boxes = res.boxes
        if boxes is not None and boxes.id is not None:
            xyxy, cls = boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)
            ids, conf = boxes.id.cpu().numpy().astype(int), boxes.conf.cpu().numpy()
            polys = res.masks.xy if getattr(res, "masks", None) is not None else None
            motos = [(ids[k], xyxy[k], float(conf[k]), k) for k in range(len(cls)) if cls[k] == 3]
            persons = [xyxy[k] for k in range(len(cls)) if cls[k] == 0]
            for tid, bx, cf, k in motos:
                x1, y1, x2, y2 = (int(max(0, bx[0])), int(max(0, bx[1])), int(min(W, bx[2])), int(min(H, bx[3])))
                if min(x2 - x1, y2 - y1) < p.min_box_side:
                    continue
                occl = float(max([_inter(bx, o[1]) / _area(bx) for o in motos if o[0] != tid] or [0.0]))
                trunc = float(x1 <= 2 or y1 <= 2 or x2 >= W - 2 or y2 >= H - 2)
                riders = sum(1 for pb in persons if _inter(pb, bx) / _area(pb) > 0.3)
                crop = frame[y1:y2, x1:x2]
                mask = None
                if polys is not None and k < len(polys) and len(polys[k]) >= 3:
                    mc = np.zeros((y2 - y1, x2 - x1), np.uint8)
                    cv2.fillPoly(mc, [np.round(polys[k] - np.array([x1, y1])).astype(np.int32)], 255)
                    mask = mc if mc.mean() > 25 else None  # ignore masks covering < 10% of the box
                score = float(cf * min(1.0, min(x2 - x1, y2 - y1) / 100) * (0.3 + 0.7 * sharpness_score(crop))
                         * (1 - 0.7 * min(1.0, occl)) * (1 - 0.5 * trunc))
                r = raw.setdefault(int(tid), {"obs": [], "heap": []})
                r["obs"].append([idx / fps, float(x1), float(y1), float(x2), float(y2), cf, riders, float(min(1.0, occl)), trunc])
                small = crop if max(crop.shape[:2]) <= 384 else cv2.resize(
                    crop, None, fx=384 / max(crop.shape[:2]), fy=384 / max(crop.shape[:2]))
                small_mask = None if mask is None else cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
                item = (score, next(counter), small.copy(), small_mask)
                if len(r["heap"]) < p.top_k_frames:
                    heapq.heappush(r["heap"], item)
                elif score > r["heap"][0][0]:
                    heapq.heapreplace(r["heap"], item)
        idx += 1
        if processed % 10 == 0:
            progress("detect", min(0.99, idx / n_frames), f"frame {idx}/{n_frames}")
    cap.release()
    elapsed = time.time() - t_start
    return {"raw": raw, "meta": {"fps": fps, "width": W, "height": H, "n_frames": idx, "stride": p.stride,
                                 "duration": idx / fps},
            "stats": {"detect_s": round(elapsed, 2), "frames_processed": processed,
                      "detector_fps": round(processed / max(elapsed, 1e-6), 1)}}


def process_video(video_id: str, path: str, p: Params, progress=Progress, force: bool = False) -> dict:
    """Detect, track and build track representations; cached on disk per (video, detector settings)."""
    emb = get_embedder()
    key = hashlib.sha1(f"{video_id}|{p.detect_key()}|{emb.name}|{FEATURE_VERSION}".encode()).hexdigest()[:16]
    d = os.path.join(cache_dir(), key)
    pk = os.path.join(d, "video.pkl")
    if os.path.exists(pk) and not force:
        progress("detect", 1.0, "using cached detections")
        with open(pk, "rb") as f:
            return pickle.load(f)
    os.makedirs(os.path.join(d, "crops"), exist_ok=True)
    det = detect_tracks(path, p, progress)
    t0 = time.time()
    tracks = []
    raw = stitch_raw({k: v for k, v in det["raw"].items() if v["obs"] and v["heap"]},
                     det["meta"]["fps"], p.stride, p.top_k_frames)
    items = {k: v for k, v in raw.items() if len(v["obs"]) >= p.min_track_obs}
    for n, (tid, v) in enumerate(sorted(items.items())):
        progress("features", n / max(1, len(items)), f"track {n + 1}/{len(items)}")
        samples = [(s, c, m) for s, _, c, m in v["heap"]]
        tr = build_track(tid, "?", sorted(v["obs"], key=lambda o: o[0]), samples, det["meta"], emb, p.enable_ocr)
        best = max(samples, key=lambda s: s[0])[1]
        cv2.imwrite(os.path.join(d, "crops", f"{tid}.jpg"), best)
        tr.thumb = f"/media/cache/{key}/crops/{tid}.jpg"
        tr.merged = v.get("merged", [tid])
        tracks.append(tr)
    out = {"video_id": video_id, "key": key, "meta": det["meta"], "tracks": tracks, "embedder": emb.name,
           "calib": list(emb.calib), "stats": {**det["stats"], "feature_s": round(time.time() - t0, 2), "n_tracks": len(tracks),
                                                "tracks_with_plate": sum(1 for t in tracks if t.plate),
                                                "ocr": ocr_state() if p.enable_ocr else "off"}}
    with open(pk, "wb") as f:
        pickle.dump(out, f)
    return out


def analyze(job_id: str, va: dict, vb: dict, p: Params, redetect: bool = False, progress=Progress,
            ground_truth: list | None = None) -> dict:
    """va / vb: {"id", "name", "path", "url"}."""
    t_total = time.time()
    timing_note = ""
    if p.timing_mode == "auto":
        sa, sb = va.get("start_ts"), vb.get("start_ts")
        if sa and sb:
            p.timing_mode, p.b_offset_s = "offset", float(sb - sa)
            timing_note = f"Auto timing: camera B starts {p.b_offset_s:.0f}s after camera A (from file names)"
        else:
            p.timing_mode = "synced"
            timing_note = "Auto timing: no timestamps in the file names, so both videos are assumed to start together"
    vids = {}
    for side, v in (("a", va), ("b", vb)):
        progress(f"detect {side.upper()}", 0.0, f"camera {side.upper()}")
        vr = process_video(v["id"], v["path"], p, lambda s, f, m="", side=side: progress(f"{s} {side.upper()}", f, m), force=redetect)
        for t in vr["tracks"]:
            t.cam = side.upper()
        vids[side] = vr
    progress("match", 0.0, "scoring candidate pairs")
    calib = calibration(p, vids["a"]["calib"])
    t_m = time.time()
    cands = rank_all(vids["a"]["tracks"], vids["b"]["tracks"], p, calib)
    ta = {t.id: t for t in vids["a"]["tracks"]}
    tb = {t.id: t for t in vids["b"]["tracks"]}
    for c in cands:
        c["review"] = review_pair(ta[c["a"]], tb[c["b"]], c, p, tuple(c["calib"]))
    match_s = time.time() - t_m

    summary = []
    for aid in ta:
        mine = sorted((c for c in cands if c["a"] == aid), key=lambda c: c["ctx"]["rank_a"])
        top = mine[0] if mine else None
        accepted = [c for c in mine if c["review"]["status"] == "accepted"]
        sel = accepted[0] if accepted else None
        summary.append({"a": aid, "top_b": top["b"] if top else None,
                        "status": (sel or top)["review"]["status"] if (sel or top) else "rejected",
                        "selected_b": sel["b"] if sel else None,
                        "no_reliable_match": sel is None,
                        "n_lookalikes": len(top["ctx"]["lookalikes_a"]) if top else 0})

    keep = [c for c in cands if c["ctx"]["rank_a"] <= 5 or c["ctx"]["rank_b"] <= 5]
    for c in keep:
        c["components"] = {k: {"score": None if v["score"] is None else round(v["score"], 3), "detail": v["detail"]}
                           for k, v in c["components"].items()}
        c["score"], c["confidence"], c["q_pair"] = round(c["score"], 3), round(c["confidence"], 3), round(c["q_pair"], 3)
        c["cos"] = round(c["cos"], 3)
    total_s = time.time() - t_total
    frames = sum(v["stats"]["frames_processed"] for v in vids.values())
    result = {
        "job_id": job_id, "params": p.to_dict(),
        "versions": {"detector": p.yolo_weights, "tracker": "bytetrack", "embedder": vids["a"]["embedder"],
                     "ocr": "easyocr" if p.enable_ocr else "off"},
        "videos": {s: {"id": v["video_id"], "name": (va if s == "a" else vb)["name"],
                       "url": (va if s == "a" else vb)["url"], **v["meta"], "stats": v["stats"],
                       "tracks": [t.summary(with_obs=True) for t in v["tracks"]]} for s, v in vids.items()},
        "candidates": keep, "summary_a": summary, "timing_note": timing_note,
        "calibration": {"embedding": cands[0]["calib"], "structure": cands[0]["shape_calib"]} if cands else {},
        "timing": {"total_s": round(total_s, 2), "match_review_s": round(match_s, 3),
                   "frames_processed": frames, "pipeline_fps": round(frames / max(total_s, 1e-6), 1)},
    }
    if ground_truth:
        result["metrics"] = evaluate(summary, cands, ground_truth)
    return json.loads(json.dumps(result, default=to_builtin))


# ------------------------------------------------------------------ export
def export_json(result: dict) -> dict:
    out = {k: result[k] for k in ("job_id", "params", "versions", "timing", "summary_a") if k in result}
    out["tracks"] = {s: [{k: v for k, v in t.items() if k != "obs"} for t in result["videos"][s]["tracks"]]
                     for s in ("a", "b")}
    out["candidate_pairs"] = [{
        "track_a": c["a"], "track_b": c["b"], "rank_for_a": c["ctx"]["rank_a"], "rank_for_b": c["ctx"]["rank_b"],
        "similarity_score": c["score"], "match_confidence": c["confidence"], "components": c["components"],
        "lookalikes_for_a": c["ctx"]["lookalikes_a"], "lookalikes_for_b": c["ctx"]["lookalikes_b"],
        "verification": {"outcome": c["review"]["status"], "confidence": c["review"]["confidence"],
                         "uncertainty": c["review"]["uncertainty"], "reasons": c["review"]["reasons"],
                         "supporting_evidence": c["review"]["supporting"],
                         "contradictory_evidence": c["review"]["contradicting"],
                         "missing_evidence": c["review"]["missing"]},
    } for c in sorted(result["candidates"], key=lambda c: (c["a"], c["ctx"]["rank_a"]))]
    if "metrics" in result:
        out["metrics"] = result["metrics"]
    return out


def export_csv(result: dict) -> str:
    buf = io.StringIO()
    comps = ["embedding", "colour", "physical", "plate", "temporal", "movement"]
    w = csv.writer(buf)
    w.writerow(["track_a", "track_b", "rank_for_a", "rank_for_b", "similarity_score", "match_confidence", "outcome",
                "review_confidence", "uncertainty", *[f"score_{c}" for c in comps], "supporting", "contradictory",
                "missing", "lookalikes"])
    for c in sorted(result["candidates"], key=lambda c: (c["a"], c["ctx"]["rank_a"])):
        r = c["review"]
        w.writerow([c["a"], c["b"], c["ctx"]["rank_a"], c["ctx"]["rank_b"], c["score"], c["confidence"], r["status"],
                    r["confidence"], r["uncertainty"], *[c["components"][k]["score"] for k in comps],
                    " | ".join(i["text"] for i in r["supporting"]), " | ".join(i["text"] for i in r["contradicting"]),
                    " | ".join(i["text"] for i in r["missing"]),
                    ";".join([f"B{b}" for b in c["ctx"]["lookalikes_a"]] + [f"A{a}" for a in c["ctx"]["lookalikes_b"]])])
    return buf.getvalue()
