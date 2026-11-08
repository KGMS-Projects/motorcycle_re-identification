# Multi-camera motorcycle re-identification and evidence review

A standalone application that tracks motorcycles in two CCTV videos, ranks cross-camera identity candidates, and
**independently reviews** each proposed match as *accepted*, *uncertain* or *rejected*, with the supporting,
conflicting and missing evidence shown for every candidate. It can say "no reliable match" instead of
auto-selecting the top candidate.

## Run it

```bash
docker compose up --build        # UI at http://localhost:8080
```

No paid credentials are needed. Model weights (YOLOv8n, DINOv2-small, EasyOCR) are downloaded at image build time.
Put clips in `./samples/` (they appear in the video dropdowns) or use **Upload** in the UI.

Without Docker: `pip install -r requirements.txt` (install CPU PyTorch first), `ffmpeg` on PATH, then
`uvicorn app.main:app --port 8080`. Tests: `pytest` (or `docker compose run --rm reid pytest`).

## Using the UI

1. Pick or upload a video for **Camera A** and **Camera B** (the first two videos are pre-selected). Change either at any time.
2. **Detect and match** runs detection and tracking on both videos; results are cached per video and detector settings.
   **Re-match** reuses the saved tracks and only re-scores, so changes in **Settings** (timing, thresholds, weights,
   direction) show up in seconds.
3. Click a motorcycle box in either video, a segment in a camera's timeline, or a track in the list. The right-hand
   evidence panel stays in view while you play the videos: verdict, confidence and uncertainty, both crops, what supports,
   conflicts with and is missing from the match, competing lookalikes, and the weighted score breakdown.
   **Jump to pair** (or `J`) seeks both videos to where each track first appears; `Up`/`Down` browse the candidates.
4. Label what you know with **Same bike** or **No match for A#n** in the evidence panel, then **Evaluate** for true and
   false matches, precision, recall, F1 and top-1 accuracy. (JSON import and export is under the labels.)
5. Download the results as JSON or CSV from the top bar.

## Pipeline

| Stage | Implementation |
|---|---|
| 1. Detection and tracking | YOLOv8n-seg (COCO motorcycle and person classes) with ByteTrack. The segmentation mask removes road and background, so colour, embedding and structure describe the motorcycle itself. Quality-ranked best frames per track are kept. Rider count comes from persons overlapping the motorcycle box. |
| 1b. Tracklet stitching | `app/stitch.py` merges fragments of one physical motorcycle inside a camera (duplicate IDs on one bike, or a track that ends where another begins with similar size and colour). Two bikes visible at the same time are never merged. |
| 2. Track representation | Per track: DINOv2-small embeddings of the best 12 frames (aggregated by quality-weighted mean, per-frame embeddings kept), HSV histogram, dominant colours, gradient-structure descriptor, plate OCR vote, motion direction, entry/exit side, and quality statistics (detection ratio, confidence, box stability, occlusion and truncation, sharpness, size, gaps). |
| 3. Cross-camera matching | Every A-track x B-track pair gets six component scores, combined with tunable weights. Missing components (no plate, timing unknown) are dropped and the remaining weights renormalised, so absence of evidence is never scored as evidence of difference. The result is scaled down for poor track quality. |
| 4. Ambiguity | Candidates are ranked from both sides. Pairs within the lookalike margin of each other are flagged, mutual-best is checked, and a Hungarian assignment prevents two tracks claiming one partner. |
| 5. Independent review | `app/reviewer.py`, described below. |
| 6. Explanation and export | Evidence lists per pair in the UI, and `export.json` / `export.csv`. |

### Component scores (initial weights, not validated: tune them)

embedding 0.30, colour 0.25, physical 0.10, plate 0.15, timing 0.10, movement 0.10. A clearly different body colour also scales the final score down.

* **embedding**: cosine of track embeddings blended with the mean of the three best frame-pair cosines, then mapped to
  0-1. By default the ramp is calibrated from the run's own pair distribution (generic embeddings give every motorcycle
  pair a high cosine, so a fixed ramp saturates). Switch to manual `emb_lo` / `emb_hi` in Settings if you prefer.
* **colour**: Bhattacharyya similarity of HSV histograms plus overlap of dominant colour names.
* **physical**: gradient-structure similarity, aspect ratio, rider count. This is a stand-in for exhaust, wheels and
  decals, which a general detector cannot resolve.
* **colour**: body colour first. Tyres, seat and shadows make most bikes partly black, so a clear chromatic colour counts as the body colour even when black covers more pixels.
* **plate**: plate-like rectangles are located first, then OCR reads are clustered and voted. OCR text similarity after normalising confusable characters (O/0, I/1, S/5, B/8, ...).
* **timing**: B-entry minus A-exit on camera A's clock, against a plausible travel window, with a clock-uncertainty
  tolerance. Timing modes: auto (reads recording start times from file names like `20261106_141615.mp4`), synced start, a manual
  B-start offset, or unknown.
* **movement**: direction cosine against the expected relationship (same or opposite image direction), only when both
  tracks are actually moving.

## The independent reviewer

The reviewer does not trust the matcher's number. It re-derives its own checks from the raw per-frame features and applies
structured rules:

* **Frame-level visual consensus**: what fraction of frames in each track has a close match in the other, rather than
  only the aggregated embedding.
* **Plates**: clearly different, confidently read plates are a *strong* contradiction.
* **Colour**: dominant hue disagreement (chromatic colours only, since lighting shifts colour).
* **Timing**: arriving in B before leaving A, beyond clock uncertainty, is a *strong* contradiction.
* **Movement, rider count, structure**: weaker cues.
* **Tracking quality and missing information**: reported as missing or unreliable evidence.
* **Ambiguity**: lookalikes within the margin and not-mutual-best rankings count against the match.

Rules, in order:

0. A confidently read, matching plate (similarity >= 0.85, both reads >= 0.45 confidence, at least 5 characters) is decisive: accepted unless something strongly contradicts it, or uncertain if tracking quality is low.
1. Otherwise, any strong contradiction, a combined score below the reject threshold, or two or more independent moderate
   contradictions give **rejected**. A high visual score cannot override this (see `test_similar_looking_but_different_plates_is_rejected`).
2. Otherwise any of these give **uncertain**: lookalikes, not mutual best, low track quality, fewer than two independent
   supporting cue types, a moderate conflict, score below the accept threshold, low reviewer confidence. The threshold
   is lowered by 0.10 when the candidate is a clear winner on both sides, or when the reviewer independently finds three or
   more agreeing cue types at 80%+ confidence.
3. Otherwise **accepted**.

Each pair also carries `confidence` (mean of matcher and reviewer confidence) and `uncertainty` (driven by track
quality, missing evidence and ambiguity).

## Export schema

`export.json` has `summary_a` (per camera-A track: top candidate, selected match or null, `no_reliable_match`),
`tracks`, and `candidate_pairs`: each with `track_a`, `track_b`, ranks, `similarity_score`, `match_confidence`, `components`,
lookalikes, and `verification` (`outcome`, `confidence`, `uncertainty`, `reasons`, `supporting_evidence`,
`contradictory_evidence`, `missing_evidence`). `export.csv` has one row per candidate pair.

## Tests

`tests/` covers timing and clock offsets, ranking, plate-contradiction rejection, lookalike handling, low quality,
missing evidence, track aggregation, metrics, export schema and API basics. Matching and review are tested on synthetic
tracks so they run without GPUs or model downloads.

## Limitations

* The COCO detector is small. Distant, occluded or night-time motorcycles can be missed or fragmented, and ByteTrack can
  split one motorcycle into several IDs. Fragmented tracks are scored as separate candidates, not merged.
* Plate OCR on CCTV is rarely readable; when it is not, the plate is reported as missing, not as a mismatch.
* "Physical" similarity is a coarse structural descriptor, not real exhaust, rim or decal recognition.
* Calibration ramps, weights and thresholds are engineering defaults. They need tuning with labelled footage, and
  the reported metrics are only as good as your ground truth.
* Direction and travel-time rules assume you supply the camera relationship (timing mode, travel window, direction).
* Colour is computed from the whole box, so background and rider clothing leak in; lighting changes shift hue.
* Videos are transcoded to H.264 at up to 720p for browser playback, and detection runs on that proxy.
* Processing is CPU-only; speed depends on the machine (the UI reports frames per second for each run).

## Technical results for the report

`python scripts/results_report.py reid_<job>.json > results.md` turns an export into the results section: outcomes per track,
accepted, uncertain and rejected examples with their evidence, false matches (when ground truth is supplied) and processing speed.

## Demo checklist (3-5 minutes)

1. Show the two videos side by side and the tracks list.
2. A correct re-identification: accepted pair, then its supporting evidence.
3. A visually similar but different motorcycle: show a plate or timing contradiction causing rejection.
4. An uncertain case: a lookalike or low-quality track.
5. Change a video or a setting and use **Re-match** to show the result update.
6. Show metrics from your ground truth (false matches included), processing speed, and the JSON export.
