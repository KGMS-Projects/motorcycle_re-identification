"""Cross-camera candidate scoring and ranking (stage 4)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .features import CHROMATIC, bhattacharyya, chromatic_share, hue_diff, plate_contained, plate_ratio, ramp
from .temporal import movement, timing

SHAPE_CALIB = (0.5, 0.95)


def calibration(p, embedder_calib):
    return (p.emb_lo if p.emb_lo is not None else embedder_calib[0],
            p.emb_hi if p.emb_hi is not None else embedder_calib[1])


def adaptive_calib(values, default, min_spread, floor_frac=0.6):
    """Spread similarity scores using this run's own pair distribution.

    Generic embeddings give every motorcycle pair a high cosine, so a fixed ramp saturates and nothing is
    distinguishable. The low end is the 20th percentile of all A x B pairs; the high end is the best pair, but never
    below an absolute floor, so the best of a bad bunch is not automatically scored as a match.
    """
    v = np.asarray(values, dtype=float)
    lo = float(np.percentile(v, 20))
    floor = default[0] + floor_frac * (default[1] - default[0])
    return lo, float(max(v.max(), lo + min_spread, floor))


def emb_cos(A, B) -> float:
    """Whole-track cosine blended with the best frame pairs (so matching views can line up)."""
    S = A.frame_embs @ B.frame_embs.T
    return 0.3 * float(A.emb @ B.emb) + 0.7 * float(np.sort(S.ravel())[-3:].mean())


def embedding_sim(A, B, calib):
    cos = emb_cos(A, B)
    return ramp(cos, *calib), cos


def primary_colour_sim(A, B) -> float:
    """Compare body colours. Black/grey parts (tyres, seat, shadow) exist on almost every bike, so a coloured body
    is only called different from a dark one when the dark track has essentially no colour at all."""
    ca, cb = A.colours[0], B.colours[0]
    ha, hb = ca["name"] in CHROMATIC, cb["name"] in CHROMATIC
    if ha and hb:
        if ca["name"] == cb["name"]:
            return 1.0
        return 0.5 if hue_diff(ca["hue"], cb["hue"]) < 35 else 0.0
    if not ha and not hb:
        if ca["name"] == cb["name"]:
            return 1.0
        return 0.6 if {ca["name"], cb["name"]} == {"black", "grey"} else 0.3
    dark_track = B if ha else A
    return 0.1 if chromatic_share(dark_track.colours) < 0.05 else 0.5


def colour_sim(A, B):
    s_hist = ramp(bhattacharyya(A.hist, B.hist), 0.45, 0.9)
    prim = primary_colour_sim(A, B)
    return 0.4 * s_hist + 0.6 * prim, f"body colour {A.colours[0]['name']} / {B.colours[0]['name']}"


def physical_sim(A, B, shape_calib=SHAPE_CALIB):
    shp = ramp(float(A.shape @ B.shape), *shape_calib)
    asp = ramp(min(A.aspect, B.aspect) / max(A.aspect, B.aspect), 0.5, 1.0)
    dr = abs(A.riders - B.riders)
    rid = 1.0 if dr == 0 else 0.5 if dr == 1 else 0.0
    return 0.6 * shp + 0.2 * asp + 0.2 * rid, f"structure {shp:.2f}, aspect {asp:.2f}, riders {A.riders}/{B.riders}"


def plate_sim(A, B):
    if not (A.plate and B.plate):
        return None, "plate not readable in both tracks"
    r = plate_ratio(A.plate["text"], B.plate["text"])
    if plate_contained(A.plate["text"], B.plate["text"]):
        r = max(r, 0.9)
    return ramp(r, 0.4, 0.9), f"{A.plate['text']} vs {B.plate['text']} (similarity {r:.2f})"


def compare(A, B, p, calib, shape_calib=SHAPE_CALIB) -> dict:
    comps = {}
    es, cos = embedding_sim(A, B, calib)
    comps["embedding"] = {"score": es, "detail": f"track cosine {cos:.2f}"}
    cs, d = colour_sim(A, B)
    comps["colour"] = {"score": cs, "detail": d}
    s, d = physical_sim(A, B, shape_calib)
    comps["physical"] = {"score": s, "detail": d}
    s, d = plate_sim(A, B)
    comps["plate"] = {"score": s, "detail": d}
    s, info = timing(A, B, p)
    comps["temporal"] = {"score": s, "detail": info["text"], "info": info}
    s, info = movement(A, B, p)
    comps["movement"] = {"score": s, "detail": info["text"], "info": info}
    # missing components are excluded and the remaining weights renormalised
    w = dict(p.weights)
    if A.plate and B.plate and min(A.plate["conf"], B.plate["conf"]) >= 0.5:
        w["plate"] = max(w.get("plate", 0.0), 0.35)  # a confidently read plate is the strongest identity cue
    avail = {k: v for k, v in comps.items() if v["score"] is not None and w.get(k, 0) > 0}
    wsum = sum(w[k] for k in avail) or 1.0
    raw = sum(w[k] * v["score"] for k, v in avail.items()) / wsum
    qp = float(np.sqrt(A.q * B.q))
    # a clearly different body colour is the most robust cue across cameras, so it also scales the final score
    gate = 0.65 + 0.35 * min(1.0, cs / 0.4)
    return {"a": A.id, "b": B.id, "score": float(raw), "q_pair": qp, "colour_gate": float(gate),
            "confidence": float(raw * (0.7 + 0.3 * qp) * gate), "components": comps, "cos": cos}


def rank_all(tracks_a, tracks_b, p, calib) -> list[dict]:
    if not tracks_a or not tracks_b:
        return []
    shape_calib = SHAPE_CALIB
    if len(tracks_a) * len(tracks_b) >= 6:  # enough pairs to estimate a distribution
        if p.emb_auto and p.emb_lo is None and p.emb_hi is None:
            calib = adaptive_calib([emb_cos(a, b) for a in tracks_a for b in tracks_b], calib, 0.12)
        shape_calib = adaptive_calib([float(a.shape @ b.shape) for a in tracks_a for b in tracks_b],
                                     SHAPE_CALIB, 0.08)
    cands = {(a.id, b.id): compare(a, b, p, calib, shape_calib) for a in tracks_a for b in tracks_b}
    ia, ib = [a.id for a in tracks_a], [b.id for b in tracks_b]
    M = np.array([[cands[(i, j)]["confidence"] for j in ib] for i in ia])
    rows, cols = linear_sum_assignment(-M)
    assigned = {(ia[r], ib[c]) for r, c in zip(rows, cols)}
    out = []
    for r, i in enumerate(ia):
        for c, j in enumerate(ib):
            row, col, v = M[r], M[:, c], M[r, c]
            rank_a = int((row > v).sum()) + 1
            rank_b = int((col > v).sum()) + 1
            others = np.sort(np.delete(row, c))[::-1]
            top_other = float(others[0]) if len(others) else 0.0
            others_b = np.sort(np.delete(col, r))[::-1]
            top_other_b = float(others_b[0]) if len(others_b) else 0.0
            look_a = [ib[k] for k in range(len(ib)) if k != c and row[k] >= v - p.margin and row[k] >= p.reject_thr]
            look_b = [ia[k] for k in range(len(ia)) if k != r and col[k] >= v - p.margin and col[k] >= p.reject_thr]
            cand = cands[(i, j)]
            cand["calib"], cand["shape_calib"] = [float(x) for x in calib], [float(x) for x in shape_calib]
            cand["ctx"] = {"rank_a": rank_a, "rank_b": rank_b, "margin_a": float(v - top_other), "margin_b": float(v - top_other_b),
                           "lookalikes_a": look_a, "lookalikes_b": look_b,
                           "mutual_best": rank_a == 1 and rank_b == 1, "assigned": (i, j) in assigned}
            out.append(cand)
    return out
