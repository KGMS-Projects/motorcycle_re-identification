"""Temporal plausibility and movement-direction consistency."""
from __future__ import annotations

import numpy as np


def timing(A, B, p):
    """Returns (score or None, info). None = evidence is missing, not 'bad'."""
    if p.timing_mode == "unknown":
        return None, {"state": "unknown", "text": "No camera timing supplied"}
    off = 0.0 if p.timing_mode == "synced" else p.b_offset_s
    travel = (B.t0 + off) - A.t1          # B entry minus A exit, on camera A's clock
    lo, hi, tol = p.min_travel_s, p.max_travel_s, p.clock_tol_s
    d = lo - travel if travel < lo else (travel - hi if travel > hi else 0.0)
    info = {"travel_s": round(float(travel), 1), "window": [lo, hi], "tolerance_s": tol}
    if d == 0:
        state, score = "consistent", 1.0
    elif d <= tol:
        state, score = "tolerated", 0.8
    elif travel < lo:
        state, score = ("marginal", 0.4) if d <= 2 * tol else ("impossible", 0.0)
    else:
        state, score = "too_long", float(max(0.0, 0.6 - (d - tol) / max(hi, 10.0)))
    msg = {"consistent": "inside the plausible travel window",
           "tolerated": "outside the window but within clock uncertainty",
           "marginal": "appears slightly before the A exit, beyond clock uncertainty",
           "impossible": "appears in B before it left A, beyond clock uncertainty",
           "too_long": "gap is longer than the plausible travel window"}[state]
    info.update(state=state, text=f"Travel time {travel:.1f}s ({lo:g}-{hi:g}s window, ±{tol:g}s clock): {msg}")
    return score, info


def movement(A, B, p):
    if p.direction_mode == "any":
        return None, {"state": "unknown", "text": "Direction mode is 'any' - movement not scored"}
    if not (A.moving and B.moving):
        return None, {"state": "unknown", "text": "One of the tracks is nearly stationary - direction unreliable"}
    c = float(np.dot(A.direction, B.direction))
    expected = 1.0 if p.direction_mode == "same" else -1.0
    s = float(0.5 * (1 + expected * c))
    state = "consistent" if s >= 0.75 else "ambiguous" if s >= 0.4 else "inconsistent"
    return s, {"state": state, "cos": round(c, 2),
               "text": f"Direction cosine {c:+.2f}, expected {'same' if expected > 0 else 'opposite'} direction: {state}"}
