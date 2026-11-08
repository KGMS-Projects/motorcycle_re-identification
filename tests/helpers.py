import numpy as np

from app.config import Params
from app.features import dominant_colours, hsv_hist, l2
from app.tracks import Track, aggregate_colours

BGR = {"red": (0, 0, 220), "blue": (220, 40, 20), "black": (20, 20, 20), "green": (30, 200, 40)}


def make_track(tid, cam="A", ident=0, noise=0.05, colour="red", plate=None, q=0.8, t0=0.0, t1=10.0,
               direction=(1.0, 0.0), moving=True, riders=1, n_obs=30, dim=64):
    """Synthetic track. Tracks with the same `ident` share an appearance (same motorcycle)."""
    rng = np.random.default_rng(ident)
    base = l2(rng.normal(size=dim))
    shape = l2(np.abs(rng.normal(size=96)))
    rng2 = np.random.default_rng(1000 + tid + (0 if cam == "A" else 500))
    frames = l2(base + noise * rng2.normal(size=(12, dim)))
    img = np.full((64, 96, 3), BGR[colour], np.uint8)
    obs = [[t0 + i * (t1 - t0) / max(1, n_obs - 1), 10, 10, 110, 90, 0.9, riders, 0.0, 0.0] for i in range(n_obs)]
    quality = {"n_obs": n_obs, "duration": t1 - t0, "q": q}
    return Track(tid, cam, 25.0, 640, 480, obs, quality, l2(frames.mean(0)), frames.astype(np.float32), shape,
                 hsv_hist(img), aggregate_colours([img], [1.0]), riders, 1.2, plate, "left", "right",
                 direction, moving, 0.0)


def params(**kw):
    return Params.from_dict(kw)
