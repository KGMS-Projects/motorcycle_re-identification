import numpy as np

from app.features import HandcraftedEmbedder
from app.tracks import build_track


def _obs(n=20):
    return [[i * 0.2, 50 + i * 8, 100, 170 + i * 8, 200, 0.85, 1, 0.0, 0.0] for i in range(n)]


def test_track_representation_aggregates_many_frames():
    rng = np.random.default_rng(0)
    samples = [(0.5 + 0.02 * i, (rng.random((80, 120, 3)) * 255).astype(np.uint8)) for i in range(8)]
    meta = {"fps": 25.0, "width": 640, "height": 480, "stride": 2}
    t = build_track(1, "A", _obs(), samples, meta, HandcraftedEmbedder(), enable_ocr=False)
    assert t.frame_embs.shape[0] == 8                       # per-frame embeddings kept
    assert abs(float(np.linalg.norm(t.emb)) - 1.0) < 1e-4   # one aggregated, normalised track embedding
    assert t.moving and t.direction[0] > 0.9                # moving right
    assert 0.0 < t.q <= 1.0 and t.quality["n_obs"] == 20


def test_short_occluded_track_scores_lower_quality():
    rng = np.random.default_rng(1)
    samples = [(0.5, (rng.random((80, 120, 3)) * 255).astype(np.uint8)) for _ in range(3)]
    meta = {"fps": 25.0, "width": 640, "height": 480, "stride": 2}
    good = build_track(1, "A", _obs(30), samples, meta, HandcraftedEmbedder(), False)
    bad_obs = [[o[0], *o[1:7], 0.8, 1.0] for o in _obs(6)]
    bad = build_track(2, "A", bad_obs, samples, meta, HandcraftedEmbedder(), False)
    assert bad.q < good.q
