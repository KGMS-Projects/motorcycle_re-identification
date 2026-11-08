import numpy as np

from app.stitch import stitch_raw

FPS, STRIDE = 25.0, 2
STEP = STRIDE / FPS


def crop(colour):
    return np.full((60, 80, 3), colour, np.uint8)


def trk(t0, n, x0, dx=4, colour=(0, 0, 210), y=100, size=60):
    obs = [[t0 + i * STEP, x0 + i * dx, y, x0 + i * dx + size, y + size, 0.8, 1, 0.0, 0.0] for i in range(n)]
    return {"obs": obs, "heap": [(0.5, 0, crop(colour), None)]}


def test_fragments_after_a_short_gap_become_one_vehicle():
    raw = {5: trk(3.0, 5, 100), 8: trk(3.0 + 5 * STEP + 0.3, 10, 125), 18: trk(5.0, 12, 170)}
    out = stitch_raw(raw, FPS, STRIDE, 12)
    assert list(out) == [5] and out[5]["merged"] == [5, 8, 18]
    assert len(out[5]["obs"]) >= 25


def test_duplicate_ids_on_the_same_bike_are_merged():
    a, b = trk(1.0, 10, 100), trk(1.0, 10, 102)
    out = stitch_raw({1: a, 2: b}, FPS, STRIDE, 12)
    assert len(out) == 1 and len(out[1]["obs"]) == 10      # one box per time step


def test_two_bikes_visible_at_once_are_never_merged():
    out = stitch_raw({1: trk(1.0, 10, 50), 2: trk(1.0, 10, 400, colour=(0, 0, 215))}, FPS, STRIDE, 12)
    assert len(out) == 2


def test_far_apart_in_time_or_colour_stay_separate():
    late = trk(10.0, 8, 100)
    differently_coloured = trk(1.0 + 8 * STEP + 0.2, 8, 130, colour=(210, 40, 20))
    out = stitch_raw({1: trk(1.0, 8, 100), 2: late, 3: differently_coloured}, FPS, STRIDE, 12)
    assert len(out) == 3
