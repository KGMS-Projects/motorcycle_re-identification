from tests.helpers import make_track, params
from app.temporal import movement, timing


def test_consistent_travel_time():
    a, b = make_track(1, "A", t1=10), make_track(1, "B", t0=14)
    s, info = timing(a, b, params(min_travel_s=3, max_travel_s=6))
    assert s == 1.0 and info["state"] == "consistent"


def test_clock_offset_is_applied():
    a, b = make_track(1, "A", t1=10), make_track(1, "B", t0=2)
    s, info = timing(a, b, params(timing_mode="offset", b_offset_s=12, min_travel_s=1, max_travel_s=6))
    assert info["travel_s"] == 4.0 and info["state"] == "consistent"


def test_b_before_a_exit_is_impossible():
    a, b = make_track(1, "A", t1=30), make_track(1, "B", t0=5)
    s, info = timing(a, b, params(clock_tol_s=3))
    assert info["state"] == "impossible" and s == 0.0


def test_clock_uncertainty_tolerated():
    a, b = make_track(1, "A", t1=10), make_track(1, "B", t0=8)
    s, info = timing(a, b, params(clock_tol_s=3))
    assert info["state"] == "tolerated" and 0 < s < 1


def test_unknown_timing_is_missing_not_bad():
    s, info = timing(make_track(1), make_track(1, "B"), params(timing_mode="unknown"))
    assert s is None and info["state"] == "unknown"


def test_movement_modes():
    a, b = make_track(1, direction=(1, 0)), make_track(1, "B", direction=(-1, 0))
    assert movement(a, b, params(direction_mode="any"))[0] is None
    assert movement(a, b, params(direction_mode="opposite"))[0] == 1.0
    assert movement(a, b, params(direction_mode="same"))[1]["state"] == "inconsistent"
