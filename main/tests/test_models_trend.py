import math
import pandas as pd

from models_trend import compute_trend_score


def test_compute_trend_score_short_series_returns_zero():
    s = pd.Series([100000, 200000])  # 长度 < 3
    score = compute_trend_score(s)
    assert score == 0.0


def test_compute_trend_score_constant_series_zero():
    s = pd.Series([200000, 200000, 200000, 200000])
    score = compute_trend_score(s)
    assert math.isclose(score, 0.0, rel_tol=1e-6)


def test_compute_trend_score_increasing_values():
    # [100, 200, 300, 400] -> mu = 250, last = 400 => raw = (400-250)/250 = 0.6
    s = pd.Series([100000, 200000, 300000, 400000])
    score = compute_trend_score(s)
    assert math.isclose(score, 0.6, rel_tol=1e-6)
    assert 0.0 <= score <= 1.0


def test_compute_trend_score_window_effect():
    # 1..20，window=5 => tail = [16..20], mu = 18, last = 20 => (20-18)/18 = 2/18
    s = pd.Series(range(1, 21))
    score = compute_trend_score(s, window=5)
    expected = (20 - 18) / 18.0
    assert math.isclose(score, expected, rel_tol=1e-6)


def test_compute_trend_score_clipped_upper_bound():
    # 极端情况让 raw > 1，返回值应被截断到 1.0
    s = pd.Series([1, 2, 1000])
    score = compute_trend_score(s)
    assert score == 1.0
