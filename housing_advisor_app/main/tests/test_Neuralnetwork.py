import numpy as np
import pytest

from Neuralnetwork import _sigmoid, _relu, _robust_01, OnlineMLPRanker


# ---------- _sigmoid ----------

@pytest.mark.parametrize("x, expected", [
    (0.0, 0.5),
    (100.0, 1.0),
    (-100.0, 0.0),
])
def test_sigmoid_scalar_values(x, expected):
    """the sigmoid function on scalar inputs."""
    out = _sigmoid(x)
    assert np.isclose(out, expected, atol=1e-6)


def test_sigmoid_vector_shape_and_range():
    """the sigmoid function on vector inputs maintains shape and outputs in [0,1]."""
    z = np.array([-100.0, 0.0, 100.0])
    out = _sigmoid(z)
    assert out.shape == z.shape
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_sigmoid_clipping_effect():
    """Very large values produce almost the same result after clipping to 35."""
    big = _sigmoid(1000.0)
    edge = _sigmoid(35.0)
    assert np.isclose(big, edge, atol=1e-8)


# ---------- _relu ----------

def test_relu_basic_vector():
    arr = np.array([-1.0, 0.0, 1.0])
    out = _relu(arr)
    assert np.array_equal(out, np.array([0.0, 0.0, 1.0]))


def test_relu_shape_and_dtype_matrix():
    """Check shape preservation and output as float."""
    arr = np.array([[-1, 2], [0, -3]], dtype=int)
    out = _relu(arr)
    assert out.shape == arr.shape
    assert out.dtype == float
    expected = np.array([[0.0, 2.0], [0.0, 0.0]])
    assert np.array_equal(out, expected)


# ---------- _robust_01 ----------

def test_robust_01_basic_range_and_endpoints():
    """Under normal numeric conditions, output is in [0,1] and endpoints are close to 0 and 1."""
    arr = np.array([0, 5, 10, 15, 20], dtype=float)
    out = _robust_01(arr)
    assert out.shape == arr.shape
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    assert np.isclose(out[0], 0.0, atol=1e-6)
    assert np.isclose(out[-1], 1.0, atol=1e-6)


def test_robust_01_empty_array():
    """An empty array returns an empty array."""
    x = np.array([], dtype=float)
    out = _robust_01(x)
    assert out.shape == (0,)
    assert out.dtype == float


def test_robust_01_constant_values_all_zero():
    """When all elements are the same, fallback is used and all zeros are returned."""
    x = np.array([5.0, 5.0, 5.0])
    out = _robust_01(x)
    assert np.array_equal(out, np.zeros_like(x, dtype=float))


def test_robust_01_with_nans_and_infs():
    """Handles NaN / Inf and still produces finite results in [0,1]."""
    x = np.array([np.nan, -10.0, 0.0, 10.0, np.inf])
    out = _robust_01(x)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_robust_01_percentile_clipping():
    """Extreme outliers are clipped by percentiles and mapped to 1."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    out = _robust_01(x, p_lo=0.0, p_hi=80.0)
    # Minimum value → 0
    assert np.isclose(out[0], 0.0, atol=1e-6)
    # The largest outlier is clipped to hi, corresponding to 1
    assert np.isclose(out[-1], 1.0, atol=1e-6)


# ---------- OnlineMLPRanker ----------

def test_ranker_predict_empty_input():
    r = OnlineMLPRanker(input_dim=5, hidden_dim=8, lr=0.1, l2=0.0, seed=0)
    X = np.zeros((0, 5), dtype=float)
    out = r.predict(X)
    assert out.shape == (0,)
    assert out.dtype == float


def test_ranker_predict_shape_and_range():
    """Predict output is one-dimensional, length equals number of samples, and values are in (0,1)."""
    r = OnlineMLPRanker(input_dim=5, hidden_dim=8, lr=0.1, l2=0.0, seed=0)
    X = np.random.RandomState(0).randn(10, 5)
    out = r.predict(X)
    assert out.shape == (10,)
    assert np.all(out > 0.0) and np.all(out < 1.0)


def test_ranker_update_moves_prob_towards_label_one():
    """After multiple updates with the same sample labeled as 1, the probability should increase."""
    r = OnlineMLPRanker(input_dim=5, hidden_dim=8, lr=0.1, l2=0.0, seed=0)
    x = np.zeros((5,), dtype=float)
    p0 = float(r.predict(x.reshape(1, -1))[0])
    for _ in range(30):
        r.update_one(x, y=1)
    p1 = float(r.predict(x.reshape(1, -1))[0])
    assert p1 > p0
    assert p1 > 0.5


def test_ranker_update_moves_prob_towards_label_zero():
    """After multiple updates with the same sample labeled as 0, the probability should decrease."""
    r = OnlineMLPRanker(input_dim=5, hidden_dim=8, lr=0.1, l2=0.0, seed=0)
    x = np.zeros((5,), dtype=float)
    p0 = float(r.predict(x.reshape(1, -1))[0])
    for _ in range(30):
        r.update_one(x, y=0)
    p1 = float(r.predict(x.reshape(1, -1))[0])
    assert p1 < p0
    assert p1 < 0.5
