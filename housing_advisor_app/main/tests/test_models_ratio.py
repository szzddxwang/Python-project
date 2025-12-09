import math
import numpy as np
import pytest

from models_ratio import (
    price_to_rent_ratio,
    ratio_score,
    affordability_score,
)


def test_price_to_rent_ratio_basic():
    # price = 300000, rent = 2500 => 300000 / (2500 * 12) = 10
    ptr = price_to_rent_ratio(price=300000, monthly_rent=2500)
    assert math.isclose(ptr, 10.0, rel_tol=1e-6)


@pytest.mark.parametrize("rent", [0, -100, None])
def test_price_to_rent_ratio_invalid_rent_returns_none(rent):
    ptr = price_to_rent_ratio(price=300000, monthly_rent=rent)
    assert ptr is None


def test_ratio_score_lower_bound():
    # <= 15 => 1.0
    assert ratio_score(10.0) == 1.0
    assert ratio_score(15.0) == 1.0


def test_ratio_score_upper_bound():
    # >= 30 => -1.0
    assert ratio_score(30.0) == -1.0
    assert ratio_score(40.0) == -1.0


def test_ratio_score_middle_linear():
    # 15~30 之间线性插值: 15 -> 1, 30 -> -1, 中点 22.5 -> 0
    s = ratio_score(22.5)
    assert math.isclose(s, 0.0, rel_tol=1e-6)


def test_ratio_score_none_or_nan_is_zero():
    assert ratio_score(None) == 0.0
    assert ratio_score(np.nan) == 0.0


def test_affordability_score_good_case():
    # price / (income * 4) <= 1.0 => 1.0
    score = affordability_score(price=800000, income=200000)
    assert score == 1.0


def test_affordability_score_bad_case():
    # price / (income * 4) >= 2.5 => -1.0
    score = affordability_score(price=2000000, income=200000)
    assert score == -1.0


def test_affordability_score_middle_linear():
    # ratio = 1.5 => Between 1 and -1
    # ratio = price / (income * 4) = 1_200_000 / (200_000 * 4) = 1.5
    score = affordability_score(price=1_200_000, income=200_000)
    # Theoretical value: 1 - (1.5 - 1) / (2.5 - 1) * 2 = 1 - (0.5 / 1.5) * 2 = 1 - 2/3 = 1/3
    assert math.isclose(score, 1.0 / 3.0, rel_tol=1e-6)


def test_affordability_score_invalid_income_is_zero():
    assert affordability_score(price=800000, income=0) == 0.0
    assert affordability_score(price=800000, income=None) == 0.0

