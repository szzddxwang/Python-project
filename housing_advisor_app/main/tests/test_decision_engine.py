import math

import pytest

from decision_engine import DecisionEngine, DecisionInputs
from config import TREND_WEIGHT, RATIO_WEIGHT, AFFORD_WEIGHT


def manual_score(x: DecisionInputs) -> float:
    """Manually compute the score once according to the formula in the decision_engine to ensure the logic is consistent."""
    return (
        TREND_WEIGHT * x.trend_score +
        RATIO_WEIGHT * x.ratio_score +
        AFFORD_WEIGHT * x.affordability_score
    )


def expected_label_from_score(s: float) -> str:
    if s >= 0.35:
        return "BUY"
    elif s <= -0.15:
        return "AVOID"
    else:
        return "WATCH"


@pytest.mark.parametrize(
    "inputs",
    [
        DecisionInputs(trend_score=1.0, ratio_score=1.0, affordability_score=1.0),
        DecisionInputs(trend_score=-1.0, ratio_score=-1.0, affordability_score=-1.0),
        DecisionInputs(trend_score=0.0, ratio_score=0.0, affordability_score=0.0),
        DecisionInputs(trend_score=0.5, ratio_score=-0.5, affordability_score=0.0),
    ],
)
def test_decision_engine_label_and_score(inputs: DecisionInputs):
    engine = DecisionEngine()
    result = engine.decide(inputs)

    s = manual_score(inputs)
    expected_label = expected_label_from_score(s)

    # The score should be consistent with the manual calculation
    assert math.isclose(result.score, s, rel_tol=1e-6)
    # The label should comply with the threshold rules.
    assert result.label == expected_label


def test_decision_engine_explanation_contains_details():
    engine = DecisionEngine()
    x = DecisionInputs(trend_score=0.1, ratio_score=0.2, affordability_score=0.3)
    result = engine.decide(x)

    assert "Composite =" in result.explanation
    #The text contains each of the sub-scores.
    assert "trend 0.10" in result.explanation
    assert "ratio 0.20" in result.explanation
    assert "afford 0.30" in result.explanation

