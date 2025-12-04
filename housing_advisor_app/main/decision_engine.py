from dataclasses import dataclass
from config import TREND_WEIGHT, RATIO_WEIGHT, AFFORD_WEIGHT

@dataclass
class DecisionInputs:
    trend_score: float
    ratio_score: float
    affordability_score: float

@dataclass
class DecisionResult:
    label: str
    score: float
    explanation: str

class DecisionEngine:
    def decide(self, x: DecisionInputs) -> DecisionResult:
        s = (
            TREND_WEIGHT * x.trend_score +
            RATIO_WEIGHT * x.ratio_score +
            AFFORD_WEIGHT * x.affordability_score
        )
        if s >= 0.35:
            label = "BUY"
        elif s <= -0.15:
            label = "AVOID"
        else:
            label = "WATCH"
        expl = (
            f"Composite = {s:.3f} (trend {x.trend_score:.2f}, "
            f"ratio {x.ratio_score:.2f}, afford {x.affordability_score:.2f})."
        )
        return DecisionResult(label=label, score=float(s), explanation=expl)
