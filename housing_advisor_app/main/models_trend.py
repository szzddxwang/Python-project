import numpy as np
import pandas as pd

def compute_trend_score(price_series: pd.Series, window: int = 12) -> float:
    s = pd.to_numeric(price_series, errors="coerce").dropna().sort_values()
    if len(s) < 3:
        return 0.0
    tail = s.tail(window) if len(s) >= window else s
    mu = tail.mean()
    last = tail.iloc[-1]
    if mu == 0 or np.isnan(mu):
        return 0.0
    raw = (last - mu) / abs(mu)
    return float(max(-1.0, min(1.0, raw)))
