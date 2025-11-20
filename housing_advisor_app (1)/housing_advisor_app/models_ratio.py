from typing import Optional
import numpy as np

def price_to_rent_ratio(price: float, monthly_rent: Optional[float]) -> Optional[float]:
    if monthly_rent is None or monthly_rent <= 0:
        return None
    annual = monthly_rent * 12.0
    if annual <= 0:
        return None
    return float(price / annual)

def ratio_score(ptr: Optional[float]) -> float:
    if ptr is None or np.isnan(ptr):
        return 0.0
    if ptr <= 15: return 1.0
    if ptr >= 30: return -1.0
    return float(1.0 - (ptr - 15) / (30 - 15) * 2.0)

def affordability_score(price: float, income: Optional[float], rate: float = 0.06) -> float:
    if income is None or income <= 0:
        return 0.0
    ratio = price / max(1.0, income * 4.0)
    if ratio <= 1.0: return 1.0
    if ratio >= 2.5: return -1.0
    return float(1.0 - (ratio - 1.0) / (2.5 - 1.0) * 2.0)
