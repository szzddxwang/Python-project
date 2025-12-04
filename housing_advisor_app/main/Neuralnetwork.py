import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from models_trend import compute_trend_score
from models_ratio import price_to_rent_ratio, ratio_score, affordability_score


# small utils
def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def _relu(x):
    return np.maximum(0.0, x)


def _robust_01(x: np.ndarray, p_lo=5.0, p_hi=95.0):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo = np.nanpercentile(x, p_lo)
    hi = np.nanpercentile(x, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        mn = np.nanmin(x)
        mx = np.nanmax(x)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return np.zeros_like(x)
        lo, hi = mn, mx
    xc = np.clip(x, lo, hi)
    return (xc - lo) / max(1e-9, (hi - lo))


# feature extraction
def extract_features(
    df: pd.DataFrame,
    desired_price: float,
    desired_rent: float,
    annual_income_before_tax: float,
    personal_income_tax_rate_pct: float,
):

    names = ["price_fit", "afford", "rent_ratio", "value", "trend"]
    if df is None or df.empty:
        return np.zeros((0, 5), dtype=float), names

    desired_price = float(desired_price) if desired_price and desired_price > 0 else float(df["price"].median())
    desired_rent = float(desired_rent) if desired_rent and desired_rent > 0 else None
    income = float(annual_income_before_tax) if annual_income_before_tax and annual_income_before_tax > 0 else None
    tax = float(personal_income_tax_rate_pct or 0.0)

    net_income = None
    if income is not None:
        net_income = income * (1.0 - tax / 100.0)

    # Trend over current filtered set
    trend_score = float(compute_trend_score(df["price"].sort_values()))
    trend01 = (trend_score + 1.0) / 2.0

    price = df["price"].astype(float).to_numpy()
    sqft = np.maximum(1.0, df["square_feet"].astype(float).to_numpy())

    # price_fit, Gaussian on log
    log_ratio = np.log(np.maximum(price, 1.0) / max(desired_price, 1.0))
    price_fit = np.exp(-(log_ratio / 0.35) ** 2)

    # afford, reuse affordability_score
    afford01 = np.asarray([(float(affordability_score(p, net_income)) + 1.0) / 2.0 for p in price], dtype=float)

    # rent_ratio, reuse ratio_score
    ratio01 = []
    for p in price:
        ptr = price_to_rent_ratio(p, desired_rent) if desired_rent is not None else None
        ratio01.append((float(ratio_score(ptr)) + 1.0) / 2.0)
    ratio01 = np.asarray(ratio01, dtype=float)

    # value, sqft per dollar, it is for robust normalized
    value01 = _robust_01(sqft / np.maximum(price, 1.0))

    #  trend constant
    trend_vec = np.full_like(price_fit, float(trend01), dtype=float)

    X = np.vstack([price_fit, afford01, ratio01, value01, trend_vec]).T
    return X, names


#  NN ranker
class OnlineMLPRanker:
    """
    2-layer MLP ranker trained online from 👍/👎:
      h = ReLU(xW1+b1)
      p = sigmoid(hW2+b2)
    """
    def __init__(self, input_dim=5, hidden_dim=12, lr=0.05, l2=0.001, seed=7):
        rng = np.random.default_rng(seed)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.l2 = float(l2)

        # init for ReLU
        self.W1 = rng.normal(0, np.sqrt(2.0 / self.input_dim), size=(self.input_dim, self.hidden_dim))
        self.b1 = np.zeros((self.hidden_dim,), dtype=float)
        self.W2 = rng.normal(0, np.sqrt(2.0 / self.hidden_dim), size=(self.hidden_dim, 1))
        self.b2 = np.zeros((1,), dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            return np.zeros((0,), dtype=float)
        z1 = X @ self.W1 + self.b1
        h = _relu(z1)
        z2 = h @ self.W2 + self.b2
        return _sigmoid(z2).reshape(-1)

    def update_one(self, x: np.ndarray, y: int):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        y = 1.0 if int(y) == 1 else 0.0

        # forward
        z1 = x @ self.W1 + self.b1
        h = _relu(z1)
        z2 = h @ self.W2 + self.b2
        p = float(_sigmoid(z2)[0, 0])

        # grads
        dz2 = (p - y)                         # scalar
        dW2 = h.T * dz2                       # (h,1)
        db2 = np.array([dz2], dtype=float)    # (1,)

        dh = (dz2 * self.W2.T)                # (1,h)
        dz1 = dh * (z1 > 0)                   # (1,h)

        dW1 = x.T @ dz1                       # (d,h)
        db1 = dz1.reshape(-1)                 # (h,)

        # SGD + L2
        self.W2 -= self.lr * (dW2 + self.l2 * self.W2)
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * (dW1 + self.l2 * self.W1)
        self.b1 -= self.lr * db1

    def state_dict(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "l2": self.l2,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }

    @classmethod
    def from_state(cls, s: dict):
        obj = cls(
            input_dim=int(s["input_dim"]),
            hidden_dim=int(s["hidden_dim"]),
            lr=float(s["lr"]),
            l2=float(s["l2"]),
            seed=7,
        )
        obj.W1 = np.asarray(s["W1"], dtype=float)
        obj.b1 = np.asarray(s["b1"], dtype=float)
        obj.W2 = np.asarray(s["W2"], dtype=float)
        obj.b2 = np.asarray(s["b2"], dtype=float)
        return obj


#  store in local
def _state_path() -> Path:
    # Always save under module folder: <project>/data/nn_ranker_state.npz
    base = Path(__file__).resolve().parent
    d = base / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d / "nn_ranker_state.npz"


def _load_state_from_disk() -> dict | None:
    p = _state_path()
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            s = {
                "input_dim": int(z["input_dim"]),
                "hidden_dim": int(z["hidden_dim"]),
                "lr": float(z["lr"]),
                "l2": float(z["l2"]),
                "W1": z["W1"],
                "b1": z["b1"],
                "W2": z["W2"],
                "b2": z["b2"],
            }
        # sanity checks
        if s["W1"].shape != (s["input_dim"], s["hidden_dim"]):
            return None
        if s["W2"].shape != (s["hidden_dim"], 1):
            return None
        return s
    except Exception:
        return None


def _save_state_to_disk(s: dict) -> None:
    p = _state_path()
    tmp = p.with_suffix(".tmp.npz")

    np.savez_compressed(
        tmp,
        input_dim=np.array(int(s["input_dim"])),
        hidden_dim=np.array(int(s["hidden_dim"])),
        lr=np.array(float(s["lr"])),
        l2=np.array(float(s["l2"])),
        W1=np.asarray(s["W1"], dtype=float),
        b1=np.asarray(s["b1"], dtype=float),
        W2=np.asarray(s["W2"], dtype=float),
        b2=np.asarray(s["b2"], dtype=float),
    )

    os.replace(str(tmp), str(p))


#  API used by app.py
def get_ranker() -> OnlineMLPRanker:

    if "nn_ranker_state" in st.session_state:
        return OnlineMLPRanker.from_state(st.session_state["nn_ranker_state"])

    disk_state = _load_state_from_disk()
    if disk_state is not None:
        st.session_state["nn_ranker_state"] = disk_state
        return OnlineMLPRanker.from_state(disk_state)

    r = OnlineMLPRanker()
    st.session_state["nn_ranker_state"] = r.state_dict()
    _save_state_to_disk(st.session_state["nn_ranker_state"])
    return r


def save_ranker(r: OnlineMLPRanker):
    s = r.state_dict()
    st.session_state["nn_ranker_state"] = s
    _save_state_to_disk(s)
