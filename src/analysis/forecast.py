import numpy as np
import pandas as pd

def linear_prediction(col, target_year=2024, eps=1e-6):
    y = col.dropna()
    if len(y) < 2: 
        return float(y.iloc[-1]) if len(y) else eps
    x = y.index.values.astype(float)
    y_ = np.clip(y.values, eps, 1 - eps)
    z = np.log(y_ / (1 - y_))
    m, b = np.polyfit(x, z, 1)
    zhat = m * target_year + b
    phat = 1 / (1 + np.exp(-zhat))
    return float(phat)


def proj_from_year_series(s):
    f = s.dropna().values
    if len(f) < 4:
        return f[-1] if len(f) else 0.0
    # Projektion: 2022 + max(Trend, 0)*2
    return f[-3] + max(0.0, f[-3] - f[-4]) * 2

