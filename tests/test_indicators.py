import math

import numpy as np
import pandas as pd
import pytest

from options_mas.indicators import EMA, SessionVWAP, ema_series, fibonacci_pivots, vwap_series
from tests.helpers import D0, D1, bar


def test_ema_sma_seed_then_recursive_matches_pandas():
    vals = [10, 11, 12, 13, 12, 11, 15, 16, 14, 13]
    out = ema_series(vals, 3)
    assert np.isnan(out[:2]).all()
    assert out[2] == pytest.approx(11.0)  # SMA seed
    # after the seed, the recursion equals pandas ewm(adjust=False) started from the seed
    s = pd.Series([11.0] + vals[3:]).ewm(alpha=0.5, adjust=False).mean().to_numpy()
    assert out[2:] == pytest.approx(s)


def test_ema_not_ready_before_period():
    e = EMA(50)
    for i in range(49):
        assert e.update(100 + i) is None
    assert e.update(100) is not None


def test_vwap_typical_price_and_session_reset():
    v = SessionVWAP()
    v.update(bar(D0, 0, 10, 12, 8, 10, v=100))      # tp 10
    r = v.update(bar(D0, 1, 10, 22, 18, 20, v=300))  # tp 20
    assert r == pytest.approx((10 * 100 + 20 * 300) / 400)
    v.reset()
    assert v.value is None
    assert v.update(bar(D1, 0, 5, 6, 4, 5, v=10)) == pytest.approx(5)


def test_vwap_zero_volume_keeps_previous_value():
    v = SessionVWAP()
    v.update(bar(D0, 0, 10, 12, 8, 10, v=100))
    assert v.update(bar(D0, 1, 50, 60, 40, 50, v=0)) == pytest.approx(10)


def test_vwap_series_resets_per_session():
    out = vwap_series([12, 22, 6], [8, 18, 4], [10, 20, 5], [100, 300, 10], ["a", "a", "b"])
    assert out[2] == pytest.approx(5)


def test_fibonacci_pivots_formula():
    p = fibonacci_pivots(120, 80, 100)
    assert p.P == pytest.approx(100)
    assert p.R1 == pytest.approx(100 + 0.382 * 40)
    assert p.R2 == pytest.approx(100 + 0.618 * 40)
    assert p.R3 == pytest.approx(140)
    assert p.S1 == pytest.approx(100 - 0.382 * 40)
    assert p.S2 == pytest.approx(100 - 0.618 * 40)
    assert p.S3 == pytest.approx(60)
    names = [n for n, _ in p.sorted_levels()]
    assert names == ["S3", "S2", "S1", "P", "R1", "R2", "R3"]
