"""選擇權對沖的估價與開關。天期、delta、價外幅度不在測試裡改。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from options_hedge import (
    bs_call,
    bs_put,
    hedge_active,
    run_overlay,
    strike_for_call_delta,
)


def _frame(n: int, close: list[float] | None = None) -> pd.DataFrame:
    idx = pd.bdate_range("2016-01-04", periods=n)
    px = np.array(close if close is not None else [100.0] * n, dtype=float)
    return pd.DataFrame({"open": px, "close": px}, index=idx)


def test_put_call_parity_and_thirty_delta_strike():
    spot, sigma, tenor = 100.0, 0.2, 21 / 252
    strike = strike_for_call_delta(spot, sigma, tenor, 0.30)
    assert strike > spot
    call = bs_call(spot, strike, tenor, sigma)
    put = bs_put(spot, strike, tenor, sigma)
    assert abs((call - put) - (spot - strike)) < 1e-8
    bumped = (bs_call(spot + 0.01, strike, tenor, sigma) - bs_call(spot - 0.01, strike, tenor, sigma)) / 0.02
    assert abs(bumped - 0.30) < 1e-3
    assert abs(spot * 0.90 - spot * (1.0 - 0.10)) < 1e-12


def test_light_red_alone_does_not_open_hedge_and_dark_red_stays_until_green():
    colors = pd.Series(["漸增淺紅", "漸增淺紅", "深紅", "漸增淺紅", "持平負區", "淺綠"])
    flag = hedge_active(colors).tolist()
    assert flag == [False, False, True, True, True, False]


def test_flat_path_without_hedge_matches_stock_round_trip():
    df = _frame(6, [100, 102, 101, 104, 103, 110])
    mask = pd.Series(True, index=df.index)
    hedge = pd.Series(False, index=df.index)
    res = run_overlay(df, hedge, "call", mask, name="none", vol=pd.Series(0.2, index=df.index), tenor=3)
    cost = 0.0005
    shares = 1.0 / (100.0 * (1.0 + cost))
    final = shares * 110.0 * (1.0 - cost)
    assert abs(float(res.equity.iloc[-1]) - final) < 1e-9
    assert res.premium_in == 0.0
    assert res.n_called == 0


def test_covered_call_caps_upside_and_counts_expiry_in_the_money():
    px = [100.0, 100.0, 100.0, 100.0, 160.0, 160.0, 160.0]
    df = _frame(len(px), px)
    mask = pd.Series(True, index=df.index)
    hedge = pd.Series(True, index=df.index)
    vol = pd.Series(0.25, index=df.index)
    plain = run_overlay(df, pd.Series(False, index=df.index), "call", mask, name="plain", vol=vol, tenor=4)
    hedged = run_overlay(df, hedge, "call", mask, name="call", vol=vol, tenor=4)
    assert float(hedged.equity.iloc[-1]) < float(plain.equity.iloc[-1])
    assert hedged.n_called >= 1
    assert hedged.premium_in > 0


def test_protective_put_loses_less_on_a_drop():
    px = [100.0] * 4 + [70.0, 70.0]
    df = _frame(len(px), px)
    mask = pd.Series(True, index=df.index)
    vol = pd.Series(0.30, index=df.index)
    plain = run_overlay(df, pd.Series(False, index=df.index), "put", mask, name="plain", vol=vol, tenor=4)
    hedged = run_overlay(df, pd.Series(True, index=df.index), "put", mask, name="put", vol=vol, tenor=4)
    assert float(hedged.equity.iloc[-1]) > float(plain.equity.iloc[-1])
    assert hedged.premium_out > 0
    assert hedged.n_called == 0


def test_later_prices_do_not_change_earlier_equity():
    base = [100.0 + i * 0.2 for i in range(12)]
    alt = base.copy()
    alt[8:] = [180.0] * 4
    df_a = _frame(12, base)
    df_b = _frame(12, alt)
    mask = pd.Series(True, index=df_a.index)
    hedge = pd.Series([False, False, True, True, True, True, True, True, True, True, True, True], index=df_a.index)
    vol = pd.Series(0.22, index=df_a.index)
    a = run_overlay(df_a, hedge, "collar", mask, name="a", vol=vol, tenor=6)
    b = run_overlay(df_b, hedge, "collar", mask, name="b", vol=vol, tenor=6)
    assert np.allclose(a.equity.iloc[:8].to_numpy(), b.equity.iloc[:8].to_numpy())
    assert a.premium_in > 0 and a.premium_out > 0
