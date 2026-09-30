"""加減碼規則與成交時點。權重不該被測試改掉。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sizing import c1_targets, c2_targets, c3_targets, run_sized_book


def _frame(colors: list[str | None], close: list[float], bull: list[bool] | None = None) -> pd.DataFrame:
    n = len(colors)
    idx = pd.bdate_range("2016-01-01", periods=n)
    px = np.array(close, dtype=float)
    if bull is None:
        bull = [True] * n
    return pd.DataFrame(
        {
            "open": px,
            "high": px + 1,
            "low": px - 1,
            "close": px,
            "m_color": colors,
            "m_green": [c in ("淺綠", "深綠") for c in colors],
            "m_dark_red": [c == "深紅" for c in colors],
            "bull": bull,
            "thdn": np.full(n, 90.0),
            "thup": np.full(n, 110.0),
        },
        index=idx,
    )


def test_c1_weights_and_flat_keeps_previous():
    df = _frame(["淺綠", "漸增淺紅", "持平負區", "深紅", "深綠"], [100, 100, 100, 100, 100])
    got = c1_targets(df, 0.0).tolist()
    assert got == [1.0, 0.5, 0.5, 0.0, 1.0]
    assert c1_targets(df, 0.25).tolist()[3] == 0.25


def test_c1_does_not_use_next_color():
    base = _frame(["淺綠", "淺綠", "深紅"], [100, 100, 100])
    changed = base.copy()
    changed.loc[changed.index[2], "m_color"] = "淺綠"
    assert c1_targets(base, 0.0).tolist()[:2] == c1_targets(changed, 0.0).tolist()[:2]


def test_c2_holds_add_until_dark_red_or_stop_not_while_only_inside_band():
    df = _frame(
        ["淺綠", "淺綠", "漸增淺紅", "深紅"],
        [100, 120, 120, 120],
    )
    # 第二天收盤在帶外（上軌 110），不該因此取消加碼。第三天漸增淺紅也不取消。
    df.loc[df.index[1], "close"] = 120
    df.loc[df.index[1], "open"] = 120
    got = c2_targets(df, 1.0, 1.5).tolist()
    assert got == [1.5, 1.5, 1.5, 1.0]


def test_c2_does_not_add_on_increasing_light_red_and_stops_below_band():
    df = _frame(["漸增淺紅", "淺綠", "淺綠"], [100, 100, 80])
    df.loc[df.index[2], "close"] = 80
    got = c2_targets(df, 0.6, 1.0).tolist()
    assert got[0] == 0.6
    assert got[1] == 1.0
    assert got[2] == 0.6


def test_c3_priority_dark_then_light_red_then_add():
    df = _frame(
        ["淺綠", "淺綠", "漸增淺紅", "深紅", "持平負區", "深綠"],
        [100, 120, 120, 120, 120, 120],
    )
    got = c3_targets(df).tolist()
    assert got[0] == 1.5
    assert got[1] == 1.5  # 離開帶內仍留著加碼
    assert got[2] == 0.5
    assert got[3] == 0.0
    assert got[4] == 0.0  # 持平不把部位加回去
    assert got[5] == 1.0  # 回綠，但收盤在帶外，不加碼


def test_rebalance_is_next_open_and_flat_target_does_not_trade_again():
    idx = pd.bdate_range("2016-01-01", periods=4)
    df = pd.DataFrame(
        {"open": [100.0, 100.0, 100.0, 110.0], "close": [100.0, 100.0, 110.0, 150.0]},
        index=idx,
    )
    # 第三根收盤才把目標改成 0，所以這根收盤仍是滿倉；下一根開盤才賣掉，之後的上漲不參與。
    target = pd.Series([1.0, 1.0, 0.0, 0.0], index=idx)
    mask = pd.Series([False, True, True, True], index=idx)
    res = run_sized_book(df, target, mask, name="lag", cost=0.0005)
    eq = res.equity.loc[mask]
    assert abs(eq.iloc[1] / eq.iloc[0] - 1.10) < 1e-9
    assert eq.iloc[2] < eq.iloc[1]
    assert eq.iloc[2] > eq.iloc[1] * 0.998
    assert float(eq.iloc[2]) < 1.2
    assert res.n_orders == 2


def test_leverage_day_tracks_one_and_a_half_and_cash_can_be_negative():
    idx = pd.bdate_range("2016-01-01", periods=2)
    df = pd.DataFrame({"open": [100.0, 100.0], "close": [110.0, 110.0]}, index=idx)
    target = pd.Series([1.5, 1.5], index=idx)
    mask = pd.Series([False, True], index=idx)
    res = run_sized_book(df, target, mask, name="lev", cost=0.0005)
    assert 1.145 < float(res.equity.iloc[1]) < 1.150


def test_round_trip_cost_is_about_two_sides():
    idx = pd.bdate_range("2016-01-01", periods=2)
    df = pd.DataFrame({"open": [100.0, 100.0], "close": [100.0, 100.0]}, index=idx)
    target = pd.Series([1.0, 1.0], index=idx)
    mask = pd.Series([False, True], index=idx)
    res = run_sized_book(df, target, mask, name="cost", cost=0.0005)
    final = float(res.equity.iloc[1])
    assert abs(final - (1.0 - 0.0005) / (1.0 + 0.0005)) < 1e-9
