"""預警層減碼。比例與優先順序不該被測試改掉。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from early_warning import e1_targets, e2_targets, e3_targets
from monthly_bx import align_weekly_bx_to_daily, weekly_bx_table
from sizing import run_sized_book


def _frame(
    close: list[float],
    dark: list[bool],
    colors: list[str],
    weekly_neg: list[bool] | None = None,
    thdn: float = 90.0,
    thup: float = 110.0,
) -> pd.DataFrame:
    n = len(close)
    idx = pd.bdate_range("2018-01-01", periods=n)
    px = np.array(close, dtype=float)
    if weekly_neg is None:
        weekly_neg = [False] * n
    return pd.DataFrame(
        {
            "open": px,
            "close": px,
            "thdn": np.full(n, thdn),
            "thup": np.full(n, thup),
            "m_dark_red": dark,
            "m_color": colors,
            "w_negative": weekly_neg,
        },
        index=idx,
    )


def test_e1_priority_dark_red_then_break_then_inside():
    df = _frame(
        [100, 80, 80, 100, 120, 100],
        [False, False, True, False, False, False],
        ["淺綠", "淺綠", "深紅", "淺綠", "淺綠", "淺綠"],
    )
    got = e1_targets(df).tolist()
    assert got == [1.0, 0.5, 0.25, 1.0, 1.0, 1.0]


def test_e1_above_band_does_not_restore_and_touching_lower_band_is_inside():
    df = _frame(
        [80, 120, 90],
        [False, False, False],
        ["淺綠", "淺綠", "淺綠"],
    )
    got = e1_targets(df).tolist()
    assert got[0] == 0.5
    assert got[1] == 0.5
    assert got[2] == 1.0


def test_e1_does_not_use_next_close():
    base = _frame([100, 100, 80], [False, False, False], ["淺綠", "淺綠", "淺綠"])
    changed = base.copy()
    changed.loc[changed.index[2], "close"] = 100
    assert e1_targets(base).tolist()[:2] == e1_targets(changed).tolist()[:2]


def test_e2_weekly_negative_holds_inside_band_and_price_break_alone_does_not_cut():
    df = _frame(
        [100, 100, 80, 100],
        [False, False, False, True],
        ["淺綠", "淺綠", "淺綠", "深紅"],
        weekly_neg=[True, False, False, True],
    )
    got = e2_targets(df).tolist()
    assert got == [0.5, 1.0, 1.0, 0.25]


def test_e2_stays_reduced_until_weekly_non_negative_and_inside_band():
    df = _frame(
        [100, 80, 80, 100],
        [True, False, False, False],
        ["深紅", "淺綠", "淺綠", "淺綠"],
        weekly_neg=[False, False, True, False],
    )
    # 離開深紅時週線不是負的、但在帶外：維持 25%，不因跌破下軌改成 50%。
    # 週線轉負後即使仍在帶外也是 50%。週線轉回非負且回到帶內才 100%。
    got = e2_targets(df).tolist()
    assert got == [0.25, 0.25, 0.5, 1.0]


def test_e3_green_first_then_inside_band_and_light_red_does_not_start_refill():
    df = _frame(
        [100, 80, 100, 100, 100, 100],
        [False, True, False, False, False, False],
        ["淺綠", "深紅", "漸增淺紅", "淺綠", "淺綠", "深綠"],
    )
    got = e3_targets(df).tolist()
    assert got == [1.0, 0.25, 0.25, 0.5, 1.0, 1.0]


def test_e3_same_day_inside_band_does_not_skip_the_fifty_step():
    df = _frame(
        [100, 100, 100],
        [True, False, False],
        ["深紅", "淺綠", "淺綠"],
    )
    got = e3_targets(df).tolist()
    assert got == [0.25, 0.5, 1.0]


def test_e3_warning_still_cuts_when_not_recovering_from_dark_red():
    df = _frame(
        [100, 80, 100],
        [False, False, False],
        ["淺綠", "淺綠", "淺綠"],
    )
    assert e3_targets(df).tolist() == [1.0, 0.5, 1.0]


def test_e3_losing_green_before_band_restarts_the_fifty_step():
    df = _frame(
        [80, 80, 80, 100],
        [True, False, False, False],
        ["深紅", "淺綠", "漸增淺紅", "淺綠"],
    )
    got = e3_targets(df).tolist()
    assert got == [0.25, 0.5, 0.25, 0.5]


def test_break_is_known_at_close_and_filled_next_open():
    idx = pd.bdate_range("2018-01-01", periods=4)
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 80.0],
            "close": [100.0, 100.0, 80.0, 80.0],
            "thdn": [90.0, 90.0, 90.0, 90.0],
            "thup": [110.0, 110.0, 110.0, 110.0],
            "m_dark_red": [False, False, False, False],
            "m_color": ["淺綠", "淺綠", "淺綠", "淺綠"],
            "w_negative": [False, False, False, False],
        },
        index=idx,
    )
    target = e1_targets(df)
    assert target.tolist() == [1.0, 1.0, 0.5, 0.5]
    mask = pd.Series([False, True, True, True], index=idx)
    res = run_sized_book(df, target, mask, name="e1", cost=0.0005)
    eq = res.equity.loc[mask]
    # 第三根收盤才跌破，這根收盤仍是滿倉，所以 100→80 的跌幅整段都在。
    assert abs(eq.iloc[1] / eq.iloc[0] - 0.80) < 1e-9
    assert res.n_orders >= 2


def _daily_close(start: str, end: str, seed: int) -> pd.Series:
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    return pd.Series(close, index=idx, name="close")


def test_incomplete_final_week_is_not_used():
    close = _daily_close("2020-01-01", "2022-06-15", seed=4)
    table = weekly_bx_table(close)
    assert not table.empty
    assert table.index.max().to_period("W-SUN") < close.index.max().to_period("W-SUN")
    aligned = align_weekly_bx_to_daily(close)
    last_week = close.index.max().to_period("W-SUN")
    tail = aligned.loc[aligned.index.to_period("W-SUN") == last_week]
    assert tail["w_bx"].notna().all()
    assert (tail["w_week_end"].dt.to_period("W-SUN") < last_week).all()


def test_mid_week_uses_previous_closed_week():
    close = _daily_close("2019-01-01", "2022-08-20", seed=5)
    aligned = align_weekly_bx_to_daily(close)
    week = pd.Period("2022-05-06", freq="W-SUN")
    days = aligned.loc[aligned.index.to_period("W-SUN") == week]
    week_end = days.index.max()
    mid = days.index[0]
    assert mid < week_end
    prev_week = week - 1
    prev_end = aligned.loc[aligned.index.to_period("W-SUN") == prev_week].index.max()
    assert aligned.loc[mid, "w_week_end"] == prev_end
    assert aligned.loc[week_end, "w_week_end"] == week_end


def test_future_week_does_not_change_past_sign():
    close = _daily_close("2018-01-01", "2022-08-12", seed=6)
    base = align_weekly_bx_to_daily(close)
    shocked = close.copy()
    shocked.loc[shocked.index >= "2022-08-01"] *= 1.5
    extra_idx = pd.bdate_range("2022-08-15", "2022-08-26")
    extra = pd.Series(float(shocked.iloc[-1]) * 1.2, index=extra_idx)
    longer = pd.concat([shocked, extra]).sort_index()
    longer = longer[~longer.index.duplicated(keep="last")]
    alt = align_weekly_bx_to_daily(longer)
    july = close.loc[close.index <= "2022-07-29"]
    assert base.loc[july.index, "w_negative"].tolist() == alt.reindex(july.index)["w_negative"].tolist()
