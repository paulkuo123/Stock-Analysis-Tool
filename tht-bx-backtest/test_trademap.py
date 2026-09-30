"""月線 Trade Map：顏色、不可偷看未來、進出場。不需網路。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine import ONE_WAY_COST, attach_metrics
from bear_windows import window_performance
from monthly_bx import align_monthly_bx_to_daily, bx_color_name, monthly_bx_table, stitch_close, stitch_ohlcv
from trademap import fair_value_touch, run_trade_map, variant_a_signals, variant_b_signals


def test_color_names_match_sign_and_slope():
    assert bx_color_name(1.0, 0.5) == "淺綠"
    assert bx_color_name(1.0, 1.0) == "淺綠"  # 正區持平沿用既有：算淺綠
    assert bx_color_name(1.0, 2.0) == "深綠"
    assert bx_color_name(-1.0, -2.0) == "漸增淺紅"
    assert bx_color_name(-1.0, -1.0) == "持平負區"
    assert bx_color_name(-2.0, -1.0) == "深紅"
    assert bx_color_name(np.nan, -1.0) is None
    assert bx_color_name(1.0, np.nan) is None


def _daily_close(start: str, end: str, seed: int = 0) -> pd.Series:
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    shock = rng.normal(0, 0.01, len(idx))
    close = 100 * np.exp(np.cumsum(shock))
    return pd.Series(close, index=idx, name="close")


def test_incomplete_final_month_is_not_used():
    close = _daily_close("2021-01-01", "2022-06-15", seed=1)
    table = monthly_bx_table(close)
    assert not table.empty
    assert table.index.max().to_period("M") < close.index.max().to_period("M")
    aligned = align_monthly_bx_to_daily(close)
    june = aligned.loc[aligned.index.to_period("M") == "2022-06"]
    assert june["m_color"].notna().all()
    assert (june["m_month_end"].dt.to_period("M") == "2022-05").all()


def test_mid_month_uses_previous_closed_month():
    close = _daily_close("2020-01-01", "2022-08-20", seed=2)
    aligned = align_monthly_bx_to_daily(close)
    # 2022-05 的月中必須仍是 4 月顏色；5 月最後一根才換成 5 月。
    may = aligned.loc[aligned.index.to_period("M") == "2022-05"]
    may_end = may.index.max()
    mid = may.index[len(may) // 2]
    assert mid < may_end
    assert aligned.loc[mid, "m_month_end"] == aligned.loc[aligned.index.to_period("M") == "2022-04"].index.max()
    assert aligned.loc[may_end, "m_month_end"] == may_end


def test_future_month_does_not_change_past_colors():
    close = _daily_close("2019-01-01", "2022-08-15", seed=3)
    base = align_monthly_bx_to_daily(close)
    shocked = close.copy()
    shocked.loc[shocked.index >= "2022-08-01"] *= 1.8
    # 再加一個完整的 9 月，讓 8 月在新序列裡變成「已收盤」，但 7 月以前不該變。
    extra_idx = pd.bdate_range("2022-08-16", "2022-09-20")
    extra = pd.Series(shocked.iloc[-1] * 1.3, index=extra_idx)
    longer = pd.concat([shocked, extra])
    longer = longer[~longer.index.duplicated(keep="last")].sort_index()
    alt = align_monthly_bx_to_daily(longer)
    july_end = close.loc[close.index.to_period("M") == "2022-07"].index.max()
    left = base.loc[:july_end, "m_color"]
    right = alt.reindex(left.index)["m_color"]
    assert left.tolist() == right.tolist()
    # 8 月月中在「8 月尚未收盤」時，仍應等於 7 月顏色。
    aug_mid = close.loc[(close.index >= "2022-08-01") & (close.index <= "2022-08-10")]
    assert (base.loc[aug_mid.index, "m_month_end"] == july_end).all()


def test_variant_b_does_not_chase_or_enter_on_dark_red():
    idx = pd.bdate_range("2021-01-01", periods=6)
    df = pd.DataFrame(
        {
            "open": [10, 10, 10, 10, 10, 10],
            "high": [11, 11, 11, 11, 11, 11],
            "low": [9, 9, 9, 9, 9, 9],
            "close": [12, 10, 10, 8, 10, 10],
            "bull": [True, True, True, True, False, True],
            "thdn": [9, 9, 9, 9, 9, 9],
            "thup": [11, 11, 11, 11, 11, 11],
            "m_cycle_ok": [True, True, False, True, True, True],
            "m_dark_red": [False, False, True, False, False, False],
            "m_color": ["淺綠", "淺綠", "深紅", "淺綠", "淺綠", "淺綠"],
            "confirm_23": [False, True, False, False, False, False],
        },
        index=idx,
    )
    touch = fair_value_touch(df)
    assert list(touch) == [False, True, True, False, True, True]
    entry_b, exit_b, inv_b = variant_b_signals(df)
    # 第 0 根收盤 12 在上軌外，不追。第 1 根回到帶內且週期有效，進場。
    assert bool(entry_b.iloc[0]) is False
    assert bool(entry_b.iloc[1]) is True
    # 第 2 根月線深紅，週期無效，不進。
    assert bool(entry_b.iloc[2]) is False
    assert bool(exit_b.iloc[2]) is True
    # 第 4 根飄帶不是綠色，不進；第 5 根綠且在帶內，狀態由假轉真，可以再進。
    assert bool(entry_b.iloc[4]) is False
    assert bool(entry_b.iloc[5]) is True
    assert np.isfinite(inv_b.iloc[1]) and inv_b.iloc[1] == 9
    entry_a, exit_a, _ = variant_a_signals(df)
    assert list(entry_a) == list(df["confirm_23"])
    assert bool(exit_a.iloc[2]) is True


def test_close_stop_exits_next_open_not_intraday():
    idx = pd.bdate_range("2021-03-01", periods=5)
    df = pd.DataFrame(
        {
            "open": [100, 100, 99, 90, 90],
            "high": [101, 101, 100, 95, 95],
            "low": [90, 80, 80, 80, 80],  # 盤中刺破不算出場
            "close": [100, 100, 89, 90, 90],
            "m_color": ["淺綠"] * 5,
        },
        index=idx,
    )
    entry = pd.Series([True, False, False, False, False], index=idx)
    exit_ = pd.Series(False, index=idx)
    inv = pd.Series([95.0, np.nan, np.nan, np.nan, np.nan], index=idx)
    mask = pd.Series(True, index=idx)
    out = run_trade_map(df, entry, exit_, inv, name="stop", analysis_mask=mask)
    assert len(out.result.trades) == 1
    trade = out.result.trades[0]
    # 訊號 3/1 收盤，3/2 開盤 100 進；3/3 收盤 89 跌破 95，3/4 開盤 90 出。
    assert trade.entry_date == idx[1]
    assert trade.exit_date == idx[3]
    assert trade.entry_price == 100
    assert trade.exit_price == 90
    assert "收盤跌破失效價" in trade.reason_exit
    # 含成本：買 100*(1+c)、賣 90*(1-c)
    c = ONE_WAY_COST
    expected = (90 * (1 - c)) / (100 * (1 + c)) - 1
    assert abs(trade.pnl_pct - expected) < 1e-12


def test_stitch_rescales_early_history_but_keeps_cache():
    idx_cache = pd.bdate_range("2020-01-02", periods=10)
    idx_early = pd.bdate_range("2019-12-01", "2020-01-31")
    cache = pd.DataFrame({"close": np.linspace(100, 110, len(idx_cache))}, index=idx_cache)
    early = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 50.0}, index=idx_early)
    # 讓重疊日的 early 收盤是 cache 的一半，比例應約為 2。
    early.loc[idx_cache, "close"] = cache["close"] / 2.0
    close, note = stitch_close(cache, early)
    assert "銜接" in note
    assert abs(close.loc[idx_cache[0]] - cache["close"].iloc[0]) < 1e-9
    before = close.loc[close.index < idx_cache[0]]
    assert abs(before.iloc[-1] - 100.0) < 1e-6  # 50 * 2


def test_stitch_ohlcv_keeps_cache_rows():
    idx_cache = pd.bdate_range("2020-01-02", periods=8)
    idx_early = pd.bdate_range("2019-12-02", "2020-01-20")
    cache = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": np.linspace(100, 107, len(idx_cache)), "volume": 1.0},
        index=idx_cache,
    )
    early = pd.DataFrame({"open": 5.0, "high": 6.0, "low": 4.0, "close": 40.0, "volume": 1.0}, index=idx_early)
    early.loc[idx_cache, "close"] = cache["close"] / 2.0
    early.loc[idx_cache, "open"] = 5.0
    out, note = stitch_ohlcv(cache, early)
    assert "銜接" in note
    assert abs(out.loc[idx_cache[0], "close"] - cache["close"].iloc[0]) < 1e-9
    before = out.loc[out.index < idx_cache[0]]
    assert abs(before["close"].iloc[-1] - 80.0) < 1e-6  # 40 * 2
    assert abs(before["open"].iloc[-1] - 10.0) < 1e-6


def test_window_performance_uses_prior_close_and_path_drawdown():
    idx = pd.bdate_range("2018-09-03", "2018-12-31")
    eq = pd.Series(100.0, index=idx)
    eq.loc["2018-10-01":"2018-11-01"] = 70.0
    eq.loc["2018-11-02":] = 90.0
    stats = window_performance(eq, "2018-10-01", "2018-12-24")
    assert stats["covered"] is True
    assert abs(stats["ret"] - (90.0 / 100.0 - 1.0)) < 1e-12
    assert abs(stats["maxdd"] - (70.0 / 100.0 - 1.0)) < 1e-12
    missing = window_performance(eq.loc["2018-11-01":], "2018-10-01", "2018-12-24")
    assert missing["covered"] is False


def test_gap_through_stop_cancels_entry():
    idx = pd.bdate_range("2021-04-01", periods=3)
    df = pd.DataFrame(
        {
            "open": [100, 80, 80],
            "high": [101, 81, 81],
            "low": [99, 79, 79],
            "close": [100, 80, 80],
            "m_color": ["淺綠"] * 3,
        },
        index=idx,
    )
    entry = pd.Series([True, False, False], index=idx)
    exit_ = pd.Series(False, index=idx)
    inv = pd.Series([90.0, np.nan, np.nan], index=idx)
    out = run_trade_map(df, entry, exit_, inv, name="gap", analysis_mask=pd.Series(True, index=idx))
    assert out.result.trades == []
    assert "取消進場" in out.details.iloc[0]["reason_exit"]
    attach_metrics(out.result, pd.Series(True, index=idx), bh_total=0.0)
    assert out.result.metrics["n_trades"] == 0
    assert abs(out.result.metrics["total_return"]) < 1e-12


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
