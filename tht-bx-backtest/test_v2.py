"""v2：ATR／RSI14／確認視窗／停損引擎。不需網路。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine import attach_metrics, run_long_only_ex
from features import compute_atr, compute_rsi_wilder, confirm_after_bars


def test_atr_positive_and_tracks_range():
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    close = pd.Series(np.linspace(100, 110, 30), index=idx)
    high = close + 2.0
    low = close - 2.0
    atr = compute_atr(high, low, close, 14)
    assert atr.dropna().min() > 0
    # 區間約 4，ATR 應落在合理帶
    assert 1.0 < float(atr.iloc[-1]) < 6.0


def test_rsi14_uptrend_high():
    close = pd.Series(np.linspace(10, 40, 60))
    rsi = compute_rsi_wilder(close, 14)
    assert (rsi.dropna() >= 0).all() and (rsi.dropna() <= 100).all()
    assert float(rsi.iloc[-1]) > 80


def test_confirm_after_bars_offsets():
    idx = pd.RangeIndex(12)
    trigger = pd.Series([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], index=idx, dtype=bool)
    still = pd.Series([1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], index=idx, dtype=bool)
    cancel = pd.Series(0, index=idx, dtype=bool)
    out = confirm_after_bars(trigger, still, offsets=(2, 3), cancel=cancel)
    # 第 1 根觸發；offset=1 的第 2 根 still=0，offset=2 的第 3 根 still=1 → 成交
    assert bool(out.iloc[3])
    assert not bool(out.iloc[2])
    # 第 6 根再觸發，offset=2 的第 8 根 still=0，offset=3 的第 9 根 still=1
    assert bool(out.iloc[9])


def test_confirm_cancel_before_ok():
    idx = pd.RangeIndex(8)
    trigger = pd.Series([1, 0, 0, 0, 0, 0, 0, 0], index=idx, dtype=bool)
    still = pd.Series(1, index=idx, dtype=bool)
    cancel = pd.Series([0, 1, 0, 0, 0, 0, 0, 0], index=idx, dtype=bool)
    out = confirm_after_bars(trigger, still, offsets=(2, 3), cancel=cancel)
    assert not out.any()


def test_fixed_stop_exits_intrabar():
    idx = pd.date_range("2021-01-01", periods=8, freq="B")
    # 訊號在第 0 根，第 1 根開盤 100 進場，第 2 根最低打到 90 → 8% 停損 92
    df = pd.DataFrame(
        {
            "open": [100, 100, 99, 99, 99, 99, 99, 99],
            "high": [101, 101, 99, 100, 100, 100, 100, 100],
            "low": [99, 99, 90, 98, 98, 98, 98, 98],
            "close": [100, 100, 91, 99, 99, 99, 99, 99],
            "atr14": [2.0] * 8,
        },
        index=idx,
    )
    entry = pd.Series([True] + [False] * 7, index=idx)
    exit_ = pd.Series(False, index=idx)
    res = run_long_only_ex(df, entry, exit_, name="SL", stop_pct=0.08)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.reason_exit == "停損(盤中)"
    assert abs(t.exit_price - 92.0) < 1e-9


def test_time_exit_next_open():
    idx = pd.date_range("2021-01-01", periods=10, freq="B")
    px = np.full(10, 100.0)
    df = pd.DataFrame(
        {"open": px, "high": px + 1, "low": px - 1, "close": px, "atr14": np.full(10, 2.0)},
        index=idx,
    )
    entry = pd.Series([True] + [False] * 9, index=idx)
    exit_ = pd.Series(False, index=idx)
    res = run_long_only_ex(df, entry, exit_, name="T", time_exit_bars=3)
    assert len(res.trades) == 1
    # 第 0 根訊號 → 第 1 根進；第 1+3=4 根收盤時間到 → 第 5 根開盤出
    assert res.trades[0].bars_held == 4
    assert "時間" in res.trades[0].reason_exit or res.trades[0].reason_exit.endswith("開盤")


def test_cooldown_skips_next_signal():
    idx = pd.date_range("2021-01-01", periods=16, freq="B")
    # 做兩段虧損：進 100 出 90，再進再出；第三個訊號應被跳過
    o = [100, 100, 90, 90, 100, 100, 90, 90, 120, 120, 130, 130, 130, 130, 130, 130]
    c = [100, 95, 90, 90, 100, 95, 90, 90, 125, 128, 130, 130, 130, 130, 130, 130]
    df = pd.DataFrame(
        {
            "open": o,
            "high": [x + 1 for x in o],
            "low": [x - 1 for x in c],
            "close": c,
            "atr14": [2.0] * 16,
        },
        index=idx,
    )
    entry = pd.Series(False, index=idx)
    entry.iloc[0] = True
    entry.iloc[4] = True
    entry.iloc[8] = True
    exit_ = pd.Series(False, index=idx)
    exit_.iloc[2] = True
    exit_.iloc[6] = True
    res = run_long_only_ex(df, entry, exit_, name="CD", cooldown_losses=2)
    # 兩筆虧後跳過第 3 個訊號 → 只有 2 筆
    assert len(res.trades) == 2
    assert all(t.pnl_pct < 0 for t in res.trades)


def test_metrics_include_payoff():
    idx = pd.date_range("2021-01-01", periods=6, freq="B")
    df = pd.DataFrame(
        {
            "open": [100, 100, 110, 110, 110, 110],
            "high": [101, 111, 111, 111, 111, 111],
            "low": [99, 99, 109, 109, 109, 109],
            "close": [100, 110, 110, 110, 110, 110],
            "atr14": [2.0] * 6,
        },
        index=idx,
    )
    entry = pd.Series([True, False, False, False, False, False], index=idx)
    exit_ = pd.Series([False, True, False, False, False, False], index=idx)
    res = run_long_only_ex(df, entry, exit_, name="M")
    mask = pd.Series(True, index=idx)
    attach_metrics(res, mask)
    assert "avg_win" in res.metrics
    assert "payoff" in res.metrics
    assert res.metrics["n_trades"] == 1


if __name__ == "__main__":
    for fn in [
        test_atr_positive_and_tracks_range,
        test_rsi14_uptrend_high,
        test_confirm_after_bars_offsets,
        test_confirm_cancel_before_ok,
        test_fixed_stop_exits_intrabar,
        test_time_exit_next_open,
        test_cooldown_skips_next_signal,
        test_metrics_include_payoff,
    ]:
        fn()
        print(f"OK {fn.__name__}")
    print("all passed")
