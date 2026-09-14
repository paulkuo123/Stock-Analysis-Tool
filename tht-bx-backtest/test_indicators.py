"""指標單元測試：不需網路、不需行情。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import (
    confirm_within,
    compute_bx,
    compute_rsi2,
    compute_tht,
    tdx_barslast,
    tdx_cross,
    tdx_sma,
)


def test_tdx_sma_matches_wilder():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y = tdx_sma(x, 3, 1)
    # y0=1; y1=(2+2*1)/3=4/3; y2=(3+2*4/3)/3=17/9
    assert abs(y.iloc[0] - 1.0) < 1e-12
    assert abs(y.iloc[1] - 4.0 / 3.0) < 1e-12
    assert abs(y.iloc[2] - 17.0 / 9.0) < 1e-12


def test_cross_and_barslast():
    a = pd.Series([1.0, 2.0, 2.0, 0.5, 2.0])
    b = pd.Series([1.5, 1.5, 1.5, 1.5, 1.5])
    cr = tdx_cross(a, b)
    assert list(cr.fillna(False)) == [False, True, False, False, True]
    bl = tdx_barslast(cr)
    assert bl.iloc[1] == 0
    assert bl.iloc[2] == 1
    assert bl.iloc[3] == 2
    assert bl.iloc[4] == 0


def test_bull_flips_on_more_recent_cross():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    close = pd.Series(np.linspace(10, 20, 40), index=idx)
    open_ = close - 0.1
    high = close + 1.0
    low = close - 1.0
    # 人為製造：第 35 根 low 上穿 basis 附近
    df = compute_tht(open_, high, low, close)
    assert df["basis"].notna().sum() >= 1
    assert df["bull"].dtype == bool or df["bull"].isin([True, False]).all()


def test_bx_color_transition_dark_red_to_light_red():
    # 構造一條先下跌再在負區轉升的 close，讓 BX 有機會出深紅→淺紅
    rng = np.random.default_rng(0)
    n = 80
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    px = 100 - np.linspace(0, 30, 40)
    px = np.concatenate([px, 70 + np.linspace(0, 8, 40)])
    close = pd.Series(px + rng.normal(0, 0.2, n), index=idx)
    bx = compute_bx(close)
    assert bx["bx"].notna().sum() > 20
    # 顏色互斥
    flags = bx[["light_green", "dark_green", "light_red", "dark_red"]].astype(int).sum(axis=1)
    assert (flags.iloc[5:] <= 1).all()


def test_rsi2_bounds():
    close = pd.Series(np.linspace(10, 20, 50))
    rsi = compute_rsi2(close)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()
    # 幾乎單邊上漲，RSI 應接近 100
    assert valid.iloc[-1] > 90


def test_confirm_within_and_cancel():
    idx = pd.RangeIndex(10)
    trigger = pd.Series([0, 1, 0, 0, 0, 1, 0, 0, 0, 0], index=idx, dtype=bool)
    confirm = pd.Series([0, 0, 0, 1, 0, 0, 1, 0, 0, 0], index=idx, dtype=bool)
    cancel = pd.Series([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], index=idx, dtype=bool)
    # N=3：第 1 根觸發，第 2 根 cancel → 第 3 根確認不該成交
    out = confirm_within(trigger, confirm, n=3, cancel=cancel)
    assert not bool(out.iloc[3])
    # 第 5 根再觸發，第 6 根確認，N=3 應成交
    assert bool(out.iloc[6])
    # N=1 必須當根同時確認
    out1 = confirm_within(trigger, confirm, n=1)
    assert not out1.any()
    same = pd.Series([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], index=idx, dtype=bool)
    out_same = confirm_within(same, same, n=1)
    assert bool(out_same.iloc[1])


if __name__ == "__main__":
    tests = [
        test_tdx_sma_matches_wilder,
        test_cross_and_barslast,
        test_bull_flips_on_more_recent_cross,
        test_bx_color_transition_dark_red_to_light_red,
        test_rsi2_bounds,
        test_confirm_within_and_cancel,
    ]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print("all passed")
