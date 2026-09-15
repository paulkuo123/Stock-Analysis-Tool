"""v2 附加特徵：ATR、均線、Donchian、標準 RSI(14)。

THT／BX／RSI2 仍由 indicators.py 計算；這裡不改那些主參數。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import _falling_edge, _rising_edge, tdx_cross, tdx_ema, tdx_sma


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder ATR（通達信 SMA(TR,N,1)）。"""
    prev_c = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_c).abs(),
            (low - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tdx_sma(tr, n, 1)
    atr.name = f"atr{n}"
    return atr


def compute_rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    """標準 Wilder RSI(n)。經典對照用，不是庭安 RSI2。"""
    diff = close.diff()
    up = diff.clip(lower=0)
    dn = (-diff).clip(lower=0)
    au = tdx_sma(up, n, 1)
    ad = tdx_sma(dn, n, 1)
    rs = au / ad.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(ad != 0, 100.0)
    rsi = rsi.where(au != 0, 0.0)
    # 上下都為 0（平盤）時維持中性
    rsi = rsi.where((au != 0) | (ad != 0), 50.0)
    rsi.name = f"rsi{n}"
    return rsi


def confirm_after_bars(
    trigger: pd.Series,
    still_ok: pd.Series,
    offsets: tuple[int, ...] = (2, 3),
    cancel: pd.Series | None = None,
) -> pd.Series:
    """觸發後第 offsets 根（不含當根）若仍成立則給進場脈衝。

    例：offsets=(2,3) → 綠飄帶後第 2 或第 3 根確認。
    同一輪觸發只取最早一根確認；若中途 cancel（例如飄帶轉紅）則取消。
    """
    trig = trigger.fillna(False).to_numpy(dtype=bool)
    ok = still_ok.fillna(False).to_numpy(dtype=bool)
    canc = (
        cancel.fillna(False).to_numpy(dtype=bool)
        if cancel is not None
        else np.zeros(len(trig), dtype=bool)
    )
    out = np.zeros(len(trig), dtype=bool)
    pending_from = -1
    max_off = max(offsets) if offsets else 0
    offs = set(int(x) for x in offsets)
    for i in range(len(trig)):
        if pending_from >= 0:
            age = i - pending_from
            if canc[i]:
                pending_from = -1
            elif age in offs and ok[i]:
                out[i] = True
                pending_from = -1
            elif age > max_off:
                pending_from = -1
        if trig[i]:
            pending_from = i
            if 0 in offs and ok[i] and not canc[i]:
                out[i] = True
                pending_from = -1
    return pd.Series(out, index=trigger.index)


def add_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    """在已有 THT／BX／RSI2 的 DataFrame 上附加經典特徵。"""
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]

    out["atr14"] = compute_atr(high, low, close, 14)
    out["atr_pct"] = out["atr14"] / close.replace(0, np.nan)
    out["atr_pct_med20"] = out["atr_pct"].rolling(20, min_periods=10).median()
    out["atr_pct_q80_60"] = out["atr_pct"].rolling(60, min_periods=20).quantile(0.80)

    out["sma50"] = close.rolling(50, min_periods=50).mean()
    out["sma200"] = close.rolling(200, min_periods=200).mean()
    out["ema20"] = tdx_ema(close, 20)
    out["ema50"] = tdx_ema(close, 50)

    out["sma50_x_sma200"] = tdx_cross(out["sma50"], out["sma200"])
    out["sma200_x_sma50"] = tdx_cross(out["sma200"], out["sma50"])
    out["ema20_x_ema50"] = tdx_cross(out["ema20"], out["ema50"])
    out["ema50_x_ema20"] = tdx_cross(out["ema50"], out["ema20"])

    out["donch_up20"] = high.shift(1).rolling(20, min_periods=20).max()
    out["donch_dn10"] = low.shift(1).rolling(10, min_periods=10).min()
    out["donch_break_up"] = close > out["donch_up20"]
    out["donch_break_dn"] = close < out["donch_dn10"]
    out["donch_entry"] = _rising_edge(out["donch_break_up"])
    out["donch_exit"] = _rising_edge(out["donch_break_dn"])

    out["rsi14"] = compute_rsi_wilder(close, 14)
    lvl30 = pd.Series(30.0, index=out.index)
    out["rsi14_x_30"] = tdx_cross(out["rsi14"], lvl30)
    out["rsi14_gt_50"] = out["rsi14"] > 50.0
    out["rsi14_gt_70"] = out["rsi14"] > 70.0

    vol_ok = out["atr_pct"] <= out["atr_pct_q80_60"]
    above_sma50 = close > out["sma50"]
    # 只在進場濾波動；出場看均線是否跌破，避免波動偶爾飆高就洗出去。
    out["trend_vol_on"] = _rising_edge(above_sma50) & vol_ok.fillna(False)
    out["trend_vol_off"] = _falling_edge(above_sma50)

    if "trend" in out.columns:
        out["trend_up"] = (out["trend"] > out["trend"].shift(1)) & (out["trend"] > 0)
    else:
        out["trend_up"] = False

    out["up_day"] = close > out["open"]
    if "basis" in out.columns:
        out["above_basis"] = close > out["basis"]
    if "thup" in out.columns:
        out["above_up1"] = close > out["thup"]

    if "bull_on" in out.columns and "bull" in out.columns:
        still = out["bull"].astype(bool) & out["up_day"]
        cancel = out["bull_off"] if "bull_off" in out.columns else None
        out["confirm_23"] = confirm_after_bars(out["bull_on"], still, offsets=(2, 3), cancel=cancel)

    out["vol_compressed"] = out["atr_pct"] < out["atr_pct_med20"]
    return out
