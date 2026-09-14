"""通達信 THT / BX / RSI2 指標（pandas／numpy 實作）。

SMA(X,N,1) 採通達信加權平滑：Y = (X + (N-1)*Y') / N，與 Wilder RMA 等價，
並非簡單移動平均。EMA 採通達信標準 alpha=2/(N+1)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 從未出現過的條件，BARSLAST 給一個很大的距離，讓比較仍成立。
_NEVER = 10**9


def tdx_sma(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """通達信 SMA(X, N, M)：Y = (M*X + (N-M)*Y') / N。

    第一個有效值當作種子。中間若遇到 NaN 則略過、維持前值。
    """
    if n <= 0:
        raise ValueError("SMA 週期 N 必須為正整數")
    x = series.to_numpy(dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    prev = np.nan
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        if np.isnan(prev):
            prev = float(v)
        else:
            prev = (m * float(v) + (n - m) * prev) / n
        out[i] = prev
    return pd.Series(out, index=series.index, name=series.name)


def tdx_ema(series: pd.Series, n: int) -> pd.Series:
    """通達信 EMA(X, N)，alpha = 2/(N+1)，adjust=False、以首值為種子。"""
    return series.ewm(span=n, adjust=False, min_periods=1).mean()


def tdx_ma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def tdx_std(series: pd.Series, n: int) -> pd.Series:
    """通達信 STD：母體標準差（ddof=0），等價 SQRT(MA(X^2,N)-MA(X,N)^2)。"""
    return series.rolling(n, min_periods=n).std(ddof=0)


def tdx_cross(a: pd.Series, b: pd.Series) -> pd.Series:
    """CROSS(A,B)：A 上穿 B（前一日 A<=B，當日 A>B）。"""
    a = a.astype(float)
    b = b.astype(float)
    return (a > b) & (a.shift(1) <= b.shift(1))


def tdx_barslast(cond: pd.Series) -> pd.Series:
    """BARSLAST(X)：距離上一次 X 為真的 K 棒數；當日為真則為 0。"""
    flags = cond.fillna(False).to_numpy(dtype=bool)
    out = np.empty(len(flags), dtype=float)
    last = -_NEVER
    for i, flag in enumerate(flags):
        if flag:
            last = i
        out[i] = i - last
    return pd.Series(out, index=cond.index)


def compute_tht(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 33,
    tw: float = 0.18,
) -> pd.DataFrame:
    """THT：N=33, TW=0.18。W1/W2 公式有宣告但未參與計算，這裡不使用。

    BULL := BARSLAST(CROSS(LOW,BASIS)) < BARSLAST(CROSS(BASIS,HIGH))
    進場關鍵：BULL 由假轉真（0→1）。
    """
    ohlc4 = (open_ + high + low + close) / 4.0
    basis = tdx_ma(ohlc4, n)
    dev = tdx_std(ohlc4, n)
    thup = basis + dev * tw
    thdn = basis - dev * tw

    cross_low_basis = tdx_cross(low, basis)
    cross_basis_high = tdx_cross(basis, high)
    bars_bullish = tdx_barslast(cross_low_basis)
    bars_bearish = tdx_barslast(cross_basis_high)
    bull = bars_bullish < bars_bearish
    # 指標尚未形成（BASIS 為 NaN）時不當成綠飄帶。
    bull = bull & basis.notna()

    prev_bull = bull.shift(1).fillna(False)
    bull_on = bull & ~prev_bull
    bull_off = ~bull & prev_bull

    return pd.DataFrame(
        {
            "ohlc4": ohlc4,
            "basis": basis,
            "dev": dev,
            "thup": thup,
            "thdn": thdn,
            "cross_low_basis": cross_low_basis.fillna(False),
            "cross_basis_high": cross_basis_high.fillna(False),
            "bull": bull,
            "bull_on": bull_on,
            "bull_off": bull_off,
        },
        index=open_.index,
    )


def compute_bx(close: pd.Series, sl1: int = 5, sl2: int = 20, sl3: int = 5) -> pd.DataFrame:
    """BX：只用 SL1=5, SL2=20, SL3=5，不用 SHORT/LONG/M/P1–P3。

    SHORTX:=EMA(CLOSE,SL1)-EMA(CLOSE,SL2)
    UP:=MAX(SHORTX-REF(SHORTX,1),0); DN:=ABS(SHORTX-REF(SHORTX,1))
    RS:=IF(SMA(DN,SL3,1)=0,50,SMA(UP,SL3,1)/SMA(DN,SL3,1)*100)
    BXRAW:=RS-50; BX:=EMA(BXRAW,3); TREND:=EMA(BX,2)
    """
    shortx = tdx_ema(close, sl1) - tdx_ema(close, sl2)
    delta = shortx - shortx.shift(1)
    up = delta.clip(lower=0)
    dn = delta.abs()
    sma_up = tdx_sma(up, sl3, 1)
    sma_dn = tdx_sma(dn, sl3, 1)
    rs = pd.Series(np.where(sma_dn.to_numpy() == 0, 50.0, sma_up / sma_dn * 100.0), index=close.index)
    bxraw = rs - 50.0
    bx = tdx_ema(bxraw, 3)
    trend = tdx_ema(bx, 2)
    prev = bx.shift(1)

    light_green = (bx >= 0) & (bx >= prev)  # 正區上升
    dark_green = (bx >= 0) & (bx < prev)  # 正區下降
    light_red = (bx < 0) & (bx >= prev)  # 負區上升
    dark_red = (bx < 0) & (bx < prev)  # 負區下降

    prev_lg = light_green.shift(1).fillna(False)
    prev_dg = dark_green.shift(1).fillna(False)
    prev_lr = light_red.shift(1).fillna(False)
    prev_dr = dark_red.shift(1).fillna(False)

    # 深紅→淺紅：負區內由下降轉為上升（買盤加強）
    dr_to_lr = prev_dr & light_red
    # 深綠→淺綠：正區內由下降轉為上升（SPEC 指定作出場；見報告假設）
    dg_to_lg = prev_dg & light_green
    # 淺綠→深綠：正區內由上升轉為下降（對照：較接近「轉弱」）
    lg_to_dg = prev_lg & dark_green
    # 淺紅→深紅：負區內再轉弱
    lr_to_dr = prev_lr & dark_red
    # 零軸穿越對照
    neg_to_pos = (prev < 0) & (bx >= 0)
    pos_to_neg = (prev >= 0) & (bx < 0)

    color = pd.Series(np.nan, index=close.index, dtype=object)
    color = color.mask(light_green, "淺綠")
    color = color.mask(dark_green, "深綠")
    color = color.mask(light_red, "淺紅")
    color = color.mask(dark_red, "深紅")

    return pd.DataFrame(
        {
            "shortx": shortx,
            "bxraw": bxraw,
            "bx": bx,
            "trend": trend,
            "light_green": light_green.fillna(False),
            "dark_green": dark_green.fillna(False),
            "light_red": light_red.fillna(False),
            "dark_red": dark_red.fillna(False),
            "color": color,
            "dr_to_lr": dr_to_lr.fillna(False),
            "dg_to_lg": dg_to_lg.fillna(False),
            "lg_to_dg": lg_to_dg.fillna(False),
            "lr_to_dr": lr_to_dr.fillna(False),
            "neg_to_pos": neg_to_pos.fillna(False),
            "pos_to_neg": pos_to_neg.fillna(False),
        },
        index=close.index,
    )


def compute_rsi2(close: pd.Series, p2: int = 12) -> pd.Series:
    """SMA 型 RSI2（P2=12）。TEMP1=上漲、TEMP2=絕對漲跌。不用 P3、不假設 P1。"""
    diff = close - close.shift(1)
    temp1 = diff.clip(lower=0)
    temp2 = diff.abs()
    sma1 = tdx_sma(temp1, p2, 1)
    sma2 = tdx_sma(temp2, p2, 1)
    rsi = sma1 / sma2.replace(0, np.nan) * 100.0
    rsi = rsi.where(sma2 != 0, 50.0)
    rsi.name = "rsi2"
    return rsi


def confirm_within(
    trigger: pd.Series,
    confirm: pd.Series,
    n: int,
    cancel: pd.Series | None = None,
) -> pd.Series:
    """觸發後（含當根）N 根內出現確認訊號才為真。

    若提供 cancel（例如 BULL 轉假），視窗在確認前結束則取消。
    確認當根即輸出進場訊號，視窗隨即關閉。
    """
    if n < 1:
        raise ValueError("N 必須 >= 1")
    trig = trigger.fillna(False).to_numpy(dtype=bool)
    conf = confirm.fillna(False).to_numpy(dtype=bool)
    canc = (
        cancel.fillna(False).to_numpy(dtype=bool)
        if cancel is not None
        else np.zeros(len(trig), dtype=bool)
    )
    out = np.zeros(len(trig), dtype=bool)
    window_end = -1
    for i in range(len(trig)):
        if window_end >= i and canc[i]:
            window_end = -1
        if trig[i]:
            window_end = i + n - 1
        if i <= window_end and conf[i]:
            out[i] = True
            window_end = -1
    return pd.Series(out, index=trigger.index)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """在 OHLCV DataFrame 上附加 THT／BX／RSI2 欄位。"""
    out = df.copy()
    tht = compute_tht(out["open"], out["high"], out["low"], out["close"])
    bx = compute_bx(out["close"])
    rsi2 = compute_rsi2(out["close"])
    out = pd.concat([out, tht, bx], axis=1)
    out["rsi2"] = rsi2
    return out
