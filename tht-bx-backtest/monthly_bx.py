"""月線 BX 對齊到日線：只用已收盤月份，不把當月未收盤的結果提前用上。

BX 本體沿用 indicators.compute_bx（SL1=5, SL2=20, SL3=5）。
顏色只用「正負」與「跟前一根月線比升降」，不再加絕對值門檻（那會變成新參數）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import compute_bx

# 官方 Trade Map 允許做多的月線顏色。持平負區不算「漸增」。
CYCLE_OK_COLORS = ("淺綠", "深綠", "漸增淺紅")
GREEN_COLORS = ("淺綠", "深綠")


def bx_color_name(bx: float, prev: float) -> str | None:
    """單根月線 BX 的顏色。前一根無效時不著色。

    - 淺綠：BX ≥ 0 且 BX ≥ 前一根（正區上升或持平）。持有。
    - 深綠：BX ≥ 0 且 BX < 前一根（正區下降）。持有。
    - 漸增淺紅：BX < 0 且 BX > 前一根（負區嚴格變高，買盤回溫）。週期仍有效。
    - 持平負區：BX < 0 且 BX == 前一根。不是漸增，也不是深紅；不進場、不出場。
    - 深紅：BX < 0 且 BX < 前一根（負區再變差）。多頭週期結束，出場。

    正區持平沿用既有 compute_bx：`bx >= prev` 算淺綠。
    負區持平不沿用「淺紅＝可進場」，因為官方寫的是「漸增」淺紅。
    """
    if not np.isfinite(bx) or not np.isfinite(prev):
        return None
    if bx >= 0 and bx >= prev:
        return "淺綠"
    if bx >= 0 and bx < prev:
        return "深綠"
    if bx < 0 and bx > prev:
        return "漸增淺紅"
    if bx < 0 and bx == prev:
        return "持平負區"
    if bx < 0 and bx < prev:
        return "深紅"
    return None


def completed_monthly_closes(close: pd.Series) -> pd.Series:
    """每個日曆月的最後一根日線收盤，且該月之後還有下個月的資料。

    整段資料的最後一個月視為尚未收盤（例如資料停在 9/14，就不能把 9 月當月線收盤）。
    索引是該月最後一個交易日的日期。
    """
    if close.empty:
        return close.copy()
    px = close.astype(float).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    periods = px.index.to_period("M")
    last_period = periods[-1]
    last_bars = px.groupby(periods).tail(1)
    done = last_bars.index.to_period("M") < last_period
    out = last_bars.loc[done]
    out.name = "monthly_close"
    return out


def monthly_bx_table(close: pd.Series) -> pd.DataFrame:
    """在已收盤月線收盤價上計算 BX 與顏色。一根月線一列。"""
    monthly_close = completed_monthly_closes(close)
    if monthly_close.empty:
        return pd.DataFrame(
            columns=["monthly_close", "bx", "prev_bx", "color", "cycle_ok", "dark_red"]
        )
    bx_df = compute_bx(monthly_close)
    prev = bx_df["bx"].shift(1)
    colors = [
        bx_color_name(float(b), float(p)) if np.isfinite(b) and np.isfinite(p) else None
        for b, p in zip(bx_df["bx"].to_numpy(), prev.to_numpy())
    ]
    table = pd.DataFrame(
        {
            "monthly_close": monthly_close.to_numpy(),
            "bx": bx_df["bx"].to_numpy(),
            "prev_bx": prev.to_numpy(),
            "color": colors,
        },
        index=monthly_close.index,
    )
    table["cycle_ok"] = table["color"].isin(CYCLE_OK_COLORS)
    table["dark_red"] = table["color"].eq("深紅")
    table["increasing_light_red"] = table["color"].eq("漸增淺紅")
    table["green"] = table["color"].isin(GREEN_COLORS)
    return table


def stitch_close(cache: pd.DataFrame, early: pd.DataFrame) -> tuple[pd.Series, str]:
    """日線策略沿用 PR #3 快取；更早的月線暖機若和快取有價差，先按重疊收盤的中位數比例縮放。

    BX 對價格水準的整體倍數不敏感，但接縫上若差一個跳空，會污染接縫之後幾個月的 BX。
    重疊日期一律採用快取，不改 PR #3 的日K。
    """
    cache = cache.sort_index()
    early = early.sort_index()
    overlap = cache.index.intersection(early.index)
    note = "沒有早段行情"
    scale = 1.0
    early_use = early
    if len(overlap) >= 5:
        ratio = (cache.loc[overlap, "close"] / early.loc[overlap, "close"]).replace([np.inf, -np.inf], np.nan)
        scale = float(ratio.median())
        if np.isfinite(scale) and abs(scale - 1.0) > 0.002:
            early_use = early.copy()
            for col in ("open", "high", "low", "close"):
                if col in early_use.columns:
                    early_use[col] = early_use[col] * scale
            note = f"早段收盤依重疊中位數比例 {scale:.6f} 銜接快取"
        else:
            note = f"早段與快取重疊中位數比 {scale:.6f}，未縮放"
    elif len(early):
        note = "早段與快取重疊不足，未縮放"
    early_only = early_use.loc[early_use.index < cache.index.min(), "close"]
    close = pd.concat([early_only, cache["close"]]).sort_index()
    close = close[~close.index.duplicated(keep="last")]
    close.name = "close"
    return close, note


def align_monthly_bx_to_daily(close: pd.Series) -> pd.DataFrame:
    """把已收盤月線顏色貼到每一根日線。

    規則（收盤才決策，沒有用到未來月份）：
    - 某日曆月的顏色，要等該月最後一根日線收盤才知道。
    - 從「該月最後一根」的收盤決策開始，一直用到下一個已收盤月份的最後一根（含）之前。
    - 月中的日子只用前一個已收盤月，不用本月至今的收盤去冒充月線。
    - 資料尾端那個還沒有下個月的月份，整月都不產生顏色。
    """
    table = monthly_bx_table(close)
    empty_cols = [
        "m_bx",
        "m_prev_bx",
        "m_color",
        "m_month_end",
        "m_cycle_ok",
        "m_dark_red",
        "m_increasing_light_red",
        "m_green",
    ]
    if table.empty:
        out = pd.DataFrame(index=close.index, columns=empty_cols)
        for col in ("m_cycle_ok", "m_dark_red", "m_increasing_light_red", "m_green"):
            out[col] = False
        return out

    # 還沒比對前一根、無法著色的月份不往日線送，避免把空值當成一種顏色。
    publish = table.loc[table["color"].notna()].copy()
    if publish.empty:
        out = pd.DataFrame(index=close.index, columns=empty_cols)
        for col in ("m_cycle_ok", "m_dark_red", "m_increasing_light_red", "m_green"):
            out[col] = False
        return out

    full_index = close.index.union(publish.index).sort_values()
    placed = pd.DataFrame(index=full_index)
    placed["m_bx"] = publish["bx"].reindex(full_index)
    placed["m_prev_bx"] = publish["prev_bx"].reindex(full_index)
    placed["m_color"] = pd.Series(publish["color"].to_numpy(), index=publish.index, dtype=object).reindex(full_index)
    placed["m_month_end"] = pd.Series(publish.index, index=publish.index).reindex(full_index)
    filled = placed.ffill()
    out = filled.reindex(close.index)
    color = out["m_color"]
    out["m_cycle_ok"] = color.isin(CYCLE_OK_COLORS).fillna(False)
    out["m_dark_red"] = color.eq("深紅").fillna(False)
    out["m_increasing_light_red"] = color.eq("漸增淺紅").fillna(False)
    out["m_green"] = color.isin(list(GREEN_COLORS)).fillna(False)
    return out
