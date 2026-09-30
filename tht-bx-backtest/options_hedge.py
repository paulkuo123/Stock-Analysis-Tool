"""月線深紅時用選擇權對沖，不減碼賣股。

這是估價模擬，不是選擇權歷史成交。本機只有日K，沒有選擇權報價。
權利金用 Black-Scholes 中價，波動度用該股過去 21 個交易日的已實現波動。

規則在寫程式時固定，不依結果改：
- 只有月線變成深紅才開新倉。
- 之後若是漸增淺紅或持平負區，對沖留到回綠。單獨的漸增淺紅不開新倉。
- 回綠（淺綠或深綠）當天收盤把選擇權用理論價平掉。
- 天期 21 個交易日。買權履約價使開倉當下的 delta 等於 0.30。賣權履約價是股價的 90%。
- 覆蓋全部持股。到期用現金結算，股票不交割。利率 0、股利 0、沒有買賣價差、沒有微笑。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

from engine import ONE_WAY_COST, TRADING_DAYS, _metrics_from_equity
from sizing import GREEN, _color_at

_NORM = NormalDist()
TENOR_DAYS = 21
CALL_DELTA = 0.30
PUT_OTM = 0.10
VOL_WINDOW = 21
VOL_FLOOR = 1e-4


def trailing_realized_vol(close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """到當天收盤為止的已實現波動，年化。不含未來報酬。"""
    rets = np.log(close.astype(float)).diff()
    return rets.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def bs_call(spot: float, strike: float, tenor: float, sigma: float, rate: float = 0.0) -> float:
    if spot <= 0 or strike <= 0:
        return 0.0
    if tenor <= 1e-12 or sigma <= 1e-8:
        return max(spot - strike * exp(-rate * max(tenor, 0.0)), 0.0)
    sig = max(float(sigma), 1e-8)
    vol_sqrt = sig * sqrt(tenor)
    d1 = (log(spot / strike) + (rate + 0.5 * sig * sig) * tenor) / vol_sqrt
    d2 = d1 - vol_sqrt
    return float(spot * exp(-rate * tenor) * _NORM.cdf(d1) - strike * exp(-rate * tenor) * _NORM.cdf(d2))


def bs_put(spot: float, strike: float, tenor: float, sigma: float, rate: float = 0.0) -> float:
    call = bs_call(spot, strike, tenor, sigma, rate)
    return float(call - spot * exp(-rate * tenor) + strike * exp(-rate * tenor))


def strike_for_call_delta(
    spot: float,
    sigma: float,
    tenor: float,
    delta: float = CALL_DELTA,
    rate: float = 0.0,
) -> float:
    """解出使買權 delta 等於目標的履約價。delta 0.30 是價外買權。"""
    sig = max(float(sigma), VOL_FLOOR)
    if spot <= 0 or tenor <= 0:
        return float(spot)
    d1 = _NORM.inv_cdf(delta)
    log_s_over_k = d1 * sig * sqrt(tenor) - (rate + 0.5 * sig * sig) * tenor
    return float(spot / exp(log_s_over_k))


def hedge_active(colors: pd.Series) -> pd.Series:
    """深紅打開對沖；回綠關掉。中間的漸增淺紅、持平負區維持原狀態。"""
    on = False
    out = np.empty(len(colors), dtype=bool)
    for i, raw in enumerate(colors.tolist()):
        color = _color_at(raw)
        if color in GREEN:
            on = False
        elif color == "深紅":
            on = True
        out[i] = on
    return pd.Series(out, index=colors.index, name="hedge_on")


@dataclass
class OverlayResult:
    name: str
    equity: pd.Series
    premium_in: float
    premium_out: float
    n_called: int
    n_early_itm: int
    n_opened_call: int
    n_opened_put: int
    avg_call_yield: float
    avg_put_yield: float
    metrics: dict


def _vol_at(vol: np.ndarray, i: int, fallback: float | None) -> float | None:
    value = vol[i] if i < len(vol) else np.nan
    if np.isfinite(value) and value > 0:
        return float(max(value, VOL_FLOOR))
    if fallback is not None and fallback > 0:
        return float(max(fallback, VOL_FLOOR))
    return None


def run_overlay(
    df: pd.DataFrame,
    hedge_on: pd.Series,
    kind: str,
    analysis_mask: pd.Series,
    name: str,
    vol: pd.Series | None = None,
    tenor: int = TENOR_DAYS,
    cost: float = ONE_WAY_COST,
    initial_cash: float = 1.0,
    bh_total: float | None = None,
) -> OverlayResult:
    """股票全段持有。選擇權只在對沖期間覆蓋全部股數，收盤估價、到期現金結算。"""
    if kind not in ("call", "put", "collar"):
        raise ValueError(f"未知的對沖種類：{kind}")
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    n = len(df)
    active = analysis_mask.reindex(df.index).fillna(False).to_numpy(dtype=bool)
    if not active.any():
        raise RuntimeError(f"{name} 沒有分析區間")
    first = int(np.argmax(active))
    last = int(n - 1 - np.argmax(active[::-1]))
    vol_s = trailing_realized_vol(df["close"]) if vol is None else vol.reindex(df.index)
    vol_a = vol_s.to_numpy(dtype=float)
    hedge = hedge_on.reindex(df.index).fillna(False).to_numpy(dtype=bool)

    cash = initial_cash
    shares = cash / (open_[first] * (1.0 + cost))
    cash = 0.0
    equity = np.full(n, np.nan, dtype=float)
    call: dict | None = None
    put: dict | None = None
    premium_in = 0.0
    premium_out = 0.0
    n_called = 0
    n_early_itm = 0
    n_opened_call = 0
    n_opened_put = 0
    call_yields: list[float] = []
    put_yields: list[float] = []

    def leg_price(is_call: bool, i: int, leg: dict, early: bool) -> float:
        days_left = int(leg["expiry"] - i)
        if (not early) or days_left <= 0:
            if is_call:
                return max(close[i] - leg["K"], 0.0)
            return max(leg["K"] - close[i], 0.0)
        sigma = _vol_at(vol_a, i, leg["vol0"])
        tenor_y = days_left / TRADING_DAYS
        if sigma is None:
            if is_call:
                return max(close[i] - leg["K"], 0.0)
            return max(leg["K"] - close[i], 0.0)
        if is_call:
            return bs_call(close[i], leg["K"], tenor_y, sigma)
        return bs_put(close[i], leg["K"], tenor_y, sigma)

    def open_call(i: int) -> None:
        nonlocal cash, premium_in, n_opened_call, call
        sigma = _vol_at(vol_a, i, None)
        if sigma is None or close[i] <= 0:
            return
        tenor_y = tenor / TRADING_DAYS
        strike = strike_for_call_delta(close[i], sigma, tenor_y, CALL_DELTA)
        prem = bs_call(close[i], strike, tenor_y, sigma)
        cash += prem * shares
        premium_in += prem * shares
        call_yields.append(prem / close[i])
        n_opened_call += 1
        call = {"K": strike, "expiry": i + tenor, "vol0": sigma, "qty": shares}

    def open_put(i: int) -> None:
        nonlocal cash, premium_out, n_opened_put, put
        sigma = _vol_at(vol_a, i, None)
        if sigma is None or close[i] <= 0:
            return
        tenor_y = tenor / TRADING_DAYS
        strike = close[i] * (1.0 - PUT_OTM)
        prem = bs_put(close[i], strike, tenor_y, sigma)
        cash -= prem * shares
        premium_out += prem * shares
        put_yields.append(prem / close[i])
        n_opened_put += 1
        put = {"K": strike, "expiry": i + tenor, "vol0": sigma, "qty": shares}

    def settle_call(i: int, early: bool) -> None:
        nonlocal cash, n_called, n_early_itm, call
        if call is None:
            return
        price = leg_price(True, i, call, early)
        cash -= price * call["qty"]
        if close[i] > call["K"]:
            if early:
                n_early_itm += 1
            else:
                n_called += 1
        call = None

    def settle_put(i: int, early: bool) -> None:
        nonlocal cash, put
        if put is None:
            return
        price = leg_price(False, i, put, early)
        cash += price * put["qty"]
        put = None

    def mark(i: int) -> float:
        call_liab = 0.0
        put_asset = 0.0
        if call is not None:
            call_liab = leg_price(True, i, call, early=True) * call["qty"]
        if put is not None:
            put_asset = leg_price(False, i, put, early=True) * put["qty"]
        return cash + shares * close[i] - call_liab + put_asset

    want_call = kind in ("call", "collar")
    want_put = kind in ("put", "collar")

    for i in range(first, last + 1):
        if call is not None and i >= call["expiry"]:
            settle_call(i, early=False)
        elif call is not None and not hedge[i]:
            settle_call(i, early=True)
        if put is not None and i >= put["expiry"]:
            settle_put(i, early=False)
        elif put is not None and not hedge[i]:
            settle_put(i, early=True)

        if i == last:
            if call is not None:
                settle_call(i, early=call["expiry"] > i)
            if put is not None:
                settle_put(i, early=put["expiry"] > i)
            cash += shares * close[i] * (1.0 - cost)
            equity[i] = cash
            continue

        if hedge[i]:
            if want_call and call is None:
                open_call(i)
            if want_put and put is None:
                open_put(i)
        equity[i] = mark(i)

    eq = pd.Series(equity, index=df.index, name=name)
    window = eq.loc[active].dropna()
    metrics = _metrics_from_equity(window, [], bh_total=bh_total, initial=initial_cash)
    return OverlayResult(
        name=name,
        equity=eq,
        premium_in=float(premium_in),
        premium_out=float(premium_out),
        n_called=int(n_called),
        n_early_itm=int(n_early_itm),
        n_opened_call=int(n_opened_call),
        n_opened_put=int(n_opened_put),
        avg_call_yield=float(np.mean(call_yields)) if call_yields else np.nan,
        avg_put_yield=float(np.mean(put_yields)) if put_yields else np.nan,
        metrics=metrics,
    )
