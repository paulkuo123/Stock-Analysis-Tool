"""官方 Trade Map 的兩種回測。

變體 A：進場維持 V_CONF23，出場改成「已收盤月線 BX 變深紅」。
變體 B（主要結果）：月線多頭確認 + 收盤回到 33 日公允價值帶才進 + 進場前鎖定下軌當失效價。

成交慣例與 PR #3 相同：訊號收盤成立，次一交易日開盤成交；單邊成本 0.05%；只做多、一次一筆。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine import ONE_WAY_COST, BacktestResult, Trade
from strategies import _and


@dataclass
class TradeMapResult:
    result: BacktestResult
    details: pd.DataFrame


def fair_value_touch(df: pd.DataFrame) -> pd.Series:
    """收盤落在 33 FVB 公允價值帶裡面：下軌 ≤ 收盤 ≤ 上軌。

    帶的定義沿用既有 THT：中軌為 33 日 OHLC4 均線，上下軌各加減 0.18 倍標準差。
    收盤在上軌之外視為離公允價值太遠，不追。收盤跌破下軌也不當「回到帶內」。
    """
    thdn = df["thdn"]
    thup = df["thup"]
    close = df["close"]
    inside = thdn.notna() & thup.notna() & (close >= thdn) & (close <= thup)
    return inside.fillna(False)


def variant_a_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """進場照 V_CONF23；出場只看月線深紅。沒有硬停損。"""
    entry = df["confirm_23"].fillna(False).astype(bool)
    exit_ = df["m_dark_red"].fillna(False).astype(bool)
    invalidation = pd.Series(np.nan, index=df.index)
    return entry, exit_, invalidation


def variant_b_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """完整官方版。

    可做多（多頭週期）要同時：
    - 日線 33 FVB 為綠色（既有 BULL）
    - 已收盤月線 BX 為綠，或漸增淺紅

    進場：上述成立，且收盤回到公允價值帶。用狀態由假轉真的那一根，避免在帶內每天重複進。
    出場：月線變深紅。另外把訊號當根的下軌鎖成失效價，之後收盤跌破就出場。
    持倉中若只是日線飄帶轉紅、或月線仍是淺紅，不單獨出場。
    """
    cycle = _and(df["bull"].astype(bool), df["m_cycle_ok"].astype(bool), fair_value_touch(df))
    prev = cycle.shift(1)
    prev = prev.where(prev.notna(), False).astype(bool)
    entry = cycle & ~prev
    exit_ = df["m_dark_red"].fillna(False).astype(bool)
    invalidation = df["thdn"].where(entry)
    return entry, exit_, invalidation


def run_trade_map(
    df: pd.DataFrame,
    entry: pd.Series,
    exit_: pd.Series,
    invalidation: pd.Series,
    name: str,
    cost: float = ONE_WAY_COST,
    analysis_mask: pd.Series | None = None,
    initial_cash: float = 1.0,
) -> TradeMapResult:
    """次日開盤進出。硬停損看收盤跌破，不看盤中最低價；跳空開盤已跌破則這筆不進。"""
    n = len(df)
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    dates = df.index
    ent = entry.reindex(df.index).fillna(False).to_numpy(dtype=bool)
    ex = exit_.reindex(df.index).fillna(False).to_numpy(dtype=bool)
    inv = invalidation.reindex(df.index).to_numpy(dtype=float)
    if analysis_mask is None:
        active = np.ones(n, dtype=bool)
    else:
        active = analysis_mask.reindex(df.index).fillna(False).to_numpy(dtype=bool)

    cash = initial_cash
    shares = 0.0
    in_pos = False
    pending_entry = False
    pending_exit = False
    pending_exit_reason = ""
    signal_stop = np.nan
    locked_stop = np.nan
    entry_i = -1
    entry_px = np.nan
    signal_i = -1
    equity = np.full(n, np.nan, dtype=float)
    trades: list[Trade] = []
    detail_rows: list[dict] = []

    def close_trade(i: int, px: float, reason: str) -> None:
        nonlocal cash, shares, in_pos, locked_stop
        proceeds = shares * px * (1.0 - cost)
        pnl = proceeds / (shares * entry_px * (1.0 + cost)) - 1.0 if entry_px > 0 else 0.0
        cash = proceeds
        shares = 0.0
        in_pos = False
        trades.append(
            Trade(
                entry_date=dates[entry_i],
                entry_price=float(entry_px),
                exit_date=dates[i],
                exit_price=float(px),
                bars_held=int(i - entry_i),
                pnl_pct=float(pnl),
                reason_entry="訊號次日開盤",
                reason_exit=reason,
            )
        )
        detail_rows.append(
            {
                "signal_date": dates[signal_i],
                "entry_date": dates[entry_i],
                "entry_price": float(entry_px),
                "exit_date": dates[i],
                "exit_price": float(px),
                "bars_held": int(i - entry_i),
                "pnl_pct": float(pnl),
                "invalidation": float(locked_stop) if np.isfinite(locked_stop) else np.nan,
                "reason_exit": reason,
                "signal_m_color": df["m_color"].iloc[signal_i] if "m_color" in df.columns else None,
                "exit_m_color": df["m_color"].iloc[i] if "m_color" in df.columns else None,
            }
        )
        locked_stop = np.nan

    for i in range(n):
        if pending_entry and not in_pos:
            px = o[i]
            stop = signal_stop
            if np.isfinite(px) and px > 0:
                if np.isfinite(stop) and px < stop:
                    detail_rows.append(
                        {
                            "signal_date": dates[signal_i],
                            "entry_date": pd.NaT,
                            "entry_price": np.nan,
                            "exit_date": dates[i],
                            "exit_price": float(px),
                            "bars_held": 0,
                            "pnl_pct": 0.0,
                            "invalidation": float(stop),
                            "reason_exit": "開盤已跌破失效價，取消進場",
                            "signal_m_color": df["m_color"].iloc[signal_i] if "m_color" in df.columns else None,
                            "exit_m_color": df["m_color"].iloc[i] if "m_color" in df.columns else None,
                        }
                    )
                else:
                    entry_px = px
                    entry_i = i
                    shares = cash / (px * (1.0 + cost))
                    cash = 0.0
                    in_pos = True
                    locked_stop = stop
            pending_entry = False
            signal_stop = np.nan
        elif pending_exit and in_pos:
            px = o[i]
            if np.isfinite(px) and px > 0:
                close_trade(i, px, pending_exit_reason)
            pending_exit = False
            pending_exit_reason = ""

        if in_pos:
            equity[i] = shares * c[i]
        else:
            equity[i] = cash

        if not active[i]:
            continue

        if in_pos:
            reasons = []
            if bool(ex[i]):
                reasons.append("月線深紅")
            if np.isfinite(locked_stop) and np.isfinite(c[i]) and c[i] < locked_stop:
                reasons.append("收盤跌破失效價")
            if reasons:
                pending_exit = True
                pending_exit_reason = "＋".join(reasons) + "，次日開盤"
        elif ent[i]:
            pending_entry = True
            signal_i = i
            signal_stop = float(inv[i]) if np.isfinite(inv[i]) else np.nan

    if in_pos:
        last = n - 1
        close_trade(last, c[last], "期末收盤強制平倉")
        equity[last] = cash

    eq = pd.Series(equity, index=dates, name=name)
    if analysis_mask is not None:
        first = int(np.argmax(active)) if active.any() else 0
        eq = eq.copy()
        eq.iloc[:first] = initial_cash

    result = BacktestResult(name=name, equity=eq, trades=trades)
    details = pd.DataFrame(detail_rows)
    return TradeMapResult(result=result, details=details)
