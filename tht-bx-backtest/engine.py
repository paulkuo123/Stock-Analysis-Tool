"""單筆做多、次日開盤成交的回測引擎與績效統計。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252
ONE_WAY_COST = 0.0005  # 單邊 0.05%


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float  # 成交開盤價（未含成本）
    exit_date: pd.Timestamp
    exit_price: float
    bars_held: int
    pnl_pct: float  # 含成本後報酬
    reason_entry: str = ""
    reason_exit: str = ""


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    trades: list[Trade] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def _metrics_from_equity(
    equity: pd.Series,
    trades: list[Trade],
    bh_total: Optional[float] = None,
    initial: float = 1.0,
) -> dict:
    eq = equity.dropna()
    n_days = max(len(eq), 1)
    years = n_days / TRADING_DAYS
    last = float(eq.iloc[-1]) if len(eq) else initial
    total = last / initial - 1.0
    ann = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 and (1.0 + total) > 0 else np.nan
    if (1.0 + total) <= 0:
        ann = -1.0
    path = np.concatenate([[initial], eq.to_numpy(dtype=float)])
    peak = np.maximum.accumulate(path)
    dd = path / peak - 1.0
    maxdd = float(dd.min()) if len(dd) else 0.0
    daily = np.diff(path) / path[:-1]
    daily = daily[np.isfinite(daily)]
    if len(daily) and float(np.std(daily, ddof=1) or 0) > 0:
        sharpe = float(np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(TRADING_DAYS))
    else:
        sharpe = np.nan
    n_trades = len(trades)
    win_pnls = [t.pnl_pct for t in trades if t.pnl_pct > 0]
    lose_pnls = [t.pnl_pct for t in trades if t.pnl_pct <= 0]
    wins = len(win_pnls)
    win_rate = wins / n_trades if n_trades else np.nan
    avg_hold = float(np.mean([t.bars_held for t in trades])) if n_trades else np.nan
    avg_pnl = float(np.mean([t.pnl_pct for t in trades])) if n_trades else np.nan
    avg_win = float(np.mean(win_pnls)) if win_pnls else np.nan
    avg_loss = float(np.mean(lose_pnls)) if lose_pnls else np.nan
    if win_pnls and lose_pnls and abs(avg_loss) > 0:
        payoff = avg_win / abs(avg_loss)
    else:
        payoff = np.nan
    gp = float(np.sum(win_pnls)) if win_pnls else 0.0
    gl = float(abs(np.sum(lose_pnls))) if lose_pnls else 0.0
    profit_factor = (gp / gl) if gl > 0 else (np.inf if gp > 0 else np.nan)
    calmar = (ann / abs(maxdd)) if maxdd < 0 and pd.notna(ann) else np.nan
    excess = (total - bh_total) if bh_total is not None else np.nan
    return {
        "n_days": int(len(eq)),
        "years": round(years, 4),
        "total_return": total,
        "ann_return": ann,
        "maxdd": maxdd,
        "sharpe": sharpe,
        "calmar": calmar,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_hold_days": avg_hold,
        "avg_trade_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff": payoff,
        "profit_factor": profit_factor,
        "excess_vs_bh": excess,
    }


def run_long_only(
    df: pd.DataFrame,
    entry: pd.Series,
    exit_: pd.Series,
    name: str,
    cost: float = ONE_WAY_COST,
    analysis_mask: Optional[pd.Series] = None,
    initial_cash: float = 1.0,
) -> BacktestResult:
    """訊號於當日收盤成立，次一交易日開盤進出。結束時仍持倉則以最後收盤平倉。

    只做多、無槓桿、一次一筆；資金全進全出。
    """
    n = len(df)
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    dates = df.index
    ent = entry.reindex(df.index).fillna(False).to_numpy(dtype=bool)
    ex = exit_.reindex(df.index).fillna(False).to_numpy(dtype=bool)
    if analysis_mask is None:
        active = np.ones(n, dtype=bool)
    else:
        active = analysis_mask.reindex(df.index).fillna(False).to_numpy(dtype=bool)

    cash = initial_cash
    shares = 0.0
    in_pos = False
    pending_entry = False
    pending_exit = False
    entry_i = -1
    entry_px = np.nan
    equity = np.full(n, np.nan, dtype=float)
    trades: list[Trade] = []

    def close_trade(i: int, px: float, reason: str) -> None:
        nonlocal cash, shares, in_pos
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
                reason_exit=reason,
            )
        )

    for i in range(n):
        if pending_entry and not in_pos:
            px = o[i]
            if np.isfinite(px) and px > 0:
                entry_px = px
                entry_i = i
                shares = cash / (px * (1.0 + cost))
                cash = 0.0
                in_pos = True
            pending_entry = False
        elif pending_exit and in_pos:
            px = o[i]
            if np.isfinite(px) and px > 0:
                close_trade(i, px, "訊號次日開盤")
            pending_exit = False

        if in_pos:
            equity[i] = shares * c[i]
        else:
            equity[i] = cash

        if not active[i]:
            continue
        if not in_pos:
            pending_entry = bool(ent[i])
        else:
            pending_exit = bool(ex[i])

    if in_pos:
        last = n - 1
        close_trade(last, c[last], "期末收盤強制平倉")
        equity[last] = cash

    eq = pd.Series(equity, index=dates, name=name)
    # 分析區間外的權益維持起始資金，避免 warmup 影響曲線。
    if analysis_mask is not None:
        first = int(np.argmax(active)) if active.any() else 0
        eq.iloc[:first] = initial_cash
        # 若 warmup 期間未進場，確保銜接
        if not active[0]:
            eq = eq.copy()
            eq.iloc[:first] = initial_cash

    return BacktestResult(name=name, equity=eq, trades=trades)


def run_buy_and_hold(
    df: pd.DataFrame,
    name: str,
    cost: float = ONE_WAY_COST,
    analysis_mask: Optional[pd.Series] = None,
    initial_cash: float = 1.0,
) -> BacktestResult:
    """分析區間第一個交易日開盤買進，最後一日收盤賣出（同樣計單邊成本）。"""
    n = len(df)
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    dates = df.index
    if analysis_mask is None:
        active = np.ones(n, dtype=bool)
    else:
        active = analysis_mask.reindex(df.index).fillna(False).to_numpy(dtype=bool)

    first = int(np.argmax(active)) if active.any() else 0
    last = n - 1
    for j in range(n - 1, -1, -1):
        if active[j]:
            last = j
            break

    cash = initial_cash
    shares = 0.0
    equity = np.full(n, initial_cash, dtype=float)
    entry_px = o[first]
    shares = cash / (entry_px * (1.0 + cost))
    cash = 0.0
    for i in range(first, last + 1):
        equity[i] = shares * c[i]
    proceeds = shares * c[last] * (1.0 - cost)
    pnl = proceeds / (shares * entry_px * (1.0 + cost)) - 1.0
    equity[last] = proceeds
    for i in range(last + 1, n):
        equity[i] = proceeds

    trade = Trade(
        entry_date=dates[first],
        entry_price=float(entry_px),
        exit_date=dates[last],
        exit_price=float(c[last]),
        bars_held=int(last - first),
        pnl_pct=float(pnl),
        reason_entry="Buy&Hold",
        reason_exit="期末收盤",
    )
    eq = pd.Series(equity, index=dates, name=name)
    return BacktestResult(name=name, equity=eq, trades=[trade])


def run_long_only_ex(
    df: pd.DataFrame,
    entry: pd.Series,
    exit_: pd.Series,
    name: str,
    cost: float = ONE_WAY_COST,
    analysis_mask: Optional[pd.Series] = None,
    initial_cash: float = 1.0,
    stop_pct: Optional[float] = None,
    atr_stop_k: Optional[float] = None,
    trail_atr_k: Optional[float] = None,
    time_exit_bars: Optional[int] = None,
    cooldown_losses: Optional[int] = None,
    atr_col: str = "atr14",
) -> BacktestResult:
    """訊號次日開盤進出，可加固定／ATR 停損、移動停利、時間出場、連虧冷卻。

    停損／移動停利：若開盤跳空穿越則以開盤成交；否則若最低價打到則以停損價成交。
    ATR 用「進場當根的前一日」數值，避免用到當日尚未走完的波動。
    """
    n = len(df)
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    dates = df.index
    ent = entry.reindex(df.index).fillna(False).to_numpy(dtype=bool)
    ex = exit_.reindex(df.index).fillna(False).to_numpy(dtype=bool)
    if atr_col in df.columns:
        atr = df[atr_col].to_numpy(dtype=float)
    else:
        atr = np.full(n, np.nan, dtype=float)
    if analysis_mask is None:
        active = np.ones(n, dtype=bool)
    else:
        active = analysis_mask.reindex(df.index).fillna(False).to_numpy(dtype=bool)

    cash = initial_cash
    shares = 0.0
    in_pos = False
    pending_entry = False
    pending_exit = False
    pending_exit_reason = "訊號次日開盤"
    entry_i = -1
    entry_px = np.nan
    stop_px = np.nan
    peak_high = np.nan
    equity = np.full(n, np.nan, dtype=float)
    trades: list[Trade] = []
    loss_streak = 0
    skip_next = False

    def close_trade(i: int, px: float, reason: str) -> None:
        nonlocal cash, shares, in_pos, loss_streak, skip_next, stop_px, peak_high
        proceeds = shares * px * (1.0 - cost)
        pnl = proceeds / (shares * entry_px * (1.0 + cost)) - 1.0 if entry_px > 0 else 0.0
        cash = proceeds
        shares = 0.0
        in_pos = False
        stop_px = np.nan
        peak_high = np.nan
        trades.append(
            Trade(
                entry_date=dates[entry_i],
                entry_price=float(entry_px),
                exit_date=dates[i],
                exit_price=float(px),
                bars_held=int(i - entry_i),
                pnl_pct=float(pnl),
                reason_exit=reason,
            )
        )
        if pnl <= 0:
            loss_streak += 1
        else:
            loss_streak = 0
        if cooldown_losses is not None and loss_streak >= int(cooldown_losses):
            skip_next = True
            loss_streak = 0

    def current_stop() -> float:
        s = stop_px
        if trail_atr_k is not None and np.isfinite(peak_high):
            a = atr[entry_i - 1] if entry_i > 0 else atr[entry_i]
            if np.isfinite(a) and a > 0:
                trail = peak_high - float(trail_atr_k) * float(a)
                s = trail if not np.isfinite(s) else max(s, trail)
        return s

    for i in range(n):
        if pending_entry and not in_pos:
            px = o[i]
            if np.isfinite(px) and px > 0:
                entry_px = px
                entry_i = i
                shares = cash / (px * (1.0 + cost))
                cash = 0.0
                in_pos = True
                # 進場當根先用成交價當高點，避免用當日最高價立刻把移動停利拉太近。
                peak_high = px
                a_ref = atr[i - 1] if i > 0 else atr[i]
                hard = np.nan
                if stop_pct is not None:
                    hard = px * (1.0 - float(stop_pct))
                if atr_stop_k is not None and np.isfinite(a_ref) and a_ref > 0:
                    atr_hard = px - float(atr_stop_k) * float(a_ref)
                    hard = atr_hard if not np.isfinite(hard) else max(hard, atr_hard)
                stop_px = hard
            pending_entry = False
        elif pending_exit and in_pos:
            px = o[i]
            if np.isfinite(px) and px > 0:
                close_trade(i, px, pending_exit_reason)
            pending_exit = False
            pending_exit_reason = "訊號次日開盤"
        elif pending_exit and not in_pos:
            pending_exit = False
            pending_exit_reason = "訊號次日開盤"

        if in_pos:
            stp = current_stop()
            if np.isfinite(stp):
                if np.isfinite(o[i]) and o[i] <= stp:
                    close_trade(i, float(o[i]), "停損(開盤缺口)")
                elif np.isfinite(low[i]) and low[i] <= stp:
                    close_trade(i, float(stp), "停損(盤中)")

        if in_pos:
            if np.isfinite(h[i]):
                peak_high = h[i] if not np.isfinite(peak_high) else max(peak_high, h[i])
            equity[i] = shares * c[i]
        else:
            equity[i] = cash

        if not active[i]:
            continue
        if not in_pos:
            if ent[i]:
                if skip_next:
                    skip_next = False
                else:
                    pending_entry = True
        else:
            time_hit = time_exit_bars is not None and (i - entry_i) >= int(time_exit_bars)
            sig_hit = bool(ex[i])
            if time_hit or sig_hit:
                pending_exit = True
                pending_exit_reason = "時間出場次日開盤" if time_hit and not sig_hit else "訊號次日開盤"

    if in_pos:
        last = n - 1
        close_trade(last, c[last], "期末收盤強制平倉")
        equity[last] = cash

    eq = pd.Series(equity, index=dates, name=name)
    if analysis_mask is not None:
        first = int(np.argmax(active)) if active.any() else 0
        eq.iloc[:first] = initial_cash
        if not active[0]:
            eq = eq.copy()
            eq.iloc[:first] = initial_cash

    return BacktestResult(name=name, equity=eq, trades=trades)


def attach_metrics(result: BacktestResult, analysis_mask: pd.Series, bh_total: Optional[float] = None) -> BacktestResult:
    mask = analysis_mask.reindex(result.equity.index).fillna(False)
    eq = result.equity.loc[mask]
    # 保證第一點為 1（若策略尚未進場）
    if len(eq) and (not np.isfinite(eq.iloc[0]) or eq.iloc[0] == 0):
        eq = eq.copy()
        eq.iloc[0] = 1.0
    result.metrics = _metrics_from_equity(eq, result.trades, bh_total=bh_total)
    result.metrics["name"] = result.name
    return result


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_date",
                "entry_price",
                "exit_date",
                "exit_price",
                "bars_held",
                "pnl_pct",
                "reason_entry",
                "reason_exit",
            ]
        )
    rows = [
        {
            "entry_date": t.entry_date,
            "entry_price": t.entry_price,
            "exit_date": t.exit_date,
            "exit_price": t.exit_price,
            "bars_held": t.bars_held,
            "pnl_pct": t.pnl_pct,
            "reason_entry": t.reason_entry,
            "reason_exit": t.reason_exit,
        }
        for t in trades
    ]
    return pd.DataFrame(rows)
