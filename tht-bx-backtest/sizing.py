"""減碼／加碼。權重在寫程式時就固定，不依回測結果改。

部位只在「目標比例」改變的下一根開盤調整，中間不每天拉回。
訊號用收盤時已經知道的月線顏色和 33 FVB；當天收盤才決定，下一個開盤才成交。
單邊成本 0.05%，只扣在實際成交的金額上。

權重（全部事先固定）：
- C1：基礎 100%。月線綠（淺綠或深綠）→ 100%。漸增淺紅 → 50%。
  深紅 → 0%（C1_0）或 25%（C1_25）。持平負區 → 維持原部位。
- C2：沒有加碼時放基礎部位。月線是綠、33 FVB 是綠、收盤回到帶內 → 加碼。
  加碼一直留到月線深紅，或收盤跌破下軌。漸增淺紅不單獨取消加碼。
  C2_150：基礎 100%，加到 150%（超過 100% 的部分視同借款，不計利息）。
  C2_60：基礎 60%，加到 100%（沒有槓桿）。
- C3：把上面拼在一起，數字不另訂。深紅 → 0%。漸增淺紅 → 50%。
  綠且回到帶內（FVB 也是綠）→ 150%，並沿用 C2 的加碼出場。
  其他綠 → 100%。持平負區 → 維持原部位。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine import ONE_WAY_COST, TRADING_DAYS, _metrics_from_equity
from trademap import fair_value_touch

# 讀結果用的門檻。不是策略參數，跑完不改。
CAPTURE_BAR = 0.80
DD_IMPROVE_BAR = 0.10

GREEN = ("淺綠", "深綠")


def _color_at(value) -> str | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    if text in ("", "None", "nan"):
        return None
    return text


def c1_targets(df: pd.DataFrame, dark_weight: float) -> pd.Series:
    """月線顏色決定部位。持平負區或還沒有顏色時，維持上一個目標。"""
    if not 0.0 <= dark_weight <= 1.0:
        raise ValueError("深紅部位必須在 0 到 1 之間")
    out = np.empty(len(df), dtype=float)
    prev = 1.0
    for i, raw in enumerate(df["m_color"].tolist()):
        color = _color_at(raw)
        if color in GREEN:
            prev = 1.0
        elif color == "漸增淺紅":
            prev = 0.5
        elif color == "深紅":
            prev = float(dark_weight)
        out[i] = prev
    return pd.Series(out, index=df.index, name="target")


def _c2_add_on(df: pd.DataFrame) -> np.ndarray:
    """加碼開關。打開要綠、飄帶綠、收盤在帶內；關掉只看深紅或跌破下軌。"""
    green = df["m_green"].fillna(False).to_numpy(dtype=bool)
    bull = df["bull"].fillna(False).to_numpy(dtype=bool)
    inside = fair_value_touch(df).to_numpy(dtype=bool)
    dark = df["m_dark_red"].fillna(False).to_numpy(dtype=bool)
    close = df["close"].to_numpy(dtype=float)
    thdn = df["thdn"].to_numpy(dtype=float)
    add = False
    flag = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        stopped = np.isfinite(thdn[i]) and close[i] < thdn[i]
        if dark[i] or (add and stopped):
            add = False
        elif (not add) and green[i] and bull[i] and inside[i]:
            add = True
        flag[i] = add
    return flag


def c2_targets(df: pd.DataFrame, base: float, added: float) -> pd.Series:
    if not 0.0 <= base <= added:
        raise ValueError("加碼部位必須不低於基礎部位，且基礎不小於 0")
    flag = _c2_add_on(df)
    out = np.where(flag, added, base).astype(float)
    return pd.Series(out, index=df.index, name="target")


def c3_targets(df: pd.DataFrame) -> pd.Series:
    """組合。順序固定：深紅、漸增淺紅，先於加碼；持平負區不改部位。"""
    green = df["m_green"].fillna(False).to_numpy(dtype=bool)
    bull = df["bull"].fillna(False).to_numpy(dtype=bool)
    inside = fair_value_touch(df).to_numpy(dtype=bool)
    close = df["close"].to_numpy(dtype=float)
    thdn = df["thdn"].to_numpy(dtype=float)
    add = False
    prev = 1.0
    out = np.empty(len(df), dtype=float)
    for i, raw in enumerate(df["m_color"].tolist()):
        color = _color_at(raw)
        stopped = np.isfinite(thdn[i]) and close[i] < thdn[i]
        if color == "深紅":
            add = False
            prev = 0.0
        elif color == "漸增淺紅":
            add = False
            prev = 0.5
        elif color == "持平負區" or color is None:
            if add and stopped:
                add = False
                prev = 1.0
        elif add and stopped:
            add = False
            prev = 1.0
        elif add:
            prev = 1.5
        elif color in GREEN and green[i] and bull[i] and inside[i]:
            add = True
            prev = 1.5
        elif color in GREEN:
            prev = 1.0
        out[i] = prev
    return pd.Series(out, index=df.index, name="target")


def target_book(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "C1_0": c1_targets(df, 0.0),
        "C1_25": c1_targets(df, 0.25),
        "C2_150": c2_targets(df, 1.0, 1.5),
        "C2_60": c2_targets(df, 0.6, 1.0),
        "C3": c3_targets(df),
    }


@dataclass
class SizeResult:
    name: str
    equity: pd.Series
    weight: pd.Series
    held_target: pd.Series
    n_orders: int
    turnover: float
    avg_weight: float
    segment_win_rate: float
    metrics: dict


def _rebalance(cash: float, shares: float, px: float, weight: float, cost: float) -> tuple[float, float]:
    """把部位調到 weight。成本從淨值扣，調完後股票市值 / 淨值 = weight。"""
    equity = cash + shares * px
    if not np.isfinite(equity) or equity <= 1e-12 or not np.isfinite(px) or px <= 0:
        return 0.0, 0.0
    if weight <= 0.0:
        cash = cash + shares * px * (1.0 - cost)
        return 0.0, cash
    stock = shares * px
    if weight * equity >= stock:
        denom = px * (1.0 + weight * cost)
    else:
        denom = px * (1.0 - weight * cost)
    delta = (weight * equity - stock) / denom
    new_shares = shares + delta
    new_cash = cash - delta * px - abs(delta) * px * cost
    if new_cash + new_shares * px <= 1e-12:
        return 0.0, 0.0
    return new_shares, new_cash


def _segment_win_rate(equity: pd.Series, held: pd.Series, initial: float = 1.0) -> float:
    """有部位的一段裡，結束淨值高於這段開始的比例。空手的段不計。"""
    if held.empty:
        return np.nan
    group_id = held.ne(held.shift()).cumsum()
    wins = 0
    n = 0
    for _, idx in held.groupby(group_id).groups.items():
        days = held.loc[idx]
        if float(days.iloc[0]) <= 0.0:
            continue
        end = float(equity.loc[days.index[-1]])
        pos = equity.index.get_loc(days.index[0])
        start = initial if pos == 0 else float(equity.iloc[pos - 1])
        if not np.isfinite(start) or start <= 0 or not np.isfinite(end):
            continue
        n += 1
        if end / start - 1.0 > 0:
            wins += 1
    return wins / n if n else np.nan


def run_sized_book(
    df: pd.DataFrame,
    target: pd.Series,
    analysis_mask: pd.Series,
    name: str,
    cost: float = ONE_WAY_COST,
    initial_cash: float = 1.0,
    bh_total: float | None = None,
) -> SizeResult:
    """目標在收盤決定，下一個開盤成交。最後一根改在收盤把部位歸零，以便和買進持有比。"""
    n = len(df)
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    tgt = target.reindex(df.index).to_numpy(dtype=float)
    active = analysis_mask.reindex(df.index).fillna(False).to_numpy(dtype=bool)
    if not active.any():
        raise RuntimeError(f"{name} 沒有分析區間")
    first = int(np.argmax(active))
    last = int(n - 1 - np.argmax(active[::-1]))

    cash = initial_cash
    shares = 0.0
    current = 0.0
    equity = np.full(n, np.nan, dtype=float)
    mtm_weight = np.full(n, np.nan, dtype=float)
    held = np.full(n, np.nan, dtype=float)
    orders = 0
    turnover = 0.0
    alive = True

    def apply_weight(i: int, px: float, weight: float) -> None:
        nonlocal cash, shares, current, orders, turnover, alive
        if not alive:
            return
        weight = float(weight)
        if abs(weight - current) < 1e-12:
            return
        new_shares, new_cash = _rebalance(cash, shares, px, weight, cost)
        cash, shares = new_cash, new_shares
        turnover += abs(weight - current)
        current = 0.0 if shares == 0.0 and cash <= 1e-12 else weight
        orders += 1
        if cash + shares * px <= 1e-12:
            alive = False
            current = 0.0

    for i in range(first, last + 1):
        if i == first:
            opening = float(tgt[first - 1]) if first > 0 and np.isfinite(tgt[first - 1]) else 1.0
        else:
            opening = float(tgt[i - 1]) if np.isfinite(tgt[i - 1]) else current
        apply_weight(i, o[i], 0.0 if not alive else opening)
        if i == last:
            mark_eq = cash + shares * c[i]
            mark_w = (shares * c[i] / mark_eq) if mark_eq > 1e-12 else 0.0
            apply_weight(i, c[i], 0.0)
            equity[i] = cash
            mtm_weight[i] = mark_w
            held[i] = opening
        else:
            equity[i] = cash + shares * c[i]
            mtm_weight[i] = (shares * c[i] / equity[i]) if equity[i] > 1e-12 else 0.0
            held[i] = current
        if equity[i] <= 1e-12:
            alive = False
            equity[i] = 0.0
            shares = 0.0
            cash = 0.0
            current = 0.0

    eq = pd.Series(equity, index=df.index, name=name)
    weight = pd.Series(mtm_weight, index=df.index, name="weight")
    held_s = pd.Series(held, index=df.index, name="held_target")
    window_eq = eq.loc[active].dropna()
    window_held = held_s.loc[window_eq.index]
    window_w = weight.loc[window_eq.index]
    metrics = _metrics_from_equity(window_eq, [], bh_total=bh_total, initial=initial_cash)
    win = _segment_win_rate(window_eq, window_held, initial=initial_cash)
    metrics["win_rate"] = win
    metrics["n_trades"] = int(orders)
    metrics["avg_weight"] = float(window_w.mean()) if len(window_w) else np.nan
    metrics["turnover"] = float(turnover)
    return SizeResult(
        name=name,
        equity=eq,
        weight=weight,
        held_target=held_s,
        n_orders=int(orders),
        turnover=float(turnover),
        avg_weight=float(metrics["avg_weight"]) if pd.notna(metrics["avg_weight"]) else np.nan,
        segment_win_rate=float(win) if pd.notna(win) else np.nan,
        metrics=metrics,
    )


def capture_ratio(total: float, bh_total: float) -> float:
    """策略總報酬 / 買進持有總報酬。買進持有不是正的就不算。"""
    if not np.isfinite(total) or not np.isfinite(bh_total) or bh_total <= 0:
        return np.nan
    return float(total / bh_total)


def drawdown_improvement(maxdd: float, bh_maxdd: float) -> float:
    """策略最大回撤減去買進持有最大回撤。正的代表這版跌得比較淺。"""
    if not np.isfinite(maxdd) or not np.isfinite(bh_maxdd):
        return np.nan
    return float(maxdd - bh_maxdd)


def meets_reading_bar(capture: float, improvement: float) -> bool:
    """事先講好的讀法：取回至少八成，而且最大回撤至少淺 10 個百分點。"""
    return bool(np.isfinite(capture) and np.isfinite(improvement) and capture >= CAPTURE_BAR and improvement >= DD_IMPROVE_BAR)


def _years_note() -> str:
    return f"年化與 Sharpe 沿用一年 {TRADING_DAYS} 個交易日。比特幣實際交易日比較多，年化會被壓低；總報酬不受這個影響。"
