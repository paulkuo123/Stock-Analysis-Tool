#!/usr/bin/env python3
"""多標的驗證：固定 V_CONF23（不改 THT／BX 主參數）。

用法：
    python run_backtest_v3_multi.py
    python run_backtest_v3_multi.py --start 2021-09-15 --end 2026-09-15
    python run_backtest_v3_multi.py --force-download
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import (  # noqa: E402
    ONE_WAY_COST,
    attach_metrics,
    run_buy_and_hold,
    run_long_only_ex,
    trades_to_frame,
)
from features import add_v2_features  # noqa: E402
from indicators import add_all_indicators  # noqa: E402
from run_backtest import DATA_DIR, FONT_PATH, _flatten_yf_columns, download_ohlcv, num, pct  # noqa: E402
from strategies_v2 import V2Spec, build_v2_specs  # noqa: E402

RESULT_DIR = ROOT / "results" / "v3"
FIG_DIR = RESULT_DIR / "figures"
REPORT_PATH = ROOT / "report-v3-multi-ticker.md"

# 顯示代號 → yfinance 代號。BTC 用加密日線 BTC-USD。
UNIVERSE: list[dict] = [
    {"id": "TSLA", "yf": "TSLA", "role": "基準", "name_zh": "特斯拉（v2 對照）"},
    {"id": "MU", "yf": "MU", "role": "驗證", "name_zh": "美光"},
    {"id": "TSM", "yf": "TSM", "role": "驗證", "name_zh": "台積電 ADR"},
    {"id": "NVDA", "yf": "NVDA", "role": "驗證", "name_zh": "輝達"},
    {"id": "BTC", "yf": "BTC-USD", "role": "驗證", "name_zh": "比特幣（BTC-USD 日線）"},
    {"id": "QQQ", "yf": "QQQ", "role": "驗證", "name_zh": "那斯達克 100 ETF"},
    {"id": "SMH", "yf": "SMH", "role": "驗證", "name_zh": "半導體 ETF"},
]

# 與 TSLA v2 對齊的「比較好打」體感門檻（寫死，不是看完數字再改）。
MIN_TRADES_FEEL = 8
MIN_WIN_RATE_FEEL = 0.50
TSLA_V2_REF = {
    "total_return": 1.990685,
    "win_rate": 0.555556,
    "maxdd": -0.218252,
    "n_trades": 18,
}

METRIC_COLS = [
    "ticker",
    "yf_symbol",
    "role",
    "name_zh",
    "status",
    "note",
    "actual_start",
    "actual_end",
    "n_bars",
    "years",
    "total_return",
    "ann_return",
    "maxdd",
    "win_rate",
    "n_trades",
    "avg_win",
    "avg_loss",
    "payoff",
    "sharpe",
    "bh_total_return",
    "bh_ann_return",
    "bh_maxdd",
    "bh_sharpe",
    "excess_vs_bh",
    "beats_bh",
    "feels_better",
]


@dataclass
class TickerRun:
    ticker: str
    yf_symbol: str
    role: str
    name_zh: str
    status: str
    note: str = ""
    actual_start: str = ""
    actual_end: str = ""
    n_bars: int = 0
    v_metrics: dict = field(default_factory=dict)
    bh_metrics: dict = field(default_factory=dict)
    equity_v: Optional[pd.Series] = None
    equity_bh: Optional[pd.Series] = None
    trades_v: Optional[pd.DataFrame] = None
    trades_bh: Optional[pd.DataFrame] = None
    window_df: Optional[pd.DataFrame] = None


def v_conf23_spec() -> V2Spec:
    specs = [s for s in build_v2_specs() if s.id == "V_CONF23"]
    if not specs:
        raise RuntimeError("strategies_v2 找不到 V_CONF23，請勿改掉這條固定策略。")
    return specs[0]


def cache_path_for(yf_symbol: str) -> Path:
    slug = yf_symbol.lower().replace("/", "-")
    return DATA_DIR / f"{slug}_ohlcv.csv"


def load_ticker_ohlcv(yf_symbol: str, warmup_start: str, end: str, force: bool) -> pd.DataFrame:
    """沿用 v1／v2 的下載＋快取；BTC-USD 等帶連字號的代號也能存檔。"""
    path = cache_path_for(yf_symbol)
    if path.exists() and not force:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = _flatten_yf_columns(df)
        print(f"使用快取行情：{path}（{len(df)} 根）")
        return df
    print(f"下載 {yf_symbol} {warmup_start}～{end} …")
    df = download_ohlcv(yf_symbol, warmup_start, end, path)
    print(f"已寫入 {path}（{len(df)} 根）")
    return df


def analysis_mask(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    in_window = (df.index >= start_ts) & (df.index <= end_ts)
    ready = df["basis"].notna() & df["bx"].notna() & df["rsi2"].notna() & df["atr14"].notna()
    return in_window & ready


def run_v_conf23_on_frame(
    raw: pd.DataFrame,
    start: str,
    end: str,
    cost: float,
) -> tuple[pd.DataFrame, pd.Series, object, object]:
    """回傳 (含指標的 df, analysis mask, V_CONF23 結果, Buy&Hold 結果)。"""
    df = add_v2_features(add_all_indicators(raw))
    mask = analysis_mask(df, start, end)
    if int(mask.sum()) < 50:
        raise RuntimeError(f"分析區間有效K棒過少：{int(mask.sum())}")
    spec = v_conf23_spec()
    bh = run_buy_and_hold(df, name="F", cost=cost, analysis_mask=mask)
    attach_metrics(bh, mask, bh_total=None)
    bh_total = bh.metrics["total_return"]
    attach_metrics(bh, mask, bh_total=bh_total)
    entry = spec.entry_fn(df)
    exit_ = spec.exit_fn(df)
    res = run_long_only_ex(
        df,
        entry,
        exit_,
        name="V_CONF23",
        cost=cost,
        analysis_mask=mask,
        stop_pct=spec.stop_pct,
        atr_stop_k=spec.atr_stop_k,
        trail_atr_k=spec.trail_atr_k,
        time_exit_bars=spec.time_exit_bars,
        cooldown_losses=spec.cooldown_losses,
    )
    attach_metrics(res, mask, bh_total=bh_total)
    return df, mask, res, bh


def feels_better_row(v: dict, bh: dict) -> bool:
    """勝率與回撤是否仍「比較好打」：至少 8 筆、勝率≥50%、回撤比該標的 B&H 不痛。"""
    n = int(v.get("n_trades") or 0)
    wr = v.get("win_rate")
    if n < MIN_TRADES_FEEL or wr is None or not np.isfinite(wr):
        return False
    if float(wr) < MIN_WIN_RATE_FEEL:
        return False
    return float(v["maxdd"]) > float(bh["maxdd"])


def setup_font() -> None:
    if Path(FONT_PATH).exists():
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK TC", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 140
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25


def md_summary_table(metrics: pd.DataFrame) -> str:
    cols = [
        ("ticker", "標的", lambda r: r["ticker"]),
        ("name_zh", "名稱", lambda r: r["name_zh"]),
        ("role", "角色", lambda r: r["role"]),
        ("status", "狀態", lambda r: r["status"]),
        ("total_return", "總報酬", lambda r: pct(r["total_return"]) if r["status"] == "ok" else "—"),
        ("ann_return", "年化", lambda r: pct(r["ann_return"]) if r["status"] == "ok" else "—"),
        ("maxdd", "MaxDD", lambda r: pct(r["maxdd"]) if r["status"] == "ok" else "—"),
        ("win_rate", "勝率", lambda r: pct(r["win_rate"]) if r["status"] == "ok" else "—"),
        ("n_trades", "筆數", lambda r: str(int(r["n_trades"])) if r["status"] == "ok" and pd.notna(r["n_trades"]) else "—"),
        ("avg_win", "平均賺", lambda r: pct(r["avg_win"]) if r["status"] == "ok" else "—"),
        ("avg_loss", "平均賠", lambda r: pct(r["avg_loss"]) if r["status"] == "ok" else "—"),
        ("payoff", "盈虧比", lambda r: num(r["payoff"]) if r["status"] == "ok" else "—"),
        ("sharpe", "Sharpe", lambda r: num(r["sharpe"]) if r["status"] == "ok" else "—"),
        ("bh_total_return", "B&H 總報酬", lambda r: pct(r["bh_total_return"]) if r["status"] == "ok" else "—"),
        ("excess_vs_bh", "超額 vs B&H", lambda r: pct(r["excess_vs_bh"]) if r["status"] == "ok" else "—"),
    ]
    header = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, r in metrics.iterrows():
        lines.append("| " + " | ".join(fn(r) for _, _, fn in cols) + " |")
    return "\n".join(lines)


def rows_from_runs(runs: list[TickerRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        v = run.v_metrics
        b = run.bh_metrics
        ok = run.status == "ok"
        row = {c: np.nan for c in METRIC_COLS}
        row.update(
            {
                "ticker": run.ticker,
                "yf_symbol": run.yf_symbol,
                "role": run.role,
                "name_zh": run.name_zh,
                "status": run.status,
                "note": run.note,
                "actual_start": run.actual_start,
                "actual_end": run.actual_end,
                "n_bars": run.n_bars,
            }
        )
        if ok:
            excess = float(v["total_return"]) - float(b["total_return"])
            row.update(
                {
                    "years": v.get("years"),
                    "total_return": v.get("total_return"),
                    "ann_return": v.get("ann_return"),
                    "maxdd": v.get("maxdd"),
                    "win_rate": v.get("win_rate"),
                    "n_trades": v.get("n_trades"),
                    "avg_win": v.get("avg_win"),
                    "avg_loss": v.get("avg_loss"),
                    "payoff": v.get("payoff"),
                    "sharpe": v.get("sharpe"),
                    "bh_total_return": b.get("total_return"),
                    "bh_ann_return": b.get("ann_return"),
                    "bh_maxdd": b.get("maxdd"),
                    "bh_sharpe": b.get("sharpe"),
                    "excess_vs_bh": excess,
                    "beats_bh": bool(excess > 0),
                    "feels_better": feels_better_row(v, b),
                }
            )
        else:
            row["beats_bh"] = False
            row["feels_better"] = False
        rows.append(row)
    return pd.DataFrame(rows)[METRIC_COLS]


def _tsla_close_to_v2(metrics: pd.DataFrame) -> str:
    t = metrics[metrics["ticker"] == "TSLA"]
    if t.empty or t.iloc[0]["status"] != "ok":
        return "TSLA 這一列沒跑成，無法對照 v2。"
    r = t.iloc[0]
    bits = []
    bits.append(f"總報酬 {pct(r['total_return'])}（v2 約 {pct(TSLA_V2_REF['total_return'])}）")
    bits.append(f"勝率 {pct(r['win_rate'])}（v2 約 {pct(TSLA_V2_REF['win_rate'])}）")
    bits.append(f"MaxDD {pct(r['maxdd'])}（v2 約 {pct(TSLA_V2_REF['maxdd'])}）")
    bits.append(f"{int(r['n_trades'])} 筆（v2 {TSLA_V2_REF['n_trades']} 筆）")
    return "；".join(bits)


def write_report(
    path: Path,
    start: str,
    end: str,
    cost: float,
    metrics: pd.DataFrame,
) -> None:
    ok = metrics[metrics["status"] == "ok"].copy()
    failed = metrics[metrics["status"] != "ok"]
    tsla = ok[ok["ticker"] == "TSLA"]
    others = ok[ok["ticker"] != "TSLA"]

    beat = ok[ok["beats_bh"] == True]  # noqa: E712
    feel = ok[ok["feels_better"] == True]  # noqa: E712
    beat_ids = "、".join(beat["ticker"].tolist()) if len(beat) else "沒有"
    feel_ids = "、".join(feel["ticker"].tolist()) if len(feel) else "沒有"

    others_beat = others[others["beats_bh"] == True]  # noqa: E712
    others_feel = others[others["feels_better"] == True]  # noqa: E712
    n_others = len(others)
    n_others_beat = len(others_beat)
    n_others_feel = len(others_feel)

    fail_lines = "沒有抓失敗的標的。"
    if len(failed):
        fail_lines = "\n".join(f"- **{r['ticker']}**（{r['yf_symbol']}）：{r['note']}" for _, r in failed.iterrows())

    per_ticker = []
    for _, r in metrics.iterrows():
        if r["status"] != "ok":
            per_ticker.append(f"### {r['ticker']} {r['name_zh']}\n\n抓資料或回測失敗：{r['note']}\n")
            continue
        beat_txt = "打贏" if r["beats_bh"] else "沒打贏"
        feel_txt = "還算比較好打" if r["feels_better"] else "體感不算比較好打"
        per_ticker.append(
            f"### {r['ticker']} {r['name_zh']}（{r['role']}）\n\n"
            f"- 區間：{r['actual_start']} ～ {r['actual_end']}（{int(r['n_bars'])} 根）\n"
            f"- V_CONF23：總報酬 {pct(r['total_return'])}、年化 {pct(r['ann_return'])}、"
            f"MaxDD {pct(r['maxdd'])}、勝率 {pct(r['win_rate'])}、{int(r['n_trades'])} 筆、"
            f"平均賺 {pct(r['avg_win'])}、平均賠 {pct(r['avg_loss'])}、盈虧比 {num(r['payoff'])}、"
            f"Sharpe {num(r['sharpe'])}\n"
            f"- Buy&Hold：總報酬 {pct(r['bh_total_return'])}、年化 {pct(r['bh_ann_return'])}、"
            f"MaxDD {pct(r['bh_maxdd'])}、Sharpe {num(r['bh_sharpe'])}\n"
            f"- 超額：{pct(r['excess_vs_bh'])} → **{beat_txt} Buy&Hold**；勝率／回撤門檻：**{feel_txt}**。\n"
        )

    # 過擬合判斷（規則寫死）
    tsla_only_like = False
    if len(tsla) and n_others:
        tsla_good = bool(tsla.iloc[0]["beats_bh"]) and bool(tsla.iloc[0]["feels_better"])
        # 「像只對 TSLA 有效」：TSLA 兩邊都過，但驗證標的打贏 B&H 不到一半，且體感過關不到 2 檔
        tsla_only_like = tsla_good and (n_others_beat < max(1, n_others / 2)) and (n_others_feel < 2)

    if tsla_only_like:
        overfit = (
            f"比較像**只對 TSLA 有效**。驗證標的 {n_others} 檔裡，打贏 Buy&Hold 的只有 {n_others_beat} 檔，"
            f"勝率／回撤仍「比較好打」的只有 {n_others_feel} 檔。這是過擬合警訊：規則是在 TSLA 上挑出來的，"
            f"換標的後優勢縮水或消失，不該當成通用進出場。"
        )
    elif n_others and n_others_beat >= max(3, n_others / 2) and n_others_feel >= 2:
        overfit = (
            f"不像「只會打 TSLA」。驗證標的 {n_others} 檔裡有 {n_others_beat} 檔打贏 Buy&Hold、"
            f"{n_others_feel} 檔勝率／回撤仍過「比較好打」門檻。不過這仍是同一段五年、同一套已看過的規則，"
            f"**不是樣本外**，不能保證以後或下一檔也這樣。"
        )
    else:
        overfit = (
            f"介於中間：不是只有 TSLA 好看，但也遠稱不上通用。"
            f"驗證標的打贏 Buy&Hold {n_others_beat}/{n_others}、體感過關 {n_others_feel}/{n_others}。"
            f"有過擬合味道，至少不該把 V_CONF23 當成「綠飄帶後確認就能到處用」的通則。"
        )

    if tsla_only_like or n_others_beat <= 1:
        verdict = (
            "不值得當通用規則：先當 TSLA 這段樣本裡比較好打的濾網，換標的要重驗，不要直接套到別檔或實盤。"
        )
    elif n_others_beat >= 3 and n_others_feel >= 2:
        verdict = (
            "可以當「多頭確認濾網」的工作假設繼續驗，但還不配當通用規則：沒做樣本外、沒改成本／滑價、也沒涵蓋空頭十年。"
        )
    else:
        verdict = (
            "現階段不值得當通用規則；頂多當盤面註解，真要交易仍要看標的本身趨勢與回撤能不能睡得著。"
        )

    tsla_line = _tsla_close_to_v2(metrics)

    beat_detail = []
    for _, r in ok.iterrows():
        mark = "贏" if r["beats_bh"] else "輸"
        feel = "體感過關" if r["feels_better"] else "體感沒過"
        beat_detail.append(
            f"- **{r['ticker']}**：相對 B&H {mark}（超額 {pct(r['excess_vs_bh'])}）；"
            f"{feel}（勝率 {pct(r['win_rate'])}、MaxDD {pct(r['maxdd'])} vs B&H {pct(r['bh_maxdd'])}、{int(r['n_trades'])} 筆）"
        )

    text = f"""# 多標的驗證：V_CONF23 是不是只會打 TSLA？

> 固定策略、固定主參數、同一段約五年日K。用來回答「v2 在 TSLA 上看起來比較好打，換標的還在不在」。**不是獲利保證**，也沒有為各標的重調 THT／BX。

## 給庭安的一句話

{verdict}

## 三句話

1. TSLA 基準重跑：{tsla_line}。規則沒改，數字應接近 v2 的約 +199%、勝率 ~56%、MaxDD ~-22%。
2. 打贏 Buy&Hold 的標的：{beat_ids}。勝率≥50% 且至少 8 筆、回撤比該檔 B&H 不痛（「比較好打」）的標的：{feel_ids}。
3. {overfit}

進出場與先前一致：**訊號收盤成立、次一交易日開盤**進出；期末未平倉用最後收盤。單邊成本 {cost*100:.2f}%。只做多、無槓桿、一次一筆。

---

## 1. 哪些標的打贏 Buy&Hold？體感還好打嗎？

「打贏」只看總報酬是否高於同期 Buy&Hold。「比較好打」寫死成：至少 {MIN_TRADES_FEEL} 筆、勝率 ≥ {MIN_WIN_RATE_FEEL:.0%}，且 MaxDD 比該標的自己的 Buy&Hold 淺。這是體感門檻，不是再優化。

{chr(10).join(beat_detail)}

### 總表

規劃區間：{start} ～ {end}。Sharpe 為日報酬、無風險利率=0、年化 √252（加密貨幣日線也用 252，方便橫向比，會略偏保守／不一致，報告有標）。

{md_summary_table(metrics)}

---

## 2. 是不是像只對 TSLA 有效？

{overfit}

V_CONF23 沒有為 MU／TSM／NVDA／BTC／QQQ／SMH 改 N、TW 或 BX 週期。若只有 TSLA 那一列漂亮，比較像「在 TSLA 上挑到剛好能避開假轉折的確認窗」，不是通則。

TSLA 對照 v2：{tsla_line}。

---

## 3. 各標的說明

{chr(10).join(per_ticker)}

### 抓資料失敗

{fail_lines}

---

## 回測設定（可重跑）

| 項目 | 內容 |
| --- | --- |
| 策略 | **V_CONF23**（沿用 `strategies_v2.py`，不改規則） |
| 進場 | THT 綠飄帶（BULL 0→1）之後，第 2～3 根仍偏多（BULL 仍真）且收紅 |
| 出場 | D3：BX 跌破零軸 **或** 飄帶轉紅（BULL 1→0） |
| THT | N=33, W1=1, W2=2（公式有宣告未參與計算）, TW=0.18 |
| BX | SL1=5, SL2=20, SL3=5（不用 SHORT/LONG 附圖） |
| 標的 | TSLA（基準）、MU、TSM、NVDA、BTC-USD、QQQ、SMH |
| 規劃區間 | {start} ～ {end} |
| 熱身 | 約 2020-01-01 起 |
| 方向 | 只做多、無槓桿、一次一筆 |
| 訊號成交 | 收盤計算，**次日開盤**；期末未平→最後收盤 |
| 成本 | 單邊 {cost*100:.2f}% |
| 對照 | 每個標的各自 Buy&Hold（區間開盤進、期末收盤出） |
| 不用 | 為各標的重掃 N／TW／SL；BX 附圖 SHORT/LONG |

```bash
cd tht-bx-backtest
pip install -r requirements.txt
python run_backtest_v3_multi.py
python test_v3.py
```

---

## 圖檔與數字

- `results/v3/summary.csv`：總表
- `results/v3/equity_curves.csv`：各標的 V_CONF23／B&H 權益
- `results/v3/trades_<TICKER>_V_CONF23.csv`：逐筆
- `results/v3/figures/equity_all_vconf23.png`：各標的策略權益疊圖
- `results/v3/figures/equity_<TICKER>.png`：單一標的策略 vs Buy&Hold
- `results/v3/figures/bars_return_vs_bh.png`：總報酬對照
- `results/v3/figures/scatter_wr_dd.png`：勝率 vs MaxDD

---

## 限制（請勿當成保證獲利）

1. 同一段五年、樣本內。規則是在 TSLA 上先看過再拿去驗，不是預先鎖死再第一次打開其他標的的嚴格樣本外。
2. 沒有滑價模型、沒有部位波動目標、沒有放空。
3. yfinance 還原價與券商／通達信未還原可能對不齊；加密貨幣日線含週末，K棒數比股票多，Sharpe 仍用 √252。
4. 沒打贏 Buy&Hold 不代表「反著做就會賺」；打贏也不代表明年還會贏。
5. **不要為了讓某檔變漂亮去改 N／TW／BX。**

## 檔案清單

| 路徑 | 說明 |
| --- | --- |
| `run_backtest_v3_multi.py` | 多標的一鍵重跑 |
| `test_v3.py` | 不需網路的單元測試 |
| `strategies_v2.py` | V_CONF23 定義（本輪未改主參數） |
| `results/v3/` | CSV、逐筆、圖 |

產生日期：腳本執行當下。資料來源：Yahoo Finance via yfinance。
"""
    path.write_text(text, encoding="utf-8")


def plot_all_equity(runs: list[TickerRun], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for run in runs:
        if run.status != "ok" or run.equity_v is None:
            continue
        lw = 2.4 if run.ticker == "TSLA" else 1.2
        ax.plot(run.equity_v.index, run.equity_v.values, label=f"{run.ticker} V_CONF23", linewidth=lw, alpha=0.9)
    ax.set_title("V_CONF23 各標的權益（起始=1）")
    ax.set_ylabel("權益")
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ticker_equity(run: TickerRun, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(run.equity_v.index, run.equity_v.values, label="V_CONF23", linewidth=2.0, color="#2ca02c")
    ax.plot(run.equity_bh.index, run.equity_bh.values, label="Buy&Hold", linewidth=1.6, color="#7f7f7f")
    ax.set_title(f"{run.ticker}：V_CONF23 vs Buy&Hold")
    ax.set_ylabel("權益（起始=1）")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_return_bars(metrics: pd.DataFrame, path: Path) -> None:
    ok = metrics[metrics["status"] == "ok"].copy()
    if ok.empty:
        return
    x = np.arange(len(ok))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.bar(x - w / 2, ok["total_return"] * 100, width=w, label="V_CONF23", color="#2ca02c")
    ax.bar(x + w / 2, ok["bh_total_return"] * 100, width=w, label="Buy&Hold", color="#7f7f7f")
    ax.set_xticks(x)
    ax.set_xticklabels(ok["ticker"].tolist())
    ax.set_ylabel("總報酬 %")
    ax.set_title("總報酬：V_CONF23 vs Buy&Hold")
    ax.axhline(0, color="#aaa", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_scatter(metrics: pd.DataFrame, path: Path) -> None:
    ok = metrics[metrics["status"] == "ok"].copy()
    if ok.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    colors = np.where(ok["beats_bh"], "#2ca02c", "#d62728")
    ax.scatter(ok["win_rate"] * 100, ok["maxdd"] * 100, s=90, c=colors, edgecolors="white", zorder=3)
    for _, r in ok.iterrows():
        ax.annotate(r["ticker"], (r["win_rate"] * 100, r["maxdd"] * 100), fontsize=9, xytext=(4, 4), textcoords="offset points")
    ax.axvline(MIN_WIN_RATE_FEEL * 100, color="#888", linestyle="--", linewidth=0.8, label=f"勝率 {MIN_WIN_RATE_FEEL:.0%}")
    ax.set_xlabel("勝率 %（愈右愈舒服）")
    ax.set_ylabel("MaxDD %（愈上愈不痛）")
    ax.set_title("V_CONF23：勝率 vs 最大回撤（綠=打贏 B&H）")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V_CONF23 多標的驗證（不改主參數）")
    p.add_argument("--start", default="2021-09-15")
    p.add_argument("--end", default="2026-09-15")
    p.add_argument("--warmup-start", default="2020-01-01")
    p.add_argument("--cost", type=float, default=ONE_WAY_COST)
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--skip-tests", action="store_true")
    return p.parse_args()


def run_unit_tests() -> None:
    import test_v3 as t

    t.main()


def run_one_universe_item(
    item: dict,
    start: str,
    end: str,
    warmup_start: str,
    cost: float,
    force: bool,
) -> TickerRun:
    run = TickerRun(
        ticker=item["id"],
        yf_symbol=item["yf"],
        role=item["role"],
        name_zh=item["name_zh"],
        status="pending",
    )
    try:
        raw = load_ticker_ohlcv(item["yf"], warmup_start, end, force)
    except Exception as exc:  # noqa: BLE001 — 單檔失敗不應炸掉整批
        run.status = "download_failed"
        run.note = f"yfinance 下載失敗：{exc}"
        print(f"[失敗] {item['id']} 下載：{exc}")
        return run
    try:
        df, mask, res, bh = run_v_conf23_on_frame(raw, start, end, cost)
    except Exception as exc:  # noqa: BLE001
        run.status = "backtest_failed"
        run.note = f"回測失敗：{exc}"
        print(f"[失敗] {item['id']} 回測：{exc}")
        return run

    idx = df.index[mask]
    run.status = "ok"
    run.actual_start = idx.min().strftime("%Y-%m-%d")
    run.actual_end = idx.max().strftime("%Y-%m-%d")
    run.n_bars = int(mask.sum())
    run.v_metrics = dict(res.metrics)
    run.bh_metrics = dict(bh.metrics)
    run.equity_v = res.equity.loc[mask]
    run.equity_bh = bh.equity.loc[mask]
    run.trades_v = trades_to_frame(res.trades)
    run.trades_bh = trades_to_frame(bh.trades)
    run.window_df = df.loc[mask]
    m = res.metrics
    wr = m["win_rate"] * 100 if pd.notna(m.get("win_rate")) else float("nan")
    print(
        f"{item['id']:6s} 報酬={m['total_return']*100:7.2f}%  "
        f"MaxDD={m['maxdd']*100:7.2f}%  勝率={wr:5.1f}%  "
        f"筆數={m['n_trades']:3d}  vsB&H={(m['excess_vs_bh']*100 if pd.notna(m.get('excess_vs_bh')) else float('nan')):7.2f}%"
    )
    return run


def main() -> int:
    args = parse_args()
    setup_font()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_tests:
        run_unit_tests()

    runs: list[TickerRun] = []
    for item in UNIVERSE:
        runs.append(
            run_one_universe_item(
                item,
                start=args.start,
                end=args.end,
                warmup_start=args.warmup_start,
                cost=args.cost,
                force=args.force_download,
            )
        )

    metrics = rows_from_runs(runs)
    metrics.to_csv(RESULT_DIR / "summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    eq_parts = {}
    for run in runs:
        if run.status != "ok":
            continue
        eq_parts[f"{run.ticker}_V_CONF23"] = run.equity_v
        eq_parts[f"{run.ticker}_BH"] = run.equity_bh
        run.trades_v.to_csv(RESULT_DIR / f"trades_{run.ticker}_V_CONF23.csv", index=False, encoding="utf-8-sig")
        run.trades_bh.to_csv(RESULT_DIR / f"trades_{run.ticker}_BH.csv", index=False, encoding="utf-8-sig")
        keep = [c for c in ["open", "high", "low", "close", "bull", "bull_on", "confirm_23", "bx", "pos_to_neg"] if c in run.window_df.columns]
        run.window_df[keep].to_csv(RESULT_DIR / f"indicators_{run.ticker}.csv", encoding="utf-8-sig")

    if eq_parts:
        pd.DataFrame(eq_parts).to_csv(RESULT_DIR / "equity_curves.csv", encoding="utf-8-sig")

    plot_all_equity(runs, FIG_DIR / "equity_all_vconf23.png")
    plot_return_bars(metrics, FIG_DIR / "bars_return_vs_bh.png")
    plot_scatter(metrics, FIG_DIR / "scatter_wr_dd.png")
    for run in runs:
        if run.status == "ok":
            plot_ticker_equity(run, FIG_DIR / f"equity_{run.ticker}.png")

    write_report(REPORT_PATH, start=args.start, end=args.end, cost=args.cost, metrics=metrics)

    readme_path = ROOT / "README.md"
    old = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    ok = metrics[metrics["status"] == "ok"]
    beat_n = int((ok["beats_bh"] == True).sum()) if len(ok) else 0  # noqa: E712
    v3_block = f"""

## v3：V_CONF23 多標的驗證

固定 **V_CONF23**（不改 THT N=33／TW=0.18、BX SL1=5 SL2=20 SL3=5），測 TSLA＋MU／TSM／NVDA／BTC-USD／QQQ／SMH。

```bash
python run_backtest_v3_multi.py
python test_v3.py
```

報告：[report-v3-multi-ticker.md](report-v3-multi-ticker.md)。產出在 `results/v3/`。
成功回測 {len(ok)} 檔，其中 {beat_n} 檔總報酬打贏各自的 Buy&Hold。**不保證獲利，不要為各標的重調主參數。**
"""
    if "## v3：" in old:
        head = old.split("## v3：", 1)[0].rstrip()
        readme_path.write_text(head + v3_block, encoding="utf-8")
    else:
        readme_path.write_text(old.rstrip() + v3_block, encoding="utf-8")

    print("已寫入 report-v3-multi-ticker.md 與 results/v3/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
