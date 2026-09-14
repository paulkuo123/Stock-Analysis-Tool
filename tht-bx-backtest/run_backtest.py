#!/usr/bin/env python3
"""TSLA 日K 約五年：THT＋BX（可選 RSI2）組合評估與回測。

用法（在 tht-bx-backtest/ 或 repo 根目錄皆可）：
    python run_backtest.py
    python run_backtest.py --ticker TSLA --start 2021-09-15 --end 2026-09-15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import (  # noqa: E402
    ONE_WAY_COST,
    attach_metrics,
    run_buy_and_hold,
    run_long_only,
    trades_to_frame,
)
from indicators import add_all_indicators  # noqa: E402
from strategies import build_combos  # noqa: E402

DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
FIG_DIR = RESULT_DIR / "figures"
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"


def setup_font() -> None:
    if Path(FONT_PATH).exists():
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK TC", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 140
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25


def run_unit_tests() -> None:
    import test_indicators as t

    t.test_tdx_sma_matches_wilder()
    t.test_cross_and_barslast()
    t.test_bull_flips_on_more_recent_cross()
    t.test_bx_color_transition_dark_red_to_light_red()
    t.test_rsi2_bounds()
    t.test_confirm_within_and_cancel()
    t.test_bull_on_is_pulse_not_state()
    print("單元測試通過。")


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]).lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
    else:
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]
    rename = {}
    for c in df.columns:
        if c in {"adj close", "adjclose"}:
            rename[c] = "close"
        elif c == "adj_close":
            rename[c] = "close"
    if rename:
        df = df.rename(columns=rename)
    need = ["open", "high", "low", "close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"行情欄位不足：{missing} / 實際={list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = np.nan
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[need + ["volume"]]


def download_ohlcv(ticker: str, warmup_start: str, end: str, cache_path: Path) -> pd.DataFrame:
    import yfinance as yf

    # yfinance 的 end 為不含當日，往後加一天把期末包進來。
    end_plus = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        ticker,
        start=warmup_start,
        end=end_plus,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance 沒有抓到 {ticker} 資料")
    df = _flatten_yf_columns(raw)
    df = df.dropna(subset=["open", "high", "low", "close"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, date_format="%Y-%m-%d")
    return df


def load_or_download(ticker: str, warmup_start: str, end: str, force: bool) -> pd.DataFrame:
    cache_path = DATA_DIR / f"{ticker.lower()}_ohlcv.csv"
    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df = _flatten_yf_columns(df)
        print(f"使用快取行情：{cache_path}（{len(df)} 根）")
        return df
    print(f"下載 {ticker} {warmup_start}～{end} …")
    df = download_ohlcv(ticker, warmup_start, end, cache_path)
    print(f"已寫入 {cache_path}（{len(df)} 根）")
    return df


def pct(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x*100:.2f}%"


def num(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def pick_recommend(metrics: pd.DataFrame) -> dict:
    """公開、可重跑的建議規則（樣本內、單一標的，不是預測）。

    overall：全部策略（不含 B&H）裡，交易次數足夠者挑 Sharpe。
    pairing：只看「THT 配 BX」組（C／D／含 BX 的 E），用來回答「怎麼搭」。
    """

    def rank(cand: pd.DataFrame) -> pd.Series:
        if cand.empty:
            raise ValueError("沒有可排名的組合")
        pool = cand
        for min_tr in (8, 5, 1):
            sub = cand[cand["n_trades"] >= min_tr]
            if len(sub):
                pool = sub
                break
        med_ret = pool["total_return"].median()
        dd_cut = pool["maxdd"].quantile(0.33)
        filt = pool[(pool["total_return"] >= med_ret) | (pool["sharpe"] >= pool["sharpe"].median())]
        filt = filt[filt["maxdd"] >= dd_cut] if len(filt) else pool
        if filt.empty:
            filt = pool
        filt = filt.sort_values(["sharpe", "calmar", "total_return"], ascending=False, na_position="last")
        return filt.iloc[0]

    ex_bh = metrics[metrics["id"] != "F"].copy()
    overall = rank(ex_bh)
    mix_ids = {
        "C1",
        "C3",
        "C5",
        "C_PRE3",
        "C_PRE5",
        "C_BXPOS",
        "C3D2",
        "C5D2",
        "D1",
        "D2",
        "D3",
        "E_C3_RSI70",
        "E_C3_RSI70_D2",
        "E_C5_RSI70_D2",
        "E_D2_RSI70",
        "E_D3_RSI70",
    }
    mix = metrics[metrics["id"].isin(mix_ids)].copy()
    pairing = rank(mix)
    tht_only = metrics[metrics["id"] == "A"].iloc[0]
    return {"overall": overall, "pairing": pairing, "tht_only": tht_only}


def plot_equity(curves: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    highlight = {"F", "A", "B1", "C3", "D2", "E_C3_RSI70_D2"}
    for col in curves.columns:
        lw = 2.2 if col.split(":")[0] in highlight or col.startswith("F") else 1.0
        alpha = 0.95 if lw > 1.5 else 0.55
        ax.plot(curves.index, curves[col], label=col, linewidth=lw, alpha=alpha)
    ax.set_title(title)
    ax.set_ylabel("權益（起始=1）")
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_recommend(df: pd.DataFrame, eq: pd.Series, trades_df: pd.DataFrame, name: str, path: Path) -> None:
    mask_eq = eq.dropna()
    peak = mask_eq.cummax()
    dd = mask_eq / peak - 1.0
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.4, 1.0]})

    ax = axes[0]
    ax.plot(df.index, df["close"], color="#444", linewidth=1.0, label="TSLA 收盤")
    if len(trades_df):
        ax.scatter(
            pd.to_datetime(trades_df["entry_date"]),
            trades_df["entry_price"],
            marker="^",
            color="#2ca02c",
            s=36,
            label="進場",
            zorder=5,
        )
        ax.scatter(
            pd.to_datetime(trades_df["exit_date"]),
            trades_df["exit_price"],
            marker="v",
            color="#d62728",
            s=36,
            label="出場",
            zorder=5,
        )
    ax.set_ylabel("價格")
    ax.set_title(f"建議組合走勢與進出點：{name}")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.plot(mask_eq.index, mask_eq.values, color="#1f77b4", linewidth=1.6)
    ax.set_ylabel("權益")

    ax = axes[2]
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.45)
    ax.set_ylabel("回撤")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_metric_bars(metrics: pd.DataFrame, path: Path) -> None:
    show = metrics.copy()
    labels = show["id"] + " " + show["name_zh"].str.slice(0, 18)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].barh(labels, show["total_return"] * 100)
    axes[0].set_xlabel("總報酬 %")
    axes[0].invert_yaxis()
    axes[1].barh(labels, show["maxdd"] * 100, color="#d62728")
    axes[1].set_xlabel("最大回撤 %")
    axes[1].invert_yaxis()
    fig.suptitle("各組合總報酬 vs 最大回撤")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def md_table(metrics: pd.DataFrame) -> str:
    cols = [
        ("id", "代號", lambda r: r["id"]),
        ("group", "組", lambda r: r["group"]),
        ("name_zh", "組合", lambda r: r["name_zh"]),
        ("total_return", "總報酬", lambda r: pct(r["total_return"])),
        ("ann_return", "年化", lambda r: pct(r["ann_return"])),
        ("maxdd", "最大回撤", lambda r: pct(r["maxdd"])),
        ("sharpe", "Sharpe", lambda r: num(r["sharpe"])),
        ("n_trades", "交易次數", lambda r: str(int(r["n_trades"]))),
        ("win_rate", "勝率", lambda r: pct(r["win_rate"])),
        ("avg_hold_days", "平均持有天", lambda r: num(r["avg_hold_days"], 1)),
        ("excess_vs_bh", "相對 B&H", lambda r: pct(r["excess_vs_bh"])),
    ]
    header = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, r in metrics.iterrows():
        lines.append("| " + " | ".join(fn(r) for _, _, fn in cols) + " |")
    return "\n".join(lines)


def _rules_zh(combo_id: str) -> str:
    rules = {
        "A": "綠飄帶（BULL 0→1）進，飄帶轉紅（BULL 1→0）出。",
        "B1": "BX 深紅→淺紅進，深綠→淺綠出（SPEC 字面）。",
        "B2": "BX 深紅→淺紅進，淺綠→深綠（正區轉弱）出。",
        "B3": "BX 由負轉正進，由正轉負出（零軸，不看四色）。",
        "B4": "BX 深紅→淺紅進，跌破零軸出。",
        "C1": "綠飄帶當根須同時出現深紅→淺紅才進；BULL 轉假出。",
        "C3": "綠飄帶後 3 根內等到深紅→淺紅才進；BULL 轉假出。",
        "C_PRE3": "BX 深紅→淺紅後 3 根內出現綠飄帶才進；BULL 轉假出。",
        "C_PRE5": "BX 深紅→淺紅後 5 根內出現綠飄帶才進；BULL 轉假出。",
        "C_BXPOS": "綠飄帶且當根 BX 已在正區才進；BULL 轉假出。",
        "E_D3_RSI70": "綠飄帶且 RSI2<70 進；BX 跌破零軸或 BULL 轉紅出。",
        "D1": "綠飄帶進；深綠→淺綠或 BULL 轉紅出。",
        "D2": "綠飄帶進；淺綠→深綠或 BULL 轉紅出。",
        "D3": "綠飄帶進；BX 跌破零軸或 BULL 轉紅出。",
        "C3D2": "綠飄帶後 3 根內深紅→淺紅才進；淺綠→深綠或 BULL 轉紅出。",
        "C5D2": "綠飄帶後 5 根內深紅→淺紅才進；淺綠→深綠或 BULL 轉紅出。",
        "E_A_RSI70": "綠飄帶且 RSI2<70 進；BULL 轉假出。",
        "E_C3_RSI70": "綠飄帶後 3 根內深紅→淺紅且 RSI2<70 進；BULL 轉假出。",
        "E_C3_RSI70_D2": "綠飄帶後 3 根內深紅→淺紅且 RSI2<70 進；轉弱或轉紅出。",
        "E_C5_RSI70_D2": "綠飄帶後 5 根內深紅→淺紅且 RSI2<70 進；轉弱或轉紅出。",
        "E_A_RSI30_70": "綠飄帶且 RSI2 在 30–70 進；BULL 轉假出。",
        "E_D2_RSI70": "綠飄帶且 RSI2<70 進；淺綠→深綠或 BULL 轉紅出。",
        "F": "分析區間開盤買進、期末收盤賣出。",
    }
    return rules.get(combo_id, combo_id)


def write_report(
    path: Path,
    ticker: str,
    start: str,
    end: str,
    actual_start: str,
    actual_end: str,
    n_bars: int,
    cost: float,
    metrics: pd.DataFrame,
    rec: pd.Series,
    rec_trades: pd.DataFrame,
    overall: pd.Series | None = None,
    facts: dict | None = None,
) -> None:
    rec_id = rec["id"]
    bh = metrics[metrics["id"] == "F"].iloc[0]
    a = metrics[metrics["id"] == "A"].iloc[0]
    b1 = metrics[metrics["id"] == "B1"].iloc[0]
    b2 = metrics[metrics["id"] == "B2"].iloc[0]
    b3 = metrics[metrics["id"] == "B3"].iloc[0]
    if overall is None:
        overall = rec
    facts = facts or {}
    on_n = facts.get("bull_on_n", "—")
    on_pos = facts.get("bull_on_bx_pos", "—")
    on_neg = facts.get("bull_on_bx_neg", "—")
    on_rsi70 = facts.get("bull_on_rsi70", "—")

    if overall["id"] == rec_id:
        headline = (
            f"**建議組合：「{rec['name_zh']}」（{rec_id}）。** "
            f"{_rules_zh(rec_id)}"
        )
    else:
        headline = (
            f"**若問這五年誰數字最好：是「{overall['name_zh']}」（{overall['id']}），"
            f"總報酬 {pct(overall['total_return'])}、年化 {pct(overall['ann_return'])}、"
            f"最大回撤 {pct(overall['maxdd'])}、Sharpe {num(overall['sharpe'])}。**\n\n"
            f"**若問 THT 怎麼配 BX：建議「{rec['name_zh']}」（{rec_id}）。** "
            f"{_rules_zh(rec_id)} "
            "「綠飄帶之後硬等深紅→淺紅才准進」在這段 TSLA 很容易漏掉波段，不建議當預設。"
        )

    text = f"""# TSLA 日K 約五年：THT＋BX（可選 RSI2）組合評估

> 這份報告是**樣本內回測**，只看 {ticker} 這一檔、日K、約五年。數字用來比較「怎麼搭指標比較合理」，**不是獲利保證**，也不能直接外推到其他股票或未來。

## 一句話建議

{headline}

建議搭配（{rec_id}）本次回測：總報酬 {pct(rec['total_return'])}、年化 {pct(rec['ann_return'])}、最大回撤 {pct(rec['maxdd'])}、Sharpe {num(rec['sharpe'])}、勝率 {pct(rec['win_rate'])}、交易 {int(rec['n_trades'])} 筆、平均持有 {num(rec['avg_hold_days'], 1)} 個交易日。同期 Buy&Hold 總報酬 {pct(bh['total_return'])}、最大回撤 {pct(bh['maxdd'])}，超額 {pct(rec['excess_vs_bh'])}。僅 THT（A）總報酬 {pct(a['total_return'])}、Sharpe {num(a['sharpe'])}。

進出場都是**訊號收盤成立、次一交易日開盤**成交；單邊成本 0.05%。

---

## 回測設定（可重跑）

| 項目 | 內容 |
| --- | --- |
| 標的 | {ticker}（yfinance，還原權息／分割後 OHLC） |
| 規劃區間 | {start} ～ {end} |
| 實際資料 | {actual_start} ～ {actual_end}，共 {n_bars} 根日K（有多少用多少） |
| 熱身 | 另向前抓約 2020-01-01 起的資料，讓 MA33／EMA20 先暖機 |
| 週期 | 日K |
| 方向 | 只做多、無槓桿、同一時間最多一筆 |
| 成交 | 訊號當日收盤計算，**次一交易日開盤**成交；期末若仍持倉改以最後收盤平倉 |
| 成本 | 單邊 {cost*100:.2f}%（進出各收一次；Buy&Hold 同樣計） |
| BX 參數 | 只用 SL1=5, SL2=20, SL3=5 |
| RSI | 只用 P2=12 的 SMA 型 RSI2；不用 P3、不假設 P1 |
| THT | N=33, TW=0.18；公式裡的 W1、W2 **沒有進計算** |

重跑：

```bash
cd tht-bx-backtest
pip install -r requirements.txt
python run_backtest.py
```

---

## 指標怎麼讀（白話）

### THT 綠飄帶

用 OHLC 四價平均做 33 日均線（BASIS）與標準差通道。  
**BULL** 的定義是：最近一次「最低價上穿 BASIS」比最近一次「BASIS 上穿最高價」更近。可以把它想成：價格最近是從軌道下方翻上來、還是從上方掉下去。

- **BULL 由 0→1**：剛轉成綠色飄帶 → 本次預設的 THT 進場。
- **BULL 由 1→0**：綠飄帶失效 → 常見出場。

### BX 四色（本次實作）

BX 本質是「短均減長均」的變化速度，再做成 0 軸附近的震盪（大約 -50～+50）。著色跟 SPEC 一致：

| 條件 | 顏色 | 白話 |
| --- | --- | --- |
| BX≥0 且 BX≥昨日 | 淺綠 | 正區、斜率向上 |
| BX≥0 且 BX<昨日 | 深綠 | 正區、斜率向下 |
| BX<0 且 BX≥昨日 | 淺紅 | 負區、斜率向上 |
| BX<0 且 BX<昨日 | 深紅 | 負區、斜率向下 |

- **深紅→淺紅**：還在 0 軸下方，但由往下改成往上 → **買盤力量開始加強**。當 BX 進場主訊號。
- **深綠→淺綠**：還在 0 軸上方，由往下改成往上。SPEC 寫「由深綠轉淺綠＝變弱」。**這與「正區斜率轉正」的字面方向相反**，比較像「正區止跌再攻」，不像「力道變弱」。本次仍按字面做了 B1／D1，並另外做兩組對照：
  - **淺綠→深綠**：正區由升轉降，較接近「轉弱」。
  - **BX 由負轉正／由正轉負**：不管顏色，只看 0 軸。

### RSI2

通達信 SMA 型、P2=12。本次當**超買濾網**（例如 RSI2<70 才准進），不是主訊號。

### 通達信 SMA(X,N,1)

採加權平滑 `Y=(X+(N-1)*Y')/N`（Wilder RMA），**不是** N 日算術平均。EMA 用 `alpha=2/(N+1)`。STD 用母體標準差（ddof=0）。

---

## 各組合績效

區間：{actual_start} ～ {actual_end}。報酬皆含單邊 0.05% 成本。Sharpe 以日報酬、無風險利率=0、年化 √252。

{md_table(metrics)}

完整數字見 `results/combo_metrics.csv`。建議組合逐筆交易見 `results/trades_{rec_id}.csv`。

---

## 怎麼看這張表（白話結論）

### 先看對照組 Buy&Hold

這五年 TSLA 波動很大，中間有一段很深的回撤。Buy&Hold 總報酬 {pct(bh['total_return'])}、年化 {pct(bh['ann_return'])}、最大回撤 {pct(bh['maxdd'])}、Sharpe {num(bh['sharpe'])}。  
波段策略如果「總報酬輸很多但回撤小很多」，仍可能比較適合睡得著；如果報酬輸、回撤又沒比較小，那這套組合在這段資料裡就沒有明顯價值。

### A. 只做 THT 綠飄帶

{a['name_zh']}：總報酬 {pct(a['total_return'])}，最大回撤 {pct(a['maxdd'])}，交易 {int(a['n_trades'])} 筆，勝率 {pct(a['win_rate'])}，平均抱 {num(a['avg_hold_days'], 1)} 天。  
這是最單純的「轉綠進、轉紅出」。優點是規則乾淨、持有時間跟趨勢段對得上；缺點是假轉折會連續挨打，也沒有用到買盤強弱。

### B. 只做 BX

- **字面著色（B1，深綠→淺綠作出場）**：總報酬 {pct(b1['total_return'])}，回撤 {pct(b1['maxdd'])}，交易 {int(b1['n_trades'])} 筆。若出場剛好是「正區再轉強」，容易賣太早或賣在不該賣的地方。
- **對照轉弱（B2，淺綠→深綠出）**：總報酬 {pct(b2['total_return'])}，回撤 {pct(b2['maxdd'])}，交易 {int(b2['n_trades'])} 筆。這組比較符合「買盤加強進、力道轉弱出」。
- **零軸對照（B3）**：總報酬 {pct(b3['total_return'])}，回撤 {pct(b3['maxdd'])}，交易 {int(b3['n_trades'])} 筆。這五年**數字最好**：比綠飄帶更勤進出，回撤明顯小於 Buy&Hold。

**四色 BX 單獨用（B1／B2）在這段 TSLA 不好用**；真正強的是「負轉正／正轉負」這條零軸規則。深紅→淺紅比較適合解釋盤面，不適合當唯一進場。

### C. 綠飄帶之後，等 BX 深紅→淺紅才進

SPEC 字面是「綠飄帶**之後** N 根內再等負區轉強」。本次 N=1／3／5 **全部 0 筆成交**。  
{on_n} 次綠飄帶裡，當根 BX 已在正區 {on_pos} 次、還在負區只有 {on_neg} 次；後面 5 根內也等不到「深紅→淺紅」。

白話：綠飄帶出現時，買盤多半**已經翻上來了**，再回頭等負區轉強，條件幾乎不會發生，**不建議這樣搭**。  
反過來「先深紅→淺紅再等綠飄帶」（C_PRE3／C_PRE5）或「綠飄帶且 BX≥0」（C_BXPOS）雖然有成交，但這段資料裡績效普通或更差，不是主角。

### D. 綠飄帶進，BX 負責出場

這組在測「要不要讓 BX 提早下車」。字面「深綠→淺綠」當出場（D1）請當假設組。  
本次結果：**D3（跌破 0 軸或飄帶轉紅）明顯最好**；D1 次之；D2（正區斜率轉弱）會切太早，總報酬甚至低於只做 THT。綠飄帶進場後，用 BX 零軸當「趨勢還沒壞」比用四色斜率當出場更禁得起這段 TSLA。

### E. 再加上 RSI2 未超買

RSI2<70 主要是擋「已經追很熱才轉綠」的進場。本次綠飄帶當根 RSI2≥70 的次數是 {on_rsi70}，所以 E 組裡很多濾網等於沒濾到，績效會跟沒加 RSI 的版本幾乎一樣。

### 建議組合為什麼是 {rec_id}

挑選分兩層，都寫在 `pick_recommend()`：**沒有對未來做參數搜尋**。

1. **overall（數字最好）**：不含 Buy&Hold，交易次數夠的組合裡頭看 Sharpe／回撤。本次是 **{overall['id']}** {overall['name_zh']}（總報酬 {pct(overall['total_return'])}、Sharpe {num(overall['sharpe'])}）。
2. **pairing（怎麼搭 THT＋BX）**：只在 C／D／含 BX 的 E 裡頭挑。本次是 **{rec_id}** {rec['name_zh']}。規則：{_rules_zh(rec_id)}

- 搭配組總報酬 {pct(rec['total_return'])}（相對 B&H {pct(rec['excess_vs_bh'])}）
- 年化 {pct(rec['ann_return'])}；最大回撤 {pct(rec['maxdd'])}（B&H 為 {pct(bh['maxdd'])}，僅 THT 為 {pct(a['maxdd'])}）
- Sharpe {num(rec['sharpe'])}；勝率 {pct(rec['win_rate'])}；{int(rec['n_trades'])} 筆；平均持有 {num(rec['avg_hold_days'], 1)} 天

**請不要因為某組在這五年數字最好，就認定它以後最好。** 換一個五年、換一檔股票，排名常常會倒過來。

---

## 建議組合的進出場細節

實際逐筆在 `results/trades_{rec_id}.csv`（共 {len(rec_trades)} 筆）。摘要：

"""
    if len(rec_trades):
        text += (
            f"- 平均單筆報酬 {pct(rec_trades['pnl_pct'].mean())}；"
            f"中位數 {pct(rec_trades['pnl_pct'].median())}\n"
            f"- 最好一筆 {pct(rec_trades['pnl_pct'].max())}；最差一筆 {pct(rec_trades['pnl_pct'].min())}\n"
            f"- 最長持有 {int(rec_trades['bars_held'].max())} 個交易日；最短 {int(rec_trades['bars_held'].min())} 個交易日\n\n"
        )
        show_n = min(8, len(rec_trades))
        text += "前幾筆（示範，完整見 CSV）：\n\n"
        text += "| 進場日 | 進場價 | 出場日 | 出場價 | 持有天 | 含成本報酬 |\n"
        text += "| --- | ---: | --- | ---: | ---: | ---: |\n"
        for _, t in rec_trades.head(show_n).iterrows():
            ed = pd.Timestamp(t["entry_date"]).date()
            xd = pd.Timestamp(t["exit_date"]).date()
            text += (
                f"| {ed} | {t['entry_price']:.2f} | {xd} | {t['exit_price']:.2f} | "
                f"{int(t['bars_held'])} | {pct(t['pnl_pct'])} |\n"
            )
        text += "\n"
    else:
        text += "此組合在區間內沒有成交（不應作為建議，請改看其他列）。\n\n"

    text += f"""
## 圖檔

- `results/figures/equity_all.png`：各組合權益曲線
- `results/figures/equity_core.png`：A／B／C／D／E／B&H 核心比較
- `results/figures/recommend_detail.png`：建議組合價格、進出點、權益與回撤
- `results/figures/metrics_bars.png`：總報酬與最大回撤條狀圖

---

## 限制與請勿過度解讀

1. **單一標的、單一區間、樣本內**。沒有做走勢外樣本、沒有滾動優化、沒有其他股票的穩健性測試。
2. **不保證獲利**。任何「建議組合」只是「在這份設定下比較好解釋、數字比較平衡」的說法。
3. TSLA 波動、跳空、新聞面極大，日K 訊號次日開盤仍會碰到缺口；成本只假設 0.05%，沒有模擬滑價、融券、稅。
4. yfinance 還原權息後的 OHLC 與券商軟體（尤其通達信未還原）會有差異，訊號日期可能對不齊。
5. BULL、BARSLAST、SMA(X,N,1) 依公開通達信定義實作；不同軟體對「從未發生過的 BARSLAST」、STD 的 ddof、EMA 種子值可能不同。
6. **「深綠→淺綠＝變弱」是 SPEC 指定假設**，與四色斜率字面不完全同向；報告已附淺綠→深綠、零軸穿越對照。實務上若你的盤面顏色定義不同，請以你軟體的圖為準重跑。
7. 只做多、一次一筆。沒有放空、沒有加碼、沒有部位波動目標。
8. W1、W2 在公式中未使用；BX 的 TREND 有計算但**沒有**拿來當進出場（SPEC 著色用 BX 本身）。

---

## 檔案清單

| 路徑 | 說明 |
| --- | --- |
| `run_backtest.py` | 一鍵重跑 |
| `indicators.py` | THT／BX／RSI2 |
| `engine.py` | 次日開盤成交與績效 |
| `strategies.py` | 組合 A–F 與對照 |
| `test_indicators.py` | 公式與視窗確認的單元測試 |
| `data/{ticker.lower()}_ohlcv.csv` | 本次使用的還原日K |
| `results/combo_metrics.csv` | 全部組合績效 |
| `results/equity_curves.csv` | 權益曲線 |
| `results/indicators.csv` | 指標與四色，方便核對盤面 |
| `results/trades_*.csv` | 各組合逐筆 |
| `results/figures/` | 圖 |

產生日期：腳本執行當下。資料來源：Yahoo Finance via yfinance。
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="THT＋BX 組合回測")
    p.add_argument("--ticker", default="TSLA")
    p.add_argument("--start", default="2021-09-15", help="分析起日")
    p.add_argument("--end", default="2026-09-15", help="分析迄日（有多少資料用多少）")
    p.add_argument("--warmup-start", default="2020-01-01")
    p.add_argument("--cost", type=float, default=ONE_WAY_COST)
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--skip-tests", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_font()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_tests:
        run_unit_tests()

    raw = load_or_download(args.ticker, args.warmup_start, args.end, args.force_download)
    df = add_all_indicators(raw)

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    in_window = (df.index >= start_ts) & (df.index <= end_ts)
    # 指標暖機：BASIS 必須有值才開始交易。
    ready = df["basis"].notna() & df["bx"].notna() & df["rsi2"].notna()
    analysis = in_window & ready
    if analysis.sum() < 50:
        raise RuntimeError(f"分析區間有效K棒過少：{int(analysis.sum())}")

    actual_start = df.index[analysis].min().strftime("%Y-%m-%d")
    actual_end = df.index[analysis].max().strftime("%Y-%m-%d")
    n_bars = int(analysis.sum())
    print(f"分析區間實際：{actual_start} ～ {actual_end}（{n_bars} 根）")

    window_df = df.loc[analysis].copy()
    ind_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "basis",
        "thup",
        "thdn",
        "bull",
        "bull_on",
        "bull_off",
        "bx",
        "color",
        "dr_to_lr",
        "dg_to_lg",
        "lg_to_dg",
        "neg_to_pos",
        "pos_to_neg",
        "rsi2",
    ]
    window_df[ind_cols].to_csv(RESULT_DIR / "indicators.csv", encoding="utf-8-sig")

    facts = {
        "bull_on_n": int(window_df["bull_on"].sum()),
        "bull_on_bx_pos": int((window_df["bull_on"] & (window_df["bx"] >= 0)).sum()),
        "bull_on_bx_neg": int((window_df["bull_on"] & (window_df["bx"] < 0)).sum()),
        "bull_on_rsi70": int((window_df["bull_on"] & (window_df["rsi2"] >= 70)).sum()),
    }

    combos = build_combos()
    results = {}
    bh = run_buy_and_hold(df, name="F", cost=args.cost, analysis_mask=analysis)
    attach_metrics(bh, analysis, bh_total=None)
    bh_total = bh.metrics["total_return"]
    attach_metrics(bh, analysis, bh_total=bh_total)
    results["F"] = bh

    for spec in combos:
        if spec.is_bh:
            continue
        entry = spec.entry_fn(df)
        exit_ = spec.exit_fn(df)
        res = run_long_only(df, entry, exit_, name=spec.id, cost=args.cost, analysis_mask=analysis)
        attach_metrics(res, analysis, bh_total=bh_total)
        results[spec.id] = res
        print(f"{spec.id:16s} 報酬={res.metrics['total_return']*100:7.2f}%  "
              f"MaxDD={res.metrics['maxdd']*100:7.2f}%  "
              f"筆數={res.metrics['n_trades']:3d}  Sharpe={res.metrics['sharpe']}")

    rows = []
    for spec in combos:
        m = dict(results[spec.id].metrics)
        m["id"] = spec.id
        m["group"] = spec.group
        m["name_zh"] = spec.name_zh
        m["note"] = spec.note
        rows.append(m)
    metrics = pd.DataFrame(rows)
    col_order = [
        "id",
        "group",
        "name_zh",
        "note",
        "total_return",
        "ann_return",
        "maxdd",
        "sharpe",
        "calmar",
        "n_trades",
        "win_rate",
        "avg_hold_days",
        "avg_trade_pnl",
        "excess_vs_bh",
        "n_days",
        "years",
    ]
    metrics = metrics[col_order]
    metrics.to_csv(RESULT_DIR / "combo_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    eq_df = pd.DataFrame({cid: results[cid].equity for cid in metrics["id"]})
    eq_df.loc[analysis].to_csv(RESULT_DIR / "equity_curves.csv", encoding="utf-8-sig")

    for spec in combos:
        trades_to_frame(results[spec.id].trades).to_csv(
            RESULT_DIR / f"trades_{spec.id}.csv", index=False, encoding="utf-8-sig"
        )

    rec_picks = pick_recommend(metrics)
    rec = rec_picks["pairing"]
    overall = rec_picks["overall"]
    rec_id = rec["id"]
    rec_trades = trades_to_frame(results[rec_id].trades)
    print(f"建議搭配：{rec_id} {rec['name_zh']}")
    print(f"數字最佳：{overall['id']} {overall['name_zh']}")

    # 圖：全部（欄位用代號＋短名）
    label_map = {r["id"]: f"{r['id']}:{r['name_zh'][:16]}" for _, r in metrics.iterrows()}
    plot_df = eq_df.loc[analysis].rename(columns=label_map)
    plot_equity(plot_df, f"{args.ticker} 各組合權益（含 Buy&Hold）", FIG_DIR / "equity_all.png")
    core_ids = ["A", "B1", "B2", "B3", "C3", "C5", "D1", "D2", "E_C3_RSI70_D2", "F", rec_id, overall["id"]]
    core_ids = list(dict.fromkeys(core_ids))
    core = eq_df.loc[analysis, [c for c in core_ids if c in eq_df.columns]].rename(columns=label_map)
    plot_equity(core, f"{args.ticker} 核心組合比較", FIG_DIR / "equity_core.png")
    plot_recommend(df.loc[analysis], eq_df.loc[analysis, rec_id], rec_trades, rec["name_zh"], FIG_DIR / "recommend_detail.png")
    plot_metric_bars(metrics, FIG_DIR / "metrics_bars.png")

    write_report(
        ROOT / "report.md",
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        actual_start=actual_start,
        actual_end=actual_end,
        n_bars=n_bars,
        cost=args.cost,
        metrics=metrics,
        rec=rec,
        rec_trades=rec_trades,
        overall=overall,
        facts=facts,
    )

    # 也寫一份精簡 README
    readme = f"""# THT＋BX 組合回測（{args.ticker} 日K）

獨立子目錄，**不依賴** repo 裡的 PyQt 選股工具。評估庭安自用指標 THT＋BX（可選 RSI2）的進出場搭法，並用建議組合回測約五年。

## 一句話

建議組合：**{rec['name_zh']}**（`{rec_id}`）。本次總報酬 {pct(rec['total_return'])}、年化 {pct(rec['ann_return'])}、最大回撤 {pct(rec['maxdd'])}、Sharpe {num(rec['sharpe'])}。同期 Buy&Hold 總報酬 {pct(bh_total)}。數字最好是 **{overall['name_zh']}**（`{overall['id']}`）。**不保證獲利。**

細節、假設與完整績效表見 [report.md](report.md)。

## 重跑

```bash
pip install -r requirements.txt
python run_backtest.py
python test_indicators.py
```

常用參數：

```bash
python run_backtest.py --ticker TSLA --start 2021-09-15 --end 2026-09-15
python run_backtest.py --force-download   # 重新抓行情
```

預設：訊號次日開盤進出、單邊成本 0.05%、只做多、一次一筆。資料來源 yfinance（還原 OHLC），並快取到 `data/`。

## 產出

- `report.md`：繁中台灣用語報告
- `results/combo_metrics.csv`、`equity_curves.csv`、`indicators.csv`、`trades_*.csv`
- `results/figures/*.png`

## 參數限制（依 SPEC）

- BX 只用 SL1=5, SL2=20, SL3=5
- RSI 只用 P2=12 的 SMA 型 RSI2
- THT 進場：BULL 0→1（綠色飄帶）
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print("已寫入 report.md 與 results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
