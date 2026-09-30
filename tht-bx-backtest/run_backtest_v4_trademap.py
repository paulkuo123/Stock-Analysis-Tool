#!/usr/bin/env python3
"""官方 Trade Map 重測：月線 BX 確認，對照 PR #3 的日線 V_CONF23。

變體 A：只把出場改成月線深紅，進場仍是 V_CONF23。
變體 B（主要結果）：月線綠或漸增淺紅 + 33 FVB 綠色，等收盤回到公允價值帶才進，並鎖下軌當失效價。

不改 THT N=33／TW=0.18、BX SL1=5／SL2=20／SL3=5。不要為了績效調參。

用法：
    python run_backtest_v4_trademap.py
    python run_backtest_v4_trademap.py --skip-download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import ONE_WAY_COST, attach_metrics  # noqa: E402
from monthly_bx import align_monthly_bx_to_daily, monthly_bx_table, stitch_close  # noqa: E402
from run_backtest import DATA_DIR, download_ohlcv, num, pct  # noqa: E402
from run_backtest_v3_multi import (  # noqa: E402
    UNIVERSE,
    analysis_mask,
    load_ticker_ohlcv,
    run_v_conf23_on_frame,
    setup_font,
    _fp,
)
from trademap import run_trade_map, variant_a_signals, variant_b_signals  # noqa: E402

RESULT_DIR = ROOT / "results" / "v4"
FIG_DIR = RESULT_DIR / "figures"
REPORT_PATH = ROOT / "report-v4-monthly-trademap.md"
V3_SUMMARY = ROOT / "results" / "v3" / "summary.csv"
WARMUP_DIR = DATA_DIR / "monthly_warmup"
MONTHLY_WARMUP_START = "2012-01-01"

# 主要結果。變體 A 只是用來拆「出場」和「整套規則」差在哪。
PRIMARY_ID = "B_OFFICIAL"
STRATEGY_ORDER = ["V_CONF23", "A_MONTHLY_EXIT", PRIMARY_ID]
STRATEGY_ZH = {
    "V_CONF23": "PR #3 舊版（日線 BX 出場）",
    "A_MONTHLY_EXIT": "變體 A：只改月線深紅出場",
    "B_OFFICIAL": "變體 B：完整官方月線版（主要結果）",
}


def _slug(yf_symbol: str) -> str:
    return yf_symbol.lower().replace("/", "-")


def load_early_ohlcv(yf_symbol: str, cache_start: pd.Timestamp, force: bool) -> pd.DataFrame | None:
    """抓快取起點之前的日K，只拿來暖月線 BX。失敗就回傳 None，日線策略仍用原快取。"""
    WARMUP_DIR.mkdir(parents=True, exist_ok=True)
    path = WARMUP_DIR / f"{_slug(yf_symbol)}_early.csv"
    end = (cache_start + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    if path.exists() and not force:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        print(f"使用月線暖機快取：{path}（{len(df)} 根）")
        return df
    try:
        print(f"下載 {yf_symbol} 月線暖機 {MONTHLY_WARMUP_START}～{end} …")
        df = download_ohlcv(yf_symbol, MONTHLY_WARMUP_START, end, path)
        print(f"已寫入 {path}（{len(df)} 根）")
        return df
    except Exception as exc:  # noqa: BLE001 — 暖機失敗時退回 2020 起的快取，並在報告標明
        print(f"[警告] {yf_symbol} 早段下載失敗，月線 BX 只用既有快取：{exc}")
        return None


def extended_close(cache: pd.DataFrame, yf_symbol: str, force: bool) -> tuple[pd.Series, str]:
    early = load_early_ohlcv(yf_symbol, cache.index.min(), force)
    if early is None or early.empty:
        close = cache["close"].astype(float).sort_index()
        return close, "沒有 2012 起的早段，月線 BX 從既有快取（約 2020-01）起算，暖機較短"
    return stitch_close(cache, early)


def _row(ticker: str, name_zh: str, strategy: str, metrics: dict, bh: dict, primary: bool) -> dict:
    total = float(metrics["total_return"])
    bh_total = float(bh["total_return"])
    return {
        "ticker": ticker,
        "name_zh": name_zh,
        "strategy": strategy,
        "strategy_zh": STRATEGY_ZH[strategy],
        "is_primary": primary,
        "total_return": total,
        "maxdd": float(metrics["maxdd"]),
        "win_rate": float(metrics["win_rate"]) if pd.notna(metrics["win_rate"]) else np.nan,
        "n_trades": int(metrics["n_trades"]),
        "avg_hold_days": float(metrics["avg_hold_days"]) if pd.notna(metrics["avg_hold_days"]) else np.nan,
        "bh_total_return": bh_total,
        "excess_vs_bh": total - bh_total,
        "beats_bh": bool(total > bh_total),
        "ann_return": float(metrics["ann_return"]) if pd.notna(metrics["ann_return"]) else np.nan,
        "sharpe": float(metrics["sharpe"]) if pd.notna(metrics["sharpe"]) else np.nan,
        "actual_start": metrics.get("actual_start", ""),
        "actual_end": metrics.get("actual_end", ""),
        "n_bars": metrics.get("n_bars", np.nan),
    }


def _fmt_hold(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.1f}"


def _fmt_wr(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return pct(x)


def _vs(flag: bool, n_trades: int) -> str:
    if n_trades <= 0:
        return "沒有交易"
    return "贏過買進持有" if flag else "輸給買進持有"


def md_primary_table(summary: pd.DataFrame) -> str:
    """主要結果：變體 B，每個標的一列。"""
    sub = summary[summary["strategy"] == PRIMARY_ID]
    header = "| 標的 | 總報酬 | 最大回撤 | 勝率 | 筆數 | 平均持有交易日 | 買進持有 | 對買進持有 |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    lines = [header, sep]
    for _, r in sub.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    r["ticker"],
                    pct(r["total_return"]),
                    pct(r["maxdd"]),
                    _fmt_wr(r["win_rate"]),
                    str(int(r["n_trades"])),
                    _fmt_hold(r["avg_hold_days"]),
                    pct(r["bh_total_return"]),
                    _vs(bool(r["beats_bh"]), int(r["n_trades"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def md_compare_table(summary: pd.DataFrame) -> str:
    """三套並排。每個標的一列，格子是「總報酬／最大回撤／勝率／筆數」。"""
    header = (
        "| 標的 | 買進持有 | PR #3 日線出場 | 變體 A 只改出場 | 變體 B 官方版（主要） | B 對買進持有 |"
    )
    sep = "| --- | ---: | ---: | ---: | ---: | --- |"
    lines = [header, sep]
    tickers = [u["id"] for u in UNIVERSE]
    for ticker in tickers:
        cells = [ticker]
        block = summary[summary["ticker"] == ticker]
        if block.empty:
            continue
        cells.append(pct(float(block["bh_total_return"].iloc[0])))
        for sid in STRATEGY_ORDER:
            r = block[block["strategy"] == sid].iloc[0]
            cells.append(
                f"{pct(r['total_return'])}／{pct(r['maxdd'])}／{_fmt_wr(r['win_rate'])}／{int(r['n_trades'])}筆"
            )
        b = block[block["strategy"] == PRIMARY_ID].iloc[0]
        cells.append(_vs(bool(b["beats_bh"]), int(b["n_trades"])))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _exit_mix(details: pd.DataFrame) -> str:
    if details is None or details.empty or "reason_exit" not in details.columns:
        return "沒有紀錄"
    traded = details[details["bars_held"].fillna(0) > 0]
    if traded.empty:
        cancelled = int(details["reason_exit"].astype(str).str.contains("取消進場").sum())
        return f"沒有成交（取消進場 {cancelled} 次）"
    counts = traded["reason_exit"].value_counts()
    parts = [f"{idx} {int(val)} 筆" for idx, val in counts.items()]
    cancelled = int(details["reason_exit"].astype(str).str.contains("取消進場").sum())
    if cancelled:
        parts.append(f"另有開盤跳空取消 {cancelled} 次")
    return "；".join(parts)


def _cell(summary: pd.DataFrame, ticker: str, strategy: str, col: str) -> float:
    hit = summary[(summary["ticker"] == ticker) & (summary["strategy"] == strategy)]
    return float(hit.iloc[0][col])


def _conclusion(summary: pd.DataFrame, stop_lines: list[str]) -> str:
    b = summary[summary["strategy"] == PRIMARY_ID].copy()
    a = summary[summary["strategy"] == "A_MONTHLY_EXIT"].copy()
    v = summary[summary["strategy"] == "V_CONF23"].copy()
    beat = b.loc[b["beats_bh"], "ticker"].tolist()
    lose = b.loc[~b["beats_bh"], "ticker"].tolist()
    med_b = float(b["n_trades"].median()) if len(b) else 0
    a_beat = a.loc[a["beats_bh"], "ticker"].tolist()
    merged = v.merge(a, on="ticker", suffixes=("_v", "_a"))
    a_better = merged.loc[merged["total_return_a"] > merged["total_return_v"], "ticker"].tolist()
    a_worse = merged.loc[merged["total_return_a"] <= merged["total_return_v"], "ticker"].tolist()
    beat_s = "、".join(beat) if beat else "沒有"
    lose_s = "、".join(lose) if lose else "沒有"
    a_beat_s = "、".join(a_beat) if a_beat else "沒有"
    a_better_s = "、".join(a_better) if a_better else "沒有"
    a_worse_s = "、".join(a_worse) if a_worse else "沒有"
    tsla_v = _cell(summary, "TSLA", "V_CONF23", "total_return")
    tsla_a = _cell(summary, "TSLA", "A_MONTHLY_EXIT", "total_return")
    tsla_b = _cell(summary, "TSLA", "B_OFFICIAL", "total_return")
    tsla_n = int(_cell(summary, "TSLA", "B_OFFICIAL", "n_trades"))
    tsla_wr = _cell(summary, "TSLA", "B_OFFICIAL", "win_rate")
    lines = [
        "不值得把這套官方月線版換成現在的預設。主要結果是變體 B。它沒有解決 PR #3「換標的就輸給買進持有」的問題，TSLA 這個舊版唯一明顯打贏的也一起掉了。",
        f"變體 B 總報酬高於該檔買進持有的只有：{beat_s}。仍輸的是：{lose_s}。交易筆數中位數約 {med_b:.0f} 筆，勝率很容易被一兩筆決定，不能說官方版比較會賺。",
        (
            f"差在哪一條規則，要用變體 A 對照。A 只把出場改成月線深紅，進場仍是 PR #3。"
            f"總報酬比舊版日線出場高的是 {a_better_s}；沒有變高的是 {a_worse_s}。"
            f"即便如此，變體 A 打贏買進持有的只有：{a_beat_s}。"
            "強勢股少被日線洗出去，會更接近一直抱著，但這段樣本裡還是多半沒有超過傻抱。"
        ),
        (
            f"TSLA 把兩件事分開了：舊版總報酬 {pct(tsla_v)}；只改月線出場變成 {pct(tsla_a)}，已經輸給買進持有；"
            f"完整官方版再變成 {pct(tsla_b)}，{tsla_n} 筆、勝率 {pct(tsla_wr)}。"
            "官方版多出來的是「等收盤回到很窄的公允價值帶」以及「用進場當根的下軌當失效價」。"
            "TSLA 這些筆幾乎都是被這條停損打掉，不是月線變深紅才走。所以 TSLA 變差，主因是進場和硬停損，不是月線顏色本身。"
        ),
        "參數沒有為了好看去調。N、TW、BX 的 SL1／SL2／SL3 都用原本的官方預設，顏色也沒有另設「要負到多少才叫深紅」。若官方圖的深紅其實還要看柱子長短，那是定義要再對，不是拿這五年的績效去湊。",
    ]
    if stop_lines:
        lines.append("變體 B 出場構成：" + "。".join(stop_lines) + "。")
    return "\n\n".join(lines)


def write_report(
    summary: pd.DataFrame,
    notes: pd.DataFrame,
    exit_notes: list[str],
    match_notes: list[str],
    stop_lines: list[str],
) -> None:
    b = summary[summary["strategy"] == PRIMARY_ID]
    n_beat = int(b["beats_bh"].sum())
    text = f"""# 官方 Trade Map 重測：月線 BX，不再用日線跌破零就出場

> 這是樣本內回測，7 檔、大約五年。**主要結果是變體 B。** 數字用來檢查規則有沒有跟官方 Trade Map 對齊，**不是獲利保證**。請不要 merge。

## 先講結論

7 檔裡，完整官方版（變體 B）有 **{n_beat} 檔**總報酬高於該檔的買進持有，**{7 - n_beat} 檔**仍然較低。交易筆數少，這個輸贏很不穩，下面有表格可以對，但不要把它讀成「以後都該這樣做」。

{_conclusion(summary, stop_lines)}

## 這次在對什麼

PR #3 的 V_CONF23 用日線：綠飄帶後第 2～3 根仍偏多且收紅才進，日線 BX 跌破零或飄帶轉紅就出。官方 Trade Map 不是這條。官方看的是**月線** BX 的顏色，而且進場要等價格回到 33 日公允價值帶，不追高。

- **33 FVB（公允價值帶）**：用開高低收四價平均做 33 日均線當中軌，上下各加減 0.18 倍標準差。帶是綠色，沿用既有程式的 BULL：最近一次低點向上穿過中軌，比最近一次中軌向上穿過高點更近。
- **月線 BX**：同一套 BX（SL1=5、SL2=20、SL3=5），但餵進去的是每月最後一根日線的收盤，不是日線 BX。
- **買進持有**：區間第一個交易日開盤買、最後一天收盤賣，單邊成本同樣 0.05%。

## 顏色怎麼定（請對著官方圖檢查）

沒有用「BX 絕對值超過多少才叫深色」。沿用 repo 既有四色的精神：只看正負，以及跟**前一根月線**比是升還是降。多一個門檻就是調參，這次不做。

| 條件 | 名稱 | 官方 Trade Map 怎麼用 |
| --- | --- | --- |
| BX ≥ 0，且 BX ≥ 前一根月線 | 淺綠 | 持有。多頭週期的月線條件成立 |
| BX ≥ 0，且 BX < 前一根月線 | 深綠 | 持有。多頭週期的月線條件仍成立 |
| BX < 0，且 BX > 前一根月線 | 漸增淺紅 | 週期仍有效，可以做多，但要盯著。不因為它是紅色就出場 |
| BX < 0，且 BX 等於前一根 | 持平負區 | 不是「漸增」，也不是深紅。不新開多單，也不當成週期結束。浮點幾乎碰不到 |
| BX < 0，且 BX < 前一根月線 | 深紅 | 多頭週期結束。收盤確認後出場，不攤平 |

正區持平（BX ≥ 0 且跟前一根一樣）沿用既有程式，算淺綠。

## 怎麼避免偷看未來

日線交易在某一天收盤做決定時，只能用**已經走完的月份**。

1. 月線收盤 = 該日曆月最後一個有成交的交易日收盤。股票就是該月最後交易日；比特幣包含週末。
2. 整段資料的最後一個月如果後面沒有下個月，視為還沒收盤。例如資料停在 2026-09-14，9 月不拿來當月線。
3. 某月的顏色要到該月最後一根收盤才知道。當月月中，用的是前一個已收盤月的顏色。
4. 月線 BX 的均線只往前滾，不會用未來月份回填過去的 BX。
5. 訊號在收盤成立，**下一根日線開盤**才買賣。所以「月線收盤變深紅就出場」的成交價，是次一交易日開盤，不是用剛收盤的那根月線價格自己成交。

日線指標（33 FVB、V_CONF23 進場）仍只用 PR #3 的行情快取，避免把新下載的早段資料改寫舊日線訊號。早段只拿來讓月線 EMA(20) 在 2021 年以前先暖完。銜接方式見下方備註。

## 兩種變體

| 代號 | 角色 | 進場 | 出場 |
| --- | --- | --- | --- |
| V_CONF23 | PR #3 舊版，用來並排 | 綠飄帶後第 2～3 根仍為多且收紅 | 日線 BX 由正轉負，或綠色飄帶消失 |
| A_MONTHLY_EXIT | 對照：只改出場 | 與 V_CONF23 相同 | 已收盤月線 BX 為深紅 |
| **B_OFFICIAL** | **主要結果** | 日線 33 FVB 為綠，且月線為綠或漸增淺紅，而且收盤回到上下軌之間。條件從「不成立」變成「成立」的那一根才進，避免在帶內每天重複買 | 月線變深紅；另外把進場訊號那根的下軌鎖成失效價，之後收盤跌破就出。持倉中日線飄帶轉紅不單獨出場 |

變體 B 的失效價是進場訊號當天的公允價值帶下軌，進場前就定死，持倉期間不跟著均線上移。官方原文沒有給第二個數字，所以用同一條下軌，沒有改成 8% 或幾倍 ATR。若開盤已經跳空跌破這個價，這筆不進。

持倉中「33 FVB 轉紅」不單獨出場。官方把「週期結束」寫在月線變深紅；「兩項都要成立才做多」用在**能不能新開倉**。這是假設，若你的圖是轉紅就該走，這版會抱得比你久。

## 回測設定（對齊 PR #3）

| 項目 | 內容 |
| --- | --- |
| 標的 | TSLA、MU、TSM、NVDA、QQQ、SMH、BTC-USD |
| 分析區間 | 2021-09-15～2026-09-15（有資料到哪裡用到哪裡） |
| 日線快取 | 沿用 `data/*_ohlcv.csv`，約 2020-01 起，讓日線 MA33 先暖機 |
| 月線暖機 | 另抓約 2012-01 起到快取起點，只算月線 BX |
| 成本 | 單邊 0.05%，進出各一次 |
| 方向 | 只做多、無槓桿、一次一筆，不攤平 |
| 成交 | 收盤訊號、次日開盤 |
| 參數 | THT N=33、TW=0.18；BX SL1=5、SL2=20、SL3=5。RSI P2=12 這次出場沒用到 |

重跑與 PR #3 公布的 V_CONF23 是否同一組數字：

{chr(10).join('- ' + s for s in match_notes)}

月線暖機：

{chr(10).join('- ' + s for s in notes['line'].tolist())}

## 主要結果：變體 B

平均持有是進場到出場中間的交易日根數，跟 PR #3 的演算法相同，不是日曆天。

{md_primary_table(summary)}

## 跟 PR #3、變體 A 並排

每一格是「總報酬／最大回撤／勝率／筆數」。買進持有只列總報酬。

{md_compare_table(summary)}

出場原因（含期末仍持有、用最後收盤平倉）：

{chr(10).join('- ' + s for s in exit_notes)}

## 為什麼不要下太重的結論

1. **筆數少。** 月線顏色一個月才變一次，五年常常只有幾次週期。勝率、平均持有在這種樣本裡很飄。
2. **這是同一段多頭。** MU、NVDA、TSM、SMH 這五年買了抱著就很高，提早下車很容易輸。輸給買進持有，不一定代表規則寫錯，只代表這段它沒有比傻抱好。
3. **沒有樣本外，也沒有為別檔重調參數。** 不要看哪一檔數字好看就把它的進場改松、停損改寬。
4. **顏色定義是可檢查的假設。** 若你的官方圖用絕對值深淺、或把「持平負區」也當漸增淺紅，請用 `results/v4/monthly_colors_*.csv` 對月份，不要直接把這次輸贏當成官方圖的成績。
5. **公允價值帶很窄**（只有 0.18 倍標準差）。變體 B 的失效價因此常常離進場不遠，很多筆可能是被下軌停掉，而不是抱到月線深紅。出場原因表就是要看這件事。這不是事後加寬停損的理由。
6. yfinance 還原股價和券商未還原的月 K 可能對不齊。成本沒有另加滑價。

## 檔案

| 路徑 | 說明 |
| --- | --- |
| `run_backtest_v4_trademap.py` | 一鍵重跑 |
| `monthly_bx.py` | 月線 BX、顏色、不可偷看未來的對齊 |
| `trademap.py` | 變體 A／B 與次日開盤回測 |
| `test_trademap.py` | 顏色、未來函數、停損時點 |
| `results/v4/summary.csv` | 三套策略的績效 |
| `results/v4/comparison.csv` | 與買進持有的對照 |
| `results/v4/monthly_colors_*.csv` | 每月 BX 與顏色，方便對官方圖 |
| `results/v4/trades_*.csv` | 逐筆 |
| `results/v4/figures/returns_compare.png` | 總報酬長條圖 |

```bash
cd tht-bx-backtest
python run_backtest_v4_trademap.py
python test_trademap.py
```
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"已寫入 {REPORT_PATH}")


def plot_returns(summary: pd.DataFrame, path: Path) -> None:
    setup_font()
    tickers = [u["id"] for u in UNIVERSE]
    x = np.arange(len(tickers))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    series = [
        ("bh_total_return", "買進持有", "#7f7f7f", -1.5),
        ("V_CONF23", "PR #3 日線出場", "#1f77b4", -0.5),
        ("A_MONTHLY_EXIT", "變體 A", "#ff7f0e", 0.5),
        (PRIMARY_ID, "變體 B 官方版", "#2ca02c", 1.5),
    ]
    for sid, label, color, shift in series:
        vals = []
        for ticker in tickers:
            block = summary[summary["ticker"] == ticker]
            if sid == "bh_total_return":
                vals.append(float(block["bh_total_return"].iloc[0]) * 100)
            else:
                vals.append(float(block.loc[block["strategy"] == sid, "total_return"].iloc[0]) * 100)
        ax.bar(x + shift * width, vals, width=width, label=label, color=color)
    ax.axhline(0, color="#444", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("總報酬 %", **_fp())
    ax.set_title("總報酬：買進持有／PR #3／變體 A／變體 B", **_fp())
    from run_backtest_v3_multi import CN_FONT

    if CN_FONT is not None:
        ax.legend(loc="upper left", prop=CN_FONT)
    else:
        ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run(start: str, end: str, cost: float, force_download: bool, skip_download: bool) -> pd.DataFrame:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    published = pd.read_csv(V3_SUMMARY) if V3_SUMMARY.exists() else None
    rows = []
    notes = []
    exit_notes = []
    match_notes = []
    stop_lines = []
    equity_parts = {}

    for item in UNIVERSE:
        ticker = item["id"]
        yf_symbol = item["yf"]
        print(f"\n=== {ticker} ===")
        raw = load_ticker_ohlcv(yf_symbol, "2020-01-01", end, force=False)
        raw = raw.loc[: pd.Timestamp(end)]
        df, mask, res_v, bh = run_v_conf23_on_frame(raw, start, end, cost)
        if published is not None:
            pub = published.loc[published["ticker"] == ticker].iloc[0]
            gap = abs(float(res_v.metrics["total_return"]) - float(pub["total_return"]))
            same = gap < 1e-6 and int(res_v.metrics["n_trades"]) == int(pub["n_trades"])
            match_notes.append(
                f"{ticker} 重跑 V_CONF23 總報酬 {pct(res_v.metrics['total_return'])}、"
                f"{int(res_v.metrics['n_trades'])} 筆；PR #3 公布 {pct(float(pub['total_return']))}、"
                f"{int(pub['n_trades'])} 筆。"
                + ("一致。" if same else f"差距 {gap:.6f}，請先查行情快取。")
            )
            if not same:
                raise RuntimeError(f"{ticker} 的 V_CONF23 與 PR #3 公布數字不一致，停止產出，避免對照表對錯基準。")

        if skip_download:
            long_close = raw["close"].astype(float)
            note = "略過早段下載，月線 BX 只用既有快取"
        else:
            long_close, note = extended_close(raw, yf_symbol, force_download)
        notes.append({"ticker": ticker, "line": f"{ticker}：{note}（月線用 {long_close.index.min().date()}～{long_close.index.max().date()}）"})

        aligned = align_monthly_bx_to_daily(long_close).reindex(df.index)
        for col in aligned.columns:
            df[col] = aligned[col]

        month_table = monthly_bx_table(long_close)
        month_table.to_csv(RESULT_DIR / f"monthly_colors_{ticker}.csv", encoding="utf-8-sig")

        window = df.loc[mask]
        actual_start = window.index.min().strftime("%Y-%m-%d")
        actual_end = window.index.max().strftime("%Y-%m-%d")
        bh.metrics["actual_start"] = actual_start
        bh.metrics["actual_end"] = actual_end
        bh.metrics["n_bars"] = int(mask.sum())

        runs = {"V_CONF23": (res_v, None)}
        for sid, signals in (
            ("A_MONTHLY_EXIT", variant_a_signals(df)),
            (PRIMARY_ID, variant_b_signals(df)),
        ):
            entry, exit_, inv = signals
            packed = run_trade_map(df, entry, exit_, inv, name=sid, cost=cost, analysis_mask=mask)
            attach_metrics(packed.result, mask, bh_total=bh.metrics["total_return"])
            runs[sid] = (packed.result, packed.details)

        for sid in STRATEGY_ORDER:
            res, details = runs[sid]
            res.metrics["actual_start"] = actual_start
            res.metrics["actual_end"] = actual_end
            res.metrics["n_bars"] = int(mask.sum())
            rows.append(_row(ticker, item["name_zh"], sid, res.metrics, bh.metrics, sid == PRIMARY_ID))
            equity_parts[f"{ticker}_{sid}"] = res.equity.loc[mask]
            if details is not None:
                details.to_csv(RESULT_DIR / f"trades_{ticker}_{sid}.csv", index=False, encoding="utf-8-sig")
                exit_notes.append(f"{ticker} {STRATEGY_ZH[sid]}：{_exit_mix(details)}")
                if sid == PRIMARY_ID:
                    traded = details[details["bars_held"].fillna(0) > 0]
                    n = int(len(traded))
                    n_stop = int(traded["reason_exit"].astype(str).str.contains("失效價").sum()) if n else 0
                    n_dark = int(traded["reason_exit"].astype(str).str.contains("月線深紅").sum()) if n else 0
                    n_end = int(traded["reason_exit"].astype(str).str.contains("期末").sum()) if n else 0
                    stop_lines.append(
                        f"{ticker} {n} 筆裡，失效價 {n_stop} 筆、月線深紅 {n_dark} 筆、期末仍持有 {n_end} 筆"
                    )
            elif sid == "V_CONF23":
                from engine import trades_to_frame

                trades_to_frame(res.trades).to_csv(
                    RESULT_DIR / f"trades_{ticker}_{sid}.csv", index=False, encoding="utf-8-sig"
                )
                reasons = pd.Series([t.reason_exit for t in res.trades]).value_counts()
                mix = "；".join(f"{k} {int(v)} 筆" for k, v in reasons.items()) or "沒有交易"
                exit_notes.append(f"{ticker} {STRATEGY_ZH[sid]}：{mix}")

        equity_parts[f"{ticker}_BH"] = bh.equity.loc[mask]
        bmet = runs[PRIMARY_ID][0].metrics
        print(
            f"{ticker:6s} B={bmet['total_return']*100:8.2f}%  "
            f"MaxDD={bmet['maxdd']*100:7.2f}%  筆數={bmet['n_trades']}  "
            f"B&H={bh.metrics['total_return']*100:8.2f}%"
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(equity_parts).to_csv(RESULT_DIR / "equity_curves.csv", encoding="utf-8-sig")
    note_df = pd.DataFrame(notes)
    note_df.to_csv(RESULT_DIR / "warmup_notes.csv", index=False, encoding="utf-8-sig")
    plot_returns(summary, FIG_DIR / "returns_compare.png")
    write_report(summary, note_df, exit_notes, match_notes, stop_lines)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="官方 Trade Map 月線 BX 重測（不改主參數）")
    p.add_argument("--start", default="2021-09-15")
    p.add_argument("--end", default="2026-09-15")
    p.add_argument("--cost", type=float, default=ONE_WAY_COST)
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--skip-download", action="store_true", help="不抓 2012 起的月線暖機")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.start, args.end, args.cost, args.force_download, args.skip_download)
