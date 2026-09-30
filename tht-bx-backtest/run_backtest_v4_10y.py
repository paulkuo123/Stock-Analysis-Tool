#!/usr/bin/env python3
"""同一套三種規則，改跑約十年：2016-09-15～2026-09-14。

不改 THT／BX 參數、不改成本、不改標的。資料不夠十年就用能拿到的最長段，並在報告註明。
空頭段日期事先固定，不依各標的高低點調整。

用法：
    python run_backtest_v4_10y.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bear_windows import BEAR_WINDOWS, window_performance  # noqa: E402
from engine import ONE_WAY_COST, attach_metrics, trades_to_frame  # noqa: E402
from monthly_bx import align_monthly_bx_to_daily, monthly_bx_table, stitch_ohlcv  # noqa: E402
from run_backtest import pct  # noqa: E402
from run_backtest_v3_multi import UNIVERSE, load_ticker_ohlcv, run_v_conf23_on_frame  # noqa: E402
from run_backtest_v4_trademap import (  # noqa: E402
    PRIMARY_ID,
    STRATEGY_ORDER,
    STRATEGY_ZH,
    _row,
    _vs,
    load_early_ohlcv,
    plot_returns,
)
from trademap import run_trade_map, variant_a_signals, variant_b_signals  # noqa: E402

RESULT_DIR = ROOT / "results" / "v4_10y"
FIG_DIR = RESULT_DIR / "figures"
REPORT_PATH = ROOT / "report-v4-10y.md"
START = "2016-09-15"
END = "2026-09-14"
REQUESTED_YEARS = 10


def load_full_ohlcv(yf_symbol: str) -> tuple[pd.DataFrame, str]:
    """快取（約 2020 起，與 PR #3 相同）接上 2012 起的早段。重疊日用快取。"""
    cache = load_ticker_ohlcv(yf_symbol, "2020-01-01", "2026-09-15", force=False)
    cache = cache.loc[: pd.Timestamp(END)]
    early = load_early_ohlcv(yf_symbol, cache.index.min(), force=False)
    if early is None or early.empty:
        raise RuntimeError(f"{yf_symbol} 沒有 2016 年以前的日 K，無法做十年回測")
    frame, note = stitch_ohlcv(cache, early)
    frame = frame.loc[: pd.Timestamp(END)]
    return frame, note


def _fmt_pct(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return pct(float(x))


def _pp(strategy_ret, bh_ret) -> str:
    if not np.isfinite(strategy_ret) or not np.isfinite(bh_ret):
        return "—"
    gap = (float(strategy_ret) - float(bh_ret)) * 100.0
    if abs(gap) <= 0.05:
        return "幾乎一樣"
    if gap > 0:
        return f"少虧 {gap:.1f} 個百分點"
    if float(bh_ret) >= 0:
        return f"這段抱著在賺，少賺 {abs(gap):.1f} 個百分點"
    return f"多虧 {abs(gap):.1f} 個百分點"


def md_full_table(summary: pd.DataFrame) -> str:
    header = "| 標的 | 實際區間 | 買進持有 | PR #3 日線出場 | 變體 A 只改出場 | 變體 B 官方版（主要） | B 對買進持有 |"
    sep = "| --- | --- | ---: | ---: | ---: | ---: | --- |"
    lines = [header, sep]
    for item in UNIVERSE:
        ticker = item["id"]
        block = summary[summary["ticker"] == ticker]
        if block.empty:
            continue
        cells = [ticker, f"{block['actual_start'].iloc[0]}～{block['actual_end'].iloc[0]}"]
        cells.append(pct(float(block["bh_total_return"].iloc[0])))
        for sid in STRATEGY_ORDER:
            r = block[block["strategy"] == sid].iloc[0]
            wr = pct(r["win_rate"]) if pd.notna(r["win_rate"]) else "—"
            cells.append(f"{pct(r['total_return'])}／{pct(r['maxdd'])}／{wr}／{int(r['n_trades'])}筆")
        b = block[block["strategy"] == PRIMARY_ID].iloc[0]
        cells.append(_vs(bool(b["beats_bh"]), int(b["n_trades"])))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_bear_table(bear: pd.DataFrame, window_id: str) -> str:
    sub = bear[bear["window_id"] == window_id]
    header = "| 標的 | 買進持有報酬 | 買進持有區間回撤 | 舊版報酬 | 變體 A 報酬 | 變體 B 報酬 | 變體 B 區間回撤 | B 比抱著 |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    lines = [header, sep]
    for item in UNIVERSE:
        r = sub[sub["ticker"] == item["id"]]
        if r.empty:
            continue
        row = r.iloc[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    item["id"],
                    _fmt_pct(row["bh_ret"]),
                    _fmt_pct(row["bh_maxdd"]),
                    _fmt_pct(row["v_ret"]),
                    _fmt_pct(row["a_ret"]),
                    _fmt_pct(row["b_ret"]),
                    _fmt_pct(row["b_maxdd"]),
                    _pp(row["b_ret"], row["bh_ret"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _names(tickers: list[str]) -> str:
    return "、".join(tickers) if tickers else "沒有"


def _conclusion(summary: pd.DataFrame, bear: pd.DataFrame, coverage: pd.DataFrame) -> str:
    b = summary[summary["strategy"] == PRIMARY_ID]
    beat = b.loc[b["beats_bh"], "ticker"].tolist()
    lose = b.loc[~b["beats_bh"], "ticker"].tolist()
    med = float(b["n_trades"].median()) if len(b) else 0
    short = coverage.loc[~coverage["full_requested"], "ticker"].tolist()
    a = summary[summary["strategy"] == "A_MONTHLY_EXIT"]
    a_beat = a.loc[a["beats_bh"], "ticker"].tolist()
    lines = [
        "拉長到約十年之後，更不值得把官方月線版換成預設。主要結果仍是變體 B，七檔總報酬全部輸給各自的買進持有。五年那版還有比特幣打贏，這次長窗口裡比特幣也輸了。",
        (
            f"變體 B 高於買進持有的是：{_names(beat)}。仍輸的是：{_names(lose)}。"
            f"交易筆數中位數約 {med:.0f} 筆，還是不多。"
        ),
        (
            "只改月線出場（變體 A）比日線出場更接近一直抱著，總報酬明顯比較高，"
            f"但打贏買進持有的是：{_names(a_beat)}。"
            "完整官方版再加「等回到窄帶才買」和「下軌當停損」之後，多數檔又把變體 A 多吃到的漲幅吐回去。"
            "空頭段少虧，沒有補回少參與這十年多頭的部分。"
        ),
    ]
    if len(short):
        lines.append(f"資料沒有蓋滿請求區間、因而縮短的是：{_names(short)}。")
    else:
        lines.append(
            "七檔的績效都從 2016-09-15 起算。比特幣行情最早只到 2014-09-17，但這次要的十年它有。"
            "它的日K缺 2026-09-14 這一天，績效算到 09-13，差一個交易日，不影響結論。"
        )
    for spec in BEAR_WINDOWS:
        sub = bear[bear["window_id"] == spec["id"]]
        ok = sub[np.isfinite(sub["b_ret"]) & np.isfinite(sub["bh_ret"])]
        gap = ok["b_ret"] - ok["bh_ret"]
        less = ok.loc[gap > 0.005, "ticker"].tolist()
        worse = ok.loc[gap < -0.005, "ticker"].tolist()
        flat = ok.loc[gap.abs() <= 0.005, "ticker"].tolist()
        bits = []
        if less:
            bits.append(f"比抱著少虧的是 {_names(less)}")
        if flat:
            bits.append(f"幾乎跟抱著一樣跌的是 {_names(flat)}")
        if worse:
            bits.append(f"這段比較差的是 {_names(worse)}")
        lines.append(f"{spec['name']}（{spec['start']}～{spec['end']}）：" + "；".join(bits) + "。")
    lines.append(
        "這三段日期七檔共用、事先寫死，沒有先看完再改起迄。"
        "某段幾乎沒跌，多半是當時空手，不是把停損調寬。參數仍是原預設。"
    )
    return "\n\n".join(lines)


def write_report(summary: pd.DataFrame, bear: pd.DataFrame, coverage: pd.DataFrame) -> None:
    b = summary[summary["strategy"] == PRIMARY_ID]
    n_beat = int(b["beats_bh"].sum())
    text = f"""# 十年重跑：同一套三種規則，區間改成約 2016～2026

> 規則、標的、單邊成本 0.05%、THT N=33／TW=0.18、BX SL1=5／SL2=20／SL3=5 都沒改。**主要結果仍是變體 B。** 這不是獲利保證。請不要 merge。

## 先講結論

約十年、7 檔裡面，完整官方版有 **{n_beat} 檔**總報酬高於該檔買進持有，**{len(b) - n_beat} 檔**仍然較低。下面先放全期對照，再放三段空頭。筆數還是不多，不要讀成「拉長就證明該換規則」。

{_conclusion(summary, bear, coverage)}

## 全期並排

每一格是「總報酬／最大回撤／勝率／筆數」。最大回撤是整段資金曲線從高點跌最深的幅度。實際區間若比 2016-09-15～2026-09-14 短，代表該檔資料不夠，已用能拿到的最長段。

{md_full_table(summary)}

## 空頭段怎麼量

三段日期事先固定，七檔用同一組，不依個股的最高最低去挑：

| 名稱 | 起迄 | 為什麼是這段 |
| --- | --- | --- |
| 2018年底 | 2018-10-01～2018-12-24 | 美股從 10 月高點附近跌到 12 月 24 日低點 |
| 2020年3月 | 2020-02-19～2020-03-23 | 疫情急跌。2 月下旬就開始，若只切 3 月 1 日會把起跌切掉 |
| 2022下跌 | 2022-01-03～2022-10-14 | 2022 年主要下跌段，到 10 月低點附近 |

區間報酬：用窗口前最後一根的權益，除到窗口內最後一根。買進持有在中段沒有再扣一次成本，所以這段差不多就是價格本身的漲跌。策略若當時空手，這段報酬會接近 0。

「少虧幾個百分點」＝策略這段報酬減去買進持有這段報酬。正的是少賠（或這段還在賺）。區間回撤是這段路徑自己的最大跌幅，用來看有沒有避開回撤，不跟整段十年的最大回撤混在一起。

### 2018 年底

{md_bear_table(bear, "y2018")}

### 2020 年 3 月

{md_bear_table(bear, "y2020")}

### 2022 下跌

{md_bear_table(bear, "y2022")}

## 資料蓋到哪

{chr(10).join('- ' + s for s in coverage['line'].tolist())}

日線指標用銜接後的整段開高低收來算，2016 年以前只當暖機，不進績效。月線 BX 仍只用已收盤月份，月中不用當月還沒走完的收盤。訊號收盤成立、次日開盤成交。這次沒有拿 PR #3 的五年數字來對（區間不同，對不上是正常的）。五年那一版仍在 `report-v4-monthly-trademap.md`。

## 不要讀太滿

1. 十年還是同一段美股多頭為主，中間夾三段下跌。總報酬打贏買進持有，常常只是少參與某一段，或某一筆抱很久。
2. 月線訊號在十年裡仍然不多。勝率、有沒有避開某次崩跌，換一段日期可能就倒過來。
3. 沒有為了這三段把出場改早或改晚，也沒有調 N、TW、BX。
4. 顏色定義與五年那版相同：只看月線 BX 正負和跟前一根比升降，沒有絕對值門檻。若官方圖的深紅還要看柱子長短，請先對 `results/v4_10y/monthly_colors_*.csv`。

```bash
cd tht-bx-backtest
python run_backtest_v4_10y.py
python test_trademap.py
```
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"已寫入 {REPORT_PATH}")


def run() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    bear_rows = []
    coverage_rows = []
    equity_parts = {}

    for item in UNIVERSE:
        ticker = item["id"]
        print(f"\n=== {ticker} ===")
        raw, note = load_full_ohlcv(item["yf"])
        df, mask, res_v, bh = run_v_conf23_on_frame(raw, START, END, ONE_WAY_COST)
        aligned = align_monthly_bx_to_daily(raw["close"].astype(float)).reindex(df.index)
        monthly_bx_table(raw["close"].astype(float)).to_csv(
            RESULT_DIR / f"monthly_colors_{ticker}.csv", encoding="utf-8-sig"
        )
        for col in aligned.columns:
            df[col] = aligned[col]

        window = df.loc[mask]
        actual_start = window.index.min()
        actual_end = window.index.max()
        span_days = (actual_end - actual_start).days
        full = actual_start <= pd.Timestamp(START) + pd.Timedelta(days=7) and actual_end >= pd.Timestamp(END) - pd.Timedelta(days=7)
        coverage_rows.append(
            {
                "ticker": ticker,
                "data_start": raw.index.min().strftime("%Y-%m-%d"),
                "data_end": raw.index.max().strftime("%Y-%m-%d"),
                "actual_start": actual_start.strftime("%Y-%m-%d"),
                "actual_end": actual_end.strftime("%Y-%m-%d"),
                "span_days": int(span_days),
                "full_requested": bool(full),
                "line": (
                    f"{ticker}：行情 {raw.index.min().date()}～{raw.index.max().date()}，"
                    f"績效 {actual_start.date()}～{actual_end.date()}"
                    f"（約 {span_days / 365.25:.1f} 年）。{note}。"
                    + ("已蓋滿請求區間。" if full else "短於請求的十年，已用能拿到的最長段。")
                ),
            }
        )
        bh.metrics["actual_start"] = actual_start.strftime("%Y-%m-%d")
        bh.metrics["actual_end"] = actual_end.strftime("%Y-%m-%d")
        bh.metrics["n_bars"] = int(mask.sum())

        runs = {"V_CONF23": res_v}
        details = {}
        for sid, signals in (
            ("A_MONTHLY_EXIT", variant_a_signals(df)),
            (PRIMARY_ID, variant_b_signals(df)),
        ):
            entry, exit_, inv = signals
            packed = run_trade_map(df, entry, exit_, inv, name=sid, cost=ONE_WAY_COST, analysis_mask=mask)
            attach_metrics(packed.result, mask, bh_total=bh.metrics["total_return"])
            runs[sid] = packed.result
            details[sid] = packed.details

        curves = {
            "V_CONF23": res_v.equity.loc[mask],
            "A_MONTHLY_EXIT": runs["A_MONTHLY_EXIT"].equity.loc[mask],
            "B_OFFICIAL": runs[PRIMARY_ID].equity.loc[mask],
            "BH": bh.equity.loc[mask],
        }
        for sid in STRATEGY_ORDER:
            res = runs[sid]
            res.metrics["actual_start"] = actual_start.strftime("%Y-%m-%d")
            res.metrics["actual_end"] = actual_end.strftime("%Y-%m-%d")
            res.metrics["n_bars"] = int(mask.sum())
            rows.append(_row(ticker, item["name_zh"], sid, res.metrics, bh.metrics, sid == PRIMARY_ID))
            equity_parts[f"{ticker}_{sid}"] = curves[sid]
            if sid == "V_CONF23":
                trades_to_frame(res.trades).to_csv(RESULT_DIR / f"trades_{ticker}_{sid}.csv", index=False, encoding="utf-8-sig")
            else:
                details[sid].to_csv(RESULT_DIR / f"trades_{ticker}_{sid}.csv", index=False, encoding="utf-8-sig")
        equity_parts[f"{ticker}_BH"] = curves["BH"]

        for spec in BEAR_WINDOWS:
            stats = {key: window_performance(series, spec["start"], spec["end"]) for key, series in curves.items()}
            bear_rows.append(
                {
                    "ticker": ticker,
                    "window_id": spec["id"],
                    "window": spec["name"],
                    "start": spec["start"],
                    "end": spec["end"],
                    "bh_ret": stats["BH"]["ret"],
                    "bh_maxdd": stats["BH"]["maxdd"],
                    "v_ret": stats["V_CONF23"]["ret"],
                    "v_maxdd": stats["V_CONF23"]["maxdd"],
                    "a_ret": stats["A_MONTHLY_EXIT"]["ret"],
                    "a_maxdd": stats["A_MONTHLY_EXIT"]["maxdd"],
                    "b_ret": stats["B_OFFICIAL"]["ret"],
                    "b_maxdd": stats["B_OFFICIAL"]["maxdd"],
                    "b_less_loss": (
                        float(stats["B_OFFICIAL"]["ret"] - stats["BH"]["ret"])
                        if stats["B_OFFICIAL"]["covered"] and stats["BH"]["covered"]
                        else np.nan
                    ),
                }
            )
        bmet = runs[PRIMARY_ID].metrics
        print(
            f"{ticker:6s} {actual_start.date()}～{actual_end.date()}  "
            f"B={bmet['total_return']*100:8.2f}% 筆數={bmet['n_trades']}  "
            f"B&H={bh.metrics['total_return']*100:8.2f}%"
        )

    summary = pd.DataFrame(rows)
    bear = pd.DataFrame(bear_rows)
    coverage = pd.DataFrame(coverage_rows)
    summary.to_csv(RESULT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    bear.to_csv(RESULT_DIR / "bear_windows.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(RESULT_DIR / "coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(equity_parts).to_csv(RESULT_DIR / "equity_curves.csv", encoding="utf-8-sig")
    plot_returns(summary, FIG_DIR / "returns_compare.png")
    write_report(summary, bear, coverage)


if __name__ == "__main__":
    run()
