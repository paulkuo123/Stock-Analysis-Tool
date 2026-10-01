#!/usr/bin/env python3
"""拿掉最近兩年大多頭，重算約八年，並單獨看這兩年與三段下跌。

規則、成本、過線標準都不改。這支程式只切日期、跑表。

八年：2016-09-15～2024-08-31，獨立從現金重跑，最後一根收盤賣掉。
最近兩年：同一條十年資金曲線上，2024-09-01 之後到資料結尾的漲跌。
三段下跌沿用先前鎖死的日期。

用法：
    python run_backtest_v4_exbull.py
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
from engine import ONE_WAY_COST, _metrics_from_equity, attach_metrics, run_buy_and_hold, run_long_only_ex  # noqa: E402
from features import add_v2_features  # noqa: E402
from indicators import add_all_indicators  # noqa: E402
from monthly_bx import align_monthly_bx_to_daily  # noqa: E402
from run_backtest import pct  # noqa: E402
from run_backtest_v3_multi import UNIVERSE, analysis_mask, v_conf23_spec  # noqa: E402
from run_backtest_v4_10y import END, START, load_full_ohlcv  # noqa: E402
from sizing import (  # noqa: E402
    CAPTURE_BAR,
    DD_IMPROVE_BAR,
    capture_ratio,
    drawdown_improvement,
    meets_reading_bar,
    run_sized_book,
    target_book,
)
from trademap import run_trade_map, variant_a_signals, variant_b_signals  # noqa: E402

RESULT_DIR = ROOT / "results" / "v4_8y_exbull"
REPORT_PATH = ROOT / "report-v4-8y-exbull.md"

EIGHT_END = "2024-08-31"
TWO_START = "2024-09-01"

ORDER = [
    "BH",
    "V_CONF23",
    "A_MONTHLY_EXIT",
    "B_OFFICIAL",
    "C1_0",
    "C1_25",
    "C2_150",
    "C2_60",
    "C3",
]
LABEL = {
    "BH": "買進持有",
    "V_CONF23": "舊版（日線出場）",
    "A_MONTHLY_EXIT": "變體 A",
    "B_OFFICIAL": "變體 B",
    "C1_0": "C1 深紅到 0%",
    "C1_25": "C1 深紅留 25%",
    "C2_150": "C2 加到 150%",
    "C2_60": "C2 60% 加到 100%",
    "C3": "C3 組合",
}
JUDGED = [s for s in ORDER if s != "BH"]


def _pp(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x * 100:+.1f} 個百分點"


def _ratio(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x * 100:.1f}%"


def _names(tickers: list[str]) -> str:
    return "、".join(tickers) if tickers else "沒有任何一檔"


def _yn(flag: bool) -> str:
    return "是" if flag else "否"


def _prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = add_v2_features(add_all_indicators(raw))
    aligned = align_monthly_bx_to_daily(raw["close"].astype(float)).reindex(df.index)
    for col in aligned.columns:
        df[col] = aligned[col]
    return df


def _run_mask(df: pd.DataFrame, mask: pd.Series) -> dict[str, pd.Series]:
    """這一窗從現金重跑。回傳各策略的資金曲線（最後一天已賣出）。"""
    if int(mask.sum()) < 50:
        raise RuntimeError(f"有效 K 棒過少：{int(mask.sum())}")
    bh = run_buy_and_hold(df, name="BH", cost=ONE_WAY_COST, analysis_mask=mask)
    attach_metrics(bh, mask, bh_total=None)
    curves = {"BH": bh.equity}
    spec = v_conf23_spec()
    old = run_long_only_ex(
        df,
        spec.entry_fn(df),
        spec.exit_fn(df),
        name="V_CONF23",
        cost=ONE_WAY_COST,
        analysis_mask=mask,
        stop_pct=spec.stop_pct,
        atr_stop_k=spec.atr_stop_k,
        trail_atr_k=spec.trail_atr_k,
        time_exit_bars=spec.time_exit_bars,
        cooldown_losses=spec.cooldown_losses,
    )
    curves["V_CONF23"] = old.equity
    for sid, signals in (
        ("A_MONTHLY_EXIT", variant_a_signals(df)),
        ("B_OFFICIAL", variant_b_signals(df)),
    ):
        entry, exit_, inv = signals
        result = run_trade_map(df, entry, exit_, inv, name=sid, cost=ONE_WAY_COST, analysis_mask=mask)
        curves[sid] = result.result.equity
    bh_total = float(bh.metrics["total_return"])
    for sid, target in target_book(df).items():
        sized = run_sized_book(df, target, mask, name=sid, cost=ONE_WAY_COST, bh_total=bh_total)
        curves[sid] = sized.equity
    return curves


def _row_from_equity(ticker: str, sid: str, equity: pd.Series, bh_equity: pd.Series, window: str) -> dict:
    metrics = _metrics_from_equity(equity.dropna(), [], bh_total=None, initial=1.0)
    bh = _metrics_from_equity(bh_equity.dropna(), [], bh_total=None, initial=1.0)
    total = float(metrics["total_return"])
    maxdd = float(metrics["maxdd"])
    bh_total = float(bh["total_return"])
    bh_dd = float(bh["maxdd"])
    capture = 1.0 if sid == "BH" else capture_ratio(total, bh_total)
    improve = 0.0 if sid == "BH" else drawdown_improvement(maxdd, bh_dd)
    eq = equity.dropna()
    return {
        "window": window,
        "ticker": ticker,
        "strategy": sid,
        "total_return": total,
        "maxdd": maxdd,
        "bh_total_return": bh_total,
        "bh_maxdd": bh_dd,
        "capture": capture,
        "dd_improve": improve,
        "gap": 0.0 if sid == "BH" else total - bh_total,
        "meets_bar": meets_reading_bar(capture, improve) if sid != "BH" else False,
        "actual_start": eq.index.min().strftime("%Y-%m-%d") if len(eq) else "",
        "actual_end": eq.index.max().strftime("%Y-%m-%d") if len(eq) else "",
        "n_bars": int(len(eq)),
    }


def _two_year_row(ticker: str, sid: str, equity: pd.Series, bh_equity: pd.Series) -> dict:
    stat = window_performance(equity, TWO_START, END)
    bh = window_performance(bh_equity, TWO_START, END)
    total = float(stat["ret"])
    bh_total = float(bh["ret"])
    capture = 1.0 if sid == "BH" else capture_ratio(total, bh_total)
    return {
        "window": "recent",
        "ticker": ticker,
        "strategy": sid,
        "total_return": total,
        "maxdd": float(stat["maxdd"]),
        "bh_total_return": bh_total,
        "bh_maxdd": float(bh["maxdd"]),
        "capture": capture,
        "dd_improve": 0.0 if sid == "BH" else drawdown_improvement(float(stat["maxdd"]), float(bh["maxdd"])),
        "gap": 0.0 if sid == "BH" else total - bh_total,
        "meets_bar": False,
        "actual_start": TWO_START,
        "actual_end": END,
        "n_bars": int(stat["n_bars"]),
    }


def _pass_line(block: pd.DataFrame, sid: str) -> str:
    sub = block[block["strategy"] == sid]
    hit = sub.loc[sub["meets_bar"], "ticker"].tolist()
    if hit:
        head = f"{LABEL[sid]} 過線的是 {_names(hit)}（{len(hit)} 檔）。"
    else:
        head = f"{LABEL[sid]} 沒有任何一檔過線。"
    return (
        head
        + f"取回比例中位數 {_ratio(float(sub['capture'].median()))}，"
        + f"最大回撤差中位數 {_pp(float(sub['dd_improve'].median()))}。"
    )


def _bull_did_not_save(full: pd.DataFrame, eight: pd.DataFrame) -> str:
    lines = []
    for sid in ("A_MONTHLY_EXIT", "B_OFFICIAL", "C1_0", "C3"):
        gained = []
        for item in UNIVERSE:
            ticker = item["id"]
            a = bool(full[(full["ticker"] == ticker) & (full["strategy"] == sid)]["meets_bar"].iloc[0])
            b = bool(eight[(eight["ticker"] == ticker) & (eight["strategy"] == sid)]["meets_bar"].iloc[0])
            if b and not a:
                gained.append(ticker)
        if not gained:
            continue
        lines.append(
            f"{LABEL[sid]} 在整段十年沒過線的檔裡，拿掉最近兩年後 {_names(gained)} 過了線。"
            "少掉一段沒跟上的上漲，取回比例才碰到 80%。這不是被大多頭撐起來，也不能改口說這版可以用。"
        )
    return "\n\n".join(lines)


def _changed(full: pd.DataFrame, eight: pd.DataFrame) -> str:
    lines = []
    for sid in JUDGED:
        lost = []
        kept = []
        gained = []
        for item in UNIVERSE:
            ticker = item["id"]
            a = bool(full[(full["ticker"] == ticker) & (full["strategy"] == sid)]["meets_bar"].iloc[0])
            b = bool(eight[(eight["ticker"] == ticker) & (eight["strategy"] == sid)]["meets_bar"].iloc[0])
            if a and b:
                kept.append(ticker)
            elif a and not b:
                lost.append(ticker)
            elif b and not a:
                gained.append(ticker)
        lines.append(
            f"{LABEL[sid]}：十年過線、拿掉最近兩年仍過線的是 {_names(kept)}。"
            f"十年過線、八年沒過的是 {_names(lost)}。"
            f"八年才過、十年沒過的是 {_names(gained)}。"
        )
    return "\n\n".join(lines)


def _c1_plain(eight: pd.DataFrame, full: pd.DataFrame, recent: pd.DataFrame) -> str:
    e = eight[eight["strategy"] == "C1_25"]
    f = full[full["strategy"] == "C1_25"]
    r = recent[recent["strategy"] == "C1_25"]
    e_hit = e.loc[e["meets_bar"], "ticker"].tolist()
    f_hit = f.loc[f["meets_bar"], "ticker"].tolist()
    if e_hit:
        hold = f"拿掉最近兩年後，C1 深紅留 25% 仍有 {_names(e_hit)} 過線，不是七檔都成立。"
    else:
        hold = "拿掉最近兩年後，C1 深紅留 25% 沒有任何一檔過線。這條在八年樣本上不成立。"
    lost = [t for t in f_hit if t not in e_hit]
    kept = [t for t in f_hit if t in e_hit]
    extra = [t for t in e_hit if t not in f_hit]
    change = (
        f"十年過線的是 {_names(f_hit)}。"
        f"其中八年仍過的是 {_names(kept)}，八年掉下去的是 {_names(lost)}。"
    )
    if extra:
        change += f"八年新過線、十年沒過的是 {_names(extra)}。"
    else:
        change += "沒有任何一檔是八年才新過線。"
    why = []
    for ticker in lost:
        fr = f[f["ticker"] == ticker].iloc[0]
        er = e[e["ticker"] == ticker].iloc[0]
        if float(er["capture"]) >= CAPTURE_BAR and float(er["dd_improve"]) < DD_IMPROVE_BAR <= float(fr["dd_improve"]):
            why.append(
                f"{ticker} 八年取回仍有 {_ratio(float(er['capture']))}，"
                f"最大回撤差從 {_pp(float(fr['dd_improve']))} 縮到 {_pp(float(er['dd_improve']))}。"
                f"C1 自己的最大回撤仍是 {pct(float(er['maxdd']))}，"
                f"十年買進持有的最大回撤是 {pct(float(fr['bh_maxdd']))}，八年是 {pct(float(er['bh_maxdd']))}。"
                "十年能過 10 個百分點，是因為買進持有在最近兩年把回撤拉得更深，不是 C1 在八年裡多擋了一截。"
            )
    near = e[(e["capture"] >= 0.70) & (e["capture"] < CAPTURE_BAR) & (e["dd_improve"] >= DD_IMPROVE_BAR)]
    if len(near):
        bits_near = [
            f"{row['ticker']} 取回 {_ratio(float(row['capture']))}、回撤差 {_pp(float(row['dd_improve']))}"
            for _, row in near.iterrows()
        ]
        why.append(
            "八年取回不到 80%、但回撤差有過 10 個百分點的是 "
            + "；".join(bits_near)
            + "。這不算過線。這幾檔的十年取回更低，是最近兩年少賺把取回拉下來，不是大多頭把他們撐過線。"
        )
    bits = []
    for item in UNIVERSE:
        ticker = item["id"]
        row = r[r["ticker"] == ticker].iloc[0]
        bits.append(
            f"{ticker} 這兩年買進持有 {pct(float(row['bh_total_return']))}、"
            f"C1 {pct(float(row['total_return']))}、報酬差 {_pp(float(row['gap']))}"
        )
    lag = r.loc[r["gap"] < -0.05, "ticker"].tolist()
    lead = r.loc[r["gap"] > 0.05, "ticker"].tolist()
    drag = (
        f"最近兩年 C1 明顯少賺的是 {_names(lag)}。"
        f"這兩年 C1 比抱著還多賺的是 {_names(lead)}。"
    )
    parts = [hold, change]
    parts.extend(why)
    parts.append(drag)
    parts.append("這兩年各檔：" + "；".join(bits) + "。")
    return "\n\n".join(parts)


def _summary_table(block: pd.DataFrame) -> str:
    header = "| 策略 | 過線檔數 | 取回比例中位數 | 最大回撤差中位數 | 策略報酬中位數 | 買進持有報酬中位數 |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: |"
    lines = [header, sep]
    bh_med = float(block[block["strategy"] == "BH"]["total_return"].median())
    for sid in JUDGED:
        sub = block[block["strategy"] == sid]
        lines.append(
            "| "
            + " | ".join(
                [
                    LABEL[sid],
                    str(int(sub["meets_bar"].sum())),
                    _ratio(float(sub["capture"].median())),
                    _pp(float(sub["dd_improve"].median())),
                    pct(float(sub["total_return"].median())),
                    pct(bh_med),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _ticker_table(block: pd.DataFrame) -> str:
    header = "| 策略 | 買進持有報酬 | 策略報酬 | 取回比例 | 最大回撤 | 最大回撤差 | 過線 |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | --- |"
    lines = [header, sep]
    bh = float(block[block["strategy"] == "BH"]["total_return"].iloc[0])
    for sid in ORDER:
        r = block[block["strategy"] == sid].iloc[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    LABEL[sid],
                    pct(bh),
                    pct(r["total_return"]),
                    _ratio(r["capture"]),
                    pct(r["maxdd"]),
                    _pp(r["dd_improve"]),
                    "—" if sid == "BH" else _yn(bool(r["meets_bar"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _recent_table(block: pd.DataFrame) -> str:
    header = "| 策略 | 買進持有這兩年 | 策略這兩年 | 報酬差 | 取回比例 | 這兩年最大回撤 |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: |"
    lines = [header, sep]
    bh = float(block[block["strategy"] == "BH"]["total_return"].iloc[0])
    for sid in ORDER:
        r = block[block["strategy"] == sid].iloc[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    LABEL[sid],
                    pct(bh),
                    pct(r["total_return"]),
                    _pp(r["gap"]),
                    _ratio(r["capture"]),
                    pct(r["maxdd"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _bear_table(bear: pd.DataFrame, window_id: str) -> str:
    sub = bear[bear["window_id"] == window_id]
    header = "| 標的 | 買進持有 | " + " | ".join(f"{LABEL[s]} 少虧" for s in JUDGED) + " |"
    sep = "| --- | ---: | " + " | ".join(["---:"] * len(JUDGED)) + " |"
    lines = [header, sep]
    for item in UNIVERSE:
        row = sub[sub["ticker"] == item["id"]].iloc[0]
        cells = [item["id"], pct(float(row["BH"]))]
        bh = float(row["BH"])
        for sid in JUDGED:
            gap = float(row[sid]) - bh
            cells.append(f"{pct(float(row[sid]))}（{_pp(gap)}）")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _stress_plain(bear: pd.DataFrame) -> str:
    lines = []
    for spec in BEAR_WINDOWS:
        sub = bear[bear["window_id"] == spec["id"]]
        less = []
        worse = []
        for _, row in sub.iterrows():
            gap = float(row["C1_25"]) - float(row["BH"])
            if gap > 0.005:
                less.append(f"{row['ticker']} 少虧 {_pp(gap)}")
            elif gap < -0.005:
                worse.append(f"{row['ticker']} 比抱著多跌 {abs(gap) * 100:.1f} 個百分點")
        lines.append(
            f"{spec['name']}：C1 深紅留 25% " + ("；".join(less) if less else "沒有少虧") + "。"
            + ("這段比抱著更差的是 " + "；".join(worse) + "。" if worse else "沒有比抱著更差的檔。")
        )
    return "\n\n".join(lines)


def _rules() -> str:
    return f"""日期在跑之前就切好，看完數字沒有改策略，也沒有改 80% 和 10 個百分點。

- 八年：{START}～{EIGHT_END}。各策略從現金重新買，最後一根收盤賣掉，單邊成本仍是 0.05%。若 8 月 31 日不是交易日，就停在該日之前最後一根。
- 最近兩年：{TWO_START}～{END}。這不是把倉位清空再重買，而是十年那條資金曲線在這段裡的漲跌，9 月沿用 8 月收盤已經有的部位。報酬差是策略這兩年的報酬減去買進持有這兩年的報酬，負的代表少賺或多虧。
- 過線只拿來判八年（以及對照用的整段十年）：策略總報酬至少是同一段買進持有的 {CAPTURE_BAR * 100:.0f}%，而且最大回撤至少比買進持有淺 {DD_IMPROVE_BAR * 100:.0f} 個百分點。兩個都要中。最近兩年太短，不拿這條線判成敗，只看拖累。
- 最大回撤差是策略最大回撤減去買進持有最大回撤，正的代表跌得比較淺。
- 三段下跌日期跟先前十年測試相同，沒有依個股高低點重挑。少虧是該段策略報酬減去買進持有報酬。
- 版本就是先前十年那一批：舊版日線出場、變體 A、變體 B、C1 深紅到 0% 與留 25%、C2 加到 150% 與 60% 加到 100%、C3 組合。權重沒有改。C2 的 150% 和 C3 的加碼仍不計借款利息。

年化算法沒有用在這張表的過線判斷。比特幣一年交易日比較多，不影響這裡的總報酬比較。"""


def write_report(summary: pd.DataFrame, bear: pd.DataFrame) -> None:
    eight = summary[summary["window"] == "eight"]
    full = summary[summary["window"] == "full"]
    recent = summary[summary["window"] == "recent"]
    eight_end = eight["actual_end"].iloc[0]
    full_end = full["actual_end"].iloc[0]
    blocks = []
    for item in UNIVERSE:
        ticker = item["id"]
        block = eight[eight["ticker"] == ticker]
        blocks.append(f"### {item['name_zh']}（{ticker}，{block['actual_start'].iloc[0]}～{block['actual_end'].iloc[0]}）\n\n{_ticker_table(block)}\n")
    recent_blocks = []
    for item in UNIVERSE:
        ticker = item["id"]
        block = recent[recent["ticker"] == ticker]
        recent_blocks.append(f"### {item['name_zh']}（{ticker}）\n\n{_recent_table(block)}\n")
    bear_blocks = []
    for spec in BEAR_WINDOWS:
        bear_blocks.append(
            f"### {spec['name']}（{spec['start']}～{spec['end']}）\n\n{spec['note']}\n\n格子是區間報酬，括號是比買進持有少虧的幅度。正的代表這段少賠。\n\n{_bear_table(bear, spec['id'])}"
        )
    text = f"""# 拿掉最近兩年大多頭之後

> 七檔、單邊成本 0.05%、規則與過線標準都不改。八年算到 {eight_end}，十年對照算到 {full_end}。**不要 merge。不是獲利保證。**

## 先講結論

{_c1_plain(eight, full, recent)}

八年各版過線：

{_pass_line(eight, "C1_25")}

{chr(10).join(_pass_line(eight, sid) for sid in JUDGED if sid != "C1_25")}

跟整段十年比，誰的過線變了：

{_changed(full, eight)}

{_bull_did_not_save(full, eight)}

三段下跌裡，C1 深紅留 25% 少虧了多少：

{_stress_plain(bear)}

同一段行情已經切過前後五年，這次再拿掉 2024-09 之後。多切幾次，總會有一種切法讓某一檔看起來還行。樣本仍是這七檔。沒有過線就不要說成還能用。C2 加到 150% 沒有扣借款利息，那一版的報酬會比真實融資帳戶好看。

## 這次怎麼切

{_rules()}

## 八年：2016-09 到 2024-08

{_summary_table(eight)}

{chr(10).join(blocks)}

## 最近兩年：2024-09 到 2026-09

這段是十年曲線的尾端，用來看大多頭對各策略是幫忙還是拖累。報酬差為負，就是這兩年少賺。

{chr(10).join(recent_blocks)}

## 三段下跌少虧多少

用的是整段十年的資金曲線。這三段都在 2024 年以前，所以跟「有沒有最近兩年」無關，只回答下跌當下少虧多少。少虧不等於過線。

{chr(10).join(bear_blocks)}

```bash
cd tht-bx-backtest
python run_backtest_v4_exbull.py
```
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"已寫入 {REPORT_PATH}")


def run() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    bear_rows: list[dict] = []
    for item in UNIVERSE:
        ticker = item["id"]
        print(f"\n=== {ticker} ===")
        raw, _note = load_full_ohlcv(item["yf"])
        df = _prepare(raw)
        full_mask = analysis_mask(df, START, END)
        eight_mask = analysis_mask(df, START, EIGHT_END) & full_mask
        full_curves = _run_mask(df, full_mask)
        eight_curves = _run_mask(df, eight_mask)
        for sid in ORDER:
            rows.append(_row_from_equity(ticker, sid, full_curves[sid].loc[full_mask], full_curves["BH"].loc[full_mask], "full"))
            rows.append(_row_from_equity(ticker, sid, eight_curves[sid].loc[eight_mask], eight_curves["BH"].loc[eight_mask], "eight"))
            rows.append(_two_year_row(ticker, sid, full_curves[sid], full_curves["BH"]))
        c1 = [r for r in rows if r["ticker"] == ticker and r["strategy"] == "C1_25" and r["window"] == "eight"][-1]
        print(
            f"  8y C1 {c1['total_return']*100:8.1f}%  bh {c1['bh_total_return']*100:8.1f}%  "
            f"cap {c1['capture']*100:5.1f}%  dd {_pp(c1['dd_improve'])}  hit={c1['meets_bar']}"
        )
        for spec in BEAR_WINDOWS:
            rec = {"ticker": ticker, "window_id": spec["id"], "window": spec["name"]}
            for sid in ORDER:
                rec[sid] = window_performance(full_curves[sid], spec["start"], spec["end"])["ret"]
            bear_rows.append(rec)

    summary = pd.DataFrame(rows)
    bear = pd.DataFrame(bear_rows)
    summary.to_csv(RESULT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    bear.to_csv(RESULT_DIR / "bear_windows.csv", index=False, encoding="utf-8-sig")
    write_report(summary, bear)


if __name__ == "__main__":
    run()
