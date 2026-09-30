#!/usr/bin/env python3
"""十年區間的日線預警減碼，並排買進持有與 C1 深紅留 25%。

規則在 early_warning.py 裡寫死。這支程式只負責跑、算表，不改比例。
前後兩半以 2021-09-15 切開，前半到 2021-09-14，後半從 2021-09-15 起。兩段不重疊。

用法：
    python run_backtest_v4_early.py
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
from early_warning import warning_book  # noqa: E402
from engine import ONE_WAY_COST, attach_metrics, run_buy_and_hold  # noqa: E402
from features import add_v2_features  # noqa: E402
from indicators import add_all_indicators  # noqa: E402
from monthly_bx import align_monthly_bx_to_daily, align_weekly_bx_to_daily  # noqa: E402
from run_backtest import pct  # noqa: E402
from run_backtest_v3_multi import UNIVERSE, analysis_mask  # noqa: E402
from run_backtest_v4_10y import END, START, load_full_ohlcv  # noqa: E402
from sizing import (  # noqa: E402
    CAPTURE_BAR,
    DD_IMPROVE_BAR,
    c1_targets,
    capture_ratio,
    drawdown_improvement,
    meets_reading_bar,
    run_sized_book,
)

RESULT_DIR = ROOT / "results" / "v4_10y_early"
REPORT_PATH = ROOT / "report-v4-10y-early.md"

HALF1_END = "2021-09-14"
HALF2_START = "2021-09-15"
HALVES = (
    ("full", "約十年", START, END),
    ("first", "前五年", START, HALF1_END),
    ("second", "後五年", HALF2_START, END),
)
ORDER = ["BH", "C1_25", "E1", "E2", "E3"]
LABEL = {
    "BH": "買進持有",
    "C1_25": "C1 深紅留 25%",
    "E1": "E1 日線跌破下軌",
    "E2": "E2 週線轉負",
    "E3": "E3 分批回補",
}
FOCUS = ("TSLA", "NVDA")


def _pp(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x * 100:+.1f} 個百分點"


def _ratio(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x * 100:.1f}%"


def _sharpe(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:.2f}"


def _weight(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x * 100:.1f}%"


def _names(tickers: list[str]) -> str:
    return "、".join(tickers) if tickers else "沒有任何一檔"


def _one_row(ticker: str, sid: str, metrics: dict, avg_weight: float, n_orders: float, bh: dict, half: str) -> dict:
    total = float(metrics["total_return"])
    maxdd = float(metrics["maxdd"])
    bh_total = float(bh["total_return"])
    bh_dd = float(bh["maxdd"])
    capture = 1.0 if sid == "BH" else capture_ratio(total, bh_total)
    improve = 0.0 if sid == "BH" else drawdown_improvement(maxdd, bh_dd)
    return {
        "half": half,
        "ticker": ticker,
        "strategy": sid,
        "label": LABEL[sid],
        "total_return": total,
        "maxdd": maxdd,
        "ann_return": float(metrics["ann_return"]) if pd.notna(metrics["ann_return"]) else np.nan,
        "sharpe": float(metrics["sharpe"]) if pd.notna(metrics["sharpe"]) else np.nan,
        "avg_weight": float(avg_weight) if pd.notna(avg_weight) else np.nan,
        "n_orders": int(n_orders),
        "capture": capture,
        "dd_improve": improve,
        "meets_bar": meets_reading_bar(capture, improve) if sid != "BH" else False,
        "actual_start": metrics.get("actual_start", ""),
        "actual_end": metrics.get("actual_end", ""),
    }


def _prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = add_v2_features(add_all_indicators(raw))
    monthly = align_monthly_bx_to_daily(raw["close"].astype(float)).reindex(df.index)
    weekly = align_weekly_bx_to_daily(raw["close"].astype(float)).reindex(df.index)
    for col in monthly.columns:
        df[col] = monthly[col]
    for col in weekly.columns:
        df[col] = weekly[col]
    return df


def _mask_for(df: pd.DataFrame, start: str, end: str, full_mask: pd.Series) -> pd.Series:
    window = analysis_mask(df, start, end)
    return window & full_mask.reindex(df.index).fillna(False)


def _run_half(df: pd.DataFrame, mask: pd.Series, ticker: str, half: str) -> tuple[list[dict], dict[str, pd.Series]]:
    if int(mask.sum()) < 50:
        raise RuntimeError(f"{ticker} {half} 有效K棒過少：{int(mask.sum())}")
    window_index = df.index[mask.to_numpy()]
    actual_start = window_index.min().strftime("%Y-%m-%d")
    actual_end = window_index.max().strftime("%Y-%m-%d")
    bh = run_buy_and_hold(df, name="BH", cost=ONE_WAY_COST, analysis_mask=mask)
    attach_metrics(bh, mask, bh_total=None)
    bh.metrics["actual_start"] = actual_start
    bh.metrics["actual_end"] = actual_end
    bh_total = float(bh.metrics["total_return"])
    rows = [_one_row(ticker, "BH", bh.metrics, 1.0, 2, bh.metrics, half)]
    curves = {"BH": bh.equity}
    targets = {"C1_25": c1_targets(df, 0.25)}
    targets.update(warning_book(df))
    for sid in ("C1_25", "E1", "E2", "E3"):
        sized = run_sized_book(df, targets[sid], mask, name=sid, cost=ONE_WAY_COST, bh_total=bh_total)
        sized.metrics["actual_start"] = actual_start
        sized.metrics["actual_end"] = actual_end
        rows.append(_one_row(ticker, sid, sized.metrics, sized.avg_weight, sized.n_orders, bh.metrics, half))
        curves[sid] = sized.equity
    return rows, curves


def _ticker_table(block: pd.DataFrame) -> str:
    header = "| 策略 | 總報酬 | 最大回撤 | 年化 | Sharpe | 平均持倉 | 換手次數 | 報酬取回 | 回撤改善 |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [header, sep]
    for sid in ORDER:
        r = block[block["strategy"] == sid].iloc[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    LABEL[sid],
                    pct(r["total_return"]),
                    pct(r["maxdd"]),
                    pct(r["ann_return"]),
                    _sharpe(r["sharpe"]),
                    _weight(r["avg_weight"]),
                    str(int(r["n_orders"])),
                    _ratio(r["capture"]),
                    _pp(r["dd_improve"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _summary_table(summary: pd.DataFrame) -> str:
    header = "| 策略 | 七檔裡同時達標 | 取回比例中位數 | 回撤改善中位數 | 總報酬中位數 | 最大回撤中位數 | 平均持倉中位數 |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [header, sep]
    for sid in ORDER:
        if sid == "BH":
            continue
        sub = summary[summary["strategy"] == sid]
        lines.append(
            "| "
            + " | ".join(
                [
                    LABEL[sid],
                    str(int(sub["meets_bar"].sum())),
                    _ratio(float(sub["capture"].median())),
                    _pp(float(sub["dd_improve"].median())),
                    pct(float(sub["total_return"].median())),
                    pct(float(sub["maxdd"].median())),
                    _weight(float(sub["avg_weight"].median())),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _bear_table(bear: pd.DataFrame, window_id: str) -> str:
    sub = bear[bear["window_id"] == window_id]
    header = "| 標的 | " + " | ".join(LABEL[s] for s in ORDER) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(ORDER)) + " |"
    lines = [header, sep]
    for item in UNIVERSE:
        row = sub[sub["ticker"] == item["id"]]
        cells = [item["id"]]
        for sid in ORDER:
            val = float(row.iloc[0][sid]) if len(row) else np.nan
            cells.append(pct(val) if np.isfinite(val) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _half_table(block: pd.DataFrame) -> str:
    header = "| 策略 | 總報酬 | 最大回撤 | 年化 | Sharpe | 平均持倉 | 換手次數 | 報酬取回 | 回撤改善 | 達標 |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    lines = [header, sep]
    for sid in ORDER:
        r = block[block["strategy"] == sid].iloc[0]
        flag = "—" if sid == "BH" else ("是" if bool(r["meets_bar"]) else "否")
        lines.append(
            "| "
            + " | ".join(
                [
                    LABEL[sid],
                    pct(r["total_return"]),
                    pct(r["maxdd"]),
                    pct(r["ann_return"]),
                    _sharpe(r["sharpe"]),
                    _weight(r["avg_weight"]),
                    str(int(r["n_orders"])),
                    _ratio(r["capture"]),
                    _pp(r["dd_improve"]),
                    flag,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _both_halves(summary: pd.DataFrame) -> str:
    lines = []
    first = summary[summary["half"] == "first"]
    second = summary[summary["half"] == "second"]
    for sid in ("C1_25", "E1", "E2", "E3"):
        both = []
        only_first = []
        only_second = []
        neither = []
        for item in UNIVERSE:
            ticker = item["id"]
            a = bool(first[(first["ticker"] == ticker) & (first["strategy"] == sid)]["meets_bar"].iloc[0])
            b = bool(second[(second["ticker"] == ticker) & (second["strategy"] == sid)]["meets_bar"].iloc[0])
            if a and b:
                both.append(ticker)
            elif a:
                only_first.append(ticker)
            elif b:
                only_second.append(ticker)
            else:
                neither.append(ticker)
        lines.append(
            f"{LABEL[sid]}：兩半都過線的是 {_names(both)}。"
            f"只在前五年過線的是 {_names(only_first)}。"
            f"只在後五年過線的是 {_names(only_second)}。"
            f"兩半都沒過的是 {_names(neither)}。"
        )
    any_all = []
    for sid in ("C1_25", "E1", "E2", "E3"):
        n_both = 0
        for item in UNIVERSE:
            ticker = item["id"]
            a = bool(first[(first["ticker"] == ticker) & (first["strategy"] == sid)]["meets_bar"].iloc[0])
            b = bool(second[(second["ticker"] == ticker) & (second["strategy"] == sid)]["meets_bar"].iloc[0])
            n_both += int(a and b)
        if n_both == len(UNIVERSE):
            any_all.append(LABEL[sid])
    if any_all:
        lines.append("七檔在兩半都過線的是 " + "、".join(any_all) + "。")
    else:
        lines.append("沒有一個版本在七檔、前後兩半都同時過線。")
    return "\n\n".join(lines)


def _focus_note(summary: pd.DataFrame, bear: pd.DataFrame) -> str:
    full = summary[summary["half"] == "full"]
    lines = []
    for ticker in FOCUS:
        block = full[full["ticker"] == ticker]
        c1 = block[block["strategy"] == "C1_25"].iloc[0]
        bits = [f"{ticker} 的 C1 取回 {_ratio(float(c1['capture']))}、回撤改善 {_pp(float(c1['dd_improve']))}，{'有' if c1['meets_bar'] else '沒有'}同時達標。"]
        for sid in ("E1", "E2", "E3"):
            row = block[block["strategy"] == sid].iloc[0]
            dd_vs_c1 = float(row["maxdd"]) - float(c1["maxdd"])
            cap_vs_c1 = float(row["capture"]) - float(c1["capture"])
            if dd_vs_c1 > 0.005 and cap_vs_c1 >= -0.005:
                how = "相對 C1，回撤更淺，取回比例沒有變低"
            elif dd_vs_c1 > 0.005:
                how = "相對 C1，回撤更淺，但取回比例比較低"
            elif dd_vs_c1 < -0.005:
                how = "相對 C1，最大回撤更深"
            else:
                how = "相對 C1，最大回撤差不多"
            bar = "有同時達標" if row["meets_bar"] else "沒有同時達標"
            bits.append(
                f"{LABEL[sid]}：{how}（回撤差 {dd_vs_c1 * 100:+.1f} 個百分點，取回差 {cap_vs_c1 * 100:+.1f} 個百分點），{bar}。"
            )
        sub = bear[bear["ticker"] == ticker]
        for spec in BEAR_WINDOWS:
            w = sub[sub["window_id"] == spec["id"]].iloc[0]
            bits.append(
                f"{spec['name']}區間報酬：買進持有 {pct(float(w['BH']))}，"
                f"C1 {pct(float(w['C1_25']))}，E1 {pct(float(w['E1']))}，"
                f"E2 {pct(float(w['E2']))}，E3 {pct(float(w['E3']))}。"
            )
        lines.append(" ".join(bits))
    return "\n\n".join(lines)


def _conclusion(summary: pd.DataFrame, bear: pd.DataFrame) -> str:
    full = summary[summary["half"] == "full"]
    lines = [
        (
            f"讀法事先固定：總報酬至少取回該段買進持有的 {CAPTURE_BAR * 100:.0f}%，"
            f"而且最大回撤至少淺 {DD_IMPROVE_BAR * 100:.0f} 個百分點。"
            "前後兩半各自跟該半的買進持有比，不是拿全期買進持有當分母。"
        )
    ]
    for sid in ("C1_25", "E1", "E2", "E3"):
        sub = full[full["strategy"] == sid]
        hit = sub.loc[sub["meets_bar"], "ticker"].tolist()
        if hit:
            head = f"約十年，{LABEL[sid]} 同時達標的是 {_names(hit)}。"
        else:
            head = f"約十年，{LABEL[sid]}：沒有任何一檔同時達標。"
        lines.append(
            head
            + f"取回比例中位數 {_ratio(float(sub['capture'].median()))}，"
            f"回撤改善中位數 {_pp(float(sub['dd_improve'].median()))}。"
        )
    lines.append(_focus_note(summary, bear))
    lines.append(_both_halves(summary))
    return "\n\n".join(lines)


def _rules_section() -> str:
    return """比例在跑之前就寫死，看完數字沒有改。

33 FVB 是 33 日均線上下各 0.18 倍標準差的帶。收盤在下軌和上軌之間叫回到帶內。跌破下軌是收盤嚴格低於下軌，用的是已經收完的那根，下一個開盤才調部位。月線 BX 要等該月最後一根日線收盤才知道，月中沿用上一個月。深紅是月線 BX 在零軸下、而且比上個月更差。回綠是淺綠或深綠，也就是月線 BX 在零軸上。週線用週日結束的那一週，最後一根日線收盤後才生效；股票多半是週五，比特幣多半是週日。週線轉負在這裡是持續狀態：最近一根已收盤週線的 BX 小於 0 就維持預警，直到週線 BX 回到大於或等於 0。

- **C1**：平時 100%。漸增淺紅 50%。深紅 25%。回綠 100%。持平負區維持原部位。這次只留先前那版「深紅 25%」，用來並排。
- **E1**：平時 100%。日線收盤跌破下軌減到 50%。月線收盤深紅減到 25%，同一天優先於 50%。回到帶內、而且月線不是深紅，回到 100%。收盤在上軌之外維持上一個目標，不因為漲離帶內就加回滿倉。月線離開深紅但仍在下軌外，停在 50%。
- **E2**：同 E1 的 100%／25%／回到帶內才回 100%，預警改成週線 BX 為負就減到 50%，而且週線維持為負的日子都停在 50%，即使收盤仍在帶內。週線不是負的時候，單靠跌破下軌不會減碼。離開深紅時若週線已不是負的、但收盤還不在帶內，維持 25%，直到回到帶內。
- **E3**：E1 再加分批回補。從深紅回來時，月線先回綠才回到 50%；回綠當天就算已經在帶內，也不直接回 100%。下一次收盤回到帶內才回 100%。漸增淺紅和持平負區不開始這段回補，部位維持 25%。若回綠之後、還沒回到帶內，月線又離開綠色，重新等下一次回綠。沒有待回補時，跌破下軌仍減到 50%。

目標在收盤決定，下一個開盤成交。比例沒變的日子不交易。單邊成本 0.05%。每一段（全期、前半、後半）都從現金 1 重新買，第一個開盤照前一日收盤已經知道的目標進場，最後一天收盤賣掉。後半不繼承前半的盈虧。前半是 2016-09-15～2021-09-14，後半是 2021-09-15～2026-09-14，中間沒有重疊的交易日。指標可以用更早的日 K 暖機。"""


def write_report(summary: pd.DataFrame, bear: pd.DataFrame) -> None:
    full = summary[summary["half"] == "full"]
    blocks = []
    for item in UNIVERSE:
        ticker = item["id"]
        block = full[full["ticker"] == ticker]
        start = block["actual_start"].iloc[0]
        end = block["actual_end"].iloc[0]
        blocks.append(f"### {item['name_zh']}（{ticker}，{start}～{end}）\n\n{_ticker_table(block)}\n")
    bear_blocks = []
    for spec in BEAR_WINDOWS:
        bear_blocks.append(
            f"### {spec['name']}（{spec['start']}～{spec['end']}）\n\n{spec['note']}\n\n{_bear_table(bear, spec['id'])}"
        )
    half_blocks = []
    for half_id, half_name, _start, _end in HALVES:
        if half_id == "full":
            continue
        half_blocks.append(f"## {half_name}\n")
        for item in UNIVERSE:
            ticker = item["id"]
            block = summary[(summary["half"] == half_id) & (summary["ticker"] == ticker)]
            start = block["actual_start"].iloc[0]
            end = block["actual_end"].iloc[0]
            half_blocks.append(f"### {item['name_zh']}（{ticker}，{start}～{end}）\n\n{_half_table(block)}\n")
    text = f"""# 日線預警減碼，以及前後五年分開看

> 七檔、約 2016-09-15～2026-09-14、單邊成本 0.05%。純股票部位。跟買進持有、C1 深紅留 25% 並排。**不要 merge。不是獲利保證。**

## 先講結論

{_conclusion(summary, bear)}

跌破下軌或週線轉負之後，收盤若停在上軌外面，部位不會回到 100%，要等收盤再進到帶內。強勢股有很長一段時間收在上軌外，預警層就會少拿那段上漲。換手也比 C1 多，成本已經扣在數字裡。

同一段十年已經拿來測過全進全出、加減碼和選擇權估價。這次又加三種預警。前後兩半是事後把同一段切開，規則沒有為了哪一半重調，切點對齊先前五年測試的 2021-09-15，不是找出最好看的切點。樣本仍是這七檔。月線一個月才變一次，週線一週才變一次，換一段日期，過線的檔數可以倒過來。不要拿這些數字去下單。

## 規則（跑之前就固定）

{_rules_section()}

報酬取回比例是策略總報酬除以同一段買進持有的總報酬。回撤改善是策略最大回撤減去同一段買進持有的最大回撤，正的代表跌得比較淺。Sharpe 是報酬除以波動後的分數。平均持倉是每天收盤實際部位的平均。換手次數是實際下單次數，含一開始買進和最後賣掉。年化與 Sharpe 沿用一年 252 個交易日。比特幣實際交易日比較多，年化會被壓低；總報酬不受這個影響。比特幣日 K 缺 2026-09-14，後半與全期績效算到資料最後一根。

## 約十年，七檔放在一起看

{_summary_table(full)}

## 約十年，每檔

{chr(10).join(blocks)}

## 三段下跌的區間報酬

日期與先前十年測試相同，七檔共用，沒有依個股高低點重挑。數字是這段裡資金曲線的漲跌，用的是約十年那條曲線，不是前後半重新起算的曲線。

{chr(10).join(bear_blocks)}

## 前後五年

每一半從現金重新買。達標仍是取回至少 80%、回撤至少淺 10 個百分點，分母是該半自己的買進持有。

{_both_halves(summary)}

{chr(10).join(half_blocks)}

```bash
cd tht-bx-backtest
python run_backtest_v4_early.py
python -m pytest test_early_warning.py -q
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
        curves = None
        for half_id, _half_name, start, end in HALVES:
            mask = _mask_for(df, start, end, full_mask)
            half_rows, half_curves = _run_half(df, mask, ticker, half_id)
            rows.extend(half_rows)
            if half_id == "full":
                curves = half_curves
            hit = [r for r in half_rows if r["strategy"] == "E1"][0]
            print(
                f"  {half_id:6s} E1 capture={hit['capture']*100:6.1f}%  "
                f"dd+={hit['dd_improve']*100:6.1f}pp  hit={hit['meets_bar']}"
            )
        assert curves is not None
        for spec in BEAR_WINDOWS:
            rec = {"ticker": ticker, "window_id": spec["id"], "window": spec["name"]}
            for sid in ORDER:
                rec[sid] = window_performance(curves[sid], spec["start"], spec["end"])["ret"]
            bear_rows.append(rec)

    summary = pd.DataFrame(rows)
    bear = pd.DataFrame(bear_rows)
    summary.to_csv(RESULT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    bear.to_csv(RESULT_DIR / "bear_windows.csv", index=False, encoding="utf-8-sig")
    write_report(summary, bear)


if __name__ == "__main__":
    run()
