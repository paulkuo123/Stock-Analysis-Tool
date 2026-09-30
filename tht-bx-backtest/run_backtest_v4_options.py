#!/usr/bin/env python3
"""十年、股票六檔：深紅時用選擇權對沖，並排買進持有與 C1 留 25%。

比特幣沒有可靠的選擇權歷史，不進這次表。權利金是估價，不是成交。

用法：
    python run_backtest_v4_options.py
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
from engine import ONE_WAY_COST  # noqa: E402
from monthly_bx import align_monthly_bx_to_daily  # noqa: E402
from options_hedge import hedge_active, run_overlay  # noqa: E402
from run_backtest import pct  # noqa: E402
from run_backtest_v3_multi import UNIVERSE, run_v_conf23_on_frame  # noqa: E402
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

RESULT_DIR = ROOT / "results" / "v4_10y_options"
REPORT_PATH = ROOT / "report-v4-10y-options.md"
STOCKS = [item for item in UNIVERSE if item["id"] != "BTC"]
ORDER = ["BH", "C1_25", "D1", "D2", "D3"]
LABEL = {
    "BH": "買進持有",
    "C1_25": "C1 深紅留 25%",
    "D1": "D1 掩護性買權",
    "D2": "D2 保護性賣權",
    "D3": "D3 領口",
}
KINDS = {"D1": "call", "D2": "put", "D3": "collar"}


def _ratio(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x * 100:.1f}%"


def _pp(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x * 100:+.1f} 個百分點"


def _sharpe(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:.2f}"


def _names(tickers: list[str]) -> str:
    return "、".join(tickers) if tickers else "沒有"


def _premium_cell(row: pd.Series) -> str:
    if row["strategy"] == "D1":
        return f"收入 {_ratio(row['premium_in'])}（每張約 {_ratio(row['avg_call_yield'])} 股價）"
    if row["strategy"] == "D2":
        return f"成本 {_ratio(row['premium_out'])}（每張約 {_ratio(row['avg_put_yield'])} 股價）"
    if row["strategy"] == "D3":
        return f"收入 {_ratio(row['premium_in'])}／成本 {_ratio(row['premium_out'])}"
    return "—"


def _called_cell(row: pd.Series) -> str:
    if row["strategy"] not in ("D1", "D3"):
        return "—"
    return f"到期價內 {int(row['n_called'])}，提前買回價內 {int(row['n_early_itm'])}"


def _row(ticker: str, sid: str, metrics: dict, bh: dict, extra: dict | None = None) -> dict:
    extra = extra or {}
    total = float(metrics["total_return"])
    maxdd = float(metrics["maxdd"])
    capture = 1.0 if sid == "BH" else capture_ratio(total, float(bh["total_return"]))
    improve = 0.0 if sid == "BH" else drawdown_improvement(maxdd, float(bh["maxdd"]))
    return {
        "ticker": ticker,
        "strategy": sid,
        "total_return": total,
        "maxdd": maxdd,
        "ann_return": float(metrics["ann_return"]) if pd.notna(metrics["ann_return"]) else np.nan,
        "sharpe": float(metrics["sharpe"]) if pd.notna(metrics["sharpe"]) else np.nan,
        "capture": capture,
        "dd_improve": improve,
        "meets_bar": meets_reading_bar(capture, improve) if sid != "BH" else False,
        "premium_in": float(extra.get("premium_in", np.nan)),
        "premium_out": float(extra.get("premium_out", np.nan)),
        "avg_call_yield": float(extra.get("avg_call_yield", np.nan)),
        "avg_put_yield": float(extra.get("avg_put_yield", np.nan)),
        "n_called": int(extra.get("n_called", 0) or 0),
        "n_early_itm": int(extra.get("n_early_itm", 0) or 0),
        "n_opened_call": int(extra.get("n_opened_call", 0) or 0),
        "n_opened_put": int(extra.get("n_opened_put", 0) or 0),
        "actual_start": metrics.get("actual_start", ""),
        "actual_end": metrics.get("actual_end", ""),
    }


def _summary_table(summary: pd.DataFrame) -> str:
    header = "| 策略 | 六檔裡同時達標 | 取回比例中位數 | 回撤改善中位數 | 總報酬中位數 | 最大回撤中位數 |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: |"
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
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _ticker_table(block: pd.DataFrame) -> str:
    header = "| 策略 | 總報酬 | 最大回撤 | 年化 | Sharpe | 權利金（相對起始資金） | 被 Call 走 | 報酬取回 | 回撤改善 |"
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
                    _premium_cell(r),
                    _called_cell(r),
                    _ratio(r["capture"]),
                    _pp(r["dd_improve"]),
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
    for item in STOCKS:
        row = sub[sub["ticker"] == item["id"]]
        cells = [item["id"]]
        for sid in ORDER:
            val = float(row.iloc[0][sid]) if len(row) else np.nan
            cells.append(pct(val) if np.isfinite(val) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _conclusion(summary: pd.DataFrame) -> str:
    lines = [
        "這些選擇權數字是 Black-Scholes 估價，不是十年裡的真實成交。不能拿去當實盤會收到或付得出的權利金。",
        (
            f"讀法和加減碼那次相同，事先固定：總報酬至少取回買進持有的 {CAPTURE_BAR * 100:.0f}%，"
            f"而且最大回撤至少淺 {DD_IMPROVE_BAR * 100:.0f} 個百分點。分母是六檔股票，比特幣不在裡面。"
        ),
    ]
    any_hit = False
    for sid in ORDER:
        if sid == "BH":
            continue
        sub = summary[summary["strategy"] == sid]
        hit = sub.loc[sub["meets_bar"], "ticker"].tolist()
        any_hit = any_hit or bool(hit)
        lines.append(
            f"{LABEL[sid]}：同時達標的是{' ' + _names(hit) if hit else '沒有任何一檔'}。"
            f"取回比例中位數 {_ratio(float(sub['capture'].median()))}，"
            f"回撤改善中位數 {_pp(float(sub['dd_improve'].median()))}。"
        )
    if not any_hit:
        lines.append("依這個讀法，買進持有以外沒有任何一檔同時達標。")
    else:
        lines.append("有達標的列在上面。那是這六檔、這十年、這個估價法下的過線，不是換一段或換成真實報價還會一樣。")
    lines.append(
        "賣出買權時，實盤的隱含波動通常高於這裡用的已實現波動，所以掩護性買權收到的權利金偏少。"
        "買進賣權時，價外加上波動率微笑，實盤通常更貴，所以保護性賣權的成本和領口的保護看起來會偏好看。"
        "買賣價差沒有扣。過線或沒過線，都不能當成券商會照這個價格成交。"
    )
    return "\n\n".join(lines)


def _assumptions() -> str:
    return """本機和這次回測用的行情檔只有日 K，沒有選擇權成交歷史。公開行情源的選擇權鏈多半是當下報價，不能回放 2016 到 2026。所以六檔股票的權利金用 Black-Scholes 中價來估。波動度用該股自己過去 21 個交易日的已實現波動，年化方式沿用一年 252 個交易日。沒用 VIX：VIX 是 S&P 500 的隱含波動，不是特斯拉或輝達的選擇權波動。

比特幣排除。沒有可靠的比特幣選擇權歷史價格，不用現貨波動度假裝那裡有一張可成交的合約。

其他假設，全部事先固定：

- 利率當 0，股利當 0。同一天所有履約價共用一個波動度，沒有波動率微笑。
- 沒有買賣價差，也沒有權利金的額外滑價。估出來的中價會比實盤好看。
- 美式提前履約和指派沒有模擬。被 Call 走只計「到期當天收盤價高於履約價」。回綠若提前用理論價買回，另外計入「提前買回價內」，股票都沒有真的交割，上方獲利用現金扣掉。
- 付權利金時若現金不夠，現金可以變負，不計利息。股票股數從區間第一天買進後就不再減碼，最後一天收盤賣掉，股票單邊成本仍是 0.05%。
- 選擇權在訊號當天收盤估價。天期固定 21 個交易日，大約一個月，沒有改成別的天數。
- 題目寫的「一口」在資金曲線裡是覆蓋全部持股，不是只對 100 股。
- 月線變成深紅才開新倉。深紅之後如果變成漸增淺紅或持平負區，對沖留到回綠。單獨的漸增淺紅不開新倉，避免多出一個沒寫死的進場。回綠是淺綠或深綠。

這是估價模擬，不等於實盤成交。"""


def write_report(summary: pd.DataFrame, bear: pd.DataFrame) -> None:
    blocks = []
    for item in STOCKS:
        block = summary[summary["ticker"] == item["id"]]
        start = block["actual_start"].iloc[0]
        end = block["actual_end"].iloc[0]
        blocks.append(f"### {item['name_zh']}（{item['id']}，{start}～{end}）\n\n{_ticker_table(block)}\n")
    bear_blocks = [
        f"### {spec['name']}（{spec['start']}～{spec['end']}）\n\n{spec['note']}\n\n{_bear_table(bear, spec['id'])}"
        for spec in BEAR_WINDOWS
    ]
    text = f"""# 選擇權對沖：估價模擬，並排買進持有與 C1 留 25%

> 六檔股票、約 2016-09-15～2026-09-14、股票單邊成本 0.05%。比特幣排除。**這是估價，不是實盤成交。不要 merge。**

## 先講結論

{_conclusion(summary)}

同一段十年已經拿來做過全進全出和加減碼。這次又加三種選擇權。樣本還是那六檔，過線很容易是估價假設造成的。

## 資料、假設與限制

{_assumptions()}

掩護性買權：手上的股票不賣，另外賣出買權收權利金，漲超過履約價的部分被封頂。保護性賣權：買進賣權，跌深時有賠償，權利金是成本。領口：兩筆同時做。Delta 是選擇權價格對股價的敏感度，0.30 大約是價外買權。Black-Scholes 是用股價、履約價、天期和波動度算理論權利金的公式。已實現波動度是過去股價實際漲跌的大小，不是選擇權市場上的隱含波動。

報酬取回比例是策略總報酬除以買進持有總報酬。回撤改善是策略最大回撤減去買進持有最大回撤，正的代表跌得比較淺。Sharpe 是報酬除以波動後的分數。權利金毛額是十年裡開倉時理論權利金的加總，相對起始資金；被履約或提前買回的損失不在這個毛額裡扣掉，那些已經進總報酬。

## 六檔放在一起看

{_summary_table(summary)}

## 每檔

{chr(10).join(blocks)}

## 三段空頭的區間報酬

日期與先前十年測試相同，六檔共用，沒有依個股高低點重挑。

{chr(10).join(bear_blocks)}

```bash
cd tht-bx-backtest
python run_backtest_v4_options.py
python -m pytest test_options_hedge.py -q
```
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"已寫入 {REPORT_PATH}")


def run() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    bear_rows = []
    for item in STOCKS:
        ticker = item["id"]
        print(f"\n=== {ticker} ===")
        raw, _note = load_full_ohlcv(item["yf"])
        df, mask, _res_v, bh = run_v_conf23_on_frame(raw, START, END, ONE_WAY_COST)
        aligned = align_monthly_bx_to_daily(raw["close"].astype(float)).reindex(df.index)
        for col in aligned.columns:
            df[col] = aligned[col]
        window_index = df.index[mask.to_numpy()]
        actual_start = window_index.min().strftime("%Y-%m-%d")
        actual_end = window_index.max().strftime("%Y-%m-%d")
        bh.metrics["actual_start"] = actual_start
        bh.metrics["actual_end"] = actual_end
        bh_total = float(bh.metrics["total_return"])
        rows.append(_row(ticker, "BH", bh.metrics, bh.metrics))

        c1 = run_sized_book(
            df,
            c1_targets(df, 0.25),
            mask,
            name="C1_25",
            cost=ONE_WAY_COST,
            bh_total=bh_total,
        )
        c1.metrics["actual_start"] = actual_start
        c1.metrics["actual_end"] = actual_end
        rows.append(_row(ticker, "C1_25", c1.metrics, bh.metrics))

        curves = {"BH": bh.equity, "C1_25": c1.equity}
        hedge = hedge_active(df["m_color"])
        for sid, kind in KINDS.items():
            overlay = run_overlay(
                df,
                hedge,
                kind,
                mask,
                name=sid,
                cost=ONE_WAY_COST,
                bh_total=bh_total,
            )
            overlay.metrics["actual_start"] = actual_start
            overlay.metrics["actual_end"] = actual_end
            rows.append(
                _row(
                    ticker,
                    sid,
                    overlay.metrics,
                    bh.metrics,
                    {
                        "premium_in": overlay.premium_in,
                        "premium_out": overlay.premium_out,
                        "avg_call_yield": overlay.avg_call_yield,
                        "avg_put_yield": overlay.avg_put_yield,
                        "n_called": overlay.n_called,
                        "n_early_itm": overlay.n_early_itm,
                        "n_opened_call": overlay.n_opened_call,
                        "n_opened_put": overlay.n_opened_put,
                    },
                )
            )
            curves[sid] = overlay.equity
            print(
                f"  {sid}: ret={overlay.metrics['total_return']*100:8.1f}%  "
                f"dd={overlay.metrics['maxdd']*100:6.1f}%  "
                f"called={overlay.n_called}  prem_in={overlay.premium_in:.3f}  prem_out={overlay.premium_out:.3f}"
            )

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
