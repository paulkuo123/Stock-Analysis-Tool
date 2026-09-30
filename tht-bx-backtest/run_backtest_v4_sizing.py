#!/usr/bin/env python3
"""十年區間的減碼／加碼，並排買進持有與先前三版。

權重在 sizing.py 裡寫死。這支程式只負責跑、算表，不改規則。

用法：
    python run_backtest_v4_sizing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bear_windows import BEAR_WINDOWS, window_performance  # noqa: E402
from engine import ONE_WAY_COST, attach_metrics, trades_to_frame  # noqa: E402
from monthly_bx import align_monthly_bx_to_daily  # noqa: E402
from run_backtest import num, pct, setup_font  # noqa: E402
from run_backtest_v3_multi import UNIVERSE  # noqa: E402
from run_backtest_v4_10y import END, START, load_full_ohlcv  # noqa: E402
from run_backtest_v3_multi import run_v_conf23_on_frame  # noqa: E402
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

RESULT_DIR = ROOT / "results" / "v4_10y_sizing"
FIG_DIR = RESULT_DIR / "figures"
REPORT_PATH = ROOT / "report-v4-10y-sizing.md"

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
    "A_MONTHLY_EXIT": "變體 A（只改月線出場）",
    "B_OFFICIAL": "變體 B（完整官方版）",
    "C1_0": "C1 減碼，深紅到 0%",
    "C1_25": "C1 減碼，深紅到 25%",
    "C2_150": "C2 加碼到 150%",
    "C2_60": "C2 無槓桿，60% 加到 100%",
    "C3": "C3 組合",
}
SIZING_IDS = ["C1_0", "C1_25", "C2_150", "C2_60", "C3"]


def _exposure(trades, index: pd.DatetimeIndex) -> float:
    if len(index) == 0:
        return np.nan
    flag = np.zeros(len(index), dtype=bool)
    for t in trades:
        flag |= (index >= t.entry_date) & (index <= t.exit_date)
    return float(flag.mean())


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


def _win(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return pct(x)


def _weight(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x * 100:.1f}%"


def _names(tickers: list[str]) -> str:
    return "、".join(tickers) if tickers else "沒有"


def _one_row(ticker: str, sid: str, metrics: dict, avg_weight: float, n_orders: float, bh: dict) -> dict:
    total = float(metrics["total_return"])
    maxdd = float(metrics["maxdd"])
    bh_total = float(bh["total_return"])
    bh_dd = float(bh["maxdd"])
    capture = 1.0 if sid == "BH" else capture_ratio(total, bh_total)
    improve = 0.0 if sid == "BH" else drawdown_improvement(maxdd, bh_dd)
    return {
        "ticker": ticker,
        "strategy": sid,
        "label": LABEL[sid],
        "total_return": total,
        "maxdd": maxdd,
        "ann_return": float(metrics["ann_return"]) if pd.notna(metrics["ann_return"]) else np.nan,
        "sharpe": float(metrics["sharpe"]) if pd.notna(metrics["sharpe"]) else np.nan,
        "win_rate": float(metrics["win_rate"]) if pd.notna(metrics["win_rate"]) else np.nan,
        "avg_weight": float(avg_weight) if pd.notna(avg_weight) else np.nan,
        "n_orders": int(n_orders),
        "capture": capture,
        "dd_improve": improve,
        "meets_bar": meets_reading_bar(capture, improve) if sid != "BH" else False,
        "actual_start": metrics.get("actual_start", ""),
        "actual_end": metrics.get("actual_end", ""),
    }


def _ticker_table(block: pd.DataFrame) -> str:
    header = "| 策略 | 總報酬 | 最大回撤 | 年化 | Sharpe | 勝率 | 平均持倉 | 換手次數 | 報酬取回 | 回撤改善 |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
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
                    _win(r["win_rate"]),
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
    header = "| 策略 | 同時達標的檔數 | 取回比例中位數 | 回撤改善中位數 | 總報酬中位數 | 最大回撤中位數 |"
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


def _bear_table(bear: pd.DataFrame, window_id: str) -> str:
    sub = bear[bear["window_id"] == window_id]
    header = "| 標的 | " + " | ".join(LABEL[s] for s in ORDER) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(ORDER)) + " |"
    lines = [header, sep]
    for item in UNIVERSE:
        row = sub[sub["ticker"] == item["id"]]
        cells = [item["id"]]
        for sid in ORDER:
            val = row.iloc[0][sid] if len(row) else np.nan
            cells.append(pct(val) if np.isfinite(val) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _bear_note(bear: pd.DataFrame) -> str:
    bits = []
    for spec in BEAR_WINDOWS:
        sub = bear[bear["window_id"] == spec["id"]]
        gap = sub["C1_25"] - sub["BH"]
        less = sub.loc[gap > 0.005, "ticker"].tolist()
        same = sub.loc[gap.abs() <= 0.005, "ticker"].tolist()
        worse = sub.loc[gap < -0.005, "ticker"].tolist()
        parts = []
        if less:
            parts.append(f"少虧的是 {_names(less)}")
        if same:
            parts.append(f"幾乎跟抱著一樣的是 {_names(same)}")
        if worse:
            parts.append(f"這段比較差的是 {_names(worse)}")
        lev = sub.loc[sub["C2_150"] < sub["BH"] - 0.005, "ticker"].tolist()
        lev_text = " " + _names(lev) if lev else "沒有"
        bits.append(
            f"{spec['name']}：C1 深紅留 25% " + "；".join(parts) + "。"
            f"加到 150% 而且這段跌得比抱著更深的是{lev_text}。"
        )
    return "\n\n".join(bits)


def _conclusion(summary: pd.DataFrame, bear: pd.DataFrame) -> str:
    counts = summary.loc[summary["strategy"] != "BH"].groupby("strategy")["meets_bar"].sum()
    best_sid = str(counts.idxmax()) if len(counts) else ""
    best_n = int(counts.max()) if len(counts) else 0
    miss = summary.loc[(summary["strategy"] == best_sid) & ~summary["meets_bar"], "ticker"].tolist()
    lines = [
        (
            f"沒有一版在七檔全都達標。過線最多的是 {LABEL.get(best_sid, best_sid)}，{best_n} 檔；"
            f"沒過的是 {_names(miss)}。這不夠拿來換成預設。"
        ),
        (
            f"讀法事先固定：總報酬至少取回買進持有的 {CAPTURE_BAR * 100:.0f}%，"
            f"而且最大回撤至少淺 {DD_IMPROVE_BAR * 100:.0f} 個百分點。兩個門檻都要過，才算「報酬還接近、回撤明顯比較小」。"
        )
    ]
    any_hit = False
    for sid in ORDER:
        if sid == "BH":
            continue
        sub = summary[summary["strategy"] == sid]
        hit = sub.loc[sub["meets_bar"], "ticker"].tolist()
        if hit:
            any_hit = True
        hit_text = "沒有任何一檔" if not hit else " " + _names(hit)
        lines.append(
            f"{LABEL[sid]}：同時達標的是{hit_text}。"
            f"取回比例中位數 {_ratio(float(sub['capture'].median()))}，"
            f"回撤改善中位數 {_pp(float(sub['dd_improve'].median()))}。"
        )
    if not any_hit:
        lines.append("依這個讀法，九種裡沒有任何一檔同時達標。沒有一版可以說成「賺得接近一直抱著，回撤又明顯比較小」。")
    else:
        lines.append("有達標的列在上面。達標只代表這十年、這七檔、這個門檻下過線，不是換一段還會一樣。")
    # 描述最接近的加減碼：回撤有改善的那些裡，取回比例最高的那一版的中位數。不另定門檻。
    sizing = summary[summary["strategy"].isin(SIZING_IDS)]
    med = (
        sizing.groupby("strategy")
        .agg(capture=("capture", "median"), improve=("dd_improve", "median"), n=("meets_bar", "sum"))
        .reset_index()
    )
    improved = med[med["improve"] > 0].sort_values("capture", ascending=False)
    if len(improved):
        top = improved.iloc[0]
        lines.append(
            f"若只看「回撤中位數有變淺」的加減碼版，取回比例中位數最高的是 {LABEL[top['strategy']]}："
            f"取回 {_ratio(float(top['capture']))}，回撤改善 {_pp(float(top['improve']))}。"
            "這是同一張表上的描述，不是事後另挑一組權重。"
        )
    lines.append(
        "C2 到 150% 和 C3 的加碼沒有計借款利息，也沒有計閒置現金的利息。這兩版的報酬會比真實融資帳戶好看。"
        "C2 的 60% 加到 100% 沒有槓桿，數字不靠這個假設。"
    )
    lines.append(_bear_note(bear))
    return "\n\n".join(lines)


def _rules_section() -> str:
    return """部位只有規則寫到的那幾種，跑完沒有改過。

月線 BX 是把日線收盤收成月線後看的買盤強弱，要等該月收完才用，月中不提前。綠色（淺綠或深綠）代表在零軸上。漸增淺紅代表還在零軸下、但比上個月高。深紅代表在零軸下、而且比上個月更差。持平負區代表在零軸下、跟上個月一樣，這次不把它當成減碼或加碼。33 FVB 是 33 日均線上下各 0.18 倍標準差的帶；綠色代表這段偏多。收盤在上下軌之間才叫回到帶內。

- **C1 減碼**：沒有減碼訊號時持有 100%。漸增淺紅改成 50%。深紅改成 0%（另測 25%）。回到綠色改回 100%。持平負區維持原來的比例。
- **C2 加碼**：一開始按基礎部位抱著。月線是綠、33 FVB 也是綠、收盤回到帶內，才加碼。加碼留到月線變深紅，或收盤跌破下軌，跟官方版的出場同一套。只是離開帶內、或變成漸增淺紅，不單獨把加碼拿掉。150% 那版基礎是 100%、加到 150%；超過 100% 視同借款，**不計利息**，上限就是 150%。另一版基礎 60%、加到 100%，沒有借款。
- **C3 組合**：數字不另訂。深紅改成 0%。漸增淺紅改成 50%。綠且回到帶內（FVB 也是綠）加到 150%，出場跟 C2 一樣。其他時候的綠色是 100%。持平負區維持原部位。

目標在收盤決定，下一個開盤才成交。比例沒變的日子不交易，也不每天把部位拉回目標。單邊成本 0.05%，只扣在實際買賣的金額。最後一天收盤把部位賣掉，跟買進持有的結尾一樣。淨值跌到 0 就停住。"""


def write_report(summary: pd.DataFrame, bear: pd.DataFrame) -> None:
    blocks = []
    for item in UNIVERSE:
        ticker = item["id"]
        block = summary[summary["ticker"] == ticker]
        start = block["actual_start"].iloc[0]
        end = block["actual_end"].iloc[0]
        blocks.append(f"### {item['name_zh']}（{ticker}，{start}～{end}）\n\n{_ticker_table(block)}\n")
    bear_blocks = []
    for spec in BEAR_WINDOWS:
        bear_blocks.append(
            f"### {spec['name']}（{spec['start']}～{spec['end']}）\n\n{spec['note']}\n\n{_bear_table(bear, spec['id'])}"
        )
    text = f"""# 十年加減碼：跟買進持有、舊版、變體 A、變體 B 並排

> 標的、區間（約 2016-09-15～2026-09-14）、單邊成本 0.05%、THT／BX 參數都沒改。權重在跑之前就寫死。**不要 merge。不是獲利保證。**

## 先講結論

{_conclusion(summary, bear)}

這批樣本就是同一段約十年、同一七檔。五種加減碼都在同一段上試，達標檔數很容易被少數幾檔帶著走。月線一個月才變一次，換一段日期，取回比例和回撤都可能倒過來。

## 規則（跑之前就固定）

{_rules_section()}

報酬取回比例是策略總報酬除以買進持有總報酬，100% 代表總報酬一樣。回撤改善是策略最大回撤減去買進持有最大回撤，正的代表跌得比較淺。Sharpe 是把報酬除以波動後的分數，交易日少的時候不穩。勝率：舊版、變體 A、變體 B 是平倉後有賺錢的筆數比例；加減碼是「這段有部位，結束時的淨值高於開始」的比例。兩種勝率不能直接比高下。平均持倉：加減碼是每天收盤實際部位的平均；舊版三套是在場天數的比例（部位只有 0 或 100%）。換手次數是實際下單的次數，含一開始買進和最後賣掉。

年化與 Sharpe 沿用一年 252 個交易日。比特幣實際交易日比較多，年化會被壓低；總報酬不受這個影響。比特幣日K缺 2026-09-14，績效算到 09-13。

## 七檔放在一起看

{_summary_table(summary)}

## 每檔

{chr(10).join(blocks)}

## 三段空頭的區間報酬

日期跟十年那版相同，七檔共用，沒有依個股高低點重挑。數字是這段裡資金曲線的漲跌。

{chr(10).join(bear_blocks)}

## 不要讀太滿

1. 門檻 80% 和 10 個百分點是讀表用的，不是把策略調到剛好過線。
2. 150% 沒有扣借款利息。真實融資會再差一截。
3. 五種加減碼加上原本三版，都是同一段行情。多試幾版，總有一版在某幾檔看起來比較順眼，這不構成該拿去實盤的理由。
4. 顏色仍只看月線 BX 的正負和跟前一個月比升降，沒有另外加絕對值門檻。

```bash
cd tht-bx-backtest
python run_backtest_v4_sizing.py
python -m pytest test_sizing.py test_trademap.py -q
```
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"已寫入 {REPORT_PATH}")


def _plot(summary: pd.DataFrame, path: Path) -> None:
    setup_font()
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {
        "V_CONF23": "#4c78a8",
        "A_MONTHLY_EXIT": "#f58518",
        "B_OFFICIAL": "#54a24b",
        "C1_0": "#e45756",
        "C1_25": "#b279a2",
        "C2_150": "#72b7b2",
        "C2_60": "#ff9da6",
        "C3": "#9d755d",
    }
    for sid, color in colors.items():
        sub = summary[summary["strategy"] == sid]
        ax.scatter(sub["capture"] * 100, sub["dd_improve"] * 100, label=sid, color=color, s=36)
        for _, r in sub.iterrows():
            ax.annotate(r["ticker"], (r["capture"] * 100, r["dd_improve"] * 100), fontsize=7, alpha=0.8)
    ax.axvline(CAPTURE_BAR * 100, color="#444", linewidth=0.6, linestyle="--")
    ax.axhline(DD_IMPROVE_BAR * 100, color="#444", linewidth=0.6, linestyle="--")
    ax.set_xlabel("return capture vs buy&hold (%)")
    ax.set_ylabel("max-drawdown improvement (percentage points)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    bear_rows = []
    equity_parts = {}

    for item in UNIVERSE:
        ticker = item["id"]
        print(f"\n=== {ticker} ===")
        raw, _note = load_full_ohlcv(item["yf"])
        df, mask, res_v, bh = run_v_conf23_on_frame(raw, START, END, ONE_WAY_COST)
        aligned = align_monthly_bx_to_daily(raw["close"].astype(float)).reindex(df.index)
        for col in aligned.columns:
            df[col] = aligned[col]
        window_index = df.index[mask.to_numpy()]
        actual_start = window_index.min().strftime("%Y-%m-%d")
        actual_end = window_index.max().strftime("%Y-%m-%d")
        bh.metrics["actual_start"] = actual_start
        bh.metrics["actual_end"] = actual_end
        bh_total = float(bh.metrics["total_return"])

        curves: dict[str, pd.Series] = {"BH": bh.equity}
        packed = {}
        for sid, signals in (
            ("A_MONTHLY_EXIT", variant_a_signals(df)),
            ("B_OFFICIAL", variant_b_signals(df)),
        ):
            entry, exit_, inv = signals
            result = run_trade_map(df, entry, exit_, inv, name=sid, cost=ONE_WAY_COST, analysis_mask=mask)
            attach_metrics(result.result, mask, bh_total=bh_total)
            packed[sid] = result.result
            curves[sid] = result.result.equity
        curves["V_CONF23"] = res_v.equity
        packed["V_CONF23"] = res_v
        packed["BH"] = bh

        for sid in ("BH", "V_CONF23", "A_MONTHLY_EXIT", "B_OFFICIAL"):
            res = packed[sid]
            res.metrics["actual_start"] = actual_start
            res.metrics["actual_end"] = actual_end
            if sid == "BH":
                avg_w, orders = 1.0, 2
            else:
                avg_w = _exposure(res.trades, window_index)
                orders = 2 * int(res.metrics["n_trades"])
            rows.append(_one_row(ticker, sid, res.metrics, avg_w, orders, bh.metrics))
            equity_parts[f"{ticker}_{sid}"] = res.equity.loc[mask]
            if sid != "BH":
                trades_to_frame(res.trades).to_csv(
                    RESULT_DIR / f"trades_{ticker}_{sid}.csv", index=False, encoding="utf-8-sig"
                )

        for sid, target in target_book(df).items():
            sized = run_sized_book(df, target, mask, name=sid, cost=ONE_WAY_COST, bh_total=bh_total)
            sized.metrics["actual_start"] = actual_start
            sized.metrics["actual_end"] = actual_end
            rows.append(_one_row(ticker, sid, sized.metrics, sized.avg_weight, sized.n_orders, bh.metrics))
            curves[sid] = sized.equity
            equity_parts[f"{ticker}_{sid}"] = sized.equity.loc[mask]
            sized.weight.loc[mask].to_csv(RESULT_DIR / f"weights_{ticker}_{sid}.csv", encoding="utf-8-sig")

        bear_row = {"ticker": ticker}
        for spec in BEAR_WINDOWS:
            stats = {sid: window_performance(curves[sid], spec["start"], spec["end"]) for sid in ORDER}
            rec = {"ticker": ticker, "window_id": spec["id"], "window": spec["name"]}
            for sid in ORDER:
                rec[sid] = stats[sid]["ret"]
            bear_rows.append(rec)
            _ = bear_row
        b = [r for r in rows if r["ticker"] == ticker and r["strategy"] == "C3"][-1]
        print(
            f"{ticker:6s} C3={b['total_return']*100:8.1f}%  "
            f"capture={b['capture']*100:5.1f}%  dd+={b['dd_improve']*100:5.1f}pp  "
            f"hit={b['meets_bar']}"
        )

    summary = pd.DataFrame(rows)
    bear = pd.DataFrame(bear_rows)
    summary.to_csv(RESULT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    bear.to_csv(RESULT_DIR / "bear_windows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(equity_parts).to_csv(RESULT_DIR / "equity_curves.csv", encoding="utf-8-sig")
    _plot(summary, FIG_DIR / "capture_vs_dd.png")
    write_report(summary, bear)


if __name__ == "__main__":
    run()
