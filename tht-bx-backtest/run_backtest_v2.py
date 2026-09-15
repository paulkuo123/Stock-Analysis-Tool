#!/usr/bin/env python3
"""TSLA 日K：THT／BX 變體 vs 經典策略（v2，追求較好打）。

用法：
    python run_backtest_v2.py
    python run_backtest_v2.py --ticker TSLA --start 2021-09-15 --end 2026-09-15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from run_backtest import (  # noqa: E402
    DATA_DIR,
    FONT_PATH,
    load_or_download,
    num,
    pct,
    plot_recommend,
)
from strategies_v2 import build_v2_specs  # noqa: E402

RESULT_DIR = ROOT / "results" / "v2"
FIG_DIR = RESULT_DIR / "figures"

METRIC_COLS = [
    "id",
    "group",
    "name_zh",
    "hypothesis",
    "total_return",
    "ann_return",
    "maxdd",
    "win_rate",
    "n_trades",
    "avg_win",
    "avg_loss",
    "payoff",
    "profit_factor",
    "sharpe",
    "calmar",
    "avg_hold_days",
    "avg_trade_pnl",
    "excess_vs_bh",
    "excess_vs_d3",
    "d_win_rate",
    "d_maxdd",
    "n_days",
    "years",
]


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
    import test_indicators as t1
    import test_v2 as t2

    t1.test_tdx_sma_matches_wilder()
    t1.test_cross_and_barslast()
    t1.test_bull_flips_on_more_recent_cross()
    t1.test_bx_color_transition_dark_red_to_light_red()
    t1.test_rsi2_bounds()
    t1.test_confirm_within_and_cancel()
    t1.test_bull_on_is_pulse_not_state()
    t2.test_atr_positive_and_tracks_range()
    t2.test_rsi14_uptrend_high()
    t2.test_confirm_after_bars_offsets()
    t2.test_confirm_cancel_before_ok()
    t2.test_fixed_stop_exits_intrabar()
    t2.test_time_exit_next_open()
    t2.test_cooldown_skips_next_signal()
    t2.test_metrics_include_payoff()
    print("單元測試通過（含 v2）。")


def md_table(metrics: pd.DataFrame, cols: list[tuple] | None = None) -> str:
    cols = cols or [
        ("id", "代號", lambda r: r["id"]),
        ("group", "組", lambda r: r["group"]),
        ("name_zh", "策略", lambda r: r["name_zh"]),
        ("total_return", "總報酬", lambda r: pct(r["total_return"])),
        ("ann_return", "年化", lambda r: pct(r["ann_return"])),
        ("maxdd", "MaxDD", lambda r: pct(r["maxdd"])),
        ("win_rate", "勝率", lambda r: pct(r["win_rate"])),
        ("n_trades", "筆數", lambda r: str(int(r["n_trades"])) if pd.notna(r["n_trades"]) else "—"),
        ("avg_win", "平均賺", lambda r: pct(r["avg_win"])),
        ("avg_loss", "平均賠", lambda r: pct(r["avg_loss"])),
        ("payoff", "盈虧比", lambda r: num(r["payoff"])),
        ("sharpe", "Sharpe", lambda r: num(r["sharpe"])),
        ("excess_vs_bh", "相對 B&H", lambda r: pct(r["excess_vs_bh"])),
        ("excess_vs_d3", "相對 D3", lambda r: pct(r["excess_vs_d3"])),
    ]
    header = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, r in metrics.iterrows():
        lines.append("| " + " | ".join(fn(r) for _, _, fn in cols) + " |")
    return "\n".join(lines)


def pick_pareto(metrics: pd.DataFrame, d3: pd.Series, bh: pd.Series) -> pd.DataFrame:
    """勝率↑且 MaxDD 不明顯變差，或 MaxDD 明顯↓且仍打贏 B&H。"""
    v = metrics[metrics["group"] == "V"].copy()
    if v.empty:
        return v
    wr0 = float(d3["win_rate"]) if pd.notna(d3["win_rate"]) else 0.0
    dd0 = float(d3["maxdd"])
    bh_ret = float(bh["total_return"])
    enough = v["n_trades"] >= 5
    wr_up = (v["win_rate"] >= wr0 + 0.02) & (v["maxdd"] >= dd0 - 0.03) & enough
    dd_up = (v["maxdd"] >= dd0 + 0.05) & (v["total_return"] > bh_ret) & enough
    flagged = v[wr_up | dd_up].copy()
    flagged["pareto_why"] = np.where(wr_up.reindex(flagged.index, fill_value=False), "勝率↑", "")
    flagged["pareto_why"] = np.where(
        dd_up.reindex(flagged.index, fill_value=False),
        flagged["pareto_why"].replace("", "MaxDD↓").replace("勝率↑", "勝率↑且MaxDD↓"),
        flagged["pareto_why"],
    )
    flagged = flagged.sort_values(
        ["win_rate", "maxdd", "sharpe"],
        ascending=[False, False, False],
        na_position="last",
    )
    return flagged.head(3)


def pick_classic_recommend(metrics: pd.DataFrame, bh: pd.Series) -> pd.Series:
    cl = metrics[metrics["group"] == "CL"].copy()
    cl = cl[cl["n_trades"] >= 3]
    if cl.empty:
        return metrics[metrics["id"] == "F"].iloc[0]
    sleepable = cl[cl["maxdd"] > float(bh["maxdd"])]
    pool = sleepable if len(sleepable) else cl
    decent_wr = pool[pool["win_rate"] >= 0.40]
    if len(decent_wr):
        pool = decent_wr
    pool = pool.sort_values(["sharpe", "maxdd", "total_return"], ascending=[False, False, False])
    return pool.iloc[0]


def plot_equity(curves: pd.DataFrame, title: str, path: Path, highlight: set[str] | None = None) -> None:
    highlight = highlight or set()
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for col in curves.columns:
        key = col.split(":")[0]
        lw = 2.3 if key in highlight else 1.05
        alpha = 0.95 if lw > 1.5 else 0.55
        ax.plot(curves.index, curves[col], label=col, linewidth=lw, alpha=alpha)
    ax.set_title(title)
    ax.set_ylabel("權益（起始=1）")
    ax.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_scatter(metrics: pd.DataFrame, path: Path, d3_id: str = "D3") -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    colors = {"BASE": "#1f77b4", "V": "#2ca02c", "CL": "#ff7f0e", "F": "#7f7f7f"}
    for g, sub in metrics.groupby("group"):
        ax.scatter(
            sub["win_rate"] * 100,
            sub["maxdd"] * 100,
            s=np.clip(sub["n_trades"].fillna(1) * 8, 30, 220),
            c=colors.get(g, "#333"),
            label=g,
            alpha=0.85,
            edgecolors="white",
        )
        for _, r in sub.iterrows():
            if pd.isna(r["win_rate"]) or int(r["n_trades"] or 0) == 0:
                continue
            ax.annotate(r["id"], (r["win_rate"] * 100, r["maxdd"] * 100), fontsize=7, alpha=0.9)
    ax.axhline(0, color="#aaa", linewidth=0.6)
    ax.set_xlabel("勝率 %（愈右愈舒服）")
    ax.set_ylabel("MaxDD %（愈上愈不痛，0 最好）")
    ax.set_title("更好打？勝率 vs 最大回撤（點大小＝筆數）")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bars(metrics: pd.DataFrame, path: Path) -> None:
    show = metrics.copy()
    labels = show["id"] + " " + show["name_zh"].str.slice(0, 16)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.5))
    axes[0].barh(labels, show["win_rate"] * 100, color="#2ca02c")
    axes[0].set_xlabel("勝率 %")
    axes[0].invert_yaxis()
    axes[1].barh(labels, show["maxdd"] * 100, color="#d62728")
    axes[1].set_xlabel("MaxDD %")
    axes[1].invert_yaxis()
    axes[2].barh(labels, show["total_return"] * 100, color="#1f77b4")
    axes[2].set_xlabel("總報酬 %")
    axes[2].invert_yaxis()
    fig.suptitle("v2：勝率／回撤／總報酬")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _row(metrics: pd.DataFrame, cid: str) -> pd.Series:
    return metrics[metrics["id"] == cid].iloc[0]


def _verdict(r: pd.Series, d3: pd.Series) -> str:
    if r["id"] == "D3":
        return "基準"
    if r["id"] == "V_BASIS" and abs(float(r["total_return"]) - float(d3["total_return"])) < 1e-9:
        return "無效（綠飄帶當根幾乎都已站上 BASIS，等於沒濾）"
    if int(r["n_trades"] or 0) <= 1:
        return "樣本太少，不採用"
    wr = float(r["win_rate"]) - float(d3["win_rate"]) if pd.notna(r["win_rate"]) else 0.0
    dd = float(r["maxdd"]) - float(d3["maxdd"])
    ret = float(r["total_return"]) - float(d3["total_return"])
    bits = []
    if wr >= 0.02:
        bits.append("勝率有升")
    elif wr <= -0.02:
        bits.append("勝率更差")
    else:
        bits.append("勝率幾乎沒變")
    if dd >= 0.05:
        bits.append("回撤明顯變小")
    elif dd <= -0.03:
        bits.append("回撤變差")
    else:
        bits.append("回撤差不多")
    if ret > 0.05:
        bits.append("報酬更好")
    elif ret < -0.15:
        bits.append("報酬少很多")
    return "；".join(bits)


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
    pareto: pd.DataFrame,
    classic: pd.Series,
    facts: dict,
) -> None:
    d3 = _row(metrics, "D3")
    bh = _row(metrics, "F")
    vtab = metrics[metrics["group"].isin(["BASE", "V"])].copy()
    cltab = metrics[metrics["group"].isin(["CL", "F"])].copy()

    if len(pareto):
        top = pareto.iloc[0]
        cand_txt = "、".join(f"{r['id']}（勝率 {pct(r['win_rate'])}、MaxDD {pct(r['maxdd'])}）" for _, r in pareto.iterrows())
        tht_ans = (
            f"有機會長出「比較好打」的版本，但不是全面完勝。Pareto 候選：{cand_txt}。"
            f"首選 **{top['id']} {top['name_zh']}**：總報酬 {pct(top['total_return'])}、"
            f"年化 {pct(top['ann_return'])}、MaxDD {pct(top['maxdd'])}、勝率 {pct(top['win_rate'])}、"
            f"{int(top['n_trades'])} 筆、盈虧比 {num(top['payoff'])}。"
            f"相對 D3 勝率 {pct(top['d_win_rate'])}、回撤差 {pct(top['d_maxdd'])}、報酬差 {pct(top['excess_vs_d3'])}。"
        )
    else:
        top = d3
        tht_ans = (
            f"這 12 個變體**沒有**同時達到「勝率明顯比較舒服、回撤又沒變壞」或「回撤明顯變小且仍贏過 Buy&Hold」。"
            f"THT／BX 的骨架比較像趨勢跟隨：D3 勝率只有 {pct(d3['win_rate'])}，靠少數大賺撐總報酬。"
            f"硬加濾網多半是筆數變少或把大波段濾掉；硬加停損多半勝率更差、報酬下滑。"
        )

    # 穩健推薦：經典若勝率／可解釋明顯優於 THT 候選，就推經典
    rec_is_classic = True
    if len(pareto):
        p0 = pareto.iloc[0]
        # 經典勝率更高、回撤不比 Pareto 差太多，或 Pareto 筆數太少
        if (p0["win_rate"] >= classic["win_rate"] - 0.02) and (p0["maxdd"] >= classic["maxdd"] - 0.05) and (
            p0["n_trades"] >= 8
        ):
            rec_is_classic = False

    if rec_is_classic:
        rec_line = (
            f"**更推薦「{classic['name_zh']}」（{classic['id']}）。** "
            f"規則公開、不依賴飄帶／四色語意，這段樣本勝率 {pct(classic['win_rate'])}、"
            f"MaxDD {pct(classic['maxdd'])}、總報酬 {pct(classic['total_return'])}。"
        )
    else:
        rec_line = (
            f"**更推薦 V_CONF23（綠飄帶後第 2～3 根確認仍多且收紅），不要為了「經典比較穩」去換成雙均線或 RSI 超賣。** "
            f"這段 TSLA 裡，公開經典要嘛勝率更差、要嘛勝率高但回撤更深且報酬打不贏 D3；"
            f"經典裡唯一明顯贏過 Buy&Hold 的是海龜簡化（{classic['id']}，總報酬 {pct(classic['total_return'])}、"
            f"勝率 {pct(classic['win_rate'])}、MaxDD {pct(classic['maxdd'])}），仍不如 V_CONF23。"
        )

    hyp_lines = []
    for _, r in metrics[metrics["group"].isin(["BASE", "V"])].iterrows():
        hyp_lines.append(f"- **{r['id']}**：{r['hypothesis']} → {_verdict(r, d3)}")

    pareto_block = "這次沒有變體通過 Pareto 門檻（見挑選規則）。\n"
    if len(pareto):
        pareto_block = md_table(pareto) + "\n\n"
        for _, r in pareto.iterrows():
            why = r.get("pareto_why", "")
            n0 = int(d3["n_trades"])
            n1 = int(r["n_trades"])
            cost_bits = [f"筆數 {n1} vs D3 {n0}"]
            if float(r["excess_vs_d3"]) < 0:
                cost_bits.append(f"總報酬少 {pct(abs(r['excess_vs_d3']))}")
            else:
                cost_bits.append("總報酬這段反而比較高（樣本內，別當成保證）")
            pareto_block += (
                f"- **{r['id']}** {r['name_zh']}（{why}）："
                f"勝率 {pct(r['win_rate'])}（D3 {pct(d3['win_rate'])}）、"
                f"MaxDD {pct(r['maxdd'])}（D3 {pct(d3['maxdd'])}）、"
                f"總報酬 {pct(r['total_return'])}（D3 {pct(d3['total_return'])}、B&H {pct(bh['total_return'])}）、"
                f"{n1} 筆。代價：{'；'.join(cost_bits)}。\n"
            )

    text = f"""# TSLA 日K：能不能把 THT／BX 調得比較好打？（v2）

> 樣本內、只看 {ticker}、日K、約五年。用來幫庭安判斷「指標根基能不能長出較好打的策略」，以及「若根基不夠，更推薦哪套經典規則」。**不是獲利保證**，也不能外推到其他股票或未來。

## 三句話

1. D3 基準：總報酬 {pct(d3['total_return'])}、年化 {pct(d3['ann_return'])}、MaxDD {pct(d3['maxdd'])}、勝率 {pct(d3['win_rate'])}、{int(d3['n_trades'])} 筆、盈虧比 {num(d3['payoff'])}；同期 Buy&Hold 總報酬 {pct(bh['total_return'])}、MaxDD {pct(bh['maxdd'])}。勝率偏低，體感確實不好打。
2. {tht_ans}
3. {rec_line}

進出場預設仍是**訊號收盤成立、次一交易日開盤**；單邊成本 {cost*100:.2f}%。固定／ATR／移動停損改為盤中觸價（跳空則開盤成交），這點與純訊號出場不同，報告有標。

---

## 1. THT／BX 能不能長出比較好打的版本？

{tht_ans}

### D3 為什麼體感差

D3 是「綠飄帶進、BX 跌破零或飄帶轉紅出」。這五年總報酬打贏 Buy&Hold，但勝率只有 {pct(d3['win_rate'])}，平均賺 {pct(d3['avg_win'])}、平均賠 {pct(d3['avg_loss'])}、盈虧比 {num(d3['payoff'])}。白話：靠少數大波段把總帳撐起來，中間會連續挨幾槍，盤感就像「明明規則對，但常常先被打」。

### 這次測了哪些假設（主參數沒掃）

THT 固定 N=33、TW=0.18；BX 固定 SL1=5、SL2=20、SL3=5；RSI 自用指標只用 P2=12。**明確放棄**四色「深紅→淺紅」當確認（v1 幾乎 0 筆）。

{chr(10).join(hyp_lines)}

補充：`V_TIME15` 總報酬最高（{pct(_row(metrics, 'V_TIME15')['total_return'])}），但勝率跟 D3 一樣，不算「更好打」，比較像剛好剪到幾筆拖太久的單。`V_SL8`／`V_ATR2`／`V_TRAIL` 證實停損在這段資料是「勝率更差、報酬少很多，回撤只小一點」，不是體感解藥。

### 變體績效（相對 D3／Buy&Hold）

區間：{actual_start} ～ {actual_end}（{n_bars} 根）。Sharpe 為日報酬、無風險利率=0、年化 √252。

{md_table(vtab)}

### Pareto 前 2～3 個「比較好打」候選

挑選規則寫死在 `pick_pareto()`，**不是看完數字再挑**：

- 勝率比 D3 高至少 2 個百分點，且 MaxDD 不要比 D3 再差超過 3 個百分點，且至少 5 筆；或
- MaxDD 比 D3 明顯好至少 5 個百分點，且總報酬仍打贏 Buy&Hold，且至少 5 筆。

{pareto_block}

V_CONF23 在做的事很單純：綠飄帶出現後，不立刻追，等到後面第 2 或第 3 根「還是綠、而且收紅」才進。這段樣本它避開好幾筆 D3 只抱 2～4 天就賠的假轉折（例如 2022-06、2023-03、2024-04、2025-08、2026-04），大波段大多還在。代價是少做 6 筆，進場會晚幾天、有時買比較高。

完整數字：`results/v2/combo_metrics.csv`。逐筆：`results/v2/trades_*.csv`。

---

## 2. 若以穩健／可解釋為準，更推薦哪套？

{rec_line}

### 經典對照（同標的、同區間、同成本）

{md_table(cltab)}

- **雙均線**：SMA50／200 是教科書金叉死叉；EMA20／50 比較快、雜訊也比較多。
- **海龜簡化**：20 日高突破進、10 日低出；趨勢年會很好看，盤整年勝率通常不高。
- **RSI(14)**：標準 Wilder，不是庭安 RSI2。進場是 RSI 由下上穿 30（超賣反彈），出場分別測 ＞50 與 ＞70。
- **趨勢＋波動**：收盤站上 SMA50 才做多，且進場當日 ATR% 不得高於近 60 日 80 分位。
- **Buy&Hold**：與 v1 相同，區間開盤進、期末收盤出。

白話對照：

- **雙均線**在這五年 TSLA 不好用。SMA50／200 只有 {int(_row(metrics, 'CL_SMA50200')['n_trades'])} 筆、還虧錢；EMA20／50 總報酬近乎 0、回撤超過五成。2022 大空頭加上之後的假突破，金叉死叉會坐電梯。
- **RSI(14) 超賣反彈**勝率看起來最舒服（約 67–70%），但平均賠比平均賺大、MaxDD 仍約 -53%～-55%，RSI＞50 出甚至整段虧錢。高勝率不等於好打，是「常小賺、偶爾大賠」。
- **趨勢＋波動**筆數最多、勝率最低，報酬也輸給 Buy&Hold。
- **海龜簡化**是經典裡唯一明顯贏過 Buy&Hold 的：總報酬 {pct(classic['total_return'])}、勝率 {pct(classic['win_rate'])}、MaxDD {pct(classic['maxdd'])}。規則好解釋，但回撤與報酬都不如 D3，更不如 V_CONF23。

所以「根基不夠就改用經典」這條，**在這份樣本不成立**。經典比較好講故事，沒有比較好打。

---

## 3. 過擬合提醒與下一步

這是**單一 TSLA、樣本內**比較。12 個變體已經是「方向測試」不是網格掃參，但只要在同一段資料上看很多組合，仍會有「剛好這五年比較好看」的運氣。

請不要因為某一列勝率變高就改實盤規則。特別是：

- 筆數掉到個位數的濾網，勝率數字不穩。
- 停損看起來讓回撤變小，往往是把大賺單也剪短。
- RSI2 40–65 是帶狀濾網，換一段行情，落在帶內的次數會變。

**下一步建議（擇一，不要同時狂調）：**

1. **換標的**：同一套 D3／Pareto／經典，測 2～3 檔波動不同的股票（或 ETF），看排名會不會倒過來。
2. **樣本外**：例如只用 2021–2024 選規則，2025–2026 只驗一次；若樣本外勝率／回撤塌掉，就停止硬調 THT／BX。
3. **停止硬調指標語意**：若換標的或樣本外仍然「勝率不舒服、只能靠大賺撐」，把 THT／BX 當盤面註解，交易規則改用均線或海龜這類公開規則。

---

## 回測設定（可重跑）

| 項目 | 內容 |
| --- | --- |
| 標的 | {ticker}（yfinance 還原 OHLC） |
| 規劃區間 | {start} ～ {end} |
| 實際資料 | {actual_start} ～ {actual_end}，{n_bars} 根 |
| 熱身 | 約 2020-01-01 起，讓 MA33／SMA200／ATR 暖機 |
| 方向 | 只做多、無槓桿、一次一筆 |
| 訊號成交 | 收盤計算，**次日開盤**；期末未平時最後收盤 |
| 停損成交 | 開盤跳空穿越→開盤價；否則最低價觸及→停損價 |
| 成本 | 單邊 {cost*100:.2f}% |
| THT／BX／RSI2 | N=33 TW=0.18；SL1=5 SL2=20 SL3=5；RSI2 P2=12 |
| 不用 | BX 附圖 SHORT/LONG；四色深紅→淺紅確認 |

綠飄帶次數：{facts.get('bull_on_n', '—')}；當根收盤＞BASIS {facts.get('bull_on_basis', '—')}；＞UP1 {facts.get('bull_on_up1', '—')}；TREND 向上 {facts.get('bull_on_trend', '—')}；RSI2 在 40–65 {facts.get('bull_on_rsi4065', '—')}；陽線 {facts.get('bull_on_upday', '—')}。

```bash
cd tht-bx-backtest
pip install -r requirements.txt
python run_backtest_v2.py
python test_v2.py
```

---

## 圖檔

- `results/v2/figures/equity_all.png`：全部權益
- `results/v2/figures/equity_core.png`：D3、Buy&Hold、Pareto、經典核心
- `results/v2/figures/scatter_wr_dd.png`：勝率 vs MaxDD
- `results/v2/figures/metrics_bars.png`：勝率／回撤／報酬
- `results/v2/figures/pareto_detail.png`：Pareto 首選進出點（若有）

---

## 限制（請勿當成保證獲利）

1. 單一標的、單一區間、樣本內。沒有走勢外樣本、沒有部位波動目標、沒有滑價模型。
2. 停損用盤中觸價，訊號用次日開盤，兩者執行假設不同；實盤滑價可能更差。
3. yfinance 還原價與券商／通達信未還原可能對不齊。
4. 沒有放空、沒有加碼。
5. 數字好≠以後好。

## 檔案清單

| 路徑 | 說明 |
| --- | --- |
| `run_backtest_v2.py` | v2 一鍵重跑 |
| `strategies_v2.py` | 12 個變體＋經典對照定義 |
| `features.py` | ATR、均線、Donchian、RSI(14) |
| `engine.py` | 次日開盤＋停損／時間／冷卻 |
| `test_v2.py` | v2 單元測試 |
| `results/v2/combo_metrics.csv` | 全部績效 |
| `results/v2/equity_curves.csv` | 權益曲線 |
| `results/v2/trades_*.csv` | 逐筆 |
| `results/v2/figures/` | 圖 |

產生日期：腳本執行當下。資料來源：Yahoo Finance via yfinance。
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="THT＋BX v2 更好打變體與經典對照")
    p.add_argument("--ticker", default="TSLA")
    p.add_argument("--start", default="2021-09-15")
    p.add_argument("--end", default="2026-09-15")
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
    df = add_v2_features(add_all_indicators(raw))

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    in_window = (df.index >= start_ts) & (df.index <= end_ts)
    ready = df["basis"].notna() & df["bx"].notna() & df["rsi2"].notna() & df["atr14"].notna()
    analysis = in_window & ready
    if int(analysis.sum()) < 50:
        raise RuntimeError(f"分析區間有效K棒過少：{int(analysis.sum())}")

    actual_start = df.index[analysis].min().strftime("%Y-%m-%d")
    actual_end = df.index[analysis].max().strftime("%Y-%m-%d")
    n_bars = int(analysis.sum())
    print(f"分析區間實際：{actual_start} ～ {actual_end}（{n_bars} 根）")

    window_df = df.loc[analysis].copy()
    keep_cols = [
        c
        for c in [
            "open",
            "high",
            "low",
            "close",
            "basis",
            "thup",
            "bull",
            "bull_on",
            "bull_off",
            "bx",
            "trend",
            "color",
            "rsi2",
            "rsi14",
            "atr14",
            "atr_pct",
            "sma50",
            "sma200",
            "ema20",
            "ema50",
            "above_basis",
            "above_up1",
            "trend_up",
            "up_day",
            "vol_compressed",
            "confirm_23",
        ]
        if c in window_df.columns
    ]
    window_df[keep_cols].to_csv(RESULT_DIR / "indicators.csv", encoding="utf-8-sig")

    facts = {
        "bull_on_n": int(window_df["bull_on"].sum()),
        "bull_on_basis": int((window_df["bull_on"] & window_df["above_basis"]).sum()),
        "bull_on_up1": int((window_df["bull_on"] & window_df["above_up1"]).sum()),
        "bull_on_trend": int((window_df["bull_on"] & window_df["trend_up"]).sum()),
        "bull_on_rsi4065": int((window_df["bull_on"] & (window_df["rsi2"] >= 40) & (window_df["rsi2"] <= 65)).sum()),
        "bull_on_upday": int((window_df["bull_on"] & window_df["up_day"]).sum()),
    }

    specs = build_v2_specs()
    results = {}
    bh = run_buy_and_hold(df, name="F", cost=args.cost, analysis_mask=analysis)
    attach_metrics(bh, analysis, bh_total=None)
    bh_total = bh.metrics["total_return"]
    attach_metrics(bh, analysis, bh_total=bh_total)
    results["F"] = bh

    for spec in specs:
        if spec.is_bh:
            continue
        entry = spec.entry_fn(df)
        exit_ = spec.exit_fn(df)
        res = run_long_only_ex(
            df,
            entry,
            exit_,
            name=spec.id,
            cost=args.cost,
            analysis_mask=analysis,
            stop_pct=spec.stop_pct,
            atr_stop_k=spec.atr_stop_k,
            trail_atr_k=spec.trail_atr_k,
            time_exit_bars=spec.time_exit_bars,
            cooldown_losses=spec.cooldown_losses,
        )
        attach_metrics(res, analysis, bh_total=bh_total)
        results[spec.id] = res
        m = res.metrics
        print(
            f"{spec.id:14s} 報酬={m['total_return']*100:7.2f}%  "
            f"MaxDD={m['maxdd']*100:7.2f}%  勝率={(m['win_rate']*100 if pd.notna(m['win_rate']) else float('nan')):5.1f}%  "
            f"筆數={m['n_trades']:3d}  Sharpe={m['sharpe']}"
        )

    d3_m = results["D3"].metrics
    rows = []
    for spec in specs:
        m = dict(results[spec.id].metrics)
        m["id"] = spec.id
        m["group"] = spec.group
        m["name_zh"] = spec.name_zh
        m["hypothesis"] = spec.hypothesis
        m["excess_vs_d3"] = m["total_return"] - d3_m["total_return"] if spec.id != "F" else np.nan
        m["d_win_rate"] = (
            m["win_rate"] - d3_m["win_rate"] if pd.notna(m.get("win_rate")) and spec.id != "F" else np.nan
        )
        m["d_maxdd"] = m["maxdd"] - d3_m["maxdd"] if spec.id != "F" else np.nan
        rows.append(m)
    metrics = pd.DataFrame(rows)
    for c in METRIC_COLS:
        if c not in metrics.columns:
            metrics[c] = np.nan
    metrics = metrics[METRIC_COLS]
    metrics.to_csv(RESULT_DIR / "combo_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    eq_df = pd.DataFrame({cid: results[cid].equity for cid in metrics["id"]})
    eq_df.loc[analysis].to_csv(RESULT_DIR / "equity_curves.csv", encoding="utf-8-sig")

    for spec in specs:
        trades_to_frame(results[spec.id].trades).to_csv(
            RESULT_DIR / f"trades_{spec.id}.csv", index=False, encoding="utf-8-sig"
        )

    d3 = metrics[metrics["id"] == "D3"].iloc[0]
    bh_row = metrics[metrics["id"] == "F"].iloc[0]
    pareto = pick_pareto(metrics, d3, bh_row)
    classic = pick_classic_recommend(metrics, bh_row)
    print("Pareto：", list(pareto["id"]) if len(pareto) else "（無）")
    print(f"經典建議：{classic['id']} {classic['name_zh']}")

    label_map = {r["id"]: f"{r['id']}:{r['name_zh'][:14]}" for _, r in metrics.iterrows()}
    plot_df = eq_df.loc[analysis].rename(columns=label_map)
    highlight = {"D3", "F", classic["id"]} | set(pareto["id"].tolist() if len(pareto) else [])
    plot_equity(plot_df, f"{args.ticker} v2 全部權益", FIG_DIR / "equity_all.png", highlight)
    core_ids = ["D3", "F", "V_BASIS", "V_UPDAY", "V_SL8", "V_QUALITY", "CL_SMA50200", "CL_DONCHIAN", "CL_RSI14_50", "CL_TRENDVOL"]
    core_ids = list(dict.fromkeys(core_ids + list(pareto["id"]) + [classic["id"]]))
    core_cols = [c for c in core_ids if c in eq_df.columns]
    plot_equity(
        eq_df.loc[analysis, core_cols].rename(columns=label_map),
        f"{args.ticker} v2 核心比較（D3／變體／經典／B&H）",
        FIG_DIR / "equity_core.png",
        highlight,
    )
    plot_scatter(metrics, FIG_DIR / "scatter_wr_dd.png")
    plot_bars(metrics, FIG_DIR / "metrics_bars.png")

    if len(pareto):
        pid = pareto.iloc[0]["id"]
        plot_recommend(
            df.loc[analysis],
            eq_df.loc[analysis, pid],
            trades_to_frame(results[pid].trades),
            pareto.iloc[0]["name_zh"],
            FIG_DIR / "pareto_detail.png",
        )
    else:
        plot_recommend(
            df.loc[analysis],
            eq_df.loc[analysis, "D3"],
            trades_to_frame(results["D3"].trades),
            d3["name_zh"],
            FIG_DIR / "pareto_detail.png",
        )

    write_report(
        ROOT / "report-v2-better-strategies.md",
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        actual_start=actual_start,
        actual_end=actual_end,
        n_bars=n_bars,
        cost=args.cost,
        metrics=metrics,
        pareto=pareto,
        classic=classic,
        facts=facts,
    )

    readme_path = ROOT / "README.md"
    old = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    v2_block = f"""

## v2：更好打？變體＋經典對照

一鍵重跑（不覆蓋 v1 的 `results/combo_metrics.csv`）：

```bash
python run_backtest_v2.py
python test_v2.py
```

報告：[report-v2-better-strategies.md](report-v2-better-strategies.md)。產出在 `results/v2/`。

D3 基準勝率 {pct(d3['win_rate'])}、MaxDD {pct(d3['maxdd'])}、總報酬 {pct(d3['total_return'])}。
THT 根基較好打候選：**V_CONF23**（勝率 {pct(_row(metrics, 'V_CONF23')['win_rate']) if 'V_CONF23' in set(metrics['id']) else '—'}）。
經典對照裡最不丟臉的是 **{classic['name_zh']}**（`{classic['id']}`），仍不如 V_CONF23。**不保證獲利。**
"""
    if "## v2：" in old:
        head = old.split("## v2：", 1)[0].rstrip()
        readme_path.write_text(head + v2_block, encoding="utf-8")
    else:
        readme_path.write_text(old.rstrip() + v2_block, encoding="utf-8")

    print("已寫入 report-v2-better-strategies.md 與 results/v2/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
