"""v2 策略：THT／BX／RSI2 變體＋經典對照。

THT N=33 TW=0.18、BX SL1=5 SL2=20 SL3=5、RSI2 P2=12 固定不掃參。
四色「深紅→淺紅」確認已放棄（v1 幾乎 0 筆）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from strategies import _and, _or


@dataclass(frozen=True)
class V2Spec:
    id: str
    group: str
    name_zh: str
    hypothesis: str
    entry_fn: Callable[[pd.DataFrame], pd.Series]
    exit_fn: Callable[[pd.DataFrame], pd.Series]
    stop_pct: Optional[float] = None
    atr_stop_k: Optional[float] = None
    trail_atr_k: Optional[float] = None
    time_exit_bars: Optional[int] = None
    cooldown_losses: Optional[int] = None
    is_bh: bool = False
    extras: dict = field(default_factory=dict)


def _d3_exit(df: pd.DataFrame) -> pd.Series:
    return _or(df["pos_to_neg"], df["bull_off"])


def _d3_entry(df: pd.DataFrame) -> pd.Series:
    return df["bull_on"].astype(bool)


def build_v2_specs() -> list[V2Spec]:
    """12 個 THT 根基變體＋D3 基準＋經典對照。每個假設寫在 hypothesis。"""

    specs: list[V2Spec] = [
        V2Spec(
            "D3",
            "BASE",
            "D3 基準：綠飄帶進／零軸或轉紅出",
            "v1 建議組，當「更好打」的比較基準，不是新優化。",
            _d3_entry,
            _d3_exit,
        ),
        V2Spec(
            "V_BASIS",
            "V",
            "綠飄帶且收盤＞BASIS",
            "假設：價格已站上中軌再跟，假轉折較少，勝率會升、回撤不會明顯變差。",
            lambda df: _and(df["bull_on"], df["above_basis"]),
            _d3_exit,
        ),
        V2Spec(
            "V_UP1",
            "V",
            "綠飄帶且收盤＞UP1",
            "假設：只要站上上軌的強勢段，避開軌道內雜訊；代價是筆數變少、可能漏大波。",
            lambda df: _and(df["bull_on"], df["above_up1"]),
            _d3_exit,
        ),
        V2Spec(
            "V_TREND",
            "V",
            "綠飄帶且 TREND(EMA BX,2) 向上",
            "假設：BX 的 2 期 EMA 在正區且比昨高，動能同向，能擋「轉綠但買盤已轉弱」。",
            lambda df: _and(df["bull_on"], df["trend_up"]),
            _d3_exit,
        ),
        V2Spec(
            "V_RSI4065",
            "V",
            "綠飄帶且 RSI2 在 40–65",
            "假設：避開過冷反彈失敗與過熱追價；v1 已知 RSI2<70 幾乎沒濾到，改縮到中段帶。",
            lambda df: _and(df["bull_on"], df["rsi2"] >= 40, df["rsi2"] <= 65),
            _d3_exit,
        ),
        V2Spec(
            "V_VOLCOMP",
            "V",
            "綠飄帶且 ATR% 低於近 20 日中位數",
            "假設：波動壓縮時的綠飄帶比較像真突破，而不是大波動裡的假翻多。",
            lambda df: _and(df["bull_on"], df["vol_compressed"]),
            _d3_exit,
        ),
        V2Spec(
            "V_CONF23",
            "V",
            "綠飄帶後第 2～3 根仍多且收紅",
            "假設：不在轉綠當根追，等 2～3 根確認收盤仍是多頭且收紅，減少隔日立刻被打。",
            lambda df: df["confirm_23"].astype(bool),
            _d3_exit,
        ),
        V2Spec(
            "V_UPDAY",
            "V",
            "綠飄帶且當日收盤＞開盤",
            "假設：轉綠當天是陽線才跟，陰線綠飄帶品質較差。",
            lambda df: _and(df["bull_on"], df["up_day"]),
            _d3_exit,
        ),
        V2Spec(
            "V_SL8",
            "V",
            "D3 進出＋固定 8% 停損",
            "假設：D3 輸家多在 -8%～-12%，硬停損能剪左尾、壓 MaxDD；勝率可能下降。",
            _d3_entry,
            _d3_exit,
            stop_pct=0.08,
        ),
        V2Spec(
            "V_ATR2",
            "V",
            "D3 進出＋進場價 −2×ATR 停損",
            "假設：用當下波動決定停損距離，比固定百分比更貼 TSLA 的脾氣。",
            _d3_entry,
            _d3_exit,
            atr_stop_k=2.0,
        ),
        V2Spec(
            "V_TRAIL",
            "V",
            "D3 進出＋最高價 −2.5×ATR 移動停利",
            "假設：賺到的波段用移動停利鎖住，回撤會小；可能提早下車少吃趨勢。",
            _d3_entry,
            _d3_exit,
            trail_atr_k=2.5,
        ),
        V2Spec(
            "V_TIME15",
            "V",
            "D3 進出＋持有滿 15 日時間出場",
            "假設：D3 平均抱約 12 日，15 日還不出代表波段沒走，先走避免變長虧。",
            _d3_entry,
            _d3_exit,
            time_exit_bars=15,
        ),
        V2Spec(
            "V_COOL2",
            "V",
            "D3 進出＋連虧 2 筆跳過下一訊號",
            "假設：連續挨打後立刻再進容易在壞環境加碼，冷卻一筆可改善體感與回撤。",
            _d3_entry,
            _d3_exit,
            cooldown_losses=2,
        ),
        V2Spec(
            "V_QUALITY",
            "V",
            "組合濾網：BASIS＋陽線＋TREND 向上",
            "假設：預先指定的品質組合（不是掃參），同時滿足站上中軌、當日收紅、動能向上。",
            lambda df: _and(df["bull_on"], df["above_basis"], df["up_day"], df["trend_up"]),
            _d3_exit,
        ),
        # —— 經典對照 ——
        V2Spec(
            "CL_SMA50200",
            "CL",
            "雙均線 SMA50／SMA200 金叉死叉",
            "經典趨勢：50 上穿 200 進、50 下穿 200 出。規則公開、不貼 THT／BX 語意。",
            lambda df: df["sma50_x_sma200"].astype(bool),
            lambda df: df["sma200_x_sma50"].astype(bool),
        ),
        V2Spec(
            "CL_EMA2050",
            "CL",
            "雙均線 EMA20／EMA50 金叉死叉",
            "較快的均線交叉，對照 SMA50／200 是否太鈍。",
            lambda df: df["ema20_x_ema50"].astype(bool),
            lambda df: df["ema50_x_ema20"].astype(bool),
        ),
        V2Spec(
            "CL_DONCHIAN",
            "CL",
            "海龜簡化：20 日高突破進／10 日低出",
            "收盤突破前 20 日最高進，收盤跌破前 10 日最低出（確認收盤、次日開盤成交）。",
            lambda df: df["donch_entry"].astype(bool),
            lambda df: df["donch_exit"].astype(bool),
        ),
        V2Spec(
            "CL_RSI14_50",
            "CL",
            "RSI(14) 上穿 30 進／RSI＞50 出",
            "標準 Wilder RSI14 超賣反彈；出場門檻 50（偏早鎖利）。",
            lambda df: df["rsi14_x_30"].astype(bool),
            lambda df: df["rsi14_gt_50"].astype(bool),
        ),
        V2Spec(
            "CL_RSI14_70",
            "CL",
            "RSI(14) 上穿 30 進／RSI＞70 出",
            "同上進場，出場改 RSI＞70，讓反彈多跑一段。",
            lambda df: df["rsi14_x_30"].astype(bool),
            lambda df: df["rsi14_gt_70"].astype(bool),
        ),
        V2Spec(
            "CL_TRENDVOL",
            "CL",
            "趨勢＋波動：收盤＞SMA50 且 ATR% 不過高",
            "站上 50 日均且 ATR% 低於近 60 日 80 分位才做多；跌破 SMA50 或波動失控出。",
            lambda df: df["trend_vol_on"].astype(bool),
            lambda df: df["trend_vol_off"].astype(bool),
        ),
        V2Spec(
            "F",
            "F",
            "Buy&Hold 對照",
            "分析區間開盤買進、期末收盤賣出，與 v1 相同規則。",
            _d3_entry,
            _d3_exit,
            is_bh=True,
        ),
    ]
    return specs
