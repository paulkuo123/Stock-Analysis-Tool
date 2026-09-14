"""THT＋BX（可選 RSI2）進出場組合定義。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from indicators import confirm_within


@dataclass(frozen=True)
class Combo:
    id: str
    group: str
    name_zh: str
    note: str
    entry_fn: Callable[[pd.DataFrame], pd.Series]
    exit_fn: Callable[[pd.DataFrame], pd.Series]
    is_bh: bool = False


def _and(*cols: pd.Series) -> pd.Series:
    out = cols[0].fillna(False).astype(bool)
    for c in cols[1:]:
        out = out & c.fillna(False).astype(bool)
    return out


def _or(*cols: pd.Series) -> pd.Series:
    out = cols[0].fillna(False).astype(bool)
    for c in cols[1:]:
        out = out | c.fillna(False).astype(bool)
    return out


def _tht_bx_confirm(df: pd.DataFrame, n: int) -> pd.Series:
    return confirm_within(df["bull_on"], df["dr_to_lr"], n=n, cancel=df["bull_off"])


def _rsi_ok(df: pd.DataFrame, lo: Optional[float], hi: Optional[float]) -> pd.Series:
    ok = pd.Series(True, index=df.index)
    if lo is not None:
        ok = ok & (df["rsi2"] > lo)
    if hi is not None:
        ok = ok & (df["rsi2"] < hi)
    return ok


def build_combos() -> list[Combo]:
    """SPEC 要求 A–F，並加上 BX 著色假設的對照組。"""

    def a_entry(df):
        return df["bull_on"]

    def a_exit(df):
        return df["bull_off"]

    def b1_entry(df):
        return df["dr_to_lr"]

    def b1_exit(df):
        return df["dg_to_lg"]

    def b2_entry(df):
        return df["dr_to_lr"]

    def b2_exit(df):
        return df["lg_to_dg"]

    def b3_entry(df):
        return df["neg_to_pos"]

    def b3_exit(df):
        return df["pos_to_neg"]

    def b4_entry(df):
        return df["dr_to_lr"]

    def b4_exit(df):
        return df["pos_to_neg"]

    def c_entry(n):
        return lambda df: _tht_bx_confirm(df, n)

    def d1_exit(df):
        return _or(df["dg_to_lg"], df["bull_off"])

    def d2_exit(df):
        return _or(df["lg_to_dg"], df["bull_off"])

    def d3_exit(df):
        return _or(df["pos_to_neg"], df["bull_off"])

    combos = [
        Combo("A", "A", "僅 THT：綠飄帶進／紅飄帶出", "BULL 0→1 進；BULL 1→0 出", a_entry, a_exit),
        Combo(
            "B1",
            "B",
            "僅 BX：深紅→淺紅進／深綠→淺綠出",
            "SPEC 字面著色組合（正區由降轉升作出場，見報告假設）",
            b1_entry,
            b1_exit,
        ),
        Combo(
            "B2",
            "B",
            "僅 BX：深紅→淺紅進／淺綠→深綠出",
            "對照：正區由升轉降視為轉弱",
            b2_entry,
            b2_exit,
        ),
        Combo(
            "B3",
            "B",
            "僅 BX：負轉正進／正轉負出",
            "零軸穿越對照，不用著色",
            b3_entry,
            b3_exit,
        ),
        Combo(
            "B4",
            "B",
            "僅 BX：深紅→淺紅進／正轉負出",
            "買盤加強進、跌破零軸出",
            b4_entry,
            b4_exit,
        ),
        Combo(
            "C1",
            "C",
            "THT 進＋BX 確認 N=1",
            "綠飄帶當根須同時深紅→淺紅；出場 BULL 轉假",
            c_entry(1),
            a_exit,
        ),
        Combo(
            "C3",
            "C",
            "THT 進＋BX 確認 N=3",
            "綠飄帶後 3 根內深紅→淺紅；出場 BULL 轉假",
            c_entry(3),
            a_exit,
        ),
        Combo(
            "C5",
            "C",
            "THT 進＋BX 確認 N=5",
            "綠飄帶後 5 根內深紅→淺紅；出場 BULL 轉假",
            c_entry(5),
            a_exit,
        ),
        Combo(
            "D1",
            "D",
            "THT 進；深綠→淺綠或 BULL 轉紅出",
            "SPEC 字面 BX 出場＋綠飄帶失效",
            a_entry,
            d1_exit,
        ),
        Combo(
            "D2",
            "D",
            "THT 進；淺綠→深綠或 BULL 轉紅出",
            "對照：正區轉弱或飄帶失效",
            a_entry,
            d2_exit,
        ),
        Combo(
            "D3",
            "D",
            "THT 進；BX 正轉負或 BULL 轉紅出",
            "零軸跌破或飄帶失效",
            a_entry,
            d3_exit,
        ),
        Combo(
            "C3D2",
            "C",
            "THT＋BX 確認 N=3；淺綠→深綠或 BULL 轉紅出",
            "確認進場＋較合理的轉弱出場",
            c_entry(3),
            d2_exit,
        ),
        Combo(
            "C5D2",
            "C",
            "THT＋BX 確認 N=5；淺綠→深綠或 BULL 轉紅出",
            "N=5 確認＋轉弱出場",
            c_entry(5),
            d2_exit,
        ),
        Combo(
            "E_A_RSI70",
            "E",
            "THT＋RSI2<70",
            "綠飄帶且 RSI2 未超買；BULL 轉假出",
            lambda df: _and(df["bull_on"], _rsi_ok(df, None, 70)),
            a_exit,
        ),
        Combo(
            "E_C3_RSI70",
            "E",
            "THT＋BX 確認 N=3＋RSI2<70",
            "確認進場再濾超買；BULL 轉假出",
            lambda df: _and(_tht_bx_confirm(df, 3), _rsi_ok(df, None, 70)),
            a_exit,
        ),
        Combo(
            "E_C3_RSI70_D2",
            "E",
            "THT＋BX 確認 N=3＋RSI2<70；轉弱或轉紅出",
            "建議候選：確認＋未超買＋轉弱出場",
            lambda df: _and(_tht_bx_confirm(df, 3), _rsi_ok(df, None, 70)),
            d2_exit,
        ),
        Combo(
            "E_C5_RSI70_D2",
            "E",
            "THT＋BX 確認 N=5＋RSI2<70；轉弱或轉紅出",
            "N=5 版本",
            lambda df: _and(_tht_bx_confirm(df, 5), _rsi_ok(df, None, 70)),
            d2_exit,
        ),
        Combo(
            "E_A_RSI30_70",
            "E",
            "THT＋RSI2 介於 30–70",
            "避開超買超賣極端；BULL 轉假出",
            lambda df: _and(df["bull_on"], _rsi_ok(df, 30, 70)),
            a_exit,
        ),
        Combo(
            "E_D2_RSI70",
            "E",
            "THT＋RSI2<70；淺綠→深綠或 BULL 轉紅出",
            "不過濾 BX 確認，只濾 RSI",
            lambda df: _and(df["bull_on"], _rsi_ok(df, None, 70)),
            d2_exit,
        ),
        Combo("F", "F", "Buy&Hold 對照", "分析區間開盤買進、期末收盤賣出", lambda df: df["bull_on"], lambda df: df["bull_off"], is_bh=True),
    ]
    return combos
