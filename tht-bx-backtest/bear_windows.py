"""事先固定的空頭段。日期不依各標的高低點調整，避免做出好看的區間。"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 七檔共用。2020 年含 2 月下旬，因為疫情急跌不是 3 月 1 日才開始。
BEAR_WINDOWS: list[dict] = [
    {
        "id": "y2018",
        "name": "2018年底",
        "start": "2018-10-01",
        "end": "2018-12-24",
        "note": "美股 10 月高點附近到 12 月 24 日低點",
    },
    {
        "id": "y2020",
        "name": "2020年3月",
        "start": "2020-02-19",
        "end": "2020-03-23",
        "note": "疫情急跌，含 2 月下旬起跌到 3 月低點",
    },
    {
        "id": "y2022",
        "name": "2022下跌",
        "start": "2022-01-03",
        "end": "2022-10-14",
        "note": "2022 年主要下跌段，到 10 月低點附近",
    },
]


def window_performance(equity: pd.Series, start: str, end: str) -> dict:
    """窗口前最後一根權益，到窗口內最後一根權益的報酬，以及這段路徑的最大回撤。

    起點用窗口前的收盤，這樣量的是這段裡發生的事，而不是從更早的高點重算。
    """
    eq = equity.dropna().sort_index()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    before = eq.loc[eq.index < start_ts]
    inside = eq.loc[(eq.index >= start_ts) & (eq.index <= end_ts)]
    if before.empty or inside.empty:
        return {"ret": np.nan, "maxdd": np.nan, "n_bars": int(len(inside)), "covered": False}
    base = float(before.iloc[-1])
    if not np.isfinite(base) or base <= 0:
        return {"ret": np.nan, "maxdd": np.nan, "n_bars": int(len(inside)), "covered": False}
    path = pd.concat([before.iloc[[-1]], inside])
    path = path[~path.index.duplicated(keep="last")]
    ret = float(path.iloc[-1] / base - 1.0)
    peak = path.cummax()
    dd = float((path / peak - 1.0).min())
    return {"ret": ret, "maxdd": dd, "n_bars": int(len(inside)), "covered": True}
