"""日線預警層減碼。比例在寫程式時就固定，不依回測結果改。

部位只在目標比例改變的下一根開盤調整。訊號用當天收盤已經知道的資料。

三條共用：
- 平時 100%。
- 月線 BX 已收盤且為深紅 → 25%。深紅優先於預警，同一天不會停在 50%。
- 收盤回到 33 FVB 帶內（下軌 ≤ 收盤 ≤ 上軌），而且月線不是深紅 → 回到 100%。
- 收盤在上軌之上不算回到帶內，維持上一個目標。跌破下軌是嚴格小於下軌；收盤剛好在下軌上仍算帶內。

E1 的預警：日線收盤跌破 33 FVB 下軌 → 50%。月線離開深紅、但仍在下軌外 → 50%，不是直接回 100%。

E2 的預警改成週線，其餘沿用上面的回補：
- 週線是週日結束的那一週，最後一根日線收盤後才知道，週中沿用上一週。
- 「轉負」做成持續狀態：最近一根已收盤週線 BX < 0 就維持 50%。
- 只在轉負那一週砍一天、若隔天仍在帶內就加回 100%，預警幾乎沒有作用，所以不採用。
- 週線 BX 回到 ≥ 0、收盤在帶內、月線不是深紅，三者同時成立才回 100%。
- 週線不是負的時候，收盤跌破下軌不會單獨減到 50%。那是 E1 的觸發。

E3 是 E1 再加分批回補，只改「從深紅回來」這一段：
- 進入深紅就設成待回補，部位 25%。
- 月線還沒回綠（淺綠或深綠）之前，維持 25%。漸增淺紅、持平負區不開始回補。
- 月線回綠的那個收盤先回到 50%，即使當天已經在帶內也不直接回 100%。
- 已經先回到 50% 之後，之後的收盤回到帶內才回 100%，並結束待回補。
- 回到 50% 之後若月線又離開綠色、但還沒再變深紅，待回補重新等下一次回綠，部位回到 25%。
- 不在待回補時，跌破下軌仍是 50%，回到帶內仍是 100%，跟 E1 一樣。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sizing import GREEN, _color_at


def _bands(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close = df["close"].to_numpy(dtype=float)
    thdn = df["thdn"].to_numpy(dtype=float)
    thup = df["thup"].to_numpy(dtype=float)
    below = np.isfinite(thdn) & (close < thdn)
    inside = np.isfinite(thdn) & np.isfinite(thup) & (close >= thdn) & (close <= thup)
    dark = df["m_dark_red"].fillna(False).to_numpy(dtype=bool)
    return below, inside, dark


def e1_targets(df: pd.DataFrame) -> pd.Series:
    below, inside, dark = _bands(df)
    prev = 1.0
    out = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        if dark[i]:
            prev = 0.25
        elif below[i]:
            prev = 0.50
        elif inside[i]:
            prev = 1.00
        out[i] = prev
    return pd.Series(out, index=df.index, name="target")


def e2_targets(df: pd.DataFrame) -> pd.Series:
    _below, inside, dark = _bands(df)
    weekly_neg = df["w_negative"].fillna(False).to_numpy(dtype=bool)
    prev = 1.0
    out = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        if dark[i]:
            prev = 0.25
        elif weekly_neg[i]:
            prev = 0.50
        elif inside[i]:
            prev = 1.00
        out[i] = prev
    return pd.Series(out, index=df.index, name="target")


def e3_targets(df: pd.DataFrame) -> pd.Series:
    below, inside, dark = _bands(df)
    colors = [_color_at(v) for v in df["m_color"].tolist()]
    prev = 1.0
    recovering = False
    stepped = False
    out = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        green = colors[i] in GREEN
        if dark[i]:
            prev = 0.25
            recovering = True
            stepped = False
        elif recovering and not green:
            prev = 0.25
            stepped = False
        elif recovering and not stepped:
            prev = 0.50
            stepped = True
        elif recovering and inside[i]:
            prev = 1.00
            recovering = False
            stepped = False
        elif recovering:
            prev = 0.50
        elif below[i]:
            prev = 0.50
        elif inside[i]:
            prev = 1.00
        out[i] = prev
    return pd.Series(out, index=df.index, name="target")


def warning_book(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "E1": e1_targets(df),
        "E2": e2_targets(df),
        "E3": e3_targets(df),
    }
