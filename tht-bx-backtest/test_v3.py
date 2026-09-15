"""v3 多標的：宇宙對照、V_CONF23 主參數未改、合成資料能跑完。不需網路。"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from engine import ONE_WAY_COST
from indicators import compute_bx, compute_tht
from run_backtest_v3_multi import (  # noqa: E402
    MIN_TRADES_FEEL,
    MIN_WIN_RATE_FEEL,
    UNIVERSE,
    TickerRun,
    cache_path_for,
    feels_better_row,
    md_summary_table,
    rows_from_runs,
    run_v_conf23_on_frame,
    v_conf23_spec,
    write_report,
)
from strategies_v2 import build_v2_specs


def test_universe_yf_symbols():
    ids = [u["id"] for u in UNIVERSE]
    assert ids[0] == "TSLA"
    assert set(ids) == {"TSLA", "MU", "TSM", "NVDA", "BTC", "QQQ", "SMH"}
    by_id = {u["id"]: u["yf"] for u in UNIVERSE}
    assert by_id["BTC"] == "BTC-USD"
    assert by_id["TSM"] == "TSM"
    assert by_id["TSLA"] == "TSLA"


def test_cache_slug_handles_crypto():
    p = cache_path_for("BTC-USD")
    assert p.name == "btc-usd_ohlcv.csv"


def test_v_conf23_spec_not_retuned():
    spec = v_conf23_spec()
    assert spec.id == "V_CONF23"
    assert spec.stop_pct is None
    assert spec.atr_stop_k is None
    assert spec.trail_atr_k is None
    assert spec.time_exit_bars is None
    # 進場必須是 confirm_23，出場必須是 D3（零軸或轉紅）
    src = inspect.getsource(spec.entry_fn)
    assert "confirm_23" in src
    d3 = next(s for s in build_v2_specs() if s.id == "D3")
    assert spec.exit_fn is d3.exit_fn or inspect.getsource(spec.exit_fn) == inspect.getsource(d3.exit_fn)


def test_tht_bx_defaults_frozen():
    tht = inspect.signature(compute_tht)
    assert tht.parameters["n"].default == 33
    assert tht.parameters["tw"].default == 0.18
    bx = inspect.signature(compute_bx)
    assert bx.parameters["sl1"].default == 5
    assert bx.parameters["sl2"].default == 20
    assert bx.parameters["sl3"].default == 5


def _synthetic_ohlcv(n: int = 280) -> pd.DataFrame:
    """足夠長的合成日K，讓 MA33／RSI／ATR 暖機後仍有分析窗。"""
    idx = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(23)
    # 慢漲＋週期雜訊，讓低點能穿越 BASIS 產生綠飄帶
    drift = np.linspace(80, 140, n)
    noise = np.sin(np.arange(n) / 7.0) * 4 + rng.normal(0, 1.2, n)
    close = drift + noise
    high = close + 1.5
    low = close - 1.5
    open_ = close + rng.normal(0, 0.4, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1_000_000},
        index=idx,
    )


def test_run_v_conf23_on_synthetic():
    raw = _synthetic_ohlcv()
    start = raw.index[80].strftime("%Y-%m-%d")
    end = raw.index[-1].strftime("%Y-%m-%d")
    df, mask, res, bh = run_v_conf23_on_frame(raw, start, end, ONE_WAY_COST)
    assert int(mask.sum()) >= 50
    assert "confirm_23" in df.columns
    assert res.name == "V_CONF23"
    for key in (
        "total_return",
        "ann_return",
        "maxdd",
        "win_rate",
        "n_trades",
        "avg_win",
        "avg_loss",
        "payoff",
        "sharpe",
        "excess_vs_bh",
    ):
        assert key in res.metrics
    assert "total_return" in bh.metrics
    assert len(res.equity) == len(raw)


def test_feels_better_gate():
    good = {"n_trades": 10, "win_rate": 0.55, "maxdd": -0.20}
    bh = {"maxdd": -0.50}
    assert feels_better_row(good, bh)
    assert not feels_better_row({"n_trades": 3, "win_rate": 0.80, "maxdd": -0.10}, bh)
    assert not feels_better_row({"n_trades": 12, "win_rate": 0.40, "maxdd": -0.10}, bh)
    assert not feels_better_row({"n_trades": 12, "win_rate": 0.60, "maxdd": -0.60}, bh)
    assert MIN_TRADES_FEEL == 8
    assert MIN_WIN_RATE_FEEL == 0.50


def test_rows_and_report_sections(tmp_path):
    ok_run = TickerRun(
        ticker="TSLA",
        yf_symbol="TSLA",
        role="基準",
        name_zh="特斯拉（v2 對照）",
        status="ok",
        actual_start="2021-09-15",
        actual_end="2026-09-14",
        n_bars=1254,
        v_metrics={
            "years": 5.0,
            "total_return": 1.99,
            "ann_return": 0.24,
            "maxdd": -0.22,
            "win_rate": 0.55,
            "n_trades": 18,
            "avg_win": 0.16,
            "avg_loss": -0.05,
            "payoff": 3.2,
            "sharpe": 1.1,
        },
        bh_metrics={"total_return": 0.46, "ann_return": 0.08, "maxdd": -0.74, "sharpe": 0.4},
    )
    bad_run = TickerRun(
        ticker="BTC",
        yf_symbol="BTC-USD",
        role="驗證",
        name_zh="比特幣",
        status="download_failed",
        note="yfinance 下載失敗：demo",
    )
    metrics = rows_from_runs([ok_run, bad_run])
    assert list(metrics["ticker"]) == ["TSLA", "BTC"]
    assert bool(metrics.iloc[0]["beats_bh"]) is True
    assert metrics.iloc[1]["status"] == "download_failed"
    table = md_summary_table(metrics)
    assert "TSLA" in table and "BTC" in table
    out = tmp_path / "report.md"
    write_report(out, "2021-09-15", "2026-09-15", ONE_WAY_COST, metrics)
    text = out.read_text(encoding="utf-8")
    assert "給庭安的一句話" in text
    assert "只對 TSLA" in text or "通用規則" in text
    assert "V_CONF23" in text
    assert "不保證" in text or "不是獲利保證" in text
    assert "N=33" in text and "SL1=5" in text


def main() -> None:
    test_universe_yf_symbols()
    test_cache_slug_handles_crypto()
    test_v_conf23_spec_not_retuned()
    test_tht_bx_defaults_frozen()
    test_run_v_conf23_on_synthetic()
    test_feels_better_gate()
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        test_rows_and_report_sections(Path(td))
    print("單元測試通過（v3 多標的）。")


if __name__ == "__main__":
    main()
