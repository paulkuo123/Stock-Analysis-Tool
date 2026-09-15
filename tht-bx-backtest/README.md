# THT＋BX 組合回測（TSLA 日K）

獨立子目錄，**不依賴** repo 裡的 PyQt 選股工具。評估庭安自用指標 THT＋BX（可選 RSI2）的進出場搭法，並用建議組合回測約五年。

## 一句話

建議組合：**THT 進；BX 正轉負或 BULL 轉紅出**（`D3`）。本次總報酬 171.53%、年化 22.23%、最大回撤 -33.09%、Sharpe 0.85。同期 Buy&Hold 總報酬 45.77%。數字最好的單策略是 BX 零軸穿越（`B3`）。**不保證獲利。**

細節、假設與完整績效表見 [report.md](report.md)。

## 重跑

```bash
pip install -r requirements.txt
python run_backtest.py
python test_indicators.py
```

常用參數：

```bash
python run_backtest.py --ticker TSLA --start 2021-09-15 --end 2026-09-15
python run_backtest.py --force-download   # 重新抓行情
```

預設：訊號次日開盤進出、單邊成本 0.05%、只做多、一次一筆。資料來源 yfinance（還原 OHLC），並快取到 `data/`。

## 產出

- `report.md`：繁中台灣用語報告
- `results/combo_metrics.csv`、`equity_curves.csv`、`indicators.csv`、`trades_*.csv`
- `results/figures/*.png`

## 參數限制（依 SPEC）

- BX 只用 SL1=5, SL2=20, SL3=5
- RSI 只用 P2=12 的 SMA 型 RSI2
- THT 進場：BULL 0→1（綠色飄帶）

## v2：更好打？變體＋經典對照

一鍵重跑（不覆蓋 v1 的 `results/combo_metrics.csv`）：

```bash
python run_backtest_v2.py
python test_v2.py
```

報告：[report-v2-better-strategies.md](report-v2-better-strategies.md)。產出在 `results/v2/`。

D3 基準勝率 45.83%、MaxDD -33.09%、總報酬 171.53%。
THT 根基較好打候選：**V_CONF23**（勝率 55.56%）。
經典對照裡最不丟臉的是 **海龜簡化：20 日高突破進／10 日低出**（`CL_DONCHIAN`），仍不如 V_CONF23。**不保證獲利。**
