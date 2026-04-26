"""全戦略パラメータの一元管理"""

# === 通貨ペア ===
SYMBOL = "JPY=X"  # yfinanceでのUSD/JPYティッカー
PAIR_NAME = "USD/JPY"

# === 時間足 ===
TIMEFRAME = "1h"  # H1
TIMEFRAME_H4 = "4h"  # H4 (上位足フィルター用)

# === Bollinger Bands ===
BB_PERIOD = 20
BB_STD = 2.0

# === RSI ===
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# === H4 RSI フィルター (上位足トレンドフィルター) ===
H4_RSI_PERIOD = 14
H4_RSI_LONG_MIN = 35   # H4 RSI がこの値以上でLONGエントリー許可
H4_RSI_SHORT_MAX = 65  # H4 RSI がこの値以下でSHORTエントリー許可

# === ATR ===
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5      # SL = ATR × この倍率
ATR_BREAKEVEN_MULTIPLIER = 1.0  # 建値移動の閾値
ATR_VOL_PERIOD = 50           # ボラティリティフィルター用MA期間
ATR_VOL_THRESHOLD = 1.5       # ATR < ATR_MA × この倍率でエントリー許可

# === セッションフィルター (UTC) ===
SESSION_START_UTC = 0   # アジアセッション開始 (00:00 UTC)
SESSION_END_UTC = 8     # アジアセッション終了 (08:00 UTC)

# === 決済条件 ===
TIME_STOP_BARS = 12  # 最大保有バー数 (H1なら12時間)

# === リスク管理 ===
RISK_PER_TRADE = 0.01      # 1トレードあたりリスク (1%)
MAX_POSITIONS = 2           # 最大同時ポジション数
DAILY_LOSS_LIMIT = 0.03    # 1日の最大損失 (3%)
INITIAL_EQUITY = 1_000_000  # 初期資金 (円)

# === バックテスト ===
# backtesting.py の spread は比率 (relative to price)
# USD/JPY 0.4 pips = 0.004円, 価格152円 → 0.004/152 ≈ 2.63e-5
SPREAD_PIPS = 0.4
SPREAD_RATIO = SPREAD_PIPS * 0.01 / 152.0  # backtesting.py用
COMMISSION = 0.0
DATA_YEARS = 2      # ヒストリカルデータ取得年数

# === 最適化パラメータ範囲 ===
OPTIMIZE_RANGES = {
    "bb_period": range(15, 26, 5),        # [15, 20, 25]
    "bb_std": [1.5, 2.0, 2.5],
    "rsi_period": [10, 14, 21],
    "rsi_oversold": [25, 30, 35],
    "rsi_overbought": [65, 70, 75],
    "atr_sl_multiplier": [1.0, 1.5, 2.0],
    "time_stop_bars": [8, 12, 16, 24],
}
