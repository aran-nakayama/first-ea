"""Tokyo Range Reversion 戦略 - backtesting.py 用 Strategy クラス

USD/JPY の平均回帰を狙う戦略。
BBバンドの外に出た後、バンド内に戻った「バウンス確認」でエントリー。
BB中央線への回帰で利確。ATRベースの動的SL。

v3: バウンス確認方式
- 前バーがBB下限以下 → 現バーがBB下限以上 = LONGエントリー
- 前バーがBB上限以上 → 現バーがBB上限以下 = SHORTエントリー
"""

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import add_indicators, rsi as calc_rsi


def prepare_data(df: pd.DataFrame, bb_period=20, bb_std=2.0,
                 rsi_period=14, atr_period=14) -> pd.DataFrame:
    """バックテスト用にインジケーター列を事前計算してDataFrameに追加する。"""
    df = add_indicators(df, bb_period, bb_std, rsi_period, atr_period)

    # H4 RSI
    h4_close = df["Close"].resample("4h").last().dropna()
    h4_rsi = calc_rsi(h4_close, rsi_period)
    df["H4_RSI"] = h4_rsi.reindex(df.index, method="ffill")

    # セッション時間
    df["Hour"] = df.index.hour

    # 前バーの値 (バウンス確認用)
    df["Close_Prev"] = df["Close"].shift(1)
    df["RSI_Prev"] = df["RSI"].shift(1)
    df["BB_Lower_Prev"] = df["BB_Lower"].shift(1)
    df["BB_Upper_Prev"] = df["BB_Upper"].shift(1)

    return df


class MeanReversionStrategy(Strategy):
    # === 最適化可能なパラメータ ===
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    rsi_oversold = 35      # 緩和 (30→35)
    rsi_overbought = 65    # 緩和 (70→65)
    atr_period = 14
    atr_sl_multiplier = 2.0  # 広め (1.5→2.0)
    atr_be_multiplier = 1.0
    atr_vol_threshold = 2.0  # 緩和 (1.5→2.0)
    time_stop_bars = 24      # 広め (12→24)
    session_start = 0
    session_end = 24         # フィルターなし
    h4_rsi_long_min = 30
    h4_rsi_short_max = 70

    def init(self):
        self.bb_upper = self.I(lambda: self.data.BB_Upper, name="BB_Upper", overlay=True)
        self.bb_middle = self.I(lambda: self.data.BB_Middle, name="BB_Middle", overlay=True)
        self.bb_lower = self.I(lambda: self.data.BB_Lower, name="BB_Lower", overlay=True)
        self.rsi_val = self.I(lambda: self.data.RSI, name="RSI")
        self.rsi_prev = self.I(lambda: self.data.RSI_Prev, name="RSI_Prev")
        self.atr_val = self.I(lambda: self.data.ATR, name="ATR")
        self.atr_ma_val = self.I(lambda: self.data.ATR_MA, name="ATR_MA")
        self.h4_rsi_val = self.I(lambda: self.data.H4_RSI, name="H4_RSI")
        self.close_prev = self.I(lambda: self.data.Close_Prev, name="Close_Prev")
        self.bb_lower_prev = self.I(lambda: self.data.BB_Lower_Prev, name="BB_Lower_Prev")
        self.bb_upper_prev = self.I(lambda: self.data.BB_Upper_Prev, name="BB_Upper_Prev")
        self._entry_bar = 0

    def next(self):
        price = self.data.Close[-1]
        prev_close = self.close_prev[-1]
        atr = self.atr_val[-1]
        bb_upper = self.bb_upper[-1]
        bb_middle = self.bb_middle[-1]
        bb_lower = self.bb_lower[-1]
        bb_lower_prev = self.bb_lower_prev[-1]
        bb_upper_prev = self.bb_upper_prev[-1]
        rsi = self.rsi_val[-1]
        rsi_prev = self.rsi_prev[-1]
        atr_ma = self.atr_ma_val[-1]
        h4_rsi = self.h4_rsi_val[-1]
        hour = self.data.Hour[-1]
        current_bar = len(self.data) - 1

        if _any_nan(bb_upper, bb_lower, bb_middle, rsi, rsi_prev, atr, atr_ma,
                    h4_rsi, prev_close, bb_lower_prev, bb_upper_prev):
            return

        # === ポジション保有中の管理 ===
        if self.position:
            bars_held = current_bar - self._entry_bar

            # 時間制限決済
            if bars_held >= self.time_stop_bars:
                self.position.close()
                return

            # TP: BB中央線への回帰
            if self.position.is_long and price >= bb_middle:
                self.position.close()
                return
            if self.position.is_short and price <= bb_middle:
                self.position.close()
                return

            # トレーリング: 建値移動
            for trade in self.trades:
                entry = trade.entry_price
                if trade.is_long:
                    if price - entry >= atr * self.atr_be_multiplier:
                        trade.sl = max(trade.sl or 0, entry)
                elif trade.is_short:
                    if entry - price >= atr * self.atr_be_multiplier:
                        trade.sl = min(trade.sl or 9999, entry)
            return

        # === フィルターチェック ===
        if self.session_end < 24:
            if not (self.session_start <= hour < self.session_end):
                return

        if atr >= atr_ma * self.atr_vol_threshold:
            return

        sl_distance = atr * self.atr_sl_multiplier
        if sl_distance <= 0:
            return

        # === バウンス確認エントリー ===
        # LONG: 前バーがBB下限以下 + 現バーがBB下限以上 (バウンス) + RSIフィルター
        long_bounce = (prev_close <= bb_lower_prev) and (price > bb_lower)
        long_rsi = rsi < self.rsi_oversold or rsi_prev < self.rsi_oversold
        long_h4 = h4_rsi >= self.h4_rsi_long_min

        if long_bounce and long_rsi and long_h4:
            self.buy(sl=price - sl_distance)
            self._entry_bar = current_bar
            return

        # SHORT: 前バーがBB上限以上 + 現バーがBB上限以下 (バウンス) + RSIフィルター
        short_bounce = (prev_close >= bb_upper_prev) and (price < bb_upper)
        short_rsi = rsi > self.rsi_overbought or rsi_prev > self.rsi_overbought
        short_h4 = h4_rsi <= self.h4_rsi_short_max

        if short_bounce and short_rsi and short_h4:
            self.sell(sl=price + sl_distance)
            self._entry_bar = current_bar


def _any_nan(*values) -> bool:
    return any(v is None or (isinstance(v, float) and np.isnan(v)) for v in values)
