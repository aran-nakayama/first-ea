"""MACD Momentum 戦略 - MACDヒストグラムのゼロクロスで勢いを捕捉

MACDは最もポピュラーなモメンタム指標のひとつ。
- MACDライン = 短期EMA - 長期EMA
- シグナルライン = MACDラインのEMA
- ヒストグラム = MACD - Signal

ヒストグラムがマイナス→プラスに転換でロング（モメンタム強まり）、
プラス→マイナス転換でショート。EMA200によるトレンドフィルター付き。
"""

import numpy as np
import pandas as pd
from backtesting import Strategy


def prepare_data(df: pd.DataFrame,
                 fast_period=12, slow_period=26, signal_period=9,
                 atr_period=14, ma_period=200):
    """H1データにMACD, ATR, MAを追加する。"""
    df = df.copy()

    close = df["Close"]
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal

    df["MACD"] = macd
    df["MACD_SIG"] = signal
    df["MACD_HIST"] = hist
    df["MACD_HIST_PREV"] = hist.shift(1)

    # MA（トレンドフィルター）
    df["MA_TREND"] = close.rolling(ma_period).mean()

    # ATR
    high = df["High"]
    low = df["Low"]
    close_prev = close.shift(1)
    tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=atr_period, adjust=False).mean()

    return df


class MACDMomentumStrategy(Strategy):
    # === 最適化可能なパラメータ ===
    atr_sl_mult = 2.5
    atr_trail_mult = 2.5
    max_hold_hours = 96
    use_ma_filter = 1  # 1=EMA200フィルター使用, 0=フィルター無し
    min_hist_threshold = 0.0  # ヒスト転換の最小幅

    def init(self):
        self.macd = self.I(lambda: self.data.MACD, name="MACD", plot=False)
        self.signal = self.I(lambda: self.data.MACD_SIG, name="Signal", plot=False)
        self.hist = self.I(lambda: self.data.MACD_HIST, name="Hist")
        self.hist_prev = self.I(lambda: self.data.MACD_HIST_PREV, name="Hist_Prev", plot=False)
        self.ma = self.I(lambda: self.data.MA_TREND, name="MA200", overlay=True)
        self.atr = self.I(lambda: self.data.ATR, name="ATR", plot=False)
        self._entry_bar = 0
        self._bar_count = 0

    def next(self):
        self._bar_count += 1
        price = self.data.Close[-1]
        atr = self.atr[-1]
        hist = self.hist[-1]
        hist_prev = self.hist_prev[-1]
        ma = self.ma[-1]

        if _any_nan(atr, hist, hist_prev, ma):
            return

        # === ポジション管理 ===
        if self.position:
            bars_held = self._bar_count - self._entry_bar

            if bars_held >= self.max_hold_hours:
                self.position.close()
                return

            # ヒストグラムの反転で早期決済
            if self.position.is_long and hist < 0 and hist_prev >= 0:
                self.position.close()
                return
            if self.position.is_short and hist > 0 and hist_prev <= 0:
                self.position.close()
                return

            # ATRトレーリング
            for trade in self.trades:
                if trade.is_long:
                    new_sl = price - atr * self.atr_trail_mult
                    trade.sl = max(trade.sl or 0, new_sl)
                else:
                    new_sl = price + atr * self.atr_trail_mult
                    trade.sl = min(trade.sl or 9999, new_sl)
            return

        sl_distance = atr * self.atr_sl_mult
        if sl_distance <= 0:
            return

        # ヒストグラム ゼロクロス検出
        cross_up = hist_prev <= 0 and hist > self.min_hist_threshold
        cross_down = hist_prev >= 0 and hist < -self.min_hist_threshold

        # MAフィルター
        long_ok = True
        short_ok = True
        if self.use_ma_filter == 1:
            long_ok = price > ma
            short_ok = price < ma

        if cross_up and long_ok:
            self.buy(sl=price - sl_distance)
            self._entry_bar = self._bar_count
            return

        if cross_down and short_ok:
            self.sell(sl=price + sl_distance)
            self._entry_bar = self._bar_count


def _any_nan(*values) -> bool:
    return any(v is None or (isinstance(v, float) and np.isnan(v)) for v in values)
