"""トレンドフォロー戦略 - Donchian Channel Breakout + ATR Trail

タートルトレーディングをベースにした戦略。
- N期間の最高値ブレイクで買い、最安値ブレイクで売り
- ATRベースのトレーリングストップで利益を伸ばす
- トレンドが出ている相場で大きく取り、レンジ相場では小さく負ける設計

長期でのエッジ:
- 勝率は低い (30-40%) が、1回の勝ちが大きい
- トレンドフォローは数十年の実績がある手法
"""

import numpy as np
import pandas as pd
from backtesting import Strategy


def prepare_data(df: pd.DataFrame, entry_period=20, exit_period=10, atr_period=14):
    """H1データにDonchianチャネルとATRを追加する。"""
    df = df.copy()

    # Donchian Channel (エントリー用: 前N期間の高値/安値。当日を含めないためshift)
    df["DC_Upper"] = df["High"].rolling(entry_period).max().shift(1)
    df["DC_Lower"] = df["Low"].rolling(entry_period).min().shift(1)

    # Donchian Channel (決済用: 短期、同様にshift)
    df["DC_Exit_Upper"] = df["High"].rolling(exit_period).max().shift(1)
    df["DC_Exit_Lower"] = df["Low"].rolling(exit_period).min().shift(1)

    # ATR (ボラティリティ)
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=atr_period, adjust=False).mean()

    # トレンドフィルター: 長期移動平均
    df["MA_Long"] = df["Close"].rolling(200).mean()

    # Hour列 (セッションフィルター用)
    if hasattr(df.index, 'hour'):
        df["Hour"] = df.index.hour

    return df


class TrendFollowStrategy(Strategy):
    # === 最適化可能なパラメータ ===
    entry_period = 20      # エントリー用Donchianチャネル期間 (H1なので20=約1日)
    exit_period = 10       # 決済用Donchianチャネル期間
    atr_sl_mult = 2.0      # SL = ATR × この倍率
    atr_trail_mult = 3.0   # トレーリングストップ = ATR × この倍率
    use_ma_filter = 1      # 1=MA200フィルター使用, 0=不使用
    max_hold_hours = 120   # 最大保有時間 (5日=120H)

    def init(self):
        self.dc_upper = self.I(lambda: self.data.DC_Upper, name="DC_Upper", overlay=True)
        self.dc_lower = self.I(lambda: self.data.DC_Lower, name="DC_Lower", overlay=True)
        self.dc_exit_upper = self.I(lambda: self.data.DC_Exit_Upper, name="DC_Exit_U", plot=False)
        self.dc_exit_lower = self.I(lambda: self.data.DC_Exit_Lower, name="DC_Exit_L", plot=False)
        self.atr = self.I(lambda: self.data.ATR, name="ATR", plot=False)
        self.ma_long = self.I(lambda: self.data.MA_Long, name="MA200", overlay=True)
        self._pending = False
        self._trail_price = None
        self._entry_bar = 0
        self._bar_count = 0

    def next(self):
        self._bar_count += 1
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]
        dc_up = self.dc_upper[-1]
        dc_low = self.dc_lower[-1]
        ma = self.ma_long[-1]

        if _any_nan(atr, dc_up, dc_low, ma):
            return

        # === ポジション保有中 ===
        if self.position:
            # エントリー直後: SLを設定
            if self._pending and len(self.trades) > 0:
                entry = self.trades[-1].entry_price
                if self.position.is_long:
                    sl = entry - atr * self.atr_sl_mult
                    self.trades[-1].sl = sl
                    self._trail_price = sl
                else:
                    sl = entry + atr * self.atr_sl_mult
                    self.trades[-1].sl = sl
                    self._trail_price = sl
                self._pending = False
                return

            # 最大保有時間の制限
            bars_held = self._bar_count - self._entry_bar
            if bars_held >= self.max_hold_hours:
                self.position.close()
                return

            # トレーリングストップの更新
            if self.position.is_long:
                new_trail = price - atr * self.atr_trail_mult
                if new_trail > self._trail_price:
                    self._trail_price = new_trail
                    if len(self.trades) > 0:
                        self.trades[-1].sl = new_trail

                # Donchian下限での決済
                if low <= self.dc_exit_lower[-1]:
                    self.position.close()
                    return

            else:  # short
                new_trail = price + atr * self.atr_trail_mult
                if new_trail < self._trail_price:
                    self._trail_price = new_trail
                    if len(self.trades) > 0:
                        self.trades[-1].sl = new_trail

                # Donchian上限での決済
                if high >= self.dc_exit_upper[-1]:
                    self.position.close()
                    return
            return

        # === 新規エントリー判定 ===
        # MAフィルター (トレンド方向のみエントリー)
        if self.use_ma_filter == 1:
            can_long = price > ma
            can_short = price < ma
        else:
            can_long = True
            can_short = True

        # LONG: 価格がDonchian上限を突破
        if can_long and high > dc_up:
            self.buy()
            self._pending = True
            self._entry_bar = self._bar_count

        # SHORT: 価格がDonchian下限を突破
        elif can_short and low < dc_low:
            self.sell()
            self._pending = True
            self._entry_bar = self._bar_count


def _any_nan(*values) -> bool:
    return any(v is None or (isinstance(v, float) and np.isnan(v)) for v in values)
