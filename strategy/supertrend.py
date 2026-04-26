"""Supertrend 戦略 - ATRベースのトレンドフォロー

Supertrendは ATR x 係数で上下のバンドを作り、価格がそれを突破したら
トレンド転換と判定する人気のトレンドフォロー指標。Donchianと異なり
ATRベースのため、ボラティリティに適応して誤シグナルを減らす。
"""

import numpy as np
import pandas as pd
from backtesting import Strategy


def _compute_supertrend(df, period=10, multiplier=3.0):
    """Supertrend値とトレンド方向を計算する。"""
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    # ATR
    tr = np.zeros(len(df))
    tr[0] = high[0] - low[0]
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values

    hl2 = (high + low) / 2
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper = np.zeros(len(df))
    lower = np.zeros(len(df))
    supertrend = np.zeros(len(df))
    direction = np.zeros(len(df), dtype=int)  # 1=uptrend, -1=downtrend

    upper[0] = upper_basic[0]
    lower[0] = lower_basic[0]
    supertrend[0] = upper[0]
    direction[0] = -1

    for i in range(1, len(df)):
        # Upper band: 更新は下がる場合のみ（前回上回り続ける間は据え置き）
        if upper_basic[i] < upper[i - 1] or close[i - 1] > upper[i - 1]:
            upper[i] = upper_basic[i]
        else:
            upper[i] = upper[i - 1]

        # Lower band: 更新は上がる場合のみ
        if lower_basic[i] > lower[i - 1] or close[i - 1] < lower[i - 1]:
            lower[i] = lower_basic[i]
        else:
            lower[i] = lower[i - 1]

        # トレンド判定
        if supertrend[i - 1] == upper[i - 1]:
            if close[i] > upper[i]:
                supertrend[i] = lower[i]
                direction[i] = 1
            else:
                supertrend[i] = upper[i]
                direction[i] = -1
        else:
            if close[i] < lower[i]:
                supertrend[i] = upper[i]
                direction[i] = -1
            else:
                supertrend[i] = lower[i]
                direction[i] = 1

    df = df.copy()
    df["SUPERTREND"] = supertrend
    df["ST_DIR"] = direction
    df["ATR_ST"] = atr
    return df


def prepare_data(df: pd.DataFrame, period=10, multiplier=3.0, atr_period=14):
    """H1データにSupertrendとATRを追加する。"""
    df = _compute_supertrend(df, period=period, multiplier=multiplier)

    # 通常のATR（SL用）
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=atr_period, adjust=False).mean()

    # トレンド方向の前バー値（シグナル転換検出用）
    df["ST_DIR_PREV"] = df["ST_DIR"].shift(1)
    return df


class SupertrendStrategy(Strategy):
    # === 最適化可能なパラメータ ===
    st_period = 10
    st_multiplier = 3.0
    atr_sl_mult = 2.0
    atr_trail_mult = 2.0
    max_hold_hours = 120

    def init(self):
        self.supertrend = self.I(lambda: self.data.SUPERTREND, name="Supertrend", overlay=True)
        self.st_dir = self.I(lambda: self.data.ST_DIR, name="ST_Dir", plot=False)
        self.st_dir_prev = self.I(lambda: self.data.ST_DIR_PREV, name="ST_Dir_Prev", plot=False)
        self.atr = self.I(lambda: self.data.ATR, name="ATR", plot=False)
        self._entry_bar = 0
        self._bar_count = 0
        self._trail_stop = None

    def next(self):
        self._bar_count += 1
        price = self.data.Close[-1]
        atr = self.atr[-1]
        st_dir = self.st_dir[-1]
        st_dir_prev = self.st_dir_prev[-1]
        supertrend = self.supertrend[-1]

        if _any_nan(atr, st_dir, st_dir_prev, supertrend):
            return

        # === ポジション管理 ===
        if self.position:
            bars_held = self._bar_count - self._entry_bar

            # 時間制限
            if bars_held >= self.max_hold_hours:
                self.position.close()
                self._trail_stop = None
                return

            # Supertrendの方向転換で決済
            if self.position.is_long and st_dir == -1:
                self.position.close()
                self._trail_stop = None
                return
            if self.position.is_short and st_dir == 1:
                self.position.close()
                self._trail_stop = None
                return

            # ATRトレーリング
            for trade in self.trades:
                if trade.is_long:
                    new_sl = price - atr * self.atr_trail_mult
                    if self._trail_stop is None or new_sl > self._trail_stop:
                        self._trail_stop = new_sl
                        trade.sl = max(trade.sl or 0, new_sl)
                else:
                    new_sl = price + atr * self.atr_trail_mult
                    if self._trail_stop is None or new_sl < self._trail_stop:
                        self._trail_stop = new_sl
                        trade.sl = min(trade.sl or 9999, new_sl)
            return

        # === エントリー判定: トレンド方向の転換で新規エントリー ===
        sl_distance = atr * self.atr_sl_mult
        if sl_distance <= 0:
            return

        # ダウン→アップへの転換で買い
        if st_dir_prev == -1 and st_dir == 1:
            self.buy(sl=price - sl_distance)
            self._entry_bar = self._bar_count
            self._trail_stop = price - sl_distance
            return

        # アップ→ダウンへの転換で売り
        if st_dir_prev == 1 and st_dir == -1:
            self.sell(sl=price + sl_distance)
            self._entry_bar = self._bar_count
            self._trail_stop = price + sl_distance


def _any_nan(*values) -> bool:
    return any(v is None or (isinstance(v, float) and np.isnan(v)) for v in values)
