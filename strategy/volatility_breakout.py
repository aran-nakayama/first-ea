"""Volatility Breakout 戦略 - 短期モメンタムキャッチ

Larry Williams のボラティリティブレイクアウトをベースにした短期戦略。
前日のレンジ (High-Low) の k倍を基準にして、日中の方向性ブレイクアウトを捕まえる。

トレンドフォローとの違い:
- TF: 数日間保有、大きなトレンドを捕まえる (低頻度・高利益)
- VB: 数時間保有、日中のモメンタムを捕まえる (高頻度・小利益)

組み合わせ効果:
- TF がレンジ相場で小さく負ける月 → VB の小さな勝ちでカバー
- TF がトレンドで大きく勝つ月 → VB も同方向で追加利益
"""

import numpy as np
import pandas as pd
from backtesting import Strategy


def prepare_data(df: pd.DataFrame, lookback=24, atr_period=14):
    """H1データにボラティリティブレイクアウト用の指標を追加する。

    Args:
        df: OHLCV DataFrame
        lookback: レンジ計算の期間 (H1本数、24=1日)
        atr_period: ATR計算期間
    """
    df = df.copy()

    # 過去N時間のレンジ (High - Low)
    df["Range_High"] = df["High"].rolling(lookback).max().shift(1)
    df["Range_Low"] = df["Low"].rolling(lookback).min().shift(1)
    df["Range"] = df["Range_High"] - df["Range_Low"]

    # 過去N時間の終値 (ブレイクアウト基準点)
    df["Pivot"] = df["Close"].shift(1)

    # ATR (SL計算用)
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=atr_period, adjust=False).mean()

    # ATR 移動平均 (ボラティリティフィルター用)
    df["ATR_MA"] = df["ATR"].rolling(50).mean()

    # ADX (トレンド強度、フィルター用)
    # 簡易ADX: +DI と -DI の差分をスムーズに
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / df["ATR"]
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / df["ATR"]
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["ADX"] = dx.ewm(span=14, adjust=False).mean()

    # 短期モメンタム (方向確認用)
    df["MOM_Short"] = df["Close"].pct_change(6)   # 6時間モメンタム
    df["MOM_Mid"] = df["Close"].pct_change(24)    # 24時間モメンタム

    # 時間帯
    if hasattr(df.index, 'hour'):
        df["Hour"] = df.index.hour

    return df


class VolatilityBreakoutStrategy(Strategy):
    # === 最適化可能なパラメータ ===
    k_factor = 0.5          # ブレイクアウト閾値 (Range × k)
    atr_sl_mult = 2.0       # SL = ATR × この倍率
    atr_tp_mult = 2.0       # TP = ATR × この倍率
    max_hold_hours = 12     # 最大保有時間
    adx_min = 15            # ADX がこの値以上でのみエントリー
    session_start = 2       # エントリー可能時間帯 (UTC)
    session_end = 20        # エントリー可能時間帯 (UTC)

    def init(self):
        self.range_high = self.I(lambda: self.data.Range_High, name="Range_H", plot=False)
        self.range_low = self.I(lambda: self.data.Range_Low, name="Range_L", plot=False)
        self.range_val = self.I(lambda: self.data.Range, name="Range", plot=False)
        self.pivot = self.I(lambda: self.data.Pivot, name="Pivot", plot=False)
        self.atr = self.I(lambda: self.data.ATR, name="ATR", plot=False)
        self.atr_ma = self.I(lambda: self.data.ATR_MA, name="ATR_MA", plot=False)
        self.adx = self.I(lambda: self.data.ADX, name="ADX")
        self.mom_short = self.I(lambda: self.data.MOM_Short, name="MOM6", plot=False)
        self.mom_mid = self.I(lambda: self.data.MOM_Mid, name="MOM24", plot=False)
        self._entry_bar = 0
        self._bar_count = 0

    def next(self):
        self._bar_count += 1
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]
        atr_ma = self.atr_ma[-1]
        rng = self.range_val[-1]
        pivot = self.pivot[-1]
        adx = self.adx[-1]
        mom_s = self.mom_short[-1]
        mom_m = self.mom_mid[-1]

        if _any_nan(atr, atr_ma, rng, pivot, adx, mom_s, mom_m):
            return

        # === ポジション保有中 ===
        if self.position:
            bars_held = self._bar_count - self._entry_bar
            if bars_held >= self.max_hold_hours:
                self.position.close()
            return

        # === フィルター ===
        # 時間帯フィルター
        if hasattr(self.data, 'Hour'):
            hour = self.data.Hour[-1]
            if not (self.session_start <= hour < self.session_end):
                return

        # ADXフィルター (最低限のトレンド強度)
        if adx < self.adx_min:
            return

        # レンジが小さすぎる場合はスキップ
        if rng <= 0:
            return

        # ブレイクアウト閾値
        threshold = rng * self.k_factor

        sl_distance = atr * self.atr_sl_mult
        tp_distance = atr * self.atr_tp_mult
        if sl_distance <= 0 or tp_distance <= 0:
            return

        # === LONG: 価格がピボット + threshold を上抜け + モメンタム確認 ===
        if high > pivot + threshold and mom_s > 0:
            sl = price - sl_distance
            tp = price + tp_distance
            self.buy(sl=sl, tp=tp)
            self._entry_bar = self._bar_count
            return

        # === SHORT: 価格がピボット - threshold を下抜け + モメンタム確認 ===
        if low < pivot - threshold and mom_s < 0:
            sl = price + sl_distance
            tp = price - tp_distance
            self.sell(sl=sl, tp=tp)
            self._entry_bar = self._bar_count


def _any_nan(*values) -> bool:
    return any(v is None or (isinstance(v, float) and np.isnan(v)) for v in values)
