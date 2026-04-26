"""Ichimoku Cloud 戦略 - 一目均衡表の雲ブレイクアウト

日本発の人気指標。JPYペアで伝統的に使われる。
- 転換線 (Tenkan-sen) = (9期間の高値 + 安値) / 2
- 基準線 (Kijun-sen) = (26期間の高値 + 安値) / 2
- 先行スパンA (Senkou A) = (転換線 + 基準線) / 2 を26期間先行
- 先行スパンB (Senkou B) = (52期間の高値 + 安値) / 2 を26期間先行
- 雲 (Kumo) = Senkou A と Senkou B の間

エントリー:
- 価格が雲を上抜け + 転換線 > 基準線 = ロング
- 価格が雲を下抜け + 転換線 < 基準線 = ショート
"""

import numpy as np
import pandas as pd
from backtesting import Strategy


def prepare_data(df: pd.DataFrame,
                 tenkan_period=9, kijun_period=26, senkou_b_period=52,
                 displacement=26, atr_period=14):
    """H1データに一目均衡表とATRを追加する。"""
    df = df.copy()

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # 転換線
    df["TENKAN"] = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2

    # 基準線
    df["KIJUN"] = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2

    # 先行スパンA (26期間先行 = 現時点で26期間前の計算値を参照)
    senkou_a = (df["TENKAN"] + df["KIJUN"]) / 2
    df["SENKOU_A"] = senkou_a.shift(displacement)

    # 先行スパンB
    senkou_b = (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2
    df["SENKOU_B"] = senkou_b.shift(displacement)

    # 雲の上下
    df["CLOUD_TOP"] = df[["SENKOU_A", "SENKOU_B"]].max(axis=1)
    df["CLOUD_BOT"] = df[["SENKOU_A", "SENKOU_B"]].min(axis=1)

    # 前バーの値（ブレイクアウト検出用）
    df["CLOSE_PREV"] = close.shift(1)
    df["CLOUD_TOP_PREV"] = df["CLOUD_TOP"].shift(1)
    df["CLOUD_BOT_PREV"] = df["CLOUD_BOT"].shift(1)

    # ATR
    close_prev = close.shift(1)
    tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=atr_period, adjust=False).mean()

    return df


class IchimokuStrategy(Strategy):
    # === 最適化可能なパラメータ ===
    atr_sl_mult = 2.5
    atr_trail_mult = 2.5
    max_hold_hours = 120
    use_tenkan_kijun = 1  # 1=転換/基準クロス条件必須

    def init(self):
        self.tenkan = self.I(lambda: self.data.TENKAN, name="Tenkan", overlay=True)
        self.kijun = self.I(lambda: self.data.KIJUN, name="Kijun", overlay=True)
        self.cloud_top = self.I(lambda: self.data.CLOUD_TOP, name="Cloud_Top", overlay=True)
        self.cloud_bot = self.I(lambda: self.data.CLOUD_BOT, name="Cloud_Bot", overlay=True)
        self.close_prev = self.I(lambda: self.data.CLOSE_PREV, name="Close_Prev", plot=False)
        self.cloud_top_prev = self.I(lambda: self.data.CLOUD_TOP_PREV, name="Cloud_Top_Prev", plot=False)
        self.cloud_bot_prev = self.I(lambda: self.data.CLOUD_BOT_PREV, name="Cloud_Bot_Prev", plot=False)
        self.atr = self.I(lambda: self.data.ATR, name="ATR", plot=False)
        self._entry_bar = 0
        self._bar_count = 0

    def next(self):
        self._bar_count += 1
        price = self.data.Close[-1]
        prev_close = self.close_prev[-1]
        atr = self.atr[-1]
        tenkan = self.tenkan[-1]
        kijun = self.kijun[-1]
        cloud_top = self.cloud_top[-1]
        cloud_bot = self.cloud_bot[-1]
        cloud_top_prev = self.cloud_top_prev[-1]
        cloud_bot_prev = self.cloud_bot_prev[-1]

        if _any_nan(atr, tenkan, kijun, cloud_top, cloud_bot,
                    cloud_top_prev, cloud_bot_prev, prev_close):
            return

        # === ポジション管理 ===
        if self.position:
            bars_held = self._bar_count - self._entry_bar

            if bars_held >= self.max_hold_hours:
                self.position.close()
                return

            # 雲への逆戻りで決済
            if self.position.is_long and price < cloud_bot:
                self.position.close()
                return
            if self.position.is_short and price > cloud_top:
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

        # === エントリー判定 ===
        # 雲ブレイクアウト（前バーは雲内/下、現バーは雲上）
        long_break = prev_close <= cloud_top_prev and price > cloud_top
        short_break = prev_close >= cloud_bot_prev and price < cloud_bot

        # 転換/基準クロス条件
        tk_long = tenkan > kijun if self.use_tenkan_kijun == 1 else True
        tk_short = tenkan < kijun if self.use_tenkan_kijun == 1 else True

        if long_break and tk_long:
            self.buy(sl=price - sl_distance)
            self._entry_bar = self._bar_count
            return

        if short_break and tk_short:
            self.sell(sl=price + sl_distance)
            self._entry_bar = self._bar_count


def _any_nan(*values) -> bool:
    return any(v is None or (isinstance(v, float) and np.isnan(v)) for v in values)
