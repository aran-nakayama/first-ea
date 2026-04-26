"""RSI Mean Reversion 戦略 - トレンドフォローの補完用

トレンドフォローが苦手な「レンジ相場」で小さく勝つ設計。
- RSIの極値（売られすぎ/買われすぎ）で逆張りエントリー
- BB中央線への回帰で利確
- 短い保有期間（最大24時間）で回転率を上げる
- 高勝率・小利益 vs トレンドフォローの低勝率・大利益

ポートフォリオ効果:
- TF がレンジで小さく負ける月 → MR が小さく勝ってカバー
- TF がトレンドで大きく勝つ月 → MR は小さく負けるが TF の利益で相殺
"""

import numpy as np
import pandas as pd
from backtesting import Strategy


def prepare_data(df: pd.DataFrame, rsi_period=14, bb_period=20, bb_std=2.0,
                 atr_period=14):
    """H1データにRSI, BB, ATRを追加する。"""
    df = df.copy()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(bb_period).mean()
    bb_std_val = df["Close"].rolling(bb_period).std()
    df["BB_Upper"] = df["BB_Middle"] + bb_std * bb_std_val
    df["BB_Lower"] = df["BB_Middle"] - bb_std * bb_std_val

    # ATR
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=atr_period, adjust=False).mean()

    # ATR移動平均（ボラティリティフィルター用）
    df["ATR_MA"] = df["ATR"].rolling(50).mean()

    # 前バーRSI（方向確認用）
    df["RSI_Prev"] = df["RSI"].shift(1)

    return df


class RSIMeanReversionStrategy(Strategy):
    # === 最適化可能なパラメータ ===
    rsi_oversold = 25       # RSIがこの値以下で買い検討
    rsi_overbought = 75     # RSIがこの値以上で売り検討
    atr_sl_mult = 2.0       # SL = ATR × この倍率
    atr_tp_mult = 1.5       # TP = ATR × この倍率 (BBミドルより先にヒットする場合)
    max_hold_hours = 24     # 最大保有時間
    vol_filter = 2.0        # ATR > ATR_MA × この倍率なら見送り (高ボラ回避)
    use_bb_tp = 1           # 1=BB中央線で利確, 0=ATR×tp_multで利確

    def init(self):
        self.rsi = self.I(lambda: self.data.RSI, name="RSI")
        self.rsi_prev = self.I(lambda: self.data.RSI_Prev, name="RSI_Prev", plot=False)
        self.bb_upper = self.I(lambda: self.data.BB_Upper, name="BB_Upper", overlay=True)
        self.bb_middle = self.I(lambda: self.data.BB_Middle, name="BB_Middle", overlay=True)
        self.bb_lower = self.I(lambda: self.data.BB_Lower, name="BB_Lower", overlay=True)
        self.atr = self.I(lambda: self.data.ATR, name="ATR", plot=False)
        self.atr_ma = self.I(lambda: self.data.ATR_MA, name="ATR_MA", plot=False)
        self._entry_bar = 0
        self._bar_count = 0

    def next(self):
        self._bar_count += 1
        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        rsi_prev = self.rsi_prev[-1]
        atr = self.atr[-1]
        atr_ma = self.atr_ma[-1]
        bb_middle = self.bb_middle[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]

        if _any_nan(rsi, rsi_prev, atr, atr_ma, bb_middle, bb_upper, bb_lower):
            return

        # === ポジション保有中 ===
        if self.position:
            bars_held = self._bar_count - self._entry_bar

            # 時間制限決済
            if bars_held >= self.max_hold_hours:
                self.position.close()
                return

            # BB中央線での利確
            if self.use_bb_tp == 1:
                if self.position.is_long and price >= bb_middle:
                    self.position.close()
                    return
                if self.position.is_short and price <= bb_middle:
                    self.position.close()
                    return
            return

        # === フィルター: 高ボラティリティ時はエントリーしない ===
        if atr > atr_ma * self.vol_filter:
            return

        sl_distance = atr * self.atr_sl_mult
        if sl_distance <= 0:
            return

        # === エントリー判定 ===
        # LONG: RSIが売られすぎ圏から反転 (RSI_prev < threshold かつ RSI > RSI_prev)
        if rsi < self.rsi_oversold and rsi > rsi_prev:
            sl = price - sl_distance
            if self.use_bb_tp == 0:
                tp = price + atr * self.atr_tp_mult
                self.buy(sl=sl, tp=tp)
            else:
                self.buy(sl=sl)
            self._entry_bar = self._bar_count
            return

        # SHORT: RSIが買われすぎ圏から反転
        if rsi > self.rsi_overbought and rsi < rsi_prev:
            sl = price + sl_distance
            if self.use_bb_tp == 0:
                tp = price - atr * self.atr_tp_mult
                self.sell(sl=sl, tp=tp)
            else:
                self.sell(sl=sl)
            self._entry_bar = self._bar_count


def _any_nan(*values) -> bool:
    return any(v is None or (isinstance(v, float) and np.isnan(v)) for v in values)
