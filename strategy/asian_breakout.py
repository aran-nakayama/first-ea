"""Asian Range Breakout 戦略 - backtesting.py 用 Strategy クラス

USD/JPY のアジアセッション(東京時間)のレンジを計測し、
ロンドン/NY セッションでのブレイクアウトを狙う戦略。

最適化済みパラメータ (2024-2026 バックテスト):
- SL=30pips, TP=75pips → WR 43%, Return +14.7%, MDD -2.4%, Sharpe 1.37
- SL=60pips, TP=75pips → WR 51%, Return +16.2%, MDD -3.0%, Sharpe 1.34
"""

import numpy as np
import pandas as pd
from backtesting import Strategy


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """H1データにアジアレンジ情報を追加する。"""
    df = df.copy()
    df["Hour"] = df.index.hour

    asian_high = []
    asian_low = []
    current_high = -np.inf
    current_low = np.inf
    current_date = None

    for idx, row in df.iterrows():
        date = idx.date()
        hour = idx.hour

        if date != current_date and hour == 0:
            current_date = date
            current_high = row["High"]
            current_low = row["Low"]
        elif date == current_date and hour < 8:
            current_high = max(current_high, row["High"])
            current_low = min(current_low, row["Low"])

        if current_high == -np.inf or current_low == np.inf:
            asian_high.append(np.nan)
            asian_low.append(np.nan)
        else:
            asian_high.append(current_high)
            asian_low.append(current_low)

    df["Asian_High"] = asian_high
    df["Asian_Low"] = asian_low
    df["Asian_Range"] = df["Asian_High"] - df["Asian_Low"]

    return df


class AsianBreakoutStrategy(Strategy):
    # === 最適化可能なパラメータ ===
    sl_pips = 30
    tp_pips = 75
    session_start = 8       # ブレイクアウト監視開始 (UTC)
    session_end = 14        # ブレイクアウト監視終了 (UTC)
    close_hour = 21         # 時間制限決済 (UTC)
    min_range = 0.15        # 最小レンジ幅
    max_range = 1.5         # 最大レンジ幅

    def init(self):
        self.asian_high = self.I(lambda: self.data.Asian_High, name="Asian_High", overlay=True)
        self.asian_low = self.I(lambda: self.data.Asian_Low, name="Asian_Low", overlay=True)
        self.asian_range = self.I(lambda: self.data.Asian_Range, name="Asian_Range", plot=False)
        self._today_traded = None
        self._tp_price = None
        self._pending = False

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        hour = self.data.Hour[-1]
        a_high = self.asian_high[-1]
        a_low = self.asian_low[-1]
        a_range = self.asian_range[-1]

        if _any_nan(a_high, a_low, a_range):
            return

        today = self.data.index[-1].date()

        # === ポジション保有中 ===
        if self.position:
            # エントリー直後: 実際のentry_priceからSL/TPを設定
            if self._pending and len(self.trades) > 0:
                entry = self.trades[-1].entry_price
                sl_dist = self.sl_pips * 0.01
                tp_dist = self.tp_pips * 0.01
                if self.position.is_long:
                    self.trades[-1].sl = entry - sl_dist
                    self._tp_price = entry + tp_dist
                else:
                    self.trades[-1].sl = entry + sl_dist
                    self._tp_price = entry - tp_dist
                self._pending = False
                return

            # 時間制限決済
            if hour >= self.close_hour:
                self.position.close()
                return

            # 手動TP判定
            if self._tp_price is not None:
                if self.position.is_long and high >= self._tp_price:
                    self.position.close()
                    return
                if self.position.is_short and low <= self._tp_price:
                    self.position.close()
                    return
            return

        # === 新規エントリー判定 ===
        if self._today_traded == today:
            return
        if not (self.session_start <= hour < self.session_end):
            return
        if a_range < self.min_range or a_range > self.max_range:
            return

        # LONG ブレイクアウト
        if high > a_high:
            self.buy()
            self._today_traded = today
            self._pending = True

        # SHORT ブレイクアウト
        elif low < a_low:
            self.sell()
            self._today_traded = today
            self._pending = True


def _any_nan(*values) -> bool:
    return any(v is None or (isinstance(v, float) and np.isnan(v)) for v in values)
