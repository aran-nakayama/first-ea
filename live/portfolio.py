"""4戦略×25%のポートフォリオ調整。

各戦略は独立して動くが、共通の口座(equity)を見て、各戦略が取れる
資金の上限 = equity × 25% × leverage となる。
"""

from typing import Dict, List
import logging
import pandas as pd

from .signals import STRATEGIES, Position, Signal
from .broker import PaperBroker, BrokerPosition

logger = logging.getLogger(__name__)


ALLOC = 0.25  # 各戦略25%
MIN_TRADE_UNITS = 1000  # 最小取引単位 (mini lot = USD 1000)


class MultiPortfolio:
    """4戦略 (TF/ST/MACD/ICHI) を協調実行するコーディネーター。"""

    def __init__(self, broker: PaperBroker, leverage: float = 1.0):
        self.broker = broker
        self.leverage = leverage
        # 戦略状態 (broker positionとは別、ロジック用)
        self._positions: Dict[str, Position] = {sid: Position() for sid in STRATEGIES}

    # ---------- ポジションサイジング ----------
    def calculate_units(self, price: float) -> int:
        """各戦略が取れる取引単位 (USD)。"""
        equity = self.broker.balance
        notional = equity * ALLOC * self.leverage   # JPYでの取引金額
        units = int(notional / price / MIN_TRADE_UNITS) * MIN_TRADE_UNITS
        return max(units, MIN_TRADE_UNITS)

    # ---------- 1バー処理 ----------
    def step(self, df: pd.DataFrame) -> List[dict]:
        """最新バー確定時に呼ばれる。各戦略のシグナルを評価して注文を出す。

        Returns: イベントログのリスト
        """
        events = []
        price = float(df["Close"].iloc[-1])
        bar_time = df.index[-1]
        prices = {sid: price for sid in STRATEGIES}

        # 1. SLヒットチェック (ブローカー側で先にやる)
        closed_by_stop = self.broker.check_stops(prices)
        for c in closed_by_stop:
            sid = c["strategy_id"]
            self._positions[sid] = Position()
            events.append({"type": "stop_hit", "strategy": sid, "record": c})

        # 2. 各戦略のシグナル評価
        for sid, spec in STRATEGIES.items():
            try:
                # 該当戦略の指標つきDataFrame
                df_strat = spec["prepare"](df)
                pos = self._positions[sid]
                # ブローカー側のbars_held同期
                bp = self.broker.get_position(sid)
                if bp:
                    pos.bars_held = bp.bars_held
                sig = spec["signal"](df_strat, pos)

                # シグナル処理
                if sig.action == "enter_long" or sig.action == "enter_short":
                    if pos.side != "flat" or self.broker.get_position(sid):
                        continue   # 既存ポジあり
                    side = "long" if sig.action == "enter_long" else "short"
                    units = self.calculate_units(price)
                    self.broker.place_order(sid, side, units, price, sl=sig.sl)
                    pos.side = side
                    pos.entry_price = price
                    pos.entry_time = bar_time
                    pos.sl = sig.sl
                    pos.trail_stop = sig.trail_stop
                    pos.units = units
                    pos.bars_held = 0
                    events.append({
                        "type": "open", "strategy": sid, "side": side,
                        "price": price, "units": units, "sl": sig.sl,
                        "reason": sig.reason, "time": str(bar_time),
                    })

                elif sig.action == "exit":
                    if pos.side != "flat" and self.broker.get_position(sid):
                        rec = self.broker.close_position(sid, price, reason=sig.reason)
                        self._positions[sid] = Position()
                        events.append({
                            "type": "close", "strategy": sid,
                            "record": rec, "time": str(bar_time),
                        })

                elif sig.action == "update_sl":
                    if pos.side != "flat" and sig.trail_stop:
                        self.broker.update_sl(sid, sig.trail_stop, sig.trail_stop)
                        pos.trail_stop = sig.trail_stop

            except Exception as e:
                logger.exception(f"{sid} 戦略でエラー: {e}")
                events.append({"type": "error", "strategy": sid, "error": str(e)})

        # 3. bars_held インクリメント
        self.broker.increment_bars()
        return events

    # ---------- 状態取得 ----------
    def status(self, current_price: float) -> dict:
        prices = {sid: current_price for sid in STRATEGIES}
        equity = self.broker.get_equity(prices)
        positions = {}
        for sid, bp in self.broker.positions.items():
            if bp.side == "long":
                pnl = (current_price - bp.entry_price) * bp.units
            else:
                pnl = (bp.entry_price - current_price) * bp.units
            positions[sid] = {
                "side": bp.side, "entry": bp.entry_price,
                "units": bp.units, "trail_stop": bp.trail_stop,
                "bars_held": bp.bars_held, "unrealized_pnl": pnl,
            }
        return {
            "balance": self.broker.balance,
            "equity": equity,
            "drawdown_pct": (equity / self.broker.initial_balance - 1) * 100,
            "positions": positions,
            "n_trades": len(self.broker.trades),
        }
