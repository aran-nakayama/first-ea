"""ブローカー抽象 - PaperBroker と OandaBroker。

すべてのブローカーは以下のインターフェースを実装する:
- get_account_balance() -> float
- get_position(strategy_id) -> Position or None
- place_order(strategy_id, side, units, sl) -> order_id
- close_position(strategy_id) -> dict
- update_sl(strategy_id, new_sl) -> bool
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Literal
from datetime import datetime
import json
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class BrokerPosition:
    strategy_id: str
    side: Literal["long", "short"]
    entry_price: float
    entry_time: str
    units: float        # 取引単位 (USDの場合 USD額面)
    sl: float = 0.0
    trail_stop: float = 0.0
    bars_held: int = 0
    order_id: str = ""


@dataclass
class TradeRecord:
    strategy_id: str
    side: str
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    units: float
    pnl: float
    reason: str
    pnl_pct: float = 0.0


# ============================================================
# Paper Broker - メモリ内シミュレーション
# ============================================================
class PaperBroker:
    """In-memoryのペーパートレード実装。実際のお金は動かない。"""

    def __init__(self, initial_balance: float = 1_000_000.0,
                 leverage: float = 1.0,
                 state_path: Optional[str] = None,
                 spread_pips: float = 0.4):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.spread_pips = spread_pips
        self.positions: Dict[str, BrokerPosition] = {}
        self.trades: list = []
        self.state_path = state_path
        if state_path:
            self._load()

    # ---------- スプレッド適用 ----------
    def _apply_spread(self, price: float, side: str) -> float:
        """エントリー時にスプレッド分不利な価格を返す。"""
        adjust = self.spread_pips * 0.01
        if side == "long":
            return price + adjust  # ASK
        return price - adjust       # BID

    # ---------- 口座情報 ----------
    def get_account_balance(self) -> float:
        return self.balance

    def get_equity(self, current_prices: Dict[str, float]) -> float:
        """現在価格から評価額を計算。"""
        eq = self.balance
        for sid, pos in self.positions.items():
            cp = current_prices.get(sid)
            if cp is None:
                continue
            if pos.side == "long":
                pnl = (cp - pos.entry_price) * pos.units
            else:
                pnl = (pos.entry_price - cp) * pos.units
            eq += pnl
        return eq

    def get_position(self, strategy_id: str) -> Optional[BrokerPosition]:
        return self.positions.get(strategy_id)

    # ---------- 注文 ----------
    def place_order(self, strategy_id: str, side: str, units: float,
                    price: float, sl: float = 0.0) -> str:
        if strategy_id in self.positions:
            raise ValueError(f"Already has position for {strategy_id}")
        adj_price = self._apply_spread(price, side)
        now = datetime.now().isoformat()
        order_id = f"P{int(datetime.now().timestamp())}-{strategy_id}"
        pos = BrokerPosition(
            strategy_id=strategy_id, side=side, entry_price=adj_price,
            entry_time=now, units=units, sl=sl, trail_stop=sl, order_id=order_id,
        )
        self.positions[strategy_id] = pos
        logger.info(f"[PAPER] OPEN {strategy_id} {side.upper()} units={units:,.0f} "
                    f"@ {adj_price:.4f} (raw={price:.4f}) SL={sl:.4f}")
        self._save()
        return order_id

    def close_position(self, strategy_id: str, price: float, reason: str = "") -> dict:
        pos = self.positions.pop(strategy_id, None)
        if pos is None:
            return {}
        # 決済時もスプレッド適用 (反対側)
        opposite = "short" if pos.side == "long" else "long"
        exit_price = self._apply_spread(price, opposite)
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.units
        else:
            pnl = (pos.entry_price - exit_price) * pos.units
        self.balance += pnl
        rec = TradeRecord(
            strategy_id=strategy_id, side=pos.side,
            entry_price=pos.entry_price, exit_price=exit_price,
            entry_time=pos.entry_time, exit_time=datetime.now().isoformat(),
            units=pos.units, pnl=pnl, reason=reason,
            pnl_pct=(pnl / (pos.entry_price * pos.units) * 100) if pos.units > 0 else 0,
        )
        self.trades.append(rec)
        logger.info(f"[PAPER] CLOSE {strategy_id} {pos.side.upper()} "
                    f"@ {exit_price:.4f} pnl={pnl:+,.0f} ({rec.pnl_pct:+.2f}%) "
                    f"reason='{reason}' balance={self.balance:,.0f}")
        self._save()
        return asdict(rec)

    def update_sl(self, strategy_id: str, new_sl: float, trail_stop: float = 0.0):
        pos = self.positions.get(strategy_id)
        if pos:
            pos.sl = new_sl
            if trail_stop:
                pos.trail_stop = trail_stop
            self._save()

    def increment_bars(self):
        """全ポジションのbars_heldを+1。バー確定時に呼ぶ。"""
        for pos in self.positions.values():
            pos.bars_held += 1
        self._save()

    def check_stops(self, prices: Dict[str, float]) -> list:
        """SL/Trailing stopに到達したポジションを決済。"""
        closed = []
        for sid in list(self.positions.keys()):
            pos = self.positions[sid]
            cp = prices.get(sid)
            if cp is None:
                continue
            stop = pos.trail_stop or pos.sl
            if stop <= 0:
                continue
            hit = (pos.side == "long" and cp <= stop) or \
                  (pos.side == "short" and cp >= stop)
            if hit:
                rec = self.close_position(sid, stop, reason=f"stop hit @ {stop:.4f}")
                closed.append(rec)
        return closed

    # ---------- 永続化 ----------
    def _save(self):
        if not self.state_path:
            return
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        data = {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "trades": [asdict(t) for t in self.trades],
        }
        with open(self.state_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _load(self):
        if not os.path.exists(self.state_path):
            return
        with open(self.state_path) as f:
            data = json.load(f)
        self.balance = data.get("balance", self.initial_balance)
        self.initial_balance = data.get("initial_balance", self.initial_balance)
        for k, v in data.get("positions", {}).items():
            self.positions[k] = BrokerPosition(**v)
        for t in data.get("trades", []):
            self.trades.append(TradeRecord(**t))


# ============================================================
# OANDA Broker - 実取引 (要 API key)
# ============================================================
class OandaBroker:
    """OANDA v20 API を使った実取引ブローカー。

    使用するには以下の環境変数が必要:
    - OANDA_API_KEY: APIトークン
    - OANDA_ACCOUNT_ID: アカウントID
    - OANDA_ENV: "practice" (デモ) or "live" (実取引)

    install: pip install --user oandapyV20
    """

    def __init__(self, api_key: str, account_id: str,
                 environment: str = "practice",
                 instrument: str = "USD_JPY"):
        try:
            from oandapyV20 import API
            from oandapyV20.endpoints.accounts import AccountSummary
        except ImportError:
            raise ImportError(
                "oandapyV20 が見つかりません。\n"
                "  pip install --user oandapyV20\n"
                "を実行してください。"
            )
        self.client = API(access_token=api_key, environment=environment)
        self.account_id = account_id
        self.instrument = instrument
        self.environment = environment
        self.positions: Dict[str, BrokerPosition] = {}  # 戦略ID → OANDAトレードIDマッピング
        # FIXME: 実装はOANDA API のラッパーをここに追加。
        # PaperBrokerと同じインターフェースを実装すれば runner.py から差し替え可能。
        raise NotImplementedError(
            "OandaBroker は雛形のみ実装されています。\n"
            "実装には以下が必要:\n"
            "  1. oandapyV20 インストール\n"
            "  2. OANDA デモアカウント取得 (https://www.oanda.jp/)\n"
            "  3. このクラスの place_order/close_position 等を完成\n"
            "Paper broker で十分検証してから実装することを推奨します。"
        )
