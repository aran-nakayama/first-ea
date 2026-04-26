"""各戦略のシグナル抽出 - バックテストロジックを純粋関数化したもの。

backtesting.py の Strategy.next() 相当のロジックを、stateful な position 情報
を引数で受け取る関数として再実装。これにより、ライブ環境でも同じ判定が可能。
"""

from dataclasses import dataclass
from typing import Optional, Literal
import math
import numpy as np
import pandas as pd

# 親プロジェクトのprepare_dataを再利用
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.trend_follow import prepare_data as prep_tf
from strategy.supertrend import prepare_data as prep_st
from strategy.macd_momentum import prepare_data as prep_macd
from strategy.ichimoku import prepare_data as prep_ichi


Side = Literal["long", "short", "flat"]


@dataclass
class Position:
    """戦略単位のポジション状態。"""
    side: Side = "flat"
    entry_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    sl: float = 0.0
    trail_stop: float = 0.0
    units: float = 0.0   # 取引単位 (USD)
    bars_held: int = 0


@dataclass
class Signal:
    """次バーで取るべきアクション。"""
    action: Literal["enter_long", "enter_short", "exit", "update_sl", "hold"]
    sl: float = 0.0           # 新しいSL価格 (enter_long/short or update_sl 時)
    trail_stop: float = 0.0   # トレーリングストップ価格
    reason: str = ""          # ログ用


def _is_nan(*vals) -> bool:
    for v in vals:
        if v is None:
            return True
        if isinstance(v, float) and math.isnan(v):
            return True
    return False


# ============================================================
# 1. Trend Follow (Donchian Channel)
# ============================================================
TF_PARAMS = dict(entry_period=15, exit_period=15, atr_sl_mult=3.0,
                 atr_trail_mult=3.0, use_ma_filter=0, max_hold_hours=72)


def prepare_tf(df: pd.DataFrame) -> pd.DataFrame:
    return prep_tf(df.copy(), entry_period=TF_PARAMS["entry_period"],
                   exit_period=TF_PARAMS["exit_period"])


def signal_tf(df: pd.DataFrame, pos: Position) -> Signal:
    """Trend Follow Donchian シグナル判定。最後のバー(直近確定足)を見る。"""
    if len(df) < 2:
        return Signal("hold")
    row = df.iloc[-1]
    price = row["Close"]
    high = row["High"]
    low = row["Low"]
    atr = row.get("ATR")
    dc_up = row.get("DC_Upper")
    dc_low = row.get("DC_Lower")
    dc_exit_up = row.get("DC_Exit_Upper")
    dc_exit_low = row.get("DC_Exit_Lower")

    if _is_nan(atr, dc_up, dc_low, dc_exit_up, dc_exit_low):
        return Signal("hold", reason="indicators not ready")

    # ポジション保有中
    if pos.side != "flat":
        # 最大保有時間チェック
        if pos.bars_held >= TF_PARAMS["max_hold_hours"]:
            return Signal("exit", reason=f"max_hold {TF_PARAMS['max_hold_hours']}h")

        # トレーリングストップ更新
        if pos.side == "long":
            new_trail = price - atr * TF_PARAMS["atr_trail_mult"]
            if new_trail > pos.trail_stop:
                pos.trail_stop = new_trail
            # Donchian下限決済
            if low <= dc_exit_low:
                return Signal("exit", reason="DC exit lower")
            # トレーリングヒット
            if low <= pos.trail_stop:
                return Signal("exit", reason="trail stop hit")
            return Signal("update_sl", trail_stop=new_trail)
        else:  # short
            new_trail = price + atr * TF_PARAMS["atr_trail_mult"]
            if pos.trail_stop == 0 or new_trail < pos.trail_stop:
                pos.trail_stop = new_trail
            if high >= dc_exit_up:
                return Signal("exit", reason="DC exit upper")
            if high >= pos.trail_stop:
                return Signal("exit", reason="trail stop hit")
            return Signal("update_sl", trail_stop=new_trail)

    # 新規エントリー判定
    if high > dc_up:
        sl = price - atr * TF_PARAMS["atr_sl_mult"]
        return Signal("enter_long", sl=sl, trail_stop=sl,
                      reason=f"break DC_Upper {dc_up:.3f}")
    if low < dc_low:
        sl = price + atr * TF_PARAMS["atr_sl_mult"]
        return Signal("enter_short", sl=sl, trail_stop=sl,
                      reason=f"break DC_Lower {dc_low:.3f}")

    return Signal("hold")


# ============================================================
# 2. Supertrend
# ============================================================
ST_PARAMS = dict(st_period=10, st_multiplier=2.0,
                 atr_sl_mult=2.0, atr_trail_mult=3.0, max_hold_hours=72)


def prepare_st(df: pd.DataFrame) -> pd.DataFrame:
    return prep_st(df.copy(), period=ST_PARAMS["st_period"],
                   multiplier=ST_PARAMS["st_multiplier"])


def signal_st(df: pd.DataFrame, pos: Position) -> Signal:
    if len(df) < 2:
        return Signal("hold")
    row = df.iloc[-1]
    price = row["Close"]
    atr = row.get("ATR")
    st_dir = row.get("ST_DIR")
    st_dir_prev = row.get("ST_DIR_PREV")

    if _is_nan(atr, st_dir, st_dir_prev):
        return Signal("hold")

    # ポジション保有中
    if pos.side != "flat":
        if pos.bars_held >= ST_PARAMS["max_hold_hours"]:
            return Signal("exit", reason="max_hold")
        # Supertrend方向転換で決済
        if pos.side == "long" and st_dir == -1:
            return Signal("exit", reason="ST flip down")
        if pos.side == "short" and st_dir == 1:
            return Signal("exit", reason="ST flip up")

        # トレーリング
        if pos.side == "long":
            new_trail = price - atr * ST_PARAMS["atr_trail_mult"]
            if new_trail > pos.trail_stop:
                pos.trail_stop = new_trail
            return Signal("update_sl", trail_stop=new_trail)
        else:
            new_trail = price + atr * ST_PARAMS["atr_trail_mult"]
            if pos.trail_stop == 0 or new_trail < pos.trail_stop:
                pos.trail_stop = new_trail
            return Signal("update_sl", trail_stop=new_trail)

    # 新規エントリー: トレンド方向転換
    sl_dist = atr * ST_PARAMS["atr_sl_mult"]
    if st_dir_prev == -1 and st_dir == 1:
        return Signal("enter_long", sl=price - sl_dist, trail_stop=price - sl_dist,
                      reason="ST flipped to UP")
    if st_dir_prev == 1 and st_dir == -1:
        return Signal("enter_short", sl=price + sl_dist, trail_stop=price + sl_dist,
                      reason="ST flipped to DOWN")
    return Signal("hold")


# ============================================================
# 3. MACD Momentum
# ============================================================
MACD_PARAMS = dict(fast_period=12, slow_period=26, signal_period=12,
                   atr_sl_mult=2.5, atr_trail_mult=2.5, use_ma_filter=0,
                   max_hold_hours=120, min_hist_threshold=0.0)


def prepare_macd(df: pd.DataFrame) -> pd.DataFrame:
    return prep_macd(df.copy(), fast_period=MACD_PARAMS["fast_period"],
                     slow_period=MACD_PARAMS["slow_period"],
                     signal_period=MACD_PARAMS["signal_period"])


def signal_macd(df: pd.DataFrame, pos: Position) -> Signal:
    if len(df) < 2:
        return Signal("hold")
    row = df.iloc[-1]
    prev_row = df.iloc[-2]
    price = row["Close"]
    atr = row.get("ATR")
    hist = row.get("MACD_HIST")
    hist_prev = prev_row.get("MACD_HIST")
    ma = row.get("MA_TREND")

    if _is_nan(atr, hist, hist_prev, ma):
        return Signal("hold")

    if pos.side != "flat":
        if pos.bars_held >= MACD_PARAMS["max_hold_hours"]:
            return Signal("exit", reason="max_hold")
        # ヒスト反対側クロスで決済
        if pos.side == "long" and hist_prev > 0 and hist <= 0:
            return Signal("exit", reason="MACD hist cross down")
        if pos.side == "short" and hist_prev < 0 and hist >= 0:
            return Signal("exit", reason="MACD hist cross up")
        # トレーリング
        if pos.side == "long":
            new_trail = price - atr * MACD_PARAMS["atr_trail_mult"]
            if new_trail > pos.trail_stop:
                pos.trail_stop = new_trail
            return Signal("update_sl", trail_stop=new_trail)
        else:
            new_trail = price + atr * MACD_PARAMS["atr_trail_mult"]
            if pos.trail_stop == 0 or new_trail < pos.trail_stop:
                pos.trail_stop = new_trail
            return Signal("update_sl", trail_stop=new_trail)

    sl_dist = atr * MACD_PARAMS["atr_sl_mult"]
    cross_up = hist_prev <= 0 and hist > MACD_PARAMS["min_hist_threshold"]
    cross_down = hist_prev >= 0 and hist < -MACD_PARAMS["min_hist_threshold"]

    long_ok = price > ma if MACD_PARAMS["use_ma_filter"] else True
    short_ok = price < ma if MACD_PARAMS["use_ma_filter"] else True

    if cross_up and long_ok:
        return Signal("enter_long", sl=price - sl_dist, trail_stop=price - sl_dist,
                      reason=f"MACD hist cross up (hist={hist:.4f})")
    if cross_down and short_ok:
        return Signal("enter_short", sl=price + sl_dist, trail_stop=price + sl_dist,
                      reason=f"MACD hist cross down (hist={hist:.4f})")
    return Signal("hold")


# ============================================================
# 4. Ichimoku Cloud
# ============================================================
ICHI_PARAMS = dict(tenkan_period=9, kijun_period=26, senkou_b_period=52,
                   atr_sl_mult=2.0, atr_trail_mult=3.0,
                   use_tenkan_kijun=1, max_hold_hours=72)


def prepare_ichi(df: pd.DataFrame) -> pd.DataFrame:
    return prep_ichi(df.copy(), tenkan_period=ICHI_PARAMS["tenkan_period"],
                     kijun_period=ICHI_PARAMS["kijun_period"],
                     senkou_b_period=ICHI_PARAMS["senkou_b_period"])


def signal_ichi(df: pd.DataFrame, pos: Position) -> Signal:
    if len(df) < 2:
        return Signal("hold")
    row = df.iloc[-1]
    price = row["Close"]
    prev_close = row.get("CLOSE_PREV")
    atr = row.get("ATR")
    tenkan = row.get("TENKAN")
    kijun = row.get("KIJUN")
    cloud_top = row.get("CLOUD_TOP")
    cloud_bot = row.get("CLOUD_BOT")
    cloud_top_prev = row.get("CLOUD_TOP_PREV")
    cloud_bot_prev = row.get("CLOUD_BOT_PREV")

    if _is_nan(atr, tenkan, kijun, cloud_top, cloud_bot,
               cloud_top_prev, cloud_bot_prev, prev_close):
        return Signal("hold")

    if pos.side != "flat":
        if pos.bars_held >= ICHI_PARAMS["max_hold_hours"]:
            return Signal("exit", reason="max_hold")
        if pos.side == "long" and price < cloud_bot:
            return Signal("exit", reason="back inside cloud (long)")
        if pos.side == "short" and price > cloud_top:
            return Signal("exit", reason="back inside cloud (short)")

        if pos.side == "long":
            new_trail = price - atr * ICHI_PARAMS["atr_trail_mult"]
            if new_trail > pos.trail_stop:
                pos.trail_stop = new_trail
            return Signal("update_sl", trail_stop=new_trail)
        else:
            new_trail = price + atr * ICHI_PARAMS["atr_trail_mult"]
            if pos.trail_stop == 0 or new_trail < pos.trail_stop:
                pos.trail_stop = new_trail
            return Signal("update_sl", trail_stop=new_trail)

    sl_dist = atr * ICHI_PARAMS["atr_sl_mult"]
    long_break = prev_close <= cloud_top_prev and price > cloud_top
    short_break = prev_close >= cloud_bot_prev and price < cloud_bot

    tk_long = tenkan > kijun if ICHI_PARAMS["use_tenkan_kijun"] else True
    tk_short = tenkan < kijun if ICHI_PARAMS["use_tenkan_kijun"] else True

    if long_break and tk_long:
        return Signal("enter_long", sl=price - sl_dist, trail_stop=price - sl_dist,
                      reason="cloud breakout up")
    if short_break and tk_short:
        return Signal("enter_short", sl=price + sl_dist, trail_stop=price + sl_dist,
                      reason="cloud breakout down")
    return Signal("hold")


# ============================================================
# 戦略レジストリ
# ============================================================
STRATEGIES = {
    "TF": {
        "name": "Trend Follow",
        "prepare": prepare_tf,
        "signal": signal_tf,
        "color": "#4263eb",
    },
    "ST": {
        "name": "Supertrend",
        "prepare": prepare_st,
        "signal": signal_st,
        "color": "#e03131",
    },
    "MACD": {
        "name": "MACD Momentum",
        "prepare": prepare_macd,
        "signal": signal_macd,
        "color": "#f59f00",
    },
    "ICHI": {
        "name": "Ichimoku",
        "prepare": prepare_ichi,
        "signal": signal_ichi,
        "color": "#9333ea",
    },
}
