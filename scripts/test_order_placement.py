"""
Order Placement Test — Synthetic

Tests the full order placement flow on MT5 Demo:
    1. Connects to MT5
    2. Verifies demo account
    3. Places a small BUY order on XAUUSD with SL/TP
    4. Verifies the order was placed (open positions)
    5. Closes the order immediately
    6. Verifies the order was closed

This is a sanity check, NOT a trading signal.

Usage:
    python scripts/test_order_placement.py

Optional:
    --symbol EURUSD     Test a different symbol
    --volume 0.01       Custom volume
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIG
# ============================================================

MT5_PATH = r"C:\Users\EnRival\AppData\Roaming\MetaTrader 5\terminal64.exe"

DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_VOLUME = 0.01        # minimum lot, tiny risk
SL_DISTANCE_PCT = 0.005      # 0.5% away
TP_DISTANCE_PCT = 0.015      # 1.5% away (3R)
MAGIC_NUMBER = 999999        # test magic

# Wait time after placing order (seconds)
WAIT_AFTER_OPEN = 3
WAIT_AFTER_CLOSE = 3


# ============================================================
# HELPERS
# ============================================================

def get_symbol_info(mt5, symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    return info


def place_test_buy(mt5, symbol, volume, sl, tp):
    """Place a market BUY order."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return None

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price": info.ask,
        "sl": round(sl, info.digits),
        "tp": round(tp, info.digits),
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "test_order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    return mt5.order_send(request)


def close_position(mt5, position):
    """Close an open position."""
    info = mt5.symbol_info(position.symbol)
    if info is None:
        return None

    # Opposite direction
    if position.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = info.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = info.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "test_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    return mt5.order_send(request)


# ============================================================
# MAIN
# ============================================================

def main():
    import MetaTrader5 as mt5

    # Parse args
    symbol = DEFAULT_SYMBOL
    volume = DEFAULT_VOLUME

    for i, arg in enumerate(sys.argv):
        if arg == "--symbol" and i + 1 < len(sys.argv):
            symbol = sys.argv[i + 1]
        if arg == "--volume" and i + 1 < len(sys.argv):
            volume = float(sys.argv[i + 1])

    print("=" * 72)
    print("ORDER PLACEMENT TEST — SYNTHETIC")
    print("=" * 72)
    print(f"Time:    {datetime.now(timezone.utc).isoformat()}")
    print(f"Symbol:  {symbol}")
    print(f"Volume:  {volume}")
    print()

    # ---- Init MT5 ----
    if not mt5.initialize(MT5_PATH):
        print(f"ERROR: mt5.initialize failed: {mt5.last_error()}")
        return

    # ---- Check account ----
    acc = mt5.account_info()
    if acc is None:
        print("ERROR: no account info")
        mt5.shutdown()
        return

    is_demo = "demo" in (acc.server or "").lower() or acc.trade_mode == 0

    print("Account:")
    print(f"  Login:    {acc.login}")
    print(f"  Server:   {acc.server}")
    print(f"  Balance:  ${acc.balance:,.2f}")
    print(f"  Demo:     {is_demo}")
    print()

    if not is_demo:
        print("ERROR: NOT a demo account. Aborting for safety.")
        mt5.shutdown()
        return

    # ---- Check symbol ----
    info = get_symbol_info(mt5, symbol)
    if info is None:
        print(f"ERROR: symbol {symbol} not found")
        mt5.shutdown()
        return

    if not info.visible:
        mt5.symbol_select(symbol, True)
        info = get_symbol_info(mt5, symbol)

    print(f"Symbol info:")
    print(f"  Bid:        {info.bid}")
    print(f"  Ask:        {info.ask}")
    print(f"  Spread:     {info.spread}")
    print(f"  Digits:     {info.digits}")
    print(f"  Volume min: {info.volume_min}")
    print(f"  Volume max: {info.volume_max}")
    print(f"  Volume step:{info.volume_step}")
    print(f"  Point:      {info.point}")
    print()

    # ---- Check existing test positions ----
    existing = mt5.positions_get(symbol=symbol)
    if existing:
        print(f"WARNING: {len(existing)} existing positions on {symbol}")
        for pos in existing:
            print(f"  Ticket: {pos.ticket}  "
                  f"Type: {pos.type}  Volume: {pos.volume}")
        print("Aborting to avoid interference.")
        mt5.shutdown()
        return

    # ---- Calculate SL/TP ----
    entry = info.ask
    sl = entry * (1 - SL_DISTANCE_PCT)
    tp = entry * (1 + TP_DISTANCE_PCT)

    print("Test order parameters:")
    print(f"  Direction: BUY")
    print(f"  Entry (ask): {entry}")
    print(f"  SL:          {sl:.{info.digits}f}  (distance {entry - sl:.5f})")
    print(f"  TP:          {tp:.{info.digits}f}  (distance {tp - entry:.5f})")
    print(f"  R:R:         {(tp - entry) / (entry - sl):.2f}")
    print()

    # ---- Place order ----
    print("Placing order ...")
    result = place_test_buy(mt5, symbol, volume, sl, tp)

    if result is None:
        print("ERROR: order_send returned None")
        mt5.shutdown()
        return

    print(f"  Retcode:  {result.retcode}")
    print(f"  Comment:  {result.comment}")

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print()
        print("❌ ORDER FAILED")
        print(f"   retcode={result.retcode}")
        mt5.shutdown()
        return

    print()
    print("✅ ORDER PLACED")
    print(f"  Deal:    {result.deal}")
    print(f"  Order:   {result.order}")
    print(f"  Volume:  {result.volume}")
    print(f"  Price:   {result.price}")
    print()

    # ---- Wait ----
    print(f"Waiting {WAIT_AFTER_OPEN}s ...")
    time.sleep(WAIT_AFTER_OPEN)

    # ---- Check position ----
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        print("❌ No open position found after order.")
        mt5.shutdown()
        return

    pos = positions[0]
    print("✅ POSITION CONFIRMED")
    print(f"  Ticket:  {pos.ticket}")
    print(f"  Symbol:  {pos.symbol}")
    print(f"  Type:    {pos.type}  (0=BUY)")
    print(f"  Volume:  {pos.volume}")
    print(f"  Price:   {pos.price_open}")
    print(f"  SL:      {pos.sl}")
    print(f"  TP:      {pos.tp}")
    print(f"  Profit:  ${pos.profit:.2f}")
    print()

    # ---- Close position ----
    print("Closing position ...")
    close_result = close_position(mt5, pos)

    if close_result is None:
        print("ERROR: close returned None")
        mt5.shutdown()
        return

    print(f"  Retcode:  {close_result.retcode}")
    print(f"  Comment:  {close_result.comment}")

    if close_result.retcode != mt5.TRADE_RETCODE_DONE:
        print()
        print("❌ CLOSE FAILED")
        print(f"   retcode={close_result.retcode}")
        mt5.shutdown()
        return

    print()
    print("✅ POSITION CLOSED")
    print(f"  Deal:    {close_result.deal}")
    print(f"  Volume:  {close_result.volume}")
    print(f"  Price:   {close_result.price}")
    print()

    # ---- Wait ----
    print(f"Waiting {WAIT_AFTER_CLOSE}s ...")
    time.sleep(WAIT_AFTER_CLOSE)

    # ---- Verify closure ----
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        print("❌ Position still open after close.")
        mt5.shutdown()
        return

    print("✅ CLOSURE CONFIRMED")
    print("  No open positions.")

    # ---- Final summary ----
    print()
    print("=" * 72)
    print("TEST SUMMARY")
    print("=" * 72)
    print("  ✅ MT5 connection          OK")
    print("  ✅ Demo account verified   OK")
    print("  ✅ Symbol found            OK")
    print("  ✅ Order placement         OK")
    print("  ✅ Position confirmation   OK")
    print("  ✅ Position closure        OK")
    print("  ✅ Closure verification    OK")
    print()
    print("  🎉 ALL SYSTEMS GO")

    acc = mt5.account_info()
    if acc:
        print()
        print(f"  Final balance: ${acc.balance:,.2f}")
        print(f"  Final equity:  ${acc.equity:,.2f}")

    mt5.shutdown()


if __name__ == "__main__":
    from pathlib import Path
    main()