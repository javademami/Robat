"""
Backtest Engine V1.

Simulates entry/exit for PositionSize objects on 5M candles.

Rules:
- SL first when a single candle touches both SL and TP (conservative).
- No look-ahead: only candles strictly after entry timestamp are used.
- Open trades at end of data are marked as "open" and excluded from
  win/loss metrics (but still recorded).
"""