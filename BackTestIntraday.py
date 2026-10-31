"""
INTRADAY Monte-Carlo backtest of the AMT VWAP + RSI Divergence strategy.

Reality: yfinance gives NO 10-year intraday history. Deepest available is
1-hour bars (~2 years). 5/15/30-min only reach ~60 days. So:

  1) 1h bars, full ~2y available   -> intraday_1h_distribution.png / _top.png
  2) Trump 2nd year (2026-01-20 ->) -> trump2_1h_distribution.png / _top.png
  3) 30m bars, last ~60 days (peek) -> intraday_30m_distribution.png / _top.png

Intraday finally lets VWAP anchor to the SESSION (daily reset) -- the way the
AMT VWAP is designed. Reuses the exact backtest engine from BackTest.py.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

from BackTest import (run_config, monte_carlo, buy_hold, START_CAPITAL,
                      plot_distribution, plot_top, show_top,
                      COMBOS, DIRECTIONS, EXITS, RSI_LENS, LOOKBACKS, HIDDEN)

SEED, N_SAMPLES = 42, 1000
TRUMP_Y2_START = "2026-01-20"
ANCHORS = ["D", "W"]   # intraday: D = session (daily reset), W = weekly


def sample_configs(rng, n):
    seen, cfgs = set(), []
    while len(cfgs) < n:
        cfg = {"combo": COMBOS[rng.integers(len(COMBOS))],
               "direction": DIRECTIONS[rng.integers(len(DIRECTIONS))],
               "exit": EXITS[rng.integers(len(EXITS))],
               "anchor": ANCHORS[rng.integers(len(ANCHORS))],
               "rsi_len": int(RSI_LENS[rng.integers(len(RSI_LENS))]),
               "lb": int(LOOKBACKS[rng.integers(len(LOOKBACKS))]),
               "use_hidden": bool(HIDDEN[rng.integers(len(HIDDEN))])}
        k = tuple(sorted(cfg.items()))
        if k not in seen:
            seen.add(k); cfgs.append(cfg)
    return cfgs


def load(interval, period):
    df = yf.Ticker("QQQ").history(period=period, interval=interval)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def run_window(df, cfgs, tag, label):
    bh, bh_ret = buy_hold(df)
    res, eq = monte_carlo(df, cfgs)
    beat = plot_distribution(res, bh_ret, f"{label}: {len(res)} strategies vs Buy & Hold",
                             f"{tag}_distribution.png")
    plot_top(res, eq, bh, bh_ret, f"{label}: top 6 configs vs Buy & Hold", f"{tag}_top.png")
    print(f"\n=== {label}  {df.index[0].date()} -> {df.index[-1].date()} "
          f"({len(df)} bars) ===   Buy & Hold {bh_ret:+.0f}%   beat: {beat}/{len(res)}")
    show_top(res)
    return res


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    cfgs = sample_configs(rng, N_SAMPLES)

    # 1) deepest intraday: 1h, full available (~2y)
    h1 = load("1h", "730d")
    run_window(h1, cfgs, "intraday_1h", "Intraday 1h (~2y)")

    # 2) Trump 2nd year, sliced from the 1h data
    t2 = h1[h1.index >= pd.Timestamp(TRUMP_Y2_START, tz=h1.index.tz)]
    if len(t2) > 60:
        run_window(t2, cfgs, "trump2_1h", "Trump 2nd year (1h)")
    else:
        print(f"\nTrump 2nd year: only {len(t2)} bars available, skipping.")

    # 3) 30m peek, last 60 days (TOO SHORT to trust -- shown for granularity only)
    m30 = load("30m", "60d")
    run_window(m30, cfgs, "intraday_30m", "30m last 60d (too short to trust)")

    print("\nSaved intraday_1h_*, trump2_1h_*, intraday_30m_* PNGs")
