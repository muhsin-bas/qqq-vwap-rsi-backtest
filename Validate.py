"""
Does the div_only/both winner survive proper scrutiny? Three real tests:

  1) OUT-OF-SAMPLE: run the EXACT winning config on windows it was NOT
     selected on (pre-Trump 2023-24, Trump 1st year 2025).
  2) PERMUTATION MONTE CARLO: keep the positions, circularly shift the return
     series 2000x to break signal<->return alignment -> null distribution of
     total return. p-value = how often random alignment beats the real one.
  3) SELECTION-BIAS MONTE CARLO: the real search kept the BEST of 1000 configs.
     Phase-randomize all 1000 position series, take the best each iteration ->
     null distribution of "best of 1000 by chance". Is the observed best inside it?

Also reports Sharpe, max drawdown, and net-of-cost for the winner.

Winning config: div_only / both / opposite / anchor W / rsi 10 / lb 7 / hidden.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

import BackTest as B
from BackTest import _signals, build_positions, sample_configs, START_CAPITAL

WIN = dict(combo="div_only", direction="both", exit="opposite",
           anchor="W", rsi_len=10, lb=7, use_hidden=True)
BARS_PER_YEAR = 252 * 6.5   # ~1h US-equity bars per year
SEED = 42


def positions(df, cfg):
    B._cache.clear()
    v, bull, bear = _signals(df, cfg["rsi_len"], cfg["lb"], cfg["anchor"], cfg["use_hidden"])
    return pd.Series(build_positions(df["Close"].values, v, bull, bear, cfg), index=df.index)


def metrics(df, pos, cost_bps=0.0):
    ret = df["Close"].pct_change().fillna(0)
    turns = pos.diff().abs().fillna(0)
    sr = pos.shift(1).fillna(0) * ret - turns * (cost_bps / 10000)
    eq = (1 + sr).cumprod()
    total = eq.iloc[-1] - 1
    sharpe = sr.mean() / sr.std() * np.sqrt(BARS_PER_YEAR) if sr.std() > 0 else 0
    maxdd = (eq / eq.cummax() - 1).min()
    return dict(ret_pct=total*100, sharpe=sharpe, maxdd_pct=maxdd*100,
                trades=int((turns > 0).sum()), expo_pct=float((pos != 0).mean())*100)


def total_return(pos_prev, ret):
    return np.prod(1 + pos_prev * ret) - 1


if __name__ == "__main__":
    h = yf.Ticker("QQQ").history(period="730d", interval="1h")[["Open","High","Low","Close","Volume"]].dropna()
    tz = h.index.tz
    windows = {
        "pre-Trump 2023-24 (OOS)":  (h.index[0],                       pd.Timestamp("2025-01-20", tz=tz)),
        "Trump 1st year 2025 (OOS)":(pd.Timestamp("2025-01-20", tz=tz),pd.Timestamp("2026-01-20", tz=tz)),
        "Trump 2nd year (SELECTED)":(pd.Timestamp("2026-01-20", tz=tz),h.index[-1]),
    }

    print("=== 1) OUT-OF-SAMPLE: the exact winner on unseen windows ===")
    print(f"{'window':<30}{'strat%':>9}{'QQQ%':>8}{'beat?':>7}{'Sharpe':>8}{'maxDD%':>8}{'trades':>8}")
    for name,(a,b_) in windows.items():
        w = h[(h.index>=a)&(h.index<b_)]
        if len(w) < 60: continue
        pos = positions(w, WIN)
        m = metrics(w, pos)
        qqq = (w["Close"].iloc[-1]/w["Close"].iloc[0]-1)*100
        beat = "YES" if m["ret_pct"]>qqq else "no"
        print(f"{name:<30}{m['ret_pct']:>+8.1f}{qqq:>+8.1f}{beat:>7}{m['sharpe']:>8.2f}{m['maxdd_pct']:>8.1f}{m['trades']:>8}")

    # --- focus window = the selected one ---
    w = h[h.index>=pd.Timestamp("2026-01-20",tz=tz)]
    ret = w["Close"].pct_change().fillna(0).values
    pos = positions(w, WIN)
    pos_prev = pos.shift(1).fillna(0).values
    actual = total_return(pos_prev, ret)

    print("\n=== Winner metrics (Trump 2nd year), net of cost ===")
    for c in [0,1,5,10]:
        m = metrics(w, pos, c)
        print(f"  cost {c:>2}bp/side: {m['ret_pct']:+6.1f}%  Sharpe {m['sharpe']:.2f}  maxDD {m['maxdd_pct']:.1f}%  trades {m['trades']}")

    # === 2) PERMUTATION MONTE CARLO ===
    rng = np.random.default_rng(SEED)
    n = len(ret); N = 2000
    null = np.empty(N)
    for i in range(N):
        k = rng.integers(50, n-50)
        null[i] = total_return(pos_prev, np.roll(ret, k))
    p_perm = (null >= actual).mean()
    print(f"\n=== 2) PERMUTATION MC ({N} circular shifts) ===")
    print(f"  actual return {actual*100:+.1f}%   null mean {null.mean()*100:+.1f}%  "
          f"null 95th pct {np.percentile(null,95)*100:+.1f}%")
    print(f"  p-value (null >= actual): {p_perm:.3f}")

    # === 3) SELECTION-BIAS MONTE CARLO (best of 1000) ===
    cfgs = sample_configs(np.random.default_rng(SEED), 1000)
    P = np.vstack([positions(w, c).shift(1).fillna(0).values for c in cfgs])   # 1000 x n
    real_best = max(total_return(P[i], ret) for i in range(len(cfgs)))
    M = 300
    idx = np.arange(n)
    best_null = np.empty(M)
    for j in range(M):
        offs = rng.integers(50, n-50, size=P.shape[0])
        rolled = P[np.arange(P.shape[0])[:,None], (idx[None,:]-offs[:,None]) % n]
        tot = np.prod(1 + rolled * ret[None,:], axis=1) - 1
        best_null[j] = tot.max()
    p_sel = (best_null >= real_best).mean()
    print(f"\n=== 3) SELECTION-BIAS MC (best of 1000, {M} iters) ===")
    print(f"  real best-of-1000 {real_best*100:+.1f}%   null best-of-1000 mean {best_null.mean()*100:+.1f}%  "
          f"95th pct {np.percentile(best_null,95)*100:+.1f}%")
    print(f"  p-value (null best >= real best): {p_sel:.3f}")

    # --- plot both null distributions ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].hist(null*100, bins=50, color="#90A4AE", edgecolor="white")
    ax[0].axvline(actual*100, color="#E53935", lw=2.5, label=f"actual {actual*100:+.0f}%")
    ax[0].set_title(f"Permutation MC  (p={p_perm:.2f})", weight="bold")
    ax[0].set_xlabel("total return under random alignment (%)"); ax[0].legend()
    ax[1].hist(best_null*100, bins=40, color="#90A4AE", edgecolor="white")
    ax[1].axvline(real_best*100, color="#E53935", lw=2.5, label=f"real best {real_best*100:+.0f}%")
    ax[1].set_title(f"Best-of-1000 under null  (p={p_sel:.2f})", weight="bold")
    ax[1].set_xlabel("best-of-1000 return by chance (%)"); ax[1].legend()
    for a in ax: a.grid(alpha=0.25)
    fig.suptitle("Does the +36% survive? (left: timing skill, right: selection luck)", fontsize=13, weight="bold")
    fig.tight_layout(); fig.savefig("validation.png", dpi=130); plt.close(fig)
    print("\nSaved validation.png")
