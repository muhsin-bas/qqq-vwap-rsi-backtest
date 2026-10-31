"""
Higher-frequency intraday backtest: Naive Bayes trigger + Ehlers Adaptive
SuperSmoother filter, on QQQ 5-minute bars (~60 days, the max 5m history free).

Signal (per bar):
  NB      : Gaussian Naive Bayes on [relative volume, momentum] -> P(up)
  Filter  : Adaptive SuperSmoother oscillator  osc = ssAdaptive - ssBase
  LONG  when P(up) > 0.5  AND osc > 0
  SHORT when P(up) < 0.5  AND osc < 0
  else FLAT   (NB and filter disagree)

Position earns the NEXT bar's return (shift(1)) -> no lookahead.

The honest centerpiece is the COST SWEEP: a 5m signal that flips often will
churn, and churn dies on spread+commission. We show net return vs cost.

Only the Naive Bayes half of the "Monte Carlo CT" indicator is used -- the
Monte Carlo polylines are a one-shot visual forecast, not a tradeable signal.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

# ---- params (from the two Pine scripts) ----
TICKER      = "QQQ"
INTERVAL    = "5m"
PERIOD      = "60d"
START_CAP   = 100_000
VOL_SMA     = 50     # relative-volume baseline
ROC_LEN     = 10     # momentum
NB_TRAIN    = 500    # Naive Bayes training lookback
SS_PERIOD   = 20     # SuperSmoother base period
RMS_LEN     = 81     # adaptive RMS length


# ---------------- Naive Bayes ----------------
def gauss_pdf(x, mean, var):
    v = np.maximum(var, 1e-6)
    return (1.0 / np.sqrt(2 * np.pi * v)) * np.exp(-((x - mean) ** 2) / (2 * v))


def nb_probability(df):
    rel_vol  = df["Volume"] / df["Volume"].rolling(VOL_SMA).mean()
    momentum = 100 * (df["Close"] - df["Close"].shift(ROC_LEN)) / df["Close"].shift(ROC_LEN)
    up = (df["Close"] > df["Close"].shift(1)).astype(float)   # target: this bar up?
    w  = NB_TRAIN

    def cond_stats(feat, mask):
        cnt  = mask.rolling(w).sum()
        s1   = (feat * mask).rolling(w).sum()
        s2   = (feat * feat * mask).rolling(w).sum()
        mean = s1 / cnt
        var  = s2 / cnt - mean ** 2
        return mean.fillna(0), np.maximum(var.fillna(0), 0)   # nz() like Pine

    dn = 1 - up
    m1_v, v1_v = cond_stats(rel_vol, up);   m1_m, v1_m = cond_stats(momentum, up)
    m0_v, v0_v = cond_stats(rel_vol, dn);   m0_m, v0_m = cond_stats(momentum, dn)
    p1 = up.rolling(w).mean().fillna(0.5)

    l1 = gauss_pdf(rel_vol, m1_v, v1_v) * gauss_pdf(momentum, m1_m, v1_m) * p1
    l0 = gauss_pdf(rel_vol, m0_v, v0_v) * gauss_pdf(momentum, m0_m, v0_m) * (1 - p1)
    prob = (l1 / (l1 + l0 + 1e-6)).fillna(0.5)
    return prob


# ---------------- Adaptive SuperSmoother (Ehlers) ----------------
def super_smoother(src, period):
    s = src.values
    n = len(s)
    res = s.copy().astype(float)
    if np.isscalar(period):
        period = np.full(n, period, float)
    for i in range(n):
        if i >= 4:
            p = max(period[i], 2)
            wv = 1.414 * np.pi / p
            q  = np.exp(-wv)
            c1 = 2.0 * q * np.cos(wv)
            c2 = q * q
            a0 = (1.0 - c1 + c2) / 2.0
            res[i] = a0 * (s[i] + s[i-1]) + c1 * res[i-1] - c2 * res[i-2]
    return pd.Series(res, index=src.index)


def adaptive_super_smoother(src, period, rms_len):
    ss = super_smoother(src, period)
    roc1 = ss.diff().fillna(0)
    rms = np.sqrt((roc1 * roc1).rolling(rms_len).mean())
    roc = np.where(rms > 0, np.minimum(np.abs(roc1 / rms.replace(0, np.nan)), 2.0), 0.0)
    roc = np.nan_to_num(roc)
    factor = (1.0 - 0.5 * roc) ** 2
    a_period = np.maximum((period * factor).astype(int), 2)
    ss_adapt = super_smoother(src, a_period)
    return ss_adapt, ss


# ---------------- strategy ----------------
def build_positions(prob, osc):
    p, o = prob.values, osc.values
    long_t  = (p > 0.5) & (o > 0)
    short_t = (p < 0.5) & (o < 0)
    pos = np.where(long_t, 1.0, np.where(short_t, -1.0, 0.0))
    return pd.Series(pos, index=prob.index)


def backtest(df, pos, cost_bps):
    ret = df["Close"].pct_change().fillna(0)
    turns = pos.diff().abs().fillna(0)
    strat = pos.shift(1).fillna(0) * ret - turns * (cost_bps / 10000)
    eq = START_CAP * (1 + strat).cumprod()
    final = eq.iloc[-1]
    return eq, {
        "ret_pct": final / START_CAP * 100 - 100,
        "max_dd_pct": (eq / eq.cummax() - 1).min() * 100,
        "exposure_pct": float((pos != 0).mean()) * 100,
        "trades": int((turns > 0).sum()),
    }


def selftest():
    # SuperSmoother of a constant series stays constant
    c = pd.Series(np.full(50, 100.0))
    assert abs(super_smoother(c, 20).iloc[-1] - 100.0) < 1e-6
    # gaussian pdf peaks at the mean
    assert gauss_pdf(0.0, 0.0, 1.0) > gauss_pdf(1.0, 0.0, 1.0)
    print("selftest ok")


if __name__ == "__main__":
    selftest()
    print(f"Downloading {TICKER} {INTERVAL} ({PERIOD})...")
    df = yf.Ticker(TICKER).history(period=PERIOD, interval=INTERVAL)[["Open", "High", "Low", "Close", "Volume"]].dropna()
    print(f"  {len(df)} bars  {df.index[0]} -> {df.index[-1]}")

    prob = nb_probability(df)
    ss_adapt, ss_base = adaptive_super_smoother(df["Close"], SS_PERIOD, RMS_LEN)
    osc = ss_adapt - ss_base
    pos = build_positions(prob, osc)

    bh = START_CAP * (df["Close"] / df["Close"].iloc[0])
    bh_ret = bh.iloc[-1] / START_CAP * 100 - 100

    # cost sweep -- the decisive test for a 5m strategy
    costs = [0, 0.5, 1, 2, 5]
    print(f"\nBuy & Hold ({INTERVAL}, {PERIOD}): {bh_ret:+.1f}%")
    _, m0 = backtest(df, pos, 0)
    print(f"Strategy: {m0['trades']} trades, {m0['exposure_pct']:.0f}% exposure, "
          f"max DD {m0['max_dd_pct']:.1f}%")
    print(f"\n{'cost/side':>10} {'net return':>12}")
    nets = []
    for c in costs:
        _, m = backtest(df, pos, c)
        nets.append(m["ret_pct"])
        print(f"{c:>8}bp {m['ret_pct']:>+11.1f}%")

    # ---- equity curve (zero-cost vs realistic 1bp) vs buy & hold ----
    eq0, _ = backtest(df, pos, 0)
    eq1, _ = backtest(df, pos, 1)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(bh.index, bh.values, color="#1565C0", lw=2, label=f"Buy & Hold ({bh_ret:+.1f}%)")
    ax.plot(eq0.index, eq0.values, color="#2E7D32", lw=1.6, label=f"Strategy 0bp ({eq0.iloc[-1]/START_CAP*100-100:+.1f}%)")
    ax.plot(eq1.index, eq1.values, color="#E53935", lw=1.6, label=f"Strategy 1bp/side ({eq1.iloc[-1]/START_CAP*100-100:+.1f}%)")
    ax.axhline(START_CAP, color="#bbb", lw=1)
    ax.set_title(f"NB + Adaptive SuperSmoother — QQQ {INTERVAL}, {PERIOD}", fontsize=14, weight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig("hft_equity.png", dpi=130); plt.close(fig)

    # ---- cost sweep ----
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(costs, nets, marker="o", lw=2, color="#2E7D32", label="strategy net return")
    ax.axhline(bh_ret, color="#1565C0", lw=2, ls="--", label=f"Buy & Hold ({bh_ret:+.1f}%)")
    ax.axhline(0, color="#999", lw=1)
    ax.set_title(f"5m strategy: net return vs trading cost ({m0['trades']} trades)", fontsize=14, weight="bold")
    ax.set_xlabel("cost per side (basis points)"); ax.set_ylabel("net return (%)")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig("hft_costs.png", dpi=130); plt.close(fig)

    print("\nSaved hft_equity.png, hft_costs.png")
