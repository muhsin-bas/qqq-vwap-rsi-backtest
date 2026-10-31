"""
This is a Trading Strategy I back tested in the ETF: QQQ, over the last ten years.
I programmed 1000 of my AMT anchored-VWAP + RSI Divergence strategy with randomized settings.
Then put all of them in a Monte Carlo Simulation

QQQ strategy research: Monte-Carlo search + honest out-of-sample test + a
Trump-second-term (noisy regime) Monte Carlo.

"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

# ---------------- params ----------------
TICKER, START_CAPITAL, YEARS = "QQQ", 100_000, "10y"
N_SAMPLES, SEED = 1000, 42
COMMISSION_BPS = 0.0
SPLIT_DATE  = "2023-01-01"   # train before, test after
TRUMP_START = "2025-01-20"

COMBOS     = ["div_vwap_filter", "div_only", "vwap_only", "vwap_div_confirm"]
DIRECTIONS = ["both", "long_only", "short_only"]
EXITS      = ["vwap_cross", "opposite"]
ANCHORS    = ["W", "M", "Q"]
RSI_LENS   = list(range(7, 22))
LOOKBACKS  = list(range(3, 9))
HIDDEN     = [False, True]
RANGE_LOW, RANGE_HIGH = 5, 60


# ---------------- indicators ----------------
def wilder_rsi(close, length):
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rma_up   = up.ewm(alpha=1/length, adjust=False).mean()
    rma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = rma_up / rma_down.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(100)


def pivots(vals, lb, kind):
    n, out = len(vals), []
    for p in range(lb, n - lb):
        w = vals[p]
        if kind == "low"  and all(w < vals[p-k] for k in range(1, lb+1)) and all(w < vals[p+k] for k in range(1, lb+1)):
            out.append(p)
        if kind == "high" and all(w > vals[p-k] for k in range(1, lb+1)) and all(w > vals[p+k] for k in range(1, lb+1)):
            out.append(p)
    return out


def anchored_vwap(df, anchor):
    grp  = df.index.tz_localize(None).to_period(anchor)
    hlc3 = (df["High"] + df["Low"] + df["Close"]) / 3
    pv   = hlc3 * df["Volume"]
    return pv.groupby(grp).cumsum() / df["Volume"].groupby(grp).cumsum()


def divergences(df, osc, lb, use_hidden):
    n = len(df)
    bull = np.zeros(n, bool); bear = np.zeros(n, bool)
    o, low, high = osc.values, df["Low"].values, df["High"].values
    pl = pivots(o, lb, "low")
    for prev, cur in zip(pl, pl[1:]):
        if RANGE_LOW <= cur - prev <= RANGE_HIGH:
            if low[cur] < low[prev] and o[cur] > o[prev]:
                bull[cur + lb] = True
            elif use_hidden and low[cur] > low[prev] and o[cur] < o[prev]:
                bull[cur + lb] = True
    ph = pivots(o, lb, "high")
    for prev, cur in zip(ph, ph[1:]):
        if RANGE_LOW <= cur - prev <= RANGE_HIGH:
            if high[cur] > high[prev] and o[cur] < o[prev]:
                bear[cur + lb] = True
            elif use_hidden and high[cur] < high[prev] and o[cur] > o[prev]:
                bear[cur + lb] = True
    return bull, bear


_cache = {}
def _signals(df, rsi_len, lb, anchor, use_hidden):
    key = (rsi_len, lb, anchor, use_hidden)
    if key not in _cache:
        osc  = wilder_rsi(df["Close"], rsi_len)
        vwap = anchored_vwap(df, anchor)
        bull, bear = divergences(df, osc, lb, use_hidden)
        _cache[key] = (vwap.values, bull, bear)
    return _cache[key]


# ---------------- backtest ----------------
def build_positions(close, v, bull, bear, cfg):
    n = len(close); pos = 0; out = np.zeros(n)
    for i in range(n):
        above, below = close[i] > v[i], close[i] < v[i]
        bn, sn = bull[i], bear[i]
        if cfg["combo"] == "div_vwap_filter":
            lt, st = bn and above, sn and below
        elif cfg["combo"] == "div_only":
            lt, st = bn, sn
        elif cfg["combo"] == "vwap_only":
            lt, st = above, below
        else:  # vwap_div_confirm
            lt, st = bn and above, sn and below
        if cfg["direction"] == "long_only":  st = False
        if cfg["direction"] == "short_only": lt = False
        if lt:   pos = 1
        elif st: pos = -1
        elif cfg["exit"] == "vwap_cross":
            if pos == 1 and below:   pos = 0
            elif pos == -1 and above: pos = 0
        else:
            if pos == 1 and sn:   pos = 0
            elif pos == -1 and bn: pos = 0
        out[i] = pos
    return out


def run_config(df, cfg):
    v, bull, bear = _signals(df, cfg["rsi_len"], cfg["lb"], cfg["anchor"], cfg["use_hidden"])
    pos_s = pd.Series(build_positions(df["Close"].values, v, bull, bear, cfg), index=df.index)
    ret = df["Close"].pct_change().fillna(0)
    strat_ret = pos_s.shift(1).fillna(0) * ret
    strat_ret -= pos_s.diff().abs().fillna(0) * (COMMISSION_BPS / 10000)
    equity = START_CAPITAL * (1 + strat_ret).cumprod()
    final = equity.iloc[-1]
    return equity, {
        "final": final, "ret_pct": final / START_CAPITAL * 100 - 100,
        "cagr_pct": ((final / START_CAPITAL) ** (252 / len(df)) - 1) * 100,
        "max_dd_pct": (equity / equity.cummax() - 1).min() * 100,
        "exposure_pct": float((pos_s != 0).mean()) * 100,
        "trades": int((pos_s.diff().abs() > 0).sum()),
    }


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


def monte_carlo(df, cfgs):
    _cache.clear()  # signals are df-specific
    rows, equities = [], {}
    for idx, cfg in enumerate(cfgs):
        eq, m = run_config(df, cfg)
        m.update(cfg); m["id"] = idx
        rows.append(m); equities[idx] = eq
    res = pd.DataFrame(rows)
    return res, equities


def buy_hold(df):
    bh = START_CAPITAL * (df["Close"] / df["Close"].iloc[0])
    return bh, bh.iloc[-1] / START_CAPITAL * 100 - 100


# ---------------- plots ----------------
def plot_distribution(res, bh_ret, title, fname):
    beat = int((res["ret_pct"] > bh_ret).sum())
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.hist(res["ret_pct"], bins=40, color="#26A69A", edgecolor="white")
    ax.axvline(bh_ret, color="#1565C0", lw=2.5, label=f"Buy & Hold {bh_ret:+.0f}%")
    ax.axvline(0, color="#999", lw=1, ls="--", label="Break-even")
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel("total return (%)"); ax.set_ylabel("number of strategies")
    ax.legend(); ax.grid(alpha=0.25)
    ax.text(0.98, 0.97, f"Beat B&H: {beat}/{len(res)}\nBest: {res['ret_pct'].max():+.0f}%\n"
            f"Median: {res['ret_pct'].median():+.0f}%\nWorst: {res['ret_pct'].min():+.0f}%",
            transform=ax.transAxes, va="top", ha="right", family="monospace", fontsize=12,
            bbox=dict(boxstyle="round", fc="white", ec="#26A69A", alpha=0.9))
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)
    return beat


def plot_top(res, equities, bh, bh_ret, title, fname, n=6):
    top = res.sort_values("ret_pct", ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(bh.index, bh.values, color="#1565C0", lw=2.5, label=f"Buy & Hold ({bh_ret:+.0f}%)")
    for _, r in top.iterrows():
        eq = equities[int(r["id"])]
        ax.plot(eq.index, eq.values, lw=1.5, alpha=0.9,
                label=f"{r['combo']}/{r['direction']} ({r['ret_pct']:+.0f}%)")
    ax.axhline(START_CAPITAL, color="#bbb", lw=1)
    ax.set_title(title, fontsize=15, weight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def show_top(res, n=8):
    cols = ["ret_pct", "cagr_pct", "max_dd_pct", "exposure_pct", "trades",
            "combo", "direction", "exit", "anchor", "rsi_len", "lb", "use_hidden"]
    top = res.sort_values("ret_pct", ascending=False).head(n)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(top[cols].to_string(index=False,
              formatters={"ret_pct": "{:+.1f}".format, "cagr_pct": "{:+.1f}".format,
                          "max_dd_pct": "{:.1f}".format, "exposure_pct": "{:.0f}".format}))


# ---------------- out-of-sample ----------------
def segment_returns(eq, i_split):
    train = eq.iloc[i_split-1] / eq.iloc[0] - 1
    test  = eq.iloc[-1] / eq.iloc[i_split-1] - 1
    return train * 100, test * 100


def out_of_sample(df, res, equities, bh):
    i_split = df.index.get_indexer([pd.Timestamp(SPLIT_DATE, tz=df.index.tz)], method="bfill")[0]
    tr, te = [], []
    for _, r in res.iterrows():
        a, b = segment_returns(equities[int(r["id"])], i_split)
        tr.append(a); te.append(b)
    res = res.copy(); res["train_ret"], res["test_ret"] = tr, te
    bh_tr, bh_te = segment_returns(bh, i_split)

    # winners chosen ONLY on train, then graded on test
    winners = res.sort_values("train_ret", ascending=False).head(10)
    split_lbl = df.index[i_split].date()
    print(f"\n=== OUT-OF-SAMPLE (train -> {split_lbl}, test {split_lbl} -> end) ===")
    print(f"Buy & Hold: train {bh_tr:+.0f}%   test {bh_te:+.0f}%")
    print(f"Top-10-by-TRAIN: avg test {winners['test_ret'].mean():+.0f}%  "
          f"(beat B&H test: {(winners['test_ret']>bh_te).sum()}/10)")

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(res["train_ret"], res["test_ret"], s=18, alpha=0.4, color="#26A69A", label="all configs")
    ax.scatter(winners["train_ret"], winners["test_ret"], s=60, color="#E53935",
               edgecolor="black", zorder=5, label="top-10 by train")
    ax.axhline(bh_te, color="#1565C0", lw=1.5, ls="--", label=f"B&H test {bh_te:+.0f}%")
    ax.axvline(bh_tr, color="#1565C0", lw=1.5, ls=":", label=f"B&H train {bh_tr:+.0f}%")
    corr = res["train_ret"].corr(res["test_ret"])
    ax.set_title(f"Out-of-sample: does TRAIN skill survive into TEST?  (corr={corr:.2f})",
                 fontsize=14, weight="bold")
    ax.set_xlabel("TRAIN return 2016-2022 (%)"); ax.set_ylabel("TEST return 2023-2026 (%)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig("oos_scatter.png", dpi=130); plt.close(fig)

    # equity of the train-winners through the TEST period only, rebased to 100k
    fig, ax = plt.subplots(figsize=(12, 6.5))
    test_idx = df.index[i_split-1:]
    bh_test = START_CAPITAL * (df["Close"].iloc[i_split-1:] / df["Close"].iloc[i_split-1])
    ax.plot(bh_test.index, bh_test.values, color="#1565C0", lw=2.5, label=f"Buy & Hold ({bh_te:+.0f}%)")
    for _, r in winners.head(6).iterrows():
        eq = equities[int(r["id"])].iloc[i_split-1:]
        eq = START_CAPITAL * eq / eq.iloc[0]
        ax.plot(eq.index, eq.values, lw=1.5, alpha=0.9,
                label=f"{r['combo']}/{r['direction']} ({r['test_ret']:+.0f}%)")
    ax.axhline(START_CAPITAL, color="#bbb", lw=1)
    ax.set_title("TEST period only: train-winners graded on unseen 2023-2026 data",
                 fontsize=14, weight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig("oos_top.png", dpi=130); plt.close(fig)


def selftest():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert wilder_rsi(up, 14).iloc[-1] > 99
    assert pivots([5, 4, 3, 2, 1, 2, 3, 4, 5], 4, "low") == [4]
    print("selftest ok")


if __name__ == "__main__":
    selftest()
    print(f"Downloading {TICKER} ({YEARS}, dividend-adjusted)...")
    df = yf.Ticker(TICKER).history(period=YEARS, auto_adjust=True)[["Open", "High", "Low", "Close", "Volume"]].dropna()
    print(f"  {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}")

    rng = np.random.default_rng(SEED)
    cfgs = sample_configs(rng, N_SAMPLES)

    # ---- 1) full 10y Monte Carlo ----
    bh, bh_ret = buy_hold(df)
    res, eq = monte_carlo(df, cfgs)
    print(f"\n=== FULL 10Y ({len(res)} configs) ===   Buy & Hold {bh_ret:+.0f}%")
    beat = plot_distribution(res, bh_ret, f"Monte Carlo: {len(res)} strategies vs Buy & Hold (10y)", "mc_distribution.png")
    plot_top(res, eq, bh, bh_ret, "Top 6 configs vs Buy & Hold (10y)", "mc_top.png")
    print(f"Beat buy & hold: {beat}/{len(res)}"); show_top(res)

    # ---- 2) out-of-sample validation (uses the same full run) ----
    out_of_sample(df, res, eq, bh)

    # ---- 3) Trump 2nd term Monte Carlo ----
    dft = df[df.index >= pd.Timestamp(TRUMP_START, tz=df.index.tz)]
    bht, bht_ret = buy_hold(dft)
    rest, eqt = monte_carlo(dft, cfgs)
    print(f"\n=== TRUMP 2ND TERM  {dft.index[0].date()} -> {dft.index[-1].date()} "
          f"({len(dft)} bars) ===   Buy & Hold {bht_ret:+.0f}%")
    beatt = plot_distribution(rest, bht_ret, f"Trump 2nd term: {len(rest)} strategies vs Buy & Hold", "trump_distribution.png")
    plot_top(rest, eqt, bht, bht_ret, "Trump 2nd term: top 6 configs vs Buy & Hold", "trump_top.png")
    print(f"Beat buy & hold: {beatt}/{len(rest)}"); show_top(rest)

    print("\nSaved: mc_distribution.png, mc_top.png, oos_scatter.png, oos_top.png, "
          "trump_distribution.png, trump_top.png")
