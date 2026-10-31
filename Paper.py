"""
Research-paper artifacts: 1000 random strategy configs, QQQ.
Generates:
  paper_10y_top.png   - 10-year daily: QQQ + top-6 strategies
  paper_1h_top.png     - hourly (~2y): QQQ + top-6 strategies
  paper_all1000.png    - all 1000 strategy equity curves + QQQ (10-year daily)
  paper_validation.png - OOS + permutation MC + selection-bias MC for the best 1h config
Prints a STATS block to transcribe into the paper.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

import BackTest as B
from BackTest import (_signals, build_positions, monte_carlo, buy_hold, START_CAPITAL,
                      COMBOS, DIRECTIONS, EXITS, RSI_LENS, LOOKBACKS, HIDDEN)

N, SEED = 1000, 42
BARS_PER_YEAR_1H = 252 * 6.5


def sample_configs(rng, n, anchors):
    seen, cfgs = set(), []
    while len(cfgs) < n:
        cfg = {"combo": COMBOS[rng.integers(len(COMBOS))],
               "direction": DIRECTIONS[rng.integers(len(DIRECTIONS))],
               "exit": EXITS[rng.integers(len(EXITS))],
               "anchor": anchors[rng.integers(len(anchors))],
               "rsi_len": int(RSI_LENS[rng.integers(len(RSI_LENS))]),
               "lb": int(LOOKBACKS[rng.integers(len(LOOKBACKS))]),
               "use_hidden": bool(HIDDEN[rng.integers(len(HIDDEN))])}
        k = tuple(sorted(cfg.items()))
        if k not in seen:
            seen.add(k); cfgs.append(cfg)
    return cfgs


def chart_top(res, eq, bh, bh_ret, title, fname):
    top = res.sort_values("ret_pct", ascending=False).head(6)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(bh.index, bh.values, color="#1565C0", lw=3, label=f"QQQ Buy & Hold ({bh_ret:+.0f}%)", zorder=10)
    for _, r in top.iterrows():
        e = eq[int(r["id"])]
        ax.plot(e.index, e.values, lw=1.5, alpha=0.9,
                label=f"{r['combo']}/{r['direction']} ({r['ret_pct']:+.0f}%)")
    ax.axhline(START_CAPITAL, color="#bbb", lw=1)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_ylabel("Account value ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def chart_all(res, eq, bh, bh_ret, title, fname):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i in eq:
        e = eq[i]
        ax.plot(e.index, e.values, color="#26A69A", lw=0.4, alpha=0.08)
    ax.plot(bh.index, bh.values, color="#1565C0", lw=3, label=f"QQQ Buy & Hold ({bh_ret:+.0f}%)", zorder=10)
    best = res.sort_values("ret_pct", ascending=False).iloc[0]
    e = eq[int(best["id"])]
    ax.plot(e.index, e.values, color="#E53935", lw=2, zorder=11,
            label=f"best of {len(res)} ({best['ret_pct']:+.0f}%)")
    ax.axhline(START_CAPITAL, color="#bbb", lw=1)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_ylabel("Account value ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def summarize(res, bh_ret, label):
    beat = int((res["ret_pct"] > bh_ret).sum())
    print(f"\n[{label}]  QQQ {bh_ret:+.1f}%   beat {beat}/{len(res)} ({100*beat/len(res):.1f}%)")
    print(f"  best {res['ret_pct'].max():+.1f}%  median {res['ret_pct'].median():+.1f}%  worst {res['ret_pct'].min():+.1f}%")
    b = res.sort_values("ret_pct", ascending=False).iloc[0]
    print(f"  best config: {b['combo']}/{b['direction']}/{b['exit']} anchor {b['anchor']} "
          f"rsi{b['rsi_len']} lb{b['lb']} hidden={b['use_hidden']}  DD {b['max_dd_pct']:.1f}% trades {b['trades']}")
    return beat


def total_return(pos_prev, ret):
    return np.prod(1 + pos_prev * ret) - 1


if __name__ == "__main__":
    print("="*70, "\nRUNNING 1000 STRATEGIES\n", "="*70)

    # ---- 10-year daily ----
    d = yf.Ticker("QQQ").history(period="10y", interval="1d")[["Open","High","Low","Close","Volume"]].dropna()
    cfgs_d = sample_configs(np.random.default_rng(SEED), N, ["W","M","Q"])
    res_d, eq_d = monte_carlo(d, cfgs_d)
    bh_d, bhr_d = buy_hold(d)
    beat_d = summarize(res_d, bhr_d, f"10Y DAILY  {d.index[0].date()}->{d.index[-1].date()}  {len(d)} bars")
    chart_top(res_d, eq_d, bh_d, bhr_d, f"10-Year Daily: QQQ vs top-6 of {N} strategies", "paper_10y_top.png")
    chart_all(res_d, eq_d, bh_d, bhr_d, f"All {N} strategies vs QQQ (10-year daily)", "paper_all1000.png")

    # ---- hourly (~2y) ----
    h = yf.Ticker("QQQ").history(period="730d", interval="1h")[["Open","High","Low","Close","Volume"]].dropna()
    cfgs_h = sample_configs(np.random.default_rng(SEED), N, ["D","W"])
    res_h, eq_h = monte_carlo(h, cfgs_h)
    bh_h, bhr_h = buy_hold(h)
    beat_h = summarize(res_h, bhr_h, f"HOURLY  {h.index[0].date()}->{h.index[-1].date()}  {len(h)} bars")
    chart_top(res_h, eq_h, bh_h, bhr_h, f"Hourly (~2y): QQQ vs top-6 of {N} strategies", "paper_1h_top.png")

    # ---- validation of the best hourly config (the headline rigor test) ----
    best = res_h.sort_values("ret_pct", ascending=False).iloc[0].to_dict()
    WIN = {k: best[k] for k in ["combo","direction","exit","anchor","rsi_len","lb","use_hidden"]}
    tz = h.index.tz
    print("\n" + "="*70, "\nVALIDATION of best hourly config:", WIN, "\n", "="*70)

    windows = {
        "pre-Trump 2023-24 (OOS)":   (h.index[0],                        pd.Timestamp("2025-01-20", tz=tz)),
        "Trump 1st year 2025 (OOS)": (pd.Timestamp("2025-01-20", tz=tz), pd.Timestamp("2026-01-20", tz=tz)),
        "Trump 2nd year (SELECTED)": (pd.Timestamp("2026-01-20", tz=tz), h.index[-1]),
    }
    for name, (a, b_) in windows.items():
        w = h[(h.index >= a) & (h.index < b_)]
        if len(w) < 60:
            continue
        B._cache.clear()
        v, bu, be = _signals(w, WIN["rsi_len"], WIN["lb"], WIN["anchor"], WIN["use_hidden"])
        pos = pd.Series(build_positions(w["Close"].values, v, bu, be, WIN), index=w.index)
        ret = w["Close"].pct_change().fillna(0)
        sr = pos.shift(1).fillna(0) * ret
        eq = (1 + sr).cumprod()
        strat = eq.iloc[-1]*100 - 100
        qqq = (w["Close"].iloc[-1]/w["Close"].iloc[0]-1)*100
        sharpe = sr.mean()/sr.std()*np.sqrt(BARS_PER_YEAR_1H) if sr.std() > 0 else 0
        print(f"  {name:<28} strat {strat:+7.1f}%  QQQ {qqq:+7.1f}%  {'BEAT' if strat>qqq else 'lost'}  "
              f"Sharpe {sharpe:5.2f}  DD {(eq/eq.cummax()-1).min()*100:6.1f}%")

    # permutation + selection MC on the selected window
    w = h[h.index >= pd.Timestamp("2026-01-20", tz=tz)]
    ret = w["Close"].pct_change().fillna(0).values
    n = len(ret)
    B._cache.clear()
    v, bu, be = _signals(w, WIN["rsi_len"], WIN["lb"], WIN["anchor"], WIN["use_hidden"])
    pos_prev = pd.Series(build_positions(w["Close"].values, v, bu, be, WIN), index=w.index).shift(1).fillna(0).values
    actual = total_return(pos_prev, ret)

    rng = np.random.default_rng(SEED)
    perm = np.array([total_return(pos_prev, np.roll(ret, rng.integers(50, n-50))) for _ in range(2000)])
    p_perm = (perm >= actual).mean()

    P = []
    for c in cfgs_h:
        B._cache.clear()
        vv, bb, ee = _signals(w, c["rsi_len"], c["lb"], c["anchor"], c["use_hidden"])
        P.append(pd.Series(build_positions(w["Close"].values, vv, bb, ee, c), index=w.index).shift(1).fillna(0).values)
    P = np.vstack(P)
    real_best = (np.prod(1 + P * ret[None, :], axis=1) - 1).max()
    idx = np.arange(n); M = 200
    best_null = np.empty(M)
    for j in range(M):
        offs = rng.integers(50, n-50, size=P.shape[0])
        rolled = P[np.arange(P.shape[0])[:, None], (idx[None, :] - offs[:, None]) % n]
        best_null[j] = (np.prod(1 + rolled * ret[None, :], axis=1) - 1).max()
    p_sel = (best_null >= real_best).mean()

    print(f"\n  PERMUTATION MC: actual {actual*100:+.1f}%  null mean {perm.mean()*100:+.1f}%  p={p_perm:.3f}")
    print(f"  SELECTION MC:   real best {real_best*100:+.1f}%  null best mean {best_null.mean()*100:+.1f}%  p={p_sel:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].hist(perm*100, bins=50, color="#90A4AE", edgecolor="white")
    ax[0].axvline(actual*100, color="#E53935", lw=2.5, label=f"actual {actual*100:+.0f}%")
    ax[0].set_title(f"Permutation MC (p={p_perm:.2f})", weight="bold")
    ax[0].set_xlabel("return under random alignment (%)"); ax[0].legend()
    ax[1].hist(best_null*100, bins=40, color="#90A4AE", edgecolor="white")
    ax[1].axvline(real_best*100, color="#E53935", lw=2.5, label=f"real best {real_best*100:+.0f}%")
    ax[1].set_title(f"Best-of-{N} under null (p={p_sel:.2f})", weight="bold")
    ax[1].set_xlabel(f"best-of-{N} return by chance (%)"); ax[1].legend()
    for a in ax: a.grid(alpha=0.25)
    fig.suptitle("Does the best strategy survive? (left: timing skill, right: selection luck)", fontsize=13, weight="bold")
    fig.tight_layout(); fig.savefig("paper_validation.png", dpi=130); plt.close(fig)

    print("\nSaved paper_10y_top.png, paper_1h_top.png, paper_all1000.png, paper_validation.png")
