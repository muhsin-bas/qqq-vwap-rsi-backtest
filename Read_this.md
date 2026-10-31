# qqq-vwap-rsi-backtest

Backtest of a VWAP + RSI-divergence trading strategy versus buy-and-hold QQQ,
with an accompanying research paper. The project is a reproducible demonstration
of **data-snooping / backtest-overfitting bias**: a randomized search over 1,000
strategy configurations is run, and the best performer is shown to fail standard
out-of-sample and selection-bias tests.

## Key finding

Across daily (10-year), hourly (~2-year), and 5-minute (~60-day) data, **fewer
than 1% of 1,000 configurations beat buy-and-hold**, the median configuration
lost money, and the single best configuration did **not** survive:

- **Out-of-sample:** inconsistent — underperformed QQQ on windows it wasn't selected on.
- **Permutation Monte Carlo:** p = 0.19 — timing indistinguishable from random.
- **Selection-bias Monte Carlo:** p = 1.00 — the best-of-1,000 is fully explained by chance.

No robust, tradeable edge was found.

## Files

| File | Purpose |
|------|---------|
| `BackTest.py` | Core engine: VWAP, RSI, divergence detection, position logic, Monte Carlo runner. Entry point for the 10-year daily study + out-of-sample + regime windows. |
| `Paper.py` | Runs 1,000 random configurations on daily and hourly data; generates the paper's charts and the validation figure. |
| `BackTestIntraday.py` | Hourly / 30-minute intraday Monte Carlo. |
| `BackTestHFT.py` | 5-minute variant: Naive Bayes + Ehlers Adaptive SuperSmoother. |
| `Validate.py` | The three robustness tests: out-of-sample, permutation MC, selection-bias MC. |

`Paper.py`, `BackTestIntraday.py`, `BackTestHFT.py`, and `Validate.py` import from
`BackTest.py`, so keep all files in the same directory.

## How to run

```bash
pip install -r requirements.txt
python Paper.py        # reproduces the paper's figures
python BackTest.py     # 10-year daily study + out-of-sample + regime windows
python Validate.py     # robustness tests on the best configuration
```

Price data is downloaded automatically from Yahoo! Finance via `yfinance`
(no API key needed). Note: Yahoo caps intraday history — hourly to ~2 years,
5/15/30-minute to ~60 days — so ten years of intraday data is not available.

## Data

QQQ (Invesco QQQ Trust), dividend-adjusted, 2016–2026, from Yahoo! Finance.

## Author

Muhsin Bas — independent research project. AI assistance was used for coding and drafting.

## License

MIT (see LICENSE).
