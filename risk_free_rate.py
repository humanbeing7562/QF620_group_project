"""
rates.py - risk-free rate used by every script in the project

Downloads daily short-term Treasury yields for the study period and averages
them into one constant, RISK_FREE_RATE. Every script does
    from rates import RISK_FREE_RATE

Sources (set SOURCE below):
    "fred"     : 1-month Treasury constant-maturity yield, FRED series DGS1MO
                 (public CSV, no API key needed). Matches our ~2-6 week expiries.
    "yfinance" : ^IRX, the 13-week (3-month) T-bill yield. yfinance has no
                 1-month series, so this is a slightly longer tenor.

The download is saved to rates_cache.csv the first time, so later runs (and
every import) work offline and always use the same numbers. Delete the cache
file or run `python rates.py --refresh` to download again.
"""
import io
import os
import sys
import urllib.request

import pandas as pd

SOURCE = "fred"
START_DATE = "2025-04-01"
END_DATE = "2025-04-10"
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rates_cache.csv")

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"


def fetch_fred(series="DGS1MO", start=START_DATE, end=END_DATE):
    """Daily yields (%) from FRED's public CSV, as a Series indexed by date."""
    url = FRED_URL.format(series=series, start=start, end=end)
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8")
    table = pd.read_csv(io.StringIO(text))

    # first column is the date ("observation_date" or "DATE"), second the yield;
    # FRED writes "." for days with no value
    dates = pd.to_datetime(table.iloc[:, 0])
    values = pd.to_numeric(table.iloc[:, 1], errors="coerce")
    rates = pd.Series(values.to_numpy(), index=dates, name=series)
    return rates.dropna()


def fetch_yfinance(ticker="^IRX", start=START_DATE, end=END_DATE):
    """Daily yields (%) from Yahoo Finance via yfinance (^IRX = 13-week T-bill)."""
    import yfinance as yf

    # yfinance's end date is exclusive, so add a day
    end_inclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    history = yf.Ticker(ticker).history(start=start, end=end_inclusive)
    if len(history) == 0:
        raise ValueError(f"yfinance returned no data for {ticker}")

    dates = []
    for stamp in history.index:
        dates.append(pd.Timestamp(stamp.date()))
    rates = pd.Series(history["Close"].to_numpy(), index=dates, name=ticker)
    return rates.dropna()


def download_rates():
    if SOURCE == "fred":
        return fetch_fred()
    if SOURCE == "yfinance":
        return fetch_yfinance()
    raise ValueError(f"Unknown SOURCE {SOURCE!r}; use 'fred' or 'yfinance'")


def load_rates(refresh=False):
    """Daily yields (%) from the cache, downloading them first if needed."""
    if os.path.exists(CACHE_FILE) and not refresh:
        cached = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        return cached.iloc[:, 0]

    try:
        rates = download_rates()
    except Exception as error:
        raise RuntimeError(
            f"rates.py could not download rates from {SOURCE} ({error}). "
            f"Run `python rates.py --refresh` with an internet connection, "
            f"or put a {os.path.basename(CACHE_FILE)} next to rates.py.") from error
    if len(rates) == 0:
        raise ValueError("Download returned no rates")
    rates.to_frame().to_csv(CACHE_FILE)
    print(f"rates.py: downloaded {len(rates)} days from {SOURCE}, saved to {CACHE_FILE}")
    return rates


def average_rate(daily_rates):
    """Average of the daily yields, as a decimal (4.37 -> 0.0437)."""
    total = 0.0
    for value in daily_rates:
        total += float(value)
    return total / len(daily_rates) / 100.0


DAILY_RATES = load_rates()

# one fixed rate for the whole period
RISK_FREE_RATE = average_rate(DAILY_RATES)


if __name__ == "__main__":
    if "--refresh" in sys.argv:
        DAILY_RATES = load_rates(refresh=True)
        RISK_FREE_RATE = average_rate(DAILY_RATES)
    print(f"source: {SOURCE}")
    for date, value in DAILY_RATES.items():
        print(f"  {date:%Y-%m-%d}  {value:.2f}%")
    print("RISK_FREE_RATE =", round(RISK_FREE_RATE, 6))