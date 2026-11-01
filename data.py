"""
data.py - QF620 group project data access / cleaning helpers
============================================================

Release notes
-------------
v0.1.0
  * Session timings filled in (naive timestamps, ET clock):
      - RTH (regular trading hours): 09:31:00 -> 16:15:00 (same day)
      - ETH (extended trading hours): 20:15:00 -> 09:15:00 (next calendar day)
    An ETH file is named by the date the session STARTS, so a 03:00 timestamp
    on 2025-04-02 is read from the ETH file dated 2025-04-01 (weekends skipped).
  * New `get_option_mid()`: for a user-specified date time and option type,
    returns the mid price ((bid + ask) / 2) at each strike given. Fill in any
    of k_itm / k_atm / k_otm (one is enough); strike choice is left to the
    caller / another function.
  * Option files are read per day and cached in memory.

Data layout (data/<ETH|RTH>/)
  <date>_assets.csv                        : SPX, ES_BID, ES_ASK, VIX
  <date>_<call|put>_<bid|ask>_<expiry>.csv : rows = timestamps, cols = strikes
  Lookup uses the last snapshot at or before the requested time.

Known gaps
  * Some option files are missing (e.g. RTH 2025-04-01 call_bid);
    `get_option_mid()` raises FileNotFoundError then.
  * Events to study are picked by the team from `events_csv`.
"""
from __future__ import annotations

import glob
import os
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
from functools import lru_cache

data_path = "QF620_group_project/data/"
events_csv = "QF620_group_project/2025 Liberation Day Tariff Main Events.csv"
datasets_dict = [
    'assets', 'calls_ask', 'calls_bid',
    "puts_ask", "puts_bid",
    ]
dateformat = "%Y-%m-%d"
timeformat = "%H:%M:%S"

trading_period = {
    "ETH": "20:15:00-09:15:00",
    "RTH": "09:31:00-16:15:00",
}

_OPTION_TYPES = {"call": "call", "c": "call", "put": "put", "p": "put"}


def _parse_period(session: str) -> tuple[time, time]:
    """Return (start, end) clock times of a session from trading_period."""
    start, end = trading_period[session].split("-")
    return (datetime.strptime(start, timeformat).time(),
            datetime.strptime(end, timeformat).time())


def get_session(timestamp: datetime) -> tuple[str, str]:
    """Return (session name, session-start date string) containing timestamp."""
    t = timestamp.time()
    rth_start, rth_end = _parse_period("RTH")
    eth_start, eth_end = _parse_period("ETH")
    if rth_start <= t <= rth_end:
        return "RTH", timestamp.strftime(dateformat)
    if t >= eth_start:
        return "ETH", timestamp.strftime(dateformat)
    if t <= eth_end:
        # Overnight session started on an earlier date; weekends have no file.
        day = timestamp.date() - timedelta(days=1)
        for _ in range(5):
            d = day.strftime(dateformat)
            if os.path.exists(f"{data_path}ETH/{d}_assets.csv"):
                return "ETH", d
            day -= timedelta(days=1)
    raise ValueError(f"{timestamp} is outside RTH {trading_period['RTH']} "
                     f"and ETH {trading_period['ETH']}")


@lru_cache(maxsize=32)
def _load_option_file(path: str) -> pd.DataFrame:
    """Load one option csv: datetime index, float strike columns (cached)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = df.columns.astype(float)
    return df.sort_index()


def _find_option_file(session: str, date: str, option_type: str, side: str) -> str:
    """Return the path of the <date>_<type>_<side>_<expiry>.csv file."""
    pattern = f"{data_path}{session}/{date}_{option_type}_{side}_*.csv"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No file matches {pattern}")
    return matches[0]


def get_option_mid(
    timestamp: datetime | str,
    option_type: str,
    k_atm: float | None = None,
    k_otm: float | None = None,
    k_itm: float | None = None,
    tolerance: str = "60s",
) -> pd.DataFrame:
    """Return mid price (bid+ask)/2 per strike given, at the last snapshot <= timestamp.

    timestamp: datetime or 'YYYY-MM-DD HH:MM:SS' (ET, inside ETH/RTH).
    option_type: 'call' or 'put'. Fill in any of k_atm/k_otm/k_itm (at least one).
    Output: DataFrame indexed by label ('atm'/'otm'/'itm') with columns strike, mid.
    """
    strikes = {"atm": k_atm, "otm": k_otm, "itm": k_itm}
    strikes = {label: float(k) for label, k in strikes.items() if k is not None}
    if not strikes:
        raise ValueError("Provide at least one of k_atm, k_otm, k_itm")
    if option_type.lower() not in _OPTION_TYPES:
        raise ValueError(f"option_type must be call or put, got {option_type!r}")
    opt = _OPTION_TYPES[option_type.lower()]
    ts = pd.Timestamp(timestamp)

    session, date = get_session(ts.to_pydatetime())
    strike_list = list(strikes.values())
    quotes = {}
    for side in ("bid", "ask"):
        df = _load_option_file(_find_option_file(session, date, opt, side))
        idx = df.index.asof(ts)
        if pd.isna(idx) or ts - idx > pd.Timedelta(tolerance):
            raise ValueError(f"No {side} snapshot within {tolerance} before {ts}")
        missing = [k for k in strike_list if k not in df.columns]
        if missing:
            raise KeyError(f"Strikes {missing} not in file; available "
                           f"{df.columns.min()}-{df.columns.max()}")
        quotes[side] = df.loc[idx, strike_list].to_numpy(dtype=float)

    return pd.DataFrame(
        {"strike": strike_list, "mid": (quotes["bid"] + quotes["ask"]) / 2},
        index=list(strikes.keys()),
    )
