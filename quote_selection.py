"""Select audited SPX option price targets for one ETH or RTH snapshot.

The option filename suffix is the last trading date of an AM-settled monthly
contract. The settlement clock is 09:30 New York time on the next trading day.
Only the two expiries present in this project's data are accepted here.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from key_strike_iv import (
    MIN_MID,
    corrected_spot,
    find_expiry,
    forward_from_parity,
    mids_from,
    read_csv,
    settlement_time,
)
from risk_free_rate import RISK_FREE_RATE
from smile_per_event import find_session


DATA_DIR = Path(__file__).resolve().parent / "data"
NEW_YORK = ZoneInfo("America/New_York")
EXPECTED_SETTLEMENT = {
    "20250416": "2025-04-17 09:30:00",
    "20250515": "2025-05-16 09:30:00",
}


@lru_cache(maxsize=10)
def _read_grid(path: str) -> pd.DataFrame:
    """Reuse large session grids across nearby event timestamps."""
    return read_csv(path)


@dataclass
class MarketSnapshot:
    requested_time: pd.Timestamp
    row_times: dict[str, pd.Timestamp]
    session: str
    file_date: str
    last_trading_date: str
    settlement_time: pd.Timestamp
    maturity: float
    rate: float
    discount: float
    forward: float
    spot: float
    spot_from_es: bool
    quotes: pd.DataFrame
    targets: pd.DataFrame
    atm_strike: float
    exclusions: dict[str, int]


def asof_row(
    frame: pd.DataFrame, timestamp: pd.Timestamp | str, max_age_seconds: float = 60
) -> tuple[pd.Series, pd.Timestamp]:
    """Return the latest source row no later than timestamp, enforcing its age."""
    if frame.empty:
        raise ValueError("source file has no rows")
    if not frame.index.is_unique:
        raise ValueError("source file has duplicate timestamps")
    wanted = pd.Timestamp(timestamp)
    row_time = frame.index.asof(wanted)
    if pd.isna(row_time):
        raise ValueError(f"no source row at or before {wanted}")
    age = (wanted - row_time).total_seconds()
    if age > max_age_seconds:
        raise ValueError(f"source row at {row_time} is {age:.1f}s old")
    return frame.loc[row_time], pd.Timestamp(row_time)


def load_quote_rows(
    session: str,
    file_date: str,
    last_trading_date: str,
    timestamp: pd.Timestamp | str,
    data_dir: Path = DATA_DIR,
    max_age_seconds: float = 60,
) -> tuple[dict[str, pd.Series], dict[str, pd.Timestamp], pd.DataFrame]:
    """Read all five grids and select each file's own recent source row."""
    folder = Path(data_dir) / session
    frames: dict[str, pd.DataFrame] = {}
    rows: dict[str, pd.Series] = {}
    times: dict[str, pd.Timestamp] = {}
    for side in ("call", "put"):
        for quote_side in ("bid", "ask"):
            name = f"{side}_{quote_side}"
            path = folder / f"{file_date}_{name}_{last_trading_date}.csv"
            frames[name] = _read_grid(str(path))
    frames["assets"] = _read_grid(str(folder / f"{file_date}_assets.csv"))

    for name, frame in frames.items():
        try:
            rows[name], times[name] = asof_row(frame, timestamp, max_age_seconds)
        except ValueError as error:
            raise ValueError(f"{session} {file_date} {name}: {error}") from error
    return rows, times, frames["assets"]


def build_quote_table(
    rows: dict[str, pd.Series]
) -> tuple[pd.DataFrame, list[float], np.ndarray, np.ndarray, dict[str, int]]:
    """Keep valid bid/ask pairs and retain the full price information."""
    labels = [rows[name].index for name in ("call_bid", "call_ask", "put_bid", "put_ask")]
    strikes_by_file = [set(float(label) for label in index) for index in labels]
    if not all(strikes == strikes_by_file[0] for strikes in strikes_by_file[1:]):
        raise ValueError("call/put bid/ask files have different strike columns")
    strikes = sorted(strikes_by_file[0])
    if not strikes:
        raise ValueError("option files have no strike columns")

    records = []
    exclusions: Counter[str] = Counter()
    mid_arrays = {}
    for side in ("call", "put"):
        bid_row = rows[f"{side}_bid"]
        ask_row = rows[f"{side}_ask"]
        bid_by_strike = {float(label): float(value) for label, value in bid_row.items()}
        ask_by_strike = {float(label): float(value) for label, value in ask_row.items()}
        bids = np.array([bid_by_strike[k] for k in strikes])
        asks = np.array([ask_by_strike[k] for k in strikes])
        mids = mids_from(bids, asks)
        mids[~(np.isfinite(bids) & np.isfinite(asks) & np.isfinite(mids))] = np.nan
        mid_arrays[side] = mids
        for k, bid, ask, mid in zip(strikes, bids, asks, mids):
            if not (math.isfinite(bid) and math.isfinite(ask)):
                exclusions["nonfinite_quote"] += 1
            elif bid <= 0 or ask <= 0:
                exclusions["nonpositive_quote"] += 1
            elif ask < bid:
                exclusions["crossed_quote"] += 1
            elif not math.isfinite(mid) or mid < MIN_MID:
                exclusions["below_min_mid"] += 1
            else:
                records.append(
                    {"side": side, "strike": k, "bid": bid, "ask": ask, "mid": mid}
                )

    quotes = pd.DataFrame(records, columns=["side", "strike", "bid", "ask", "mid"])
    if quotes.empty:
        raise ValueError("no valid option quotes at this snapshot")
    return quotes, strikes, mid_arrays["call"], mid_arrays["put"], dict(exclusions)


def select_calibration_targets(
    quotes: pd.DataFrame, forward: float, max_atm_distance: float = 0.01
) -> tuple[pd.DataFrame, float, dict[str, int]]:
    """Use OTM prices and replace the nearest OTM quote with its ATM straddle."""
    if not math.isfinite(forward) or forward <= 0:
        raise ValueError("parity forward must be positive and finite")
    by_key = {(row.side, row.strike): row for row in quotes.itertuples(index=False)}
    common = sorted(
        {k for side, k in by_key if side == "call"}
        & {k for side, k in by_key if side == "put"},
        key=lambda k: (abs(k - forward), k),
    )
    if not common:
        raise ValueError("no strike has both valid call and put quotes for a straddle")
    atm_strike = common[0]
    if abs(atm_strike / forward - 1.0) > max_atm_distance:
        raise ValueError(f"nearest valid straddle strike {atm_strike:g} is too far from F={forward:g}")

    targets = []
    skipped = 0
    for strike in sorted({k for _, k in by_key}):
        if strike == atm_strike:
            call = by_key[("call", strike)]
            put = by_key[("put", strike)]
            targets.append(
                {"kind": "straddle", "strike": strike,
                 "bid": call.bid + put.bid, "ask": call.ask + put.ask,
                 "mid": call.mid + put.mid}
            )
        else:
            side = "put" if strike < forward else "call"
            quote = by_key.get((side, strike))
            if quote is None:
                skipped += 1
                continue
            targets.append(
                {"kind": side, "strike": strike, "bid": quote.bid,
                 "ask": quote.ask, "mid": quote.mid}
            )
    selected = pd.DataFrame(targets, columns=["kind", "strike", "bid", "ask", "mid"])
    if len(selected) < 10 or not {"call", "put", "straddle"}.issubset(set(selected["kind"])):
        raise ValueError("calibration needs at least ten targets, both option sides, and a straddle")
    return selected, float(atm_strike), {"missing_otm_side": skipped}


def load_market_snapshot(
    timestamp: pd.Timestamp | str,
    data_dir: Path = DATA_DIR,
    r: float = RISK_FREE_RATE,
    max_age_seconds: float = 60,
) -> MarketSnapshot:
    """Build one auditable, AM-settled SPX price-calibration snapshot."""
    requested = pd.Timestamp(timestamp)
    if requested.tzinfo is not None:
        requested = requested.tz_convert(NEW_YORK).tz_localize(None)
    if not math.isfinite(r):
        raise ValueError("risk-free rate must be finite")
    data_dir = Path(data_dir).resolve()
    data_folder = str(data_dir) + "/"
    found = find_session(requested, data_folder)
    if found is None:
        raise ValueError(f"{requested} is outside the available ETH/RTH sessions")
    session, file_date = found
    last_trading_date = find_expiry(session, file_date, data_folder)
    if last_trading_date not in EXPECTED_SETTLEMENT:
        raise ValueError(f"unreviewed SPX AM expiry code {last_trading_date}")
    settlement = settlement_time(last_trading_date)
    if str(settlement) != EXPECTED_SETTLEMENT[last_trading_date]:
        raise ValueError(f"settlement mapping disagrees for {last_trading_date}")
    elapsed = (
        settlement.tz_localize(NEW_YORK).tz_convert("UTC")
        - requested.tz_localize(NEW_YORK).tz_convert("UTC")
    ).total_seconds()
    if elapsed <= 0:
        raise ValueError("snapshot is at or after AM settlement")
    maturity = elapsed / (365 * 24 * 3600)
    discount = math.exp(-r * maturity)

    rows, row_times, assets = load_quote_rows(
        session, file_date, last_trading_date, requested, data_dir, max_age_seconds
    )
    quotes, strikes, call_mids, put_mids, exclusions = build_quote_table(rows)
    forward = float(forward_from_parity(strikes, call_mids, put_mids, discount))
    targets, atm_strike, target_exclusions = select_calibration_targets(quotes, forward)
    exclusions.update(target_exclusions)

    spot_values, spot_from_es = corrected_spot(assets, session, file_date, data_folder)
    asset_position = assets.index.get_loc(row_times["assets"])
    spot = float(spot_values[asset_position])
    if not math.isfinite(spot) or spot <= 0:
        raise ValueError("spot diagnostic is not positive and finite")

    return MarketSnapshot(
        requested, row_times, session, file_date, last_trading_date, settlement,
        maturity, float(r), discount, forward, spot, bool(spot_from_es), quotes,
        targets, atm_strike, exclusions,
    )
