"""
key_delta_iv.py - implied vol at fixed deltas over time (QF620 group project)

Delta-based companion to key_strike_iv.py (which tracks fixed moneyness).
Same data, same forward, same implied vols; only the x-axis changes.

For every snapshot (~3 s) in every session:
    1. call / put mids, forward F from put-call parity, Black-Scholes IV at
       every strike from the OTM option (all reused from key_strike_iv.py)
    2. Black-Scholes forward delta at each OTM strike, using that strike's own IV:
           d1 = (ln(F/K) + 0.5 vol^2 T) / (vol sqrt(T))
           call delta = N(d1)          (calls at or above F, 0 < delta < 0.5)
           put delta  = N(d1) - 1      (puts below F, -0.5 < delta < 0)
    3. IV at fixed deltas, interpolated linearly in IV between the two
       neighbouring strikes whose deltas bracket the target:
           puts : -25 and -35 delta
           calls:  25 and  35 delta
           ATM   : IV at K = spot, ATMF: IV at K = F (same as key_strike_iv.py)
       Targets beyond the last usable strike (e.g. the 25-delta put below 4600
       at the lows on 7-9 April) are left blank (NaN), not extrapolated.
       10-delta is not tracked: the 10-delta put needs a strike below 4600 for
       almost all of 4-10 April, so it would be blank most of the time.
    4. risk reversal and butterfly at each level (25 and 35 delta):
           rr_25 = iv_25c - iv_25p
           bf_25 = (iv_25c + iv_25p) / 2 - iv_atmf
       35-delta is tracked as well because the 25-delta put is blank for much of
       7-9 April (4600 is less than 25 delta out of the money there); the
       35-delta put stays inside the strike grid almost all the time.

Why delta as well as moneyness: a fixed delta adjusts for the vol level and
time to expiry, so it stays comparable across the 6 April expiry roll
(~15 -> ~39 days), where a fixed % of F does not.

Output: one CSV and two charts (puts / calls) per session in key_delta_output/.
"""
import math
import os

import numpy as np
import pandas as pd
from scipy.special import ndtr

from risk_free_rate import RISK_FREE_RATE
from key_strike_iv import (
    DATA_FOLDER,
    bs_implied_vol,
    corrected_spot,
    find_expiry,
    forward_from_parity,
    interpolate_iv,
    list_sessions,
    mids_from,
    plot_key_strikes,
    read_csv,
    settlement_time,
)

OUTPUT_FOLDER = "key_delta_output/"

DELTA_LEVELS = [25, 35]


# ---------------------------------------------------------------- delta helpers

def forward_delta(F, K, T, vol, is_call):
    """Black-Scholes forward (undiscounted) delta for one strike."""
    if math.isnan(vol) or vol <= 0:
        return float("nan")
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / (vol * math.sqrt(T))
    if is_call:
        return float(ndtr(d1))
    return float(ndtr(d1)) - 1.0


def interpolate_by_delta(deltas, ivs, target):
    """Linear interpolation in IV between the two strikes whose deltas bracket target.

    deltas and ivs are lists in strike order. Neighbouring strikes with a missing
    IV break the bracket, so the result is NaN rather than a jump over a gap.
    """
    for i in range(len(deltas) - 1):
        d1 = deltas[i]
        d2 = deltas[i + 1]
        if math.isnan(d1) or math.isnan(d2):
            continue
        low = min(d1, d2)
        high = max(d1, d2)
        if low <= target <= high:
            if d1 == d2:
                return ivs[i]
            weight = (target - d1) / (d2 - d1)
            return ivs[i] + weight * (ivs[i + 1] - ivs[i])
    return float("nan")


# ---------------------------------------------------------------- one session

def build_key_delta_series(session, date, r=RISK_FREE_RATE, data_folder=DATA_FOLDER):
    """Fixed-delta implied vols at every snapshot of one session."""
    expiry = find_expiry(session, date, data_folder)
    folder = f"{data_folder}{session}/"

    call_bid = read_csv(f"{folder}{date}_call_bid_{expiry}.csv")
    call_ask = read_csv(f"{folder}{date}_call_ask_{expiry}.csv")
    put_bid = read_csv(f"{folder}{date}_put_bid_{expiry}.csv")
    put_ask = read_csv(f"{folder}{date}_put_ask_{expiry}.csv")
    assets = read_csv(f"{folder}{date}_assets.csv")

    times = call_ask.index
    if len(times) == 0:
        return pd.DataFrame()

    frames = [call_bid, put_bid, put_ask, assets]
    for i in range(len(frames)):
        if not frames[i].index.equals(times):
            frames[i] = frames[i].reindex(times, method="ffill")
    call_bid, put_bid, put_ask, assets = frames

    strikes = []
    for column in call_ask.columns:
        strikes.append(float(column))

    call_mid = mids_from(call_bid.to_numpy(dtype=float), call_ask.to_numpy(dtype=float))
    put_mid = mids_from(put_bid.to_numpy(dtype=float), put_ask.to_numpy(dtype=float))

    spot, spot_fixed = corrected_spot(assets, session, date, data_folder)

    settle = settlement_time(expiry)
    T = []
    for t in times:
        T.append((settle - t).total_seconds() / (365.0 * 24.0 * 3600.0))
    T = np.array(T)
    discount = np.exp(-r * T)

    forward = np.full(len(times), np.nan)
    for row in range(len(times)):
        forward[row] = forward_from_parity(strikes, call_mid[row], put_mid[row], discount[row])
    q = r - np.log(forward / spot) / T

    call_iv = np.full(call_mid.shape, np.nan)
    put_iv = np.full(put_mid.shape, np.nan)
    for j in range(len(strikes)):
        call_iv[:, j] = bs_implied_vol(call_mid[:, j], spot, strikes[j], T, r, q, True)
        put_iv[:, j] = bs_implied_vol(put_mid[:, j], spot, strikes[j], T, r, q, False)

    records = []
    for row in range(len(times)):
        F = forward[row]
        if math.isnan(F):
            continue

        # OTM puts (below F) and OTM calls (at or above F), each in strike order
        put_deltas = []
        put_ivs = []
        call_deltas = []
        call_ivs = []
        otm_iv = []
        for j in range(len(strikes)):
            K = strikes[j]
            if K < F:
                vol = put_iv[row, j]
                put_ivs.append(vol)
                put_deltas.append(forward_delta(F, K, T[row], vol, False))
                otm_iv.append(vol)
            else:
                vol = call_iv[row, j]
                call_ivs.append(vol)
                call_deltas.append(forward_delta(F, K, T[row], vol, True))
                otm_iv.append(vol)

        record = {
            "time": times[row],
            "spot": spot[row],
            "forward": F,
            "T": T[row],
            "iv_atm": interpolate_iv(strikes, otm_iv, spot[row]),
            "iv_atmf": interpolate_iv(strikes, otm_iv, F),
        }
        for level in DELTA_LEVELS:
            record[f"iv_{level}p"] = interpolate_by_delta(put_deltas, put_ivs, -level / 100.0)
            record[f"iv_{level}c"] = interpolate_by_delta(call_deltas, call_ivs, level / 100.0)
        records.append(record)

    result = pd.DataFrame(records).set_index("time")
    for level in DELTA_LEVELS:
        result[f"rr_{level}"] = result[f"iv_{level}c"] - result[f"iv_{level}p"]
        result[f"bf_{level}"] = 0.5 * (result[f"iv_{level}c"] + result[f"iv_{level}p"]) - result["iv_atmf"]
    result["spot_from_es"] = spot_fixed
    result["expiry"] = expiry

    # columns from the far put wing to the far call wing
    ordered = ["spot", "forward", "T"]
    for level in DELTA_LEVELS:
        ordered.append(f"iv_{level}p")
    ordered += ["iv_atm", "iv_atmf"]
    for level in reversed(DELTA_LEVELS):
        ordered.append(f"iv_{level}c")
    for level in DELTA_LEVELS:
        ordered += [f"rr_{level}", f"bf_{level}"]
    ordered += ["spot_from_es", "expiry"]
    return result[ordered]


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # wings darker, near-ATM lighter; ATMF in grey (same scheme as key_strike_iv.py)
    put_columns = ["iv_25p", "iv_35p", "iv_atm", "iv_atmf"]
    put_labels = ["25d put", "35d put", "ATM (spot)", "ATMF"]
    put_colors = ["#2f6fbe", "#86b6ef", "#e08a3c", "#52514e"]

    call_columns = ["iv_atm", "iv_atmf", "iv_35c", "iv_25c"]
    call_labels = ["ATM (spot)", "ATMF", "35d call", "25d call"]
    call_colors = ["#e08a3c", "#52514e", "#7fd1ae", "#3fa77a"]

    for session, day in list_sessions():
        print(f"{session} {day} ...")
        try:
            series = build_key_delta_series(session, day)
        except FileNotFoundError as error:
            print("  skipped:", error)
            continue
        if len(series) == 0:
            print("  no data - skipped")
            continue

        name = f"iv_delta_{session}_{day}"
        series.to_csv(OUTPUT_FOLDER + name + ".csv")

        plot_key_strikes(series, put_columns, put_labels, put_colors,
                         OUTPUT_FOLDER + name + "_puts.png",
                         f"Implied vol by delta, OTM puts and ATMF - {session} session starting {day}")
        plot_key_strikes(series, call_columns, call_labels, call_colors,
                         OUTPUT_FOLDER + name + "_calls.png",
                         f"Implied vol by delta, OTM calls and ATMF - {session} session starting {day}")
        print(f"  saved {name}: {len(series)} rows")