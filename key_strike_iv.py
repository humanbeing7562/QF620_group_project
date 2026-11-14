"""
key_strike_iv.py - implied vol at key moneyness points over time (QF620 group project)

For every snapshot (~3 s) in every session:
    1. call / put mids at every strike
    2. forward F from put-call parity (same method as vix_index.py)
    3. Black-Scholes implied vol at each strike, from the OTM option
       (put below F, call at or above F)
    4. IV at key points, interpolated linearly in IV between the two
       neighbouring strikes:
           90%, 95%, 97.5% of F, ATM (spot), ATMF (= F), 102.5%, 105%, 110% of F
       Points outside the 4600-6600 strike grid, or next to quotes stuck near the
       minimum tick (mid < 0.10), are left blank (NaN), not extrapolated.

Black-Scholes uses spot S and a dividend yield q backed out from the parity
forward, q = r - ln(F/S) / T, so the prices are consistent with the market forward.

Output: one CSV and two charts (downside / upside) per session in key_strike_output/.
"""
import glob
import math
import os

import numpy as np
import pandas as pd
from scipy.special import ndtr   # standard normal CDF that works on whole arrays

DATA_FOLDER = "data/"
OUTPUT_FOLDER = "key_strike_output/"
RISK_FREE_RATE = 0.043    # TODO: replace with 1-month T-bill, same as the other files
EXPIRY_TIME = "09:30:00"  # AM settlement on the trading day after the file date

SESSION_TIMES = {
    "RTH": ("09:31:00", "16:15:00"),
    "ETH": ("20:15:00", "09:15:00"),
}

# (column name, multiple of the forward); ATM (spot) is handled separately
MONEYNESS_POINTS = [
    ("iv_90", 0.90),
    ("iv_95", 0.95),
    ("iv_97_5", 0.975),
    ("iv_atmf", 1.00),
    ("iv_102_5", 1.025),
    ("iv_105", 1.05),
    ("iv_110", 1.10),
]

# quotes this close to the minimum tick (0.05) say little about vol; we keep anything
# from 0.10 up so 110% calls exist on calm days too (they are noisy, read with care)
MIN_MID = 0.10

# a session's SPX column normally sits ~35-43 points below the ES mid;
# beyond this we treat the SPX column as broken and rebuild spot from ES
MAX_NORMAL_BASIS = 100.0


# ---------------------------------------------------------------- dates / files

def settlement_time(expiry):
    """File date is the last trading day; settlement is 09:30 on the next weekday."""
    last_trading_day = pd.to_datetime(expiry, format="%Y%m%d")
    settle_day = last_trading_day + pd.Timedelta(days=1)
    while settle_day.weekday() >= 5:
        settle_day = settle_day + pd.Timedelta(days=1)
    return pd.Timestamp(f"{settle_day.date()} {EXPIRY_TIME}")


def list_sessions(data_folder=DATA_FOLDER):
    """All (session, date) pairs that have an assets file, in time order."""
    found = []
    for session in ["RTH", "ETH"]:
        for path in glob.glob(f"{data_folder}{session}/*_assets.csv"):
            date = os.path.basename(path).split("_")[0]
            start = pd.Timestamp(f"{date} {SESSION_TIMES[session][0]}")
            found.append((start, session, date))
    found.sort()

    ordered = []
    for start, session, date in found:
        ordered.append((session, date))
    return ordered


def find_expiry(session, date, data_folder=DATA_FOLDER):
    """Expiry date from the option file names, e.g. "20250416"."""
    matches = glob.glob(f"{data_folder}{session}/{date}_call_bid_*.csv")
    if len(matches) == 0:
        raise FileNotFoundError(f"No option files for {session} {date}")
    name = os.path.basename(matches[0])
    return name.split("_")[-1].replace(".csv", "")


def read_csv(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


# ---------------------------------------------------------------- spot fix

def session_basis(session, date, data_folder=DATA_FOLDER):
    """Average ES mid minus SPX over a session (NaN if the session is empty)."""
    path = f"{data_folder}{session}/{date}_assets.csv"
    if not os.path.exists(path):
        return float("nan")
    assets = read_csv(path)
    if len(assets) == 0:
        return float("nan")
    es_mid = (assets["ES_BID"] + assets["ES_ASK"]) / 2.0
    return float((es_mid - assets["SPX"]).mean())


def corrected_spot(assets, session, date, data_folder=DATA_FOLDER):
    """SPX spot for every row; rebuilt from ES when the SPX column is off.

    In ETH 2025-04-06 the SPX column sits ~286 points ABOVE the ES mid, which
    can't be right. For such a session we use ES mid minus the average basis
    of the sessions just before and after it.
    """
    es_mid = (assets["ES_BID"] + assets["ES_ASK"]) / 2.0
    basis = float((es_mid - assets["SPX"]).mean())
    if abs(basis) <= MAX_NORMAL_BASIS:
        return assets["SPX"].to_numpy(dtype=float), False

    sessions = list_sessions(data_folder)
    position = sessions.index((session, date))
    neighbour_bases = []
    for step in [-1, 1]:
        j = position + step
        if 0 <= j < len(sessions):
            value = session_basis(sessions[j][0], sessions[j][1], data_folder)
            if not math.isnan(value) and abs(value) <= MAX_NORMAL_BASIS:
                neighbour_bases.append(value)
    fixed_basis = sum(neighbour_bases) / len(neighbour_bases)
    print(f"  {session} {date}: SPX column off (ES - SPX = {basis:.1f}); "
          f"using ES mid - {fixed_basis:.2f} as spot")
    return (es_mid - fixed_basis).to_numpy(dtype=float), True


# ---------------------------------------------------------------- Black-Scholes

def bs_price(S, K, T, r, q, vol, is_call):
    """Black-Scholes price; S, q, vol can be numpy arrays (one value per row)."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T
    if is_call:
        return S * np.exp(-q * T) * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)
    return K * np.exp(-r * T) * ndtr(-d2) - S * np.exp(-q * T) * ndtr(-d1)


def bs_implied_vol(price, S, K, T, r, q, is_call, iterations=60):
    """Implied vol by bisection, for a whole column of rows at once.

    price, S, T, q are arrays (one value per row); K is a single strike.
    Returns NaN where the price is missing or outside what any vol can produce.
    """
    low = np.full(len(price), 0.001)
    high = np.full(len(price), 5.0)

    price_low = bs_price(S, K, T, r, q, low, is_call)
    price_high = bs_price(S, K, T, r, q, high, is_call)
    valid = (price > 0) & (price >= price_low) & (price <= price_high)

    for _ in range(iterations):
        middle = (low + high) / 2.0
        price_middle = bs_price(S, K, T, r, q, middle, is_call)
        too_low = price_middle < price
        low = np.where(too_low, middle, low)
        high = np.where(too_low, high, middle)

    vol = (low + high) / 2.0
    vol[~valid] = np.nan
    return vol


# ---------------------------------------------------------------- per-row helpers

def forward_from_parity(strikes, call_mids, put_mids, discount):
    """F = K + (C - P) / D at the strike where |C - P| is smallest (one row)."""
    best_index = -1
    best_gap = None
    for j in range(len(strikes)):
        c = call_mids[j]
        p = put_mids[j]
        if not (c > 0) or not (p > 0):
            continue
        gap = abs(c - p)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_index = j
    if best_index < 0:
        return float("nan")
    return strikes[best_index] + (call_mids[best_index] - put_mids[best_index]) / discount


def interpolate_iv(strikes, ivs, target):
    """Linear interpolation in IV between the two strikes around target."""
    if math.isnan(target):
        return float("nan")
    for i in range(len(strikes) - 1):
        k1 = strikes[i]
        k2 = strikes[i + 1]
        if k1 <= target <= k2:
            if math.isnan(ivs[i]) or math.isnan(ivs[i + 1]):
                return float("nan")
            weight = (target - k1) / (k2 - k1)
            return ivs[i] + weight * (ivs[i + 1] - ivs[i])
    return float("nan")


def mids_from(bid, ask):
    """Mid where the quote is usable, else NaN.

    Usable: bid > 0, ask >= bid, and mid at least MIN_MID (not stuck at the tick).
    """
    mid = (bid + ask) / 2.0
    usable = (bid > 0) & (ask >= bid) & (mid >= MIN_MID)
    return np.where(usable, mid, np.nan)


# ---------------------------------------------------------------- one session

def build_key_strike_series(session, date, r=RISK_FREE_RATE, data_folder=DATA_FOLDER):
    """Key-point implied vols at every snapshot of one session."""
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

    # line everything up on the same timestamps (normally already identical)
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

    # forward and implied dividend yield, row by row
    forward = np.full(len(times), np.nan)
    for row in range(len(times)):
        forward[row] = forward_from_parity(strikes, call_mid[row], put_mid[row], discount[row])
    q = r - np.log(forward / spot) / T

    # implied vol at every strike, one strike column at a time (all rows at once)
    call_iv = np.full(call_mid.shape, np.nan)
    put_iv = np.full(put_mid.shape, np.nan)
    for j in range(len(strikes)):
        call_iv[:, j] = bs_implied_vol(call_mid[:, j], spot, strikes[j], T, r, q, True)
        put_iv[:, j] = bs_implied_vol(put_mid[:, j], spot, strikes[j], T, r, q, False)

    # key points, row by row, using the OTM option at each strike
    records = []
    for row in range(len(times)):
        F = forward[row]
        if math.isnan(F):
            continue

        otm_iv = []
        for j in range(len(strikes)):
            if strikes[j] < F:
                otm_iv.append(put_iv[row, j])
            else:
                otm_iv.append(call_iv[row, j])

        record = {
            "time": times[row],
            "spot": spot[row],
            "forward": F,
            "T": T[row],
            "iv_atm": interpolate_iv(strikes, otm_iv, spot[row]),
        }
        for name, multiple in MONEYNESS_POINTS:
            record[name] = interpolate_iv(strikes, otm_iv, multiple * F)
        records.append(record)

    result = pd.DataFrame(records).set_index("time")
    result["skew_90_atmf"] = result["iv_90"] - result["iv_atmf"]
    result["skew_95_105"] = result["iv_95"] - result["iv_105"]
    result["spot_from_es"] = spot_fixed
    result["expiry"] = expiry

    # put the columns in moneyness order
    ordered = ["spot", "forward", "T", "iv_90", "iv_95", "iv_97_5", "iv_atm",
               "iv_atmf", "iv_102_5", "iv_105", "iv_110",
               "skew_90_atmf", "skew_95_105", "spot_from_es", "expiry"]
    return result[ordered]


# ---------------------------------------------------------------- charts

def plot_key_strikes(series, columns, labels, colors, output_path, title):
    """IV (in %) of the chosen key points over one session."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink = "#0b0b0b"
    muted = "#52514e"
    background = "#fcfcfb"

    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=background)
    ax.set_facecolor(background)

    for i in range(len(columns)):
        ax.plot(series.index, series[columns[i]] * 100.0, color=colors[i],
                linewidth=1.3, label=labels[i])

    ax.set_title(title, loc="left", fontsize=11, color=ink)
    ax.set_ylabel("implied vol (%)", color=muted)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9c8c3")
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.tick_params(colors=muted)
    ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # puts get darker the further out they are; same for calls; ATMF in grey
    downside_columns = ["iv_90", "iv_95", "iv_97_5", "iv_atm", "iv_atmf"]
    downside_labels = ["90% F", "95% F", "97.5% F", "ATM (spot)", "ATMF"]
    downside_colors = ["#08306b", "#2f6fbe", "#8cbde6", "#e08a3c", "#52514e"]

    upside_columns = ["iv_atmf", "iv_102_5", "iv_105", "iv_110"]
    upside_labels = ["ATMF", "102.5% F", "105% F", "110% F"]
    upside_colors = ["#52514e", "#9fd8bf", "#3fa77a", "#0b5a39"]

    for session, day in list_sessions():
        print(f"{session} {day} ...")
        try:
            series = build_key_strike_series(session, day)
        except FileNotFoundError as error:
            print("  skipped:", error)
            continue
        if len(series) == 0:
            print("  no data - skipped")
            continue

        name = f"iv_{session}_{day}"
        series.to_csv(OUTPUT_FOLDER + name + ".csv")

        plot_key_strikes(series, downside_columns, downside_labels, downside_colors,
                         OUTPUT_FOLDER + name + "_downside.png",
                         f"Implied vol, ATM and OTM puts - {session} session starting {day}")
        plot_key_strikes(series, upside_columns, upside_labels, upside_colors,
                         OUTPUT_FOLDER + name + "_upside.png",
                         f"Implied vol, ATMF and OTM calls - {session} session starting {day}")
        print(f"  saved {name}: {len(series)} rows")