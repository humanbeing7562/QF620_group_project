"""
vix_index.py - VIX-style volatility index from SPX option mids (QF620 group project)

Uses the Cboe VIX formula (Session 10 notes) on a single expiry:

    sigma^2 = (2/T) * sum( dK_i / K_i^2 * exp(rT) * Q(K_i) ) - (1/T) * (F/K0 - 1)^2
    index   = 100 * sigma

Model-free: works directly on market mid prices, no SABR involved.
Note: the real VIX interpolates two expiries to 30 days; we only have one
expiry per day, so this is a single-maturity version.
"""
import glob
import math
import os

import numpy as np
import pandas as pd

DATA_FOLDER = "data/"
EVENTS_CSV = "2025 Liberation Day Tariff Main Events.csv"
from risk_free_rate import RISK_FREE_RATE
EXPIRY_TIME = "09:30:00" # AM settlement on the day after the file date

# session clock times (ET); an ETH file is named by the date the session starts
SESSION_TIMES = {
    "RTH": ("09:31:00", "16:15:00"),
    "ETH": ("20:15:00", "09:15:00"),
}

# csv files are big (~7 MB), so keep each one in memory after the first read
_file_cache = {}


def _read_csv_cached(path):
    if path not in _file_cache:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        _file_cache[path] = df.sort_index()
    return _file_cache[path]


def _row_at(df, timestamp, max_gap_seconds):
    """Last row at or before timestamp, or None if there is no recent enough row."""
    if len(df) == 0:
        return None, None
    wanted = pd.Timestamp(timestamp)
    row_time = df.index.asof(wanted)
    if pd.isna(row_time):
        return None, None
    if (wanted - row_time).total_seconds() > max_gap_seconds:
        return None, None
    return df.loc[row_time], row_time


def settlement_time(expiry):
    """Settlement moment for the options in a file.

    SPX monthlies are AM-settled: the file date is the LAST TRADING DAY, and the
    options settle at the 09:30 ET open on the next trading day.
        "20250416" (Wed) -> Thu 2025-04-17 09:30   (Good Friday moved it)
        "20250515" (Thu) -> Fri 2025-05-16 09:30
    """
    last_trading_day = pd.to_datetime(expiry, format="%Y%m%d")
    settle_day = last_trading_day + pd.Timedelta(days=1)
    while settle_day.weekday() >= 5:   # skip Saturday / Sunday
        settle_day = settle_day + pd.Timedelta(days=1)
    return pd.Timestamp(f"{settle_day.date()} {EXPIRY_TIME}")


def time_to_expiry(timestamp, expiry):
    """Years from the snapshot time to settlement (calendar time, ACT/365).

    timestamp : snapshot time, e.g. "2025-04-02 15:45:00" (ET)
    expiry    : date exactly as in the file names, e.g. "20250416"
    """
    expiry_moment = settlement_time(expiry)
    snapshot = pd.Timestamp(timestamp)

    seconds_left = (expiry_moment - snapshot).total_seconds()
    if seconds_left <= 0:
        raise ValueError(f"Snapshot {snapshot} is not before settlement {expiry_moment}")

    return seconds_left / (365.0 * 24.0 * 3600.0)


def find_expiry(session, date, data_folder=DATA_FOLDER):
    """Read the expiry date from the option file names for one session.

    e.g. data/RTH/2025-04-02_call_bid_20250416.csv -> "20250416"
    Raises an error if there is no option file, or more than one expiry.
    """
    pattern = f"{data_folder}{session}/{date}_call_bid_*.csv"
    matches = glob.glob(pattern)
    if len(matches) == 0:
        raise FileNotFoundError(f"No option file matches {pattern}")

    expiries = []
    for path in matches:
        name = os.path.basename(path)            # 2025-04-02_call_bid_20250416.csv
        expiry = name.split("_")[-1].replace(".csv", "")
        if expiry not in expiries:
            expiries.append(expiry)

    if len(expiries) > 1:
        raise ValueError(f"More than one expiry for {session} {date}: {expiries}")
    return expiries[0]


def read_option_mids(session, date, expiry, timestamp, max_gap_seconds=60,
                     data_folder=DATA_FOLDER):
    """Call and put mids at every strike, at the last snapshot <= timestamp.

    session : "RTH" or "ETH"
    date    : file date, e.g. "2025-04-02" (for ETH, the date the session starts)
    expiry  : from the file name, e.g. "20250416"

    A mid is set to NaN when the bid is zero or missing, so those strikes are
    skipped by the index (Cboe also drops zero-bid options).

    Returns (strikes, call_mids, put_mids, row_time), or None if no data.
    """
    rows = {}
    row_time = None
    for opt in ["call", "put"]:
        for side in ["bid", "ask"]:
            path = f"{data_folder}{session}/{date}_{opt}_{side}_{expiry}.csv"
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            row, t = _row_at(_read_csv_cached(path), timestamp, max_gap_seconds)
            if row is None:
                return None
            rows[f"{opt}_{side}"] = row
            row_time = t

    strikes = []
    call_mids = []
    put_mids = []
    for column in rows["call_ask"].index:
        k = float(column)
        mids = {}
        for opt in ["call", "put"]:
            bid = rows[f"{opt}_bid"][column]
            ask = rows[f"{opt}_ask"][column]
            if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask < bid:
                mids[opt] = float("nan")
            else:
                mids[opt] = (bid + ask) / 2.0
        strikes.append(k)
        call_mids.append(mids["call"])
        put_mids.append(mids["put"])

    return strikes, call_mids, put_mids, row_time


def read_real_vix(session, date, timestamp, max_gap_seconds=60, data_folder=DATA_FOLDER):
    """The actual Cboe VIX at timestamp, from the VIX column of <date>_assets.csv."""
    path = f"{data_folder}{session}/{date}_assets.csv"
    row, row_time = _row_at(_read_csv_cached(path), timestamp, max_gap_seconds)
    if row is None:
        return float("nan")
    return float(row["VIX"])


def forward_from_parity(strikes, call_mids, put_mids, r, T):
    """Forward index level from option prices, the way Cboe does it for the VIX.

    Find the strike where |call - put| is smallest, then
    F = K + exp(r * T) * (call - put) at that strike.
    """
    best_index = None
    best_gap = None
    for i in range(len(strikes)):
        if not (call_mids[i] > 0) or not (put_mids[i] > 0):
            continue
        gap = abs(call_mids[i] - put_mids[i])
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_index = i

    k = strikes[best_index]
    return k + math.exp(r * T) * (call_mids[best_index] - put_mids[best_index])


def vix_style_index(strikes, call_mids, put_mids, T, r, forward=None):
    """Model-free volatility index for one expiry (Cboe VIX formula, one maturity).

    sigma^2 = (2/T) * sum( dK_i / K_i^2 * exp(rT) * Q(K_i) ) - (1/T) * (F/K0 - 1)^2

    strikes   : sorted strikes
    call_mids : call mid at each strike
    put_mids  : put mid at each strike
    T, r      : time to expiry (years) and risk-free rate
    forward   : optional; if not given it is backed out from put-call parity

    Strikes whose OTM mid is missing or not positive are skipped.
    Returns (index, details) where index is in VIX units (e.g. 25.3).
    """
    if forward is None:
        forward = forward_from_parity(strikes, call_mids, put_mids, r, T)

    # K0 = first strike at or below the forward
    k0 = None
    for k in strikes:
        if k <= forward:
            k0 = k
    if k0 is None:
        raise ValueError(f"No strike below the forward {forward:.2f}")

    # Q(K): put below K0, call above K0, average of both at K0
    used_strikes = []
    used_q = []
    for i in range(len(strikes)):
        k = strikes[i]
        if k < k0:
            q_value = put_mids[i]
        elif k > k0:
            q_value = call_mids[i]
        else:
            q_value = (put_mids[i] + call_mids[i]) / 2.0

        if q_value is None or not (q_value > 0):
            continue
        used_strikes.append(k)
        used_q.append(q_value)

    # strike intervals: half the gap to the neighbours, one-sided at the ends
    n = len(used_strikes)
    total = 0.0
    for i in range(n):
        if i == 0:
            dK = used_strikes[1] - used_strikes[0]
        elif i == n - 1:
            dK = used_strikes[n - 1] - used_strikes[n - 2]
        else:
            dK = (used_strikes[i + 1] - used_strikes[i - 1]) / 2.0
        k = used_strikes[i]
        total += dK / (k * k) * math.exp(r * T) * used_q[i]

    variance = (2.0 / T) * total - (1.0 / T) * (forward / k0 - 1.0) ** 2
    index = 100.0 * math.sqrt(variance)

    details = {
        "forward": forward,
        "k0": k0,
        "strikes_used": n,
        "variance": variance,
    }
    return index, details


def session_bounds(session, date):
    """Start and end timestamps of a session; ETH ends on a later calendar day."""
    start_clock, end_clock = SESSION_TIMES[session]
    start = pd.Timestamp(f"{date} {start_clock}")
    end = pd.Timestamp(f"{date} {end_clock}")
    if end <= start:
        # overnight session; the next file date tells us the end day
        # (Friday-night and weekend sessions are not in the data)
        end = end + pd.Timedelta(days=1)
    return start, end


def build_vix_series(session, date, step_minutes=5, r=RISK_FREE_RATE,
                     data_folder=DATA_FOLDER):
    """Our VIX-style index and the real VIX every step_minutes through one session.

    The expiry is read from the option file names, so only session and date are needed.

    Returns a DataFrame indexed by time with columns: our_index, real_vix, forward, T.
    Timestamps with no data (gaps, empty files) are skipped.
    """
    expiry = find_expiry(session, date, data_folder=data_folder)
    start, end = session_bounds(session, date)
    times = pd.date_range(start, end, freq=f"{step_minutes}min")

    records = []
    for ts in times:
        snapshot = read_option_mids(session, date, expiry, ts, data_folder=data_folder)
        if snapshot is None:
            continue
        strikes, call_mids, put_mids, row_time = snapshot
        T = time_to_expiry(ts, expiry)
        try:
            index, details = vix_style_index(strikes, call_mids, put_mids, T, r)
        except (ValueError, TypeError):
            continue
        records.append({
            "time": ts,
            "our_index": index,
            "real_vix": read_real_vix(session, date, ts, data_folder=data_folder),
            "forward": details["forward"],
            "T": T,
        })

    if len(records) == 0:
        return pd.DataFrame(columns=["our_index", "real_vix", "forward", "T"])
    return pd.DataFrame(records).set_index("time")


def plot_vix_series(series, output_path="vix_series.png", title=None, events=None):
    """Line chart of our index vs the real VIX over one session.

    events: optional DataFrame from load_events(); events inside the session's
            time range are drawn as dashed lines with a short label.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ours_color = "#2a78d6"
    real_color = "#eb6834"
    ink = "#0b0b0b"
    muted = "#52514e"
    background = "#fcfcfb"

    # Cboe only calculates the VIX overnight from 03:15 ET; before that the
    # column repeats the previous close, so leave those points out
    real = []
    for i in range(len(series)):
        t = series.index[i]
        clock = t.hour * 60 + t.minute
        if clock >= 20 * 60 or clock < 3 * 60 + 15:
            real.append(float("nan"))
        else:
            real.append(series["real_vix"].iloc[i])

    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=background)
    ax.set_facecolor(background)

    ax.plot(series.index, real, color=real_color, linewidth=1.4, label="Cboe VIX (30-day)")
    ax.plot(series.index, series["our_index"], color=ours_color, linewidth=1.4,
            label="Our index (single expiry)")

    if events is not None:
        start = series.index[0]
        end = series.index[-1]
        low = min(series["our_index"].min(), series["real_vix"].min())
        for i in range(len(events)):
            when = events.loc[i, "time"]
            if when < start or when > end:
                continue
            label = f"{when:%H:%M} {events.loc[i, 'Country_or_Actor']}: {events.loc[i, 'Event_Type']}"
            ax.axvline(when, color=muted, linewidth=1, linestyle="--", zorder=0)
            ax.text(when, low, " " + label, color=muted, fontsize=8, rotation=90,
                    va="bottom", ha="right")

    if title is None:
        title = "VIX-style index vs Cboe VIX"
    ax.set_title(title, loc="left", fontsize=11, color=ink)
    ax.set_ylabel("index level", color=muted)
    ax.legend(frameon=False, loc="best")

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


def list_sessions(data_folder=DATA_FOLDER):
    """All (session, date) pairs that have an assets file, in time order."""
    sessions = []
    for session in ["RTH", "ETH"]:
        pattern = f"{data_folder}{session}/*_assets.csv"
        for path in glob.glob(pattern):
            date = os.path.basename(path).split("_")[0]
            start, end = session_bounds(session, date)
            sessions.append((start, session, date))
    sessions.sort()

    ordered = []
    for start, session, date in sessions:
        ordered.append((session, date))
    return ordered


def build_full_series(step_minutes=5, r=RISK_FREE_RATE, data_folder=DATA_FOLDER):
    """Run build_vix_series over every session and stitch them in time order.

    Adds a 'session' column (e.g. "RTH 2025-04-02") and an 'expiry' column.
    Sessions with no usable data (e.g. the empty ETH 2025-04-09 files) are skipped.
    """
    pieces = []
    for session, date in list_sessions(data_folder):
        try:
            expiry = find_expiry(session, date, data_folder=data_folder)
            series = build_vix_series(session, date, step_minutes=step_minutes,
                                      r=r, data_folder=data_folder)
        except (FileNotFoundError, ValueError) as error:
            print(f"skipping {session} {date}: {error}")
            continue
        if len(series) == 0:
            print(f"skipping {session} {date}: no data")
            continue
        series["session"] = f"{session} {date}"
        series["expiry"] = expiry
        pieces.append(series)
        print(f"done {session} {date}: {len(series)} points")

    full = pd.concat(pieces)
    return full.sort_index()


def load_events(events_csv=EVENTS_CSV):
    """Event list with a proper ET timestamp column."""
    events = pd.read_csv(events_csv, encoding="utf-8-sig")
    times = []
    for i in range(len(events)):
        times.append(pd.Timestamp(f"{events.loc[i, 'Date_ET']} {events.loc[i, 'Time_ET']}"))
    events["time"] = times
    return events


def _value_near(series, column, when, max_gap_minutes):
    """Last value at or before 'when', if it is no older than max_gap_minutes."""
    row_time = series.index.asof(when)
    if pd.isna(row_time):
        return float("nan"), None
    if (when - row_time).total_seconds() > max_gap_minutes * 60:
        return float("nan"), None
    return float(series.loc[row_time, column]), row_time


def event_reactions(series, events, after_minutes=30, step_minutes=5):
    """Index just before each event vs up to after_minutes later.

    'before' is the last point at or before the event time.
    'after'  is the last point at or before event + after_minutes, in the SAME
             session; if the session closes sooner, the window is cut short
             and 'window_min' shows how many minutes it actually covers.
    Events that fall outside the trading sessions get NaN.
    """
    rows = []
    for i in range(len(events)):
        event_time = events.loc[i, "time"]
        before, before_time = _value_near(series, "our_index", event_time, step_minutes)

        after = float("nan")
        window = float("nan")
        if before_time is not None:
            session = series.loc[before_time, "session"]
            same_session = series[series["session"] == session]
            target = event_time + pd.Timedelta(minutes=after_minutes)
            after_time = same_session.index.asof(target)
            if not pd.isna(after_time) and after_time > before_time:
                after = float(same_session.loc[after_time, "our_index"])
                window = (after_time - event_time).total_seconds() / 60.0
            else:
                before = float("nan")

        rows.append({
            "time": event_time,
            "actor": events.loc[i, "Country_or_Actor"],
            "event_type": events.loc[i, "Event_Type"],
            "index_before": round(before, 2),
            "index_after": round(after, 2),
            "change": round(after - before, 2),
            "window_min": window,
        })
    return pd.DataFrame(rows)


def plot_full_series(series, events, output_path="vix_full_period.png", label_times=None):
    """Our index vs Cboe VIX over 1-10 Apr, with every event as a thin vertical line.

    label_times: optional dict {event time string: short label} for the few
    events worth naming on the chart.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ours_color = "#2a78d6"
    real_color = "#eb6834"
    ink = "#0b0b0b"
    muted = "#52514e"
    background = "#fcfcfb"

    # break the lines between sessions so they don't draw across closed hours
    times = list(series.index)
    ours = []
    real = []
    plot_times = []
    for i in range(len(times)):
        if i > 0 and series["session"].iloc[i] != series["session"].iloc[i - 1]:
            plot_times.append(times[i - 1] + pd.Timedelta(minutes=1))
            ours.append(float("nan"))
            real.append(float("nan"))
        plot_times.append(times[i])
        ours.append(series["our_index"].iloc[i])
        # Cboe only calculates the VIX overnight from 03:15 ET; before that the
        # column just repeats the previous close, so leave it out of the chart
        clock = times[i].hour * 60 + times[i].minute
        frozen = clock >= 20 * 60 or clock < 3 * 60 + 15
        if frozen:
            real.append(float("nan"))
        else:
            real.append(series["real_vix"].iloc[i])

    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=background)
    ax.set_facecolor(background)

    for i in range(len(events)):
        ax.axvline(events.loc[i, "time"], color="#d6d5d0", linewidth=0.8, zorder=0)

    ax.plot(plot_times, real, color=real_color, linewidth=1.6, label="Cboe VIX (30-day)")
    ax.plot(plot_times, ours, color=ours_color, linewidth=1.6, label="Our index (single expiry)")

    top = max(series["our_index"].max(), series["real_vix"].max())
    if label_times is not None:
        for when, label in label_times.items():
            ax.axvline(pd.Timestamp(when), color=muted, linewidth=1, linestyle="--", zorder=1)
            ax.text(pd.Timestamp(when), top * 1.02, label, color=muted, fontsize=9,
                    rotation=0, ha="center")

    ax.set_title("VIX-style index vs Cboe VIX, 1-10 April 2025 (grey lines = events)",
                 loc="left", fontsize=11, color=ink, pad=22)
    ax.set_ylabel("index level", color=muted)
    ax.set_ylim(0, top * 1.12)
    ax.legend(frameon=False, loc="lower center")

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


def build_vix_series_every_row(session, date, r=RISK_FREE_RATE, data_folder=DATA_FOLDER):
    """Our index and the real VIX at EVERY snapshot in one session (about every 3 s).

    Same output columns as build_vix_series, but instead of looking up chosen
    times it walks through the rows of the files directly, which is much faster
    at full resolution.
    """
    expiry = find_expiry(session, date, data_folder=data_folder)

    frames = {}
    for opt in ["call", "put"]:
        for side in ["bid", "ask"]:
            path = f"{data_folder}{session}/{date}_{opt}_{side}_{expiry}.csv"
            frames[f"{opt}_{side}"] = _read_csv_cached(path)
    assets = _read_csv_cached(f"{data_folder}{session}/{date}_assets.csv")

    times = frames["call_ask"].index
    if len(times) == 0:
        return pd.DataFrame(columns=["our_index", "real_vix", "forward", "T"])

    # line every file up on the same timestamps (they normally already match)
    for name in frames:
        if not frames[name].index.equals(times):
            frames[name] = frames[name].reindex(times, method="ffill")
    if not assets.index.equals(times):
        assets = assets.reindex(times, method="ffill")

    columns = list(frames["call_ask"].columns)
    strikes = []
    for column in columns:
        strikes.append(float(column))

    call_bid = frames["call_bid"].to_numpy(dtype=float)
    call_ask = frames["call_ask"].to_numpy(dtype=float)
    put_bid = frames["put_bid"].to_numpy(dtype=float)
    put_ask = frames["put_ask"].to_numpy(dtype=float)
    real_vix = assets["VIX"].to_numpy(dtype=float)

    records = []
    for row in range(len(times)):
        call_mids = []
        put_mids = []
        for j in range(len(strikes)):
            cb, ca = call_bid[row, j], call_ask[row, j]
            pb, pa = put_bid[row, j], put_ask[row, j]
            if cb > 0 and ca >= cb:
                call_mids.append((cb + ca) / 2.0)
            else:
                call_mids.append(float("nan"))
            if pb > 0 and pa >= pb:
                put_mids.append((pb + pa) / 2.0)
            else:
                put_mids.append(float("nan"))

        T = time_to_expiry(times[row], expiry)
        try:
            index, details = vix_style_index(strikes, call_mids, put_mids, T, r)
        except (ValueError, TypeError):
            continue

        records.append({
            "time": times[row],
            "our_index": index,
            "real_vix": real_vix[row],
            "forward": details["forward"],
            "T": T,
        })

    return pd.DataFrame(records).set_index("time")


def build_full_series_every_row(r=RISK_FREE_RATE, data_folder=DATA_FOLDER):
    """Full-resolution version of build_full_series: every snapshot, every session."""
    pieces = []
    for session, date in list_sessions(data_folder):
        try:
            expiry = find_expiry(session, date, data_folder=data_folder)
            series = build_vix_series_every_row(session, date, r=r, data_folder=data_folder)
        except (FileNotFoundError, ValueError) as error:
            print(f"skipping {session} {date}: {error}")
            continue
        if len(series) == 0:
            print(f"skipping {session} {date}: no data")
            continue
        series["session"] = f"{session} {date}"
        series["expiry"] = expiry
        pieces.append(series)
        print(f"done {session} {date}: {len(series)} points")
        _file_cache.clear()   # free memory before the next session

    full = pd.concat(pieces)
    return full.sort_index()


if __name__ == "__main__":
    output_folder = "vix_output/"
    os.makedirs(output_folder, exist_ok=True)
    events = load_events()

    for session, day in list_sessions():
        series = build_vix_series_every_row(session, day)
        if len(series) == 0:
            print("no data for", session, day, "- skipped")
            continue

        name = "vix_" + session + "_" + day
        series.to_csv(output_folder + name + ".csv")
        plot_vix_series(series, output_folder + name + ".png",
                        title="VIX-style index vs Cboe VIX, " + session + " session starting " + day
                              + " (every snapshot)",
                        events=events)
        print("saved", name, "-", len(series), "rows")
        _file_cache.clear()   # free memory before the next session