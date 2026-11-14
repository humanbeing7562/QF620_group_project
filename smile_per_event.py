"""
smile_gif.py - animated implied vol smile around each event (QF620 group project)

For every event in the events file:
    - take one snapshot per minute from 10 minutes before to 10 minutes after
    - at each snapshot: mids -> parity forward -> Black-Scholes IV at every strike
      (OTM option: put below F, call at or above F)
    - make a GIF with one frame per minute; each frame shows the current smile
      and, in grey, the smile 10 minutes before the event for reference
    - also a static PNG with all the minutes overlaid (for the printed report)
    - save the numbers to a CSV as well

The x-axis is moneyness K / F, so smiles stay comparable while SPX moves.
Minutes that fall outside the session (e.g. after 16:15) are skipped, and events
with no data at all (weekend, the empty ETH 2025-04-09 session) are skipped.

Uses the helpers in key_strike_iv.py, so run it from the repo folder.
Output goes to smile_output/.
"""
import math
import os

import numpy as np
import pandas as pd
from risk_free_rate import RISK_FREE_RATE
from key_strike_iv import (DATA_FOLDER, SESSION_TIMES, bs_implied_vol,
                           corrected_spot, find_expiry, forward_from_parity, list_sessions,
                           mids_from, read_csv, settlement_time)

EVENTS_CSV = "2025 Liberation Day Tariff Main Events.csv"
OUTPUT_FOLDER = "smile_output/"
MINUTES_BEFORE = 10
MINUTES_AFTER = 10
MAX_GAP_SECONDS = 60          # snapshot must be at most this old
FRAME_SECONDS = 0.6           # how long each minute is shown in the GIF
MONEYNESS_RANGE = (0.85, 1.15)


# ---------------------------------------------------------------- which session

def find_session(timestamp, data_folder=DATA_FOLDER):
    """(session, file date) whose time range contains the timestamp, or None."""
    t = pd.Timestamp(timestamp)
    for session, date in list_sessions(data_folder):
        start_clock, end_clock = SESSION_TIMES[session]
        start = pd.Timestamp(f"{date} {start_clock}")
        end = pd.Timestamp(f"{date} {end_clock}")
        if end <= start:
            end = end + pd.Timedelta(days=1)    # overnight session
        if start <= t <= end:
            return session, date
    return None


def load_session(session, date, data_folder=DATA_FOLDER):
    """All the files of one session, lined up on the same timestamps."""
    expiry = find_expiry(session, date, data_folder)
    folder = f"{data_folder}{session}/"
    data = {
        "expiry": expiry,
        "call_bid": read_csv(f"{folder}{date}_call_bid_{expiry}.csv"),
        "call_ask": read_csv(f"{folder}{date}_call_ask_{expiry}.csv"),
        "put_bid": read_csv(f"{folder}{date}_put_bid_{expiry}.csv"),
        "put_ask": read_csv(f"{folder}{date}_put_ask_{expiry}.csv"),
        "assets": read_csv(f"{folder}{date}_assets.csv"),
    }
    times = data["call_ask"].index
    for name in ["call_bid", "put_bid", "put_ask", "assets"]:
        if not data[name].index.equals(times):
            data[name] = data[name].reindex(times, method="ffill")
    if len(times) > 0:
        data["spot"], _ = corrected_spot(data["assets"], session, date, data_folder)
    return data


# ---------------------------------------------------------------- one smile

def smile_at(data, timestamp, r=RISK_FREE_RATE):
    """Smile at the last snapshot <= timestamp: DataFrame of strike, moneyness, iv.

    Returns (smile, row_time, forward), or (None, None, None) if there is no
    snapshot within MAX_GAP_SECONDS.
    """
    times = data["call_ask"].index
    wanted = pd.Timestamp(timestamp)
    row_time = times.asof(wanted)
    if pd.isna(row_time) or (wanted - row_time).total_seconds() > MAX_GAP_SECONDS:
        return None, None, None
    row = times.get_loc(row_time)

    strikes = []
    for column in data["call_ask"].columns:
        strikes.append(float(column))

    call_mid = mids_from(data["call_bid"].iloc[row].to_numpy(dtype=float),
                         data["call_ask"].iloc[row].to_numpy(dtype=float))
    put_mid = mids_from(data["put_bid"].iloc[row].to_numpy(dtype=float),
                        data["put_ask"].iloc[row].to_numpy(dtype=float))

    T = (settlement_time(data["expiry"]) - row_time).total_seconds() / (365.0 * 24.0 * 3600.0)
    forward = forward_from_parity(strikes, call_mid, put_mid, math.exp(-r * T))
    if math.isnan(forward):
        return None, None, None
    spot = data["spot"][row]
    q = r - math.log(forward / spot) / T

    records = []
    for j in range(len(strikes)):
        k = strikes[j]
        if k < forward:
            price = put_mid[j]
            is_call = False
        else:
            price = call_mid[j]
            is_call = True
        iv = bs_implied_vol(np.array([price]), np.array([spot]), k, np.array([T]),
                            r, np.array([q]), is_call)[0]
        if math.isnan(iv):
            continue
        records.append({"strike": k, "moneyness": k / forward, "iv": iv})

    return pd.DataFrame(records), row_time, forward


# ---------------------------------------------------------------- one event

def event_smiles(event_time, data_folder=DATA_FOLDER):
    """Smiles for every minute in the window around one event (long-format DataFrame)."""
    found = find_session(event_time, data_folder)
    if found is None:
        return None, "outside the trading sessions"
    session, date = found
    data = load_session(session, date, data_folder)
    if len(data["call_ask"]) == 0:
        return None, f"{session} {date} has no data"

    pieces = []
    for offset in range(-MINUTES_BEFORE, MINUTES_AFTER + 1):
        when = event_time + pd.Timedelta(minutes=offset)
        smile, row_time, forward = smile_at(data, when)
        if smile is None or len(smile) == 0:
            continue
        smile["minute"] = offset
        smile["time"] = when
        smile["row_time"] = row_time
        smile["forward"] = forward
        pieces.append(smile)

    if len(pieces) == 0:
        return None, "no snapshots in the window"
    result = pd.concat(pieces, ignore_index=True)
    result["session"] = f"{session} {date}"
    return result, ""


def make_gif(smiles, event_time, label, output_path):
    """One frame per minute; grey line = first minute of the window, for reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    before_color = "#2a78d6"
    after_color = "#eb6834"
    reference_color = "#b4b2a9"
    ink = "#0b0b0b"
    muted = "#52514e"
    background = "#fcfcfb"

    minutes = []
    for m in smiles["minute"]:
        if m not in minutes:
            minutes.append(m)
    minutes.sort()

    # fixed axes for every frame, so movement is real movement
    inside = smiles[(smiles["moneyness"] >= MONEYNESS_RANGE[0]) &
                    (smiles["moneyness"] <= MONEYNESS_RANGE[1])]
    if len(inside) == 0:
        inside = smiles
    y_low = inside["iv"].min() * 100.0
    y_high = inside["iv"].max() * 100.0
    pad = (y_high - y_low) * 0.08

    first = smiles[smiles["minute"] == minutes[0]]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=background)
    ax.set_facecolor(background)
    ax.plot(first["moneyness"] * 100.0, first["iv"] * 100.0, color=reference_color,
            linewidth=1.5, label=f"{minutes[0]:+d} min (reference)")
    line, = ax.plot([], [], color=before_color, linewidth=2.2, marker="o", markersize=3)
    ax.axvline(100.0, color=muted, linewidth=0.8, linestyle=":")

    ax.set_xlim(MONEYNESS_RANGE[0] * 100.0, MONEYNESS_RANGE[1] * 100.0)
    ax.set_ylim(y_low - pad, y_high + pad)
    ax.set_xlabel("strike / forward (%)", color=muted)
    ax.set_ylabel("implied vol (%)", color=muted)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9c8c3")
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.tick_params(colors=muted)
    ax.grid(color="#e8e7e3", linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle(label, x=0.07, ha="left", fontsize=10, color=ink)
    title = ax.set_title("", loc="left", fontsize=11, color=ink)

    def draw(frame):
        m = minutes[frame]
        smile = smiles[smiles["minute"] == m]
        line.set_data(smile["moneyness"] * 100.0, smile["iv"] * 100.0)
        if m < 0:
            line.set_color(before_color)
        else:
            line.set_color(after_color)
        when = smile["time"].iloc[0]
        forward = smile["forward"].iloc[0]
        title.set_text(f"{when:%Y-%m-%d %H:%M} ET   ({m:+d} min from event)   F = {forward:,.1f}")
        return line, title

    # hold the last frame a bit longer by repeating it
    frame_order = list(range(len(minutes))) + [len(minutes) - 1] * 4
    animation = FuncAnimation(fig, lambda i: draw(frame_order[i]),
                              frames=len(frame_order), blit=False)
    animation.save(output_path, writer=PillowWriter(fps=1.0 / FRAME_SECONDS))
    plt.close(fig)
    return output_path


def make_overlay(smiles, label, output_path):
    """All minutes on one static chart.

    Before the event: light to dark blue as the event gets closer.
    The event minute: black.
    After the event: light to dark orange/red as time goes on.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ink = "#0b0b0b"
    muted = "#52514e"
    background = "#fcfcfb"
    before_ramp = LinearSegmentedColormap.from_list("before", ["#c6dbef", "#08306b"])
    after_ramp = LinearSegmentedColormap.from_list("after", ["#fdd0a2", "#7f2704"])

    minutes = []
    for m in smiles["minute"]:
        if m not in minutes:
            minutes.append(m)
    minutes.sort()

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=background)
    ax.set_facecolor(background)

    for m in minutes:
        smile = smiles[smiles["minute"] == m]
        if m < 0:
            color = before_ramp((m + MINUTES_BEFORE) / max(MINUTES_BEFORE - 1, 1))
            width = 1.0
            z = 1
        elif m == 0:
            color = ink
            width = 2.0
            z = 3
        else:
            color = after_ramp((m - 1) / max(MINUTES_AFTER - 1, 1))
            width = 1.0
            z = 2
        ax.plot(smile["moneyness"] * 100.0, smile["iv"] * 100.0, color=color,
                linewidth=width, zorder=z)

    # legend: just the three groups
    ax.plot([], [], color="#4a90d0", linewidth=2, label=f"-{MINUTES_BEFORE} to -1 min (light to dark)")
    ax.plot([], [], color=ink, linewidth=2, label="event minute")
    ax.plot([], [], color="#d9611c", linewidth=2, label=f"+1 to +{MINUTES_AFTER} min (light to dark)")
    ax.legend(frameon=False, loc="upper right")

    ax.axvline(100.0, color=muted, linewidth=0.8, linestyle=":")
    ax.set_xlim(MONEYNESS_RANGE[0] * 100.0, MONEYNESS_RANGE[1] * 100.0)
    ax.set_xlabel("strike / forward (%)", color=muted)
    ax.set_ylabel("implied vol (%)", color=muted)
    ax.set_title(label, loc="left", fontsize=10, color=ink)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9c8c3")
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.tick_params(colors=muted)
    ax.grid(color="#e8e7e3", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def load_events(events_csv=EVENTS_CSV):
    events = pd.read_csv(events_csv, encoding="utf-8-sig")
    times = []
    for i in range(len(events)):
        times.append(pd.Timestamp(f"{events.loc[i, 'Date_ET']} {events.loc[i, 'Time_ET']}"))
    events["time"] = times
    return events


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    events = load_events()

    # events at the same minute (e.g. two at 2025-04-09 00:01) share one GIF
    done_times = []
    for i in range(len(events)):
        event_time = events.loc[i, "time"]
        if event_time in done_times:
            continue
        done_times.append(event_time)

        same_time = events[events["time"] == event_time]
        names = []
        for j in same_time.index:
            names.append(f"{same_time.loc[j, 'Country_or_Actor']}: {same_time.loc[j, 'Event_Type']}")
        label = f"{event_time:%d %b %H:%M} ET - " + " / ".join(names)

        smiles, reason = event_smiles(event_time)
        if smiles is None:
            print(f"skipped {label}: {reason}")
            continue

        name = f"smile_{event_time:%Y-%m-%d_%H%M}"
        smiles.to_csv(OUTPUT_FOLDER + name + ".csv", index=False)
        make_gif(smiles, event_time, label, OUTPUT_FOLDER + name + ".gif")
        make_overlay(smiles, label, OUTPUT_FOLDER + name + ".png")
        minutes_found = smiles["minute"].nunique()
        print(f"saved {name}: {minutes_found} minutes ({smiles['session'].iloc[0]})")