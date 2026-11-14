"""
spx_es_prices.py - SPX and ES price behaviour over 1-10 April 2025 (QF620 group project)

Reads the <date>_assets.csv files for every RTH and ETH session and makes:
    - one chart of SPX over the whole period, with event lines
    - a table of open / high / low / close per session

Note: during ETH the SPX column moves too; it sits a fixed distance below the
ES mid within a session, so it looks derived from ES futures overnight.
"""
import glob
import os

import pandas as pd

DATA_FOLDER = "data/"
EVENTS_CSV = "2025 Liberation Day Tariff Main Events.csv"

SESSION_TIMES = {
    "RTH": ("09:31:00", "16:15:00"),
    "ETH": ("20:15:00", "09:15:00"),
}


def session_start(session, date):
    start_clock = SESSION_TIMES[session][0]
    return pd.Timestamp(f"{date} {start_clock}")


def list_sessions(data_folder=DATA_FOLDER):
    """All (session, date) pairs that have an assets file, in time order."""
    found = []
    for session in ["RTH", "ETH"]:
        for path in glob.glob(f"{data_folder}{session}/*_assets.csv"):
            date = os.path.basename(path).split("_")[0]
            found.append((session_start(session, date), session, date))
    found.sort()

    ordered = []
    for start, session, date in found:
        ordered.append((session, date))
    return ordered


def load_prices(data_folder=DATA_FOLDER):
    """Every assets row from every session, stitched in time order.

    Columns: SPX, ES_BID, ES_ASK, VIX, ES_MID, session.
    Empty sessions (e.g. ETH 2025-04-09) are skipped.
    """
    pieces = []
    for session, date in list_sessions(data_folder):
        path = f"{data_folder}{session}/{date}_assets.csv"
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) == 0:
            print(f"skipping {session} {date}: no data")
            continue
        df = df.sort_index()
        df["ES_MID"] = (df["ES_BID"] + df["ES_ASK"]) / 2.0
        df["session"] = f"{session} {date}"
        pieces.append(df)

    prices = pd.concat(pieces)
    return prices.sort_index()


def session_summary(prices):
    """Open / high / low / close of SPX per session, plus the % change."""
    rows = []
    for name, group in prices.groupby("session", sort=False):
        spx = group["SPX"]
        first = float(spx.iloc[0])
        last = float(spx.iloc[-1])
        rows.append({
            "session": name,
            "start": group.index[0],
            "end": group.index[-1],
            "open": first,
            "high": float(spx.max()),
            "low": float(spx.min()),
            "close": last,
            "change_pct": round(100.0 * (last / first - 1.0), 2),
            "es_minus_spx": round(float((group["ES_MID"] - spx).mean()), 2),
        })
    return pd.DataFrame(rows)


def load_events(events_csv=EVENTS_CSV):
    """Event list with a proper ET timestamp column."""
    events = pd.read_csv(events_csv, encoding="utf-8-sig")
    times = []
    for i in range(len(events)):
        times.append(pd.Timestamp(f"{events.loc[i, 'Date_ET']} {events.loc[i, 'Time_ET']}"))
    events["time"] = times
    return events


def plot_spx(prices, output_path="spx_full_period.png", events=None, label_times=None):
    """SPX over the whole period; lines are broken between sessions.

    events      : optional DataFrame from load_events(), drawn as thin grey lines
    label_times : optional dict {time string: short label} for a few key events
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    line_color = "#2a78d6"
    ink = "#0b0b0b"
    muted = "#52514e"
    background = "#fcfcfb"

    # insert a gap between sessions so the line doesn't cross closed hours
    times = list(prices.index)
    sessions = list(prices["session"])
    spx = list(prices["SPX"])
    plot_times = []
    plot_spx = []
    for i in range(len(times)):
        if i > 0 and sessions[i] != sessions[i - 1]:
            plot_times.append(times[i - 1] + pd.Timedelta(seconds=1))
            plot_spx.append(float("nan"))
        plot_times.append(times[i])
        plot_spx.append(spx[i])

    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=background)
    ax.set_facecolor(background)

    if events is not None:
        for i in range(len(events)):
            ax.axvline(events.loc[i, "time"], color="#d6d5d0", linewidth=0.8, zorder=0)

    ax.plot(plot_times, plot_spx, color=line_color, linewidth=1.2)

    top = prices["SPX"].max()
    bottom = prices["SPX"].min()
    if label_times is not None:
        for when, label in label_times.items():
            ax.axvline(pd.Timestamp(when), color=muted, linewidth=1, linestyle="--", zorder=1)
            ax.text(pd.Timestamp(when), top + (top - bottom) * 0.04, label,
                    color=muted, fontsize=9, ha="center")

    ax.set_title("SPX, 1-10 April 2025 (RTH and ETH sessions; grey lines = events)",
                 loc="left", fontsize=11, color=ink, pad=22)
    ax.set_ylabel("SPX", color=muted)
    ax.set_ylim(bottom - (top - bottom) * 0.05, top + (top - bottom) * 0.12)

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


def plot_spx_session(group, output_path, title, events=None):
    """SPX over one session, with any events inside the session as dashed lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    line_color = "#2a78d6"
    ink = "#0b0b0b"
    muted = "#52514e"
    background = "#fcfcfb"

    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=background)
    ax.set_facecolor(background)

    ax.plot(group.index, group["SPX"], color=line_color, linewidth=1.4)

    if events is not None:
        start = group.index[0]
        end = group.index[-1]
        low = group["SPX"].min()
        for i in range(len(events)):
            when = events.loc[i, "time"]
            if when < start or when > end:
                continue
            label = f"{when:%H:%M} {events.loc[i, 'Country_or_Actor']}: {events.loc[i, 'Event_Type']}"
            ax.axvline(when, color=muted, linewidth=1, linestyle="--", zorder=0)
            ax.text(when, low, " " + label, color=muted, fontsize=8, rotation=90,
                    va="bottom", ha="right")

    ax.set_title(title, loc="left", fontsize=11, color=ink)
    ax.set_ylabel("SPX", color=muted)

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
    output_folder = "spx_es_output/"
    os.makedirs(output_folder, exist_ok=True)

    prices = load_prices()

    summary = session_summary(prices)
    pd.set_option("display.width", 160)
    print(summary.to_string(index=False))
    summary.to_csv(output_folder + "spx_session_summary.csv", index=False)

    if os.path.exists(EVENTS_CSV):
        events = load_events()
    else:
        print("events file not found, charts will have no event lines")
        events = None

    for name, group in prices.groupby("session", sort=False):
        session, day = name.split(" ")
        change = 100.0 * (group["SPX"].iloc[-1] / group["SPX"].iloc[0] - 1.0)
        title = f"SPX, {session} session starting {day} ({change:+.2f}% open to close)"

        # flag sessions where SPX doesn't line up with ES (normally ~35-43 below)
        basis = float((group["ES_MID"] - group["SPX"]).mean())
        if abs(basis) > 100:
            title = title + f"  [check: ES - SPX = {basis:.0f}]"

        path = plot_spx_session(group, output_folder + "spx_es_" + session + "_" + day + ".png",
                                title, events)
        print("saved", path)