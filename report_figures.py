"""
report_figures.py - figures for the QF620 report, following report-outline.md

    Figure 1   overview 1-10 Apr: SPX & ES, VIX-style index vs Cboe VIX, ATMF IV,
               25- and 35-delta risk reversals
    Figure 2-6 one figure per graphed event (or event cluster), smiles against delta:
                 left  : (a) SPX & ES, (b) VIX-style vs Cboe VIX,
                         (c) IV at 25/35-delta puts, ATMF, 35/25-delta calls
                 right : (d) smile every minute -10..+10, (e) BL density at -10 and +10 min,
                         one (d)+(e) pair per event
    Weekend    smile by delta across the 5-6 Apr weekend and expiry roll (Section 3.3)
    Period end risk-neutral densities overlaid

Inputs: key_delta_output/, key_strike_output/ (spot only), vix_output/, data/*/<date>_assets.csv, smile_output/,
event_density_output/densities_per_minute.csv, period_end_densities.csv
Output: report_figures/fig*.png
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


OUTPUT_FOLDER = "report_figures/"
EVENTS_CSV = "2025 Liberation Day Tariff Main Events.csv"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e8e7e3"
AXIS = "#c9c8c3"
EVENT_LINE = "#8a8984"

BEFORE_RAMP = LinearSegmentedColormap.from_list("before", ["#b7d3f6", "#104281"])
AFTER_RAMP = LinearSegmentedColormap.from_list("after", ["#fbd3c0", "#8a2e0b"])

GAP_WIDTH = 25.0   # minutes of x-axis used to show a break between sessions

def pick_font():
    """First of these fonts that is installed (Inter is not on most Windows machines)."""
    from matplotlib import font_manager
    installed = set()
    for font in font_manager.fontManager.ttflist:
        installed.add(font.name)
    for name in ["Inter", "Arial", "Calibri", "DejaVu Sans"]:
        if name in installed:
            return name
    return "sans-serif"


plt.rcParams.update({
    "font.family": pick_font(),
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.facecolor": "white",
    "legend.edgecolor": "white",
    "savefig.dpi": 220,
})


# ---------------------------------------------------------------- data

def load_folder(pattern):
    frames = []
    for path in sorted(glob.glob(pattern)):
        frames.append(pd.read_csv(path, index_col=0, parse_dates=True))
    data = pd.concat(frames)
    data = data[~data.index.duplicated(keep="last")]
    return data.sort_index()


def load_all():
    iv = load_folder("key_strike_output/iv_*.csv")          # spot (ES-corrected) for panel (a)
    kd = load_folder("key_delta_output/iv_delta_*.csv")     # IVs by delta, risk reversals
    vix = load_folder("vix_output/vix_*.csv")
    assets = load_folder("data/*/*_assets.csv")
    es = (assets["ES_BID"] + assets["ES_ASK"]) / 2.0
    # Cboe does not calculate the VIX between 16:15 and 03:15, so the column just
    # repeats the last value overnight; blank it out
    clock = vix.index.hour * 60 + vix.index.minute
    overnight = (clock > 16 * 60 + 15) | (clock < 3 * 60 + 15)
    vix.loc[overnight, "real_vix"] = np.nan
    densities = pd.read_csv("event_density_output/densities_per_minute.csv")
    events = pd.read_csv(EVENTS_CSV, encoding="utf-8-sig")
    event_times = []
    for i in range(len(events)):
        event_times.append(pd.Timestamp(f"{events['Date_ET'][i]} {events['Time_ET'][i]}"))
    return iv, kd, vix, es, densities, event_times


# ---------------------------------------------------------------- broken time axis

class TimeAxis:
    """Maps timestamps to x positions with the gaps between segments squeezed out."""

    def __init__(self, segments):
        self.segments = []
        offset = 0.0
        for start, end in segments:
            start = pd.Timestamp(start)
            end = pd.Timestamp(end)
            self.segments.append((start, end, offset))
            offset += (end - start).total_seconds() / 60.0 + GAP_WIDTH

    def x(self, t):
        t = pd.Timestamp(t)
        for start, end, offset in self.segments:
            if start <= t <= end:
                return offset + (t - start).total_seconds() / 60.0
        return None

    def xs(self, index):
        start0, _, _ = self.segments[0]
        positions = []
        for t in index:
            positions.append(self.x(t))
        return np.array(positions, dtype=float)

    def plot(self, ax, series, **style):
        """Plot a time series segment by segment, so lines don't cross the gaps."""
        label = style.pop("label", None)
        first = True
        last_point = None
        for start, end, offset in self.segments:
            # keep the NaNs so the line breaks where data is missing
            # (e.g. the 25-delta put when no strike is that far out)
            piece = series[start:end]
            if piece.notna().sum() == 0:
                continue
            x = offset + (piece.index - start).total_seconds() / 60.0
            ax.plot(x, piece.values, label=label if first else None, **style)
            first = False
            valid = piece.dropna()
            last_x = offset + (valid.index[-1] - start).total_seconds() / 60.0
            last_point = (last_x, valid.values[-1])
        return last_point

    def finish(self, ax, tick_every_minutes, label_format, show_labels=True):
        ticks = []
        labels = []
        for start, end, offset in self.segments:
            t = start.ceil(f"{tick_every_minutes}min")
            while t <= end:
                ticks.append(self.x(t))
                labels.append(t.strftime(label_format))
                t = t + pd.Timedelta(minutes=tick_every_minutes)
        ax.set_xticks(ticks)
        if show_labels:
            ax.set_xticklabels(labels)
        else:
            ax.set_xticklabels([])
        last_start, last_end, last_offset = self.segments[-1]
        ax.set_xlim(0, last_offset + (last_end - last_start).total_seconds() / 60.0)
        # shade the squeezed gaps
        for i in range(len(self.segments) - 1):
            start, end, offset = self.segments[i]
            gap_from = offset + (end - start).total_seconds() / 60.0
            ax.axvspan(gap_from, gap_from + GAP_WIDTH, color="#f1f0ec", linewidth=0, zorder=0)

    def event_lines(self, ax, times, color=EVENT_LINE, style="--", width=0.8):
        for t in times:
            x = self.x(t)
            if x is not None:
                ax.axvline(x, color=color, linestyle=style, linewidth=width, zorder=3)


# ---------------------------------------------------------------- panels

def direct_labels(ax, points):
    """Write each series name just right of its last point (instead of a legend)."""
    # sort by height and push labels apart so they don't overlap
    usable = []
    for name, point in points:
        if point is not None:
            usable.append((point[1], name, point))
    usable.sort()
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    min_gap = 0.11 * y_range
    placed = []
    for value, name, point in usable:
        y = value
        if len(placed) > 0 and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        ax.annotate(name, xy=point, xytext=(point[0], y), textcoords="data",
                    va="center", fontsize=6, color=INK, annotation_clip=False)
    # small offset to the right of the line end
    for text in ax.texts[-len(usable):]:
        text.set_x(text.get_position()[0] + 2)


def panel_prices(ax, axis, iv, es, labels=False):
    a = axis.plot(ax, iv["spot"], color=BLUE, label="SPX")
    b = axis.plot(ax, es, color=ORANGE, label="ES")
    ax.set_ylabel("index level")
    if labels:
        direct_labels(ax, [("SPX", a), ("ES", b)])


def panel_vix(ax, axis, vix, labels=False):
    a = axis.plot(ax, vix["our_index"], color=BLUE, label="VIX-style index")
    b = axis.plot(ax, vix["real_vix"], color=ORANGE, label="Cboe VIX")
    ax.set_ylabel("vol index")
    if labels:
        direct_labels(ax, [("VIX-style", a), ("Cboe VIX", b)])


def panel_iv(ax, axis, kd, labels=False):
    """IV at fixed deltas. 25-delta put is blank when 4600 is less than 25 delta out."""
    a = axis.plot(ax, kd["iv_25p"] * 100, color=BLUE, label="25\u0394 put")
    b = axis.plot(ax, kd["iv_35p"] * 100, color=BLUE, linestyle=(0, (3, 1.5)), label="35\u0394 put")
    c = axis.plot(ax, kd["iv_atmf"] * 100, color=ORANGE, label="ATMF")
    d = axis.plot(ax, kd["iv_35c"] * 100, color=AQUA, linestyle=(0, (3, 1.5)), label="35\u0394 call")
    e = axis.plot(ax, kd["iv_25c"] * 100, color=AQUA, label="25\u0394 call")
    ax.set_ylabel("implied vol (%)")
    if labels:
        direct_labels(ax, [("25P", a), ("35P", b), ("ATMF", c), ("35C", d), ("25C", e)])


def panel_smile(ax, smile_csv, title, show_legend=False, show_xlabel=True, x_mode="moneyness"):
    # call_delta is stored by smile_per_event.py (N(d1) at each strike's own IV)
    smiles = pd.read_csv(smile_csv)
    minutes = sorted(smiles["minute"].unique())
    for m in minutes:
        if x_mode == "delta":
            piece = smiles[smiles["minute"] == m].sort_values("call_delta")
            piece = piece[(piece["call_delta"] >= 0.05) & (piece["call_delta"] <= 0.95)]
            x = piece["call_delta"]
        else:
            piece = smiles[smiles["minute"] == m].sort_values("moneyness")
            piece = piece[(piece["moneyness"] >= 0.88) & (piece["moneyness"] <= 1.10)]
            x = piece["moneyness"]
        if m <= 0:
            color = BEFORE_RAMP((m + 10) / 10.0)
        else:
            color = AFTER_RAMP((m - 1) / 9.0)
        width = 1.6 if m in (minutes[0], minutes[-1]) else 0.7
        ax.plot(x, piece["iv"] * 100, color=color, linewidth=width)
    ax.plot([], [], color=BEFORE_RAMP(0.8), linewidth=1.6, label=f"before ({minutes[0]:+d} to 0 min)")
    ax.plot([], [], color=AFTER_RAMP(0.8), linewidth=1.6, label=f"after (+1 to {minutes[-1]:+d} min)")
    if x_mode == "delta":
        # low strikes (puts) on the left, like the moneyness version
        ax.set_xlim(0.95, 0.05)
        ax.set_xticks([0.9, 0.75, 0.5, 0.25, 0.1])
        ax.set_xticklabels(["10P", "25P", "50", "25C", "10C"])
    if show_xlabel:
        ax.set_xlabel("delta" if x_mode == "delta" else "moneyness K/F")
    ax.set_ylabel("implied vol (%)")
    ax.set_title(title, loc="left")
    if show_legend:
        # the smile slopes down to the right, so the top right corner is empty
        ax.legend(loc="upper right", handlelength=1.2, fontsize=5.8, borderpad=0.3,
                  labelspacing=0.3)


# P(S_T < 4600) = 1 - mass on the strike grid, from the calibration run
BELOW_4600 = {}
_params = pd.read_csv("event_density_output/sabr_params_per_minute.csv")
for _i in range(len(_params)):
    BELOW_4600[(_params["event"][_i], int(_params["minute"][_i]))] = (1.0 - _params["total_mass"][_i]) * 100


def panel_density(ax, densities, event, show_xlabel=True):
    piece = densities[densities["event"] == event]
    first_minute = piece["minute"].min()
    last_minute = piece["minute"].max()
    for minute, color in ((first_minute, BLUE), (last_minute, ORANGE)):
        curve = piece[piece["minute"] == minute].sort_values("strike")
        curve = curve[curve["strike"] <= 6400]
        below = BELOW_4600[(event, minute)]
        ax.plot(curve["strike"], curve["density"] * 1000, color=color,
                label=f"{minute:+d} min ({below:.1f}%)")
    if show_xlabel:
        ax.set_xlabel("SPX at expiry")
    ax.set_ylabel("density (x10$^{-3}$)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.8)
    ax.legend(loc="upper left", handlelength=1.4)


# ---------------------------------------------------------------- figures

def overview_figure(iv, kd, vix, es, event_times, graphed_times, path):
    # one segment per session with data, 1-10 April
    segments = []
    starts = iv.index.to_series().diff() > pd.Timedelta(minutes=10)
    session_starts = [iv.index[0]] + list(iv.index[starts])
    for i in range(len(session_starts)):
        start = session_starts[i]
        if i + 1 < len(session_starts):
            end = iv.index[iv.index < session_starts[i + 1]][-1]
        else:
            end = iv.index[-1]
        if start >= pd.Timestamp("2025-04-11"):
            continue
        segments.append((start, min(end, pd.Timestamp("2025-04-10 16:15"))))
    axis = TimeAxis(segments)

    iv5 = iv.resample("5min").last()
    kd5 = kd.resample("5min").last()
    vix5 = vix.resample("5min").last()
    es5 = es.resample("5min").last()

    fig, axes = plt.subplots(4, 1, figsize=(6.3, 4.8), sharex=True)
    panel_prices(axes[0], axis, iv5, es5)
    panel_vix(axes[1], axis, vix5)
    axis.plot(axes[2], kd5["iv_atmf"] * 100, color=ORANGE, label="ATMF implied vol")
    axes[2].set_ylabel("implied vol (%)")
    axis.plot(axes[3], kd5["rr_25"] * 100, color=BLUE, label="25\u0394 risk reversal")
    axis.plot(axes[3], kd5["rr_35"] * 100, color=AQUA, label="35\u0394 risk reversal")
    axes[3].set_ylabel("vol points")
    axes[0].set_title("(a) SPX and ES", loc="left")
    axes[1].set_title("(b) VIX-style index vs Cboe VIX", loc="left")
    axes[2].set_title("(c) ATMF implied vol", loc="left")
    axes[3].set_title("(d) risk reversal, call IV minus put IV at the same delta (25\u0394 blank when no strike reaches it)", loc="left")

    roll_x = axis.x(segments[0][0])
    for start, end, offset in axis.segments:
        if start >= pd.Timestamp("2025-04-06"):
            roll_x = offset - GAP_WIDTH / 2.0
            break
    for ax in axes:
        axis.event_lines(ax, event_times, color="#d6d5d0", style="-", width=0.5)
        axis.event_lines(ax, graphed_times, color=EVENT_LINE, style="--", width=0.7)
        ax.axvline(roll_x, color=INK, linewidth=0.9)
        if ax is axes[0] or ax is axes[3]:
            ax.legend(loc="lower left", ncol=2)
        else:
            ax.legend(loc="upper left", ncol=2)
    axes[0].text(roll_x + 8, axes[0].get_ylim()[1], "expiry roll 16 Apr \u2192 15 May",
                 fontsize=6, color=INK, va="top")

    # one tick per trading day, at the start of RTH
    ticks = []
    labels = []
    for start, end, offset in axis.segments:
        if start.hour == 9:
            ticks.append(offset)
            labels.append(start.strftime("%d %b"))
    axes[3].set_xticks(ticks)
    axes[3].set_xticklabels(labels)
    last_start, last_end, last_offset = axis.segments[-1]
    axes[3].set_xlim(0, last_offset + (last_end - last_start).total_seconds() / 60.0)
    for i in range(len(axis.segments) - 1):
        start, end, offset = axis.segments[i]
        gap_from = offset + (end - start).total_seconds() / 60.0
        for ax in axes:
            ax.axvspan(gap_from, gap_from + GAP_WIDTH, color="#f1f0ec", linewidth=0, zorder=0)
    fig.text(0.01, 0.005, "Grey bands: breaks between sessions (squeezed out). Dashed lines: graphed events; "
             "light lines: other events.", fontsize=6, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def event_figure(iv, kd, vix, es, densities, segments, tick_minutes, events, path, height, x_mode="moneyness"):
    """events: list of (short label, event time, smile csv, density event name)."""
    axis = TimeAxis(segments)
    n = len(events)
    fig = plt.figure(figsize=(6.3, height))
    grid = fig.add_gridspec(6, 3, width_ratios=[1.3, 1, 1], wspace=0.7, hspace=2.3)

    rows_per_event = 6 // n
    left = [fig.add_subplot(grid[0:2, 0])]
    left.append(fig.add_subplot(grid[2:4, 0], sharex=left[0]))
    left.append(fig.add_subplot(grid[4:6, 0], sharex=left[0]))

    first = pd.Timestamp(segments[0][0])
    last = pd.Timestamp(segments[-1][1])
    iv1 = iv[first - pd.Timedelta(minutes=5):last].resample("30s").last()
    kd1 = kd[first - pd.Timedelta(minutes=5):last].resample("30s").last()
    vix1 = vix[first - pd.Timedelta(minutes=5):last].resample("30s").last()
    es1 = es[first - pd.Timedelta(minutes=5):last].resample("30s").last()

    panel_prices(left[0], axis, iv1, es1, labels=True)
    panel_vix(left[1], axis, vix1, labels=True)
    panel_iv(left[2], axis, kd1, labels=True)
    left[0].set_title("(a) SPX and ES", loc="left")
    left[1].set_title("(b) VIX-style index vs Cboe VIX", loc="left")
    left[2].set_title("(c) implied vol by delta", loc="left")

    event_times = []
    for label, t, smile_csv, density_event in events:
        event_times.append(pd.Timestamp(t))
    for i in range(3):
        axis.event_lines(left[i], event_times)
        axis.finish(left[i], tick_minutes, "%H:%M")
    for i in range(2):
        plt.setp(left[i].get_xticklabels(), visible=False)
    # label the event lines on the top panel
    if n > 1:
        top = left[0].get_ylim()[1]
        for label, t, smile_csv, density_event in events:
            left[0].text(axis.x(t) + 1, 0.97, label.split(" ")[0], fontsize=5.5,
                         color=MUTED, va="top", transform=left[0].get_xaxis_transform())

    for j in range(n):
        label, t, smile_csv, density_event = events[j]
        rows = slice(j * rows_per_event, (j + 1) * rows_per_event)
        ax_smile = fig.add_subplot(grid[rows, 1])
        ax_density = fig.add_subplot(grid[rows, 2])
        prefix = "(d)" if n == 1 else f"(d{j + 1})"
        last_row = (j == n - 1)
        panel_smile(ax_smile, smile_csv, f"{prefix} smile, {label}", show_legend=False,
                    show_xlabel=last_row, x_mode=x_mode)
        panel_density(ax_density, densities, density_event, show_xlabel=last_row)
        prefix = "(e)" if n == 1 else f"(e{j + 1})"
        ax_density.set_title(f"{prefix} risk-neutral density", loc="left")

    # one shared key for the smile colours, under the right-hand panels
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BEFORE_RAMP(0.85), linewidth=2),
               Line2D([], [], color=AFTER_RAMP(0.85), linewidth=2)]
    labels = ["smile before event (\u221210 to 0 min)", "smile after event (+1 to +10 min)"]
    # room under the axes: ~0.42 in for tick labels and "delta", ~0.18 in for the key
    bottom = 0.60 / height
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.64, 0.01), ncol=2,
               fontsize=6, frameon=False, handlelength=1.8, columnspacing=1.5)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=bottom)
    fig.savefig(path)
    plt.close(fig)


PERIOD_ENDS = [
    ("End of P1 (2 Apr 15:50)", "2025-04-02 15:50"),
    ("End of P2 (4 Apr 16:14)", "2025-04-04 16:14"),
    ("End of P3 (9 Apr 13:00)", "2025-04-09 13:00"),
    ("End of P4 (10 Apr 16:14)", "2025-04-10 16:14"),
]
PERIOD_END_CSV = "period_end_densities.csv"


def build_period_end_densities():
    """SABR fit + BL density at each period end (same steps as event_densities.py)."""
    from breeden_litzenberger import density_checks
    from event_densities import BETA, density_for_snapshot
    from quote_selection import load_market_snapshot
    from sabr_calibration import calibrate_snapshot

    rows = []
    for label, timestamp in PERIOD_ENDS:
        snapshot = load_market_snapshot(pd.Timestamp(timestamp))
        result = calibrate_snapshot(snapshot, BETA)
        strikes, density = density_for_snapshot(snapshot, result)
        checks = density_checks(strikes, density, snapshot.forward)
        below = 1.0 - checks["total_mass"]
        for k, value in zip(strikes, density):
            rows.append({"label": label, "strike": float(k), "density": float(value),
                         "forward": snapshot.forward, "p4600": below})
        print(f"  {label}: forward {snapshot.forward:,.0f}, P(<4600) {below * 100:.1f}%")
    data = pd.DataFrame(rows)
    data.to_csv(PERIOD_END_CSV, index=False)
    return data


def period_end_figure(path):
    if os.path.exists(PERIOD_END_CSV):
        data = pd.read_csv(PERIOD_END_CSV)
    else:
        print(f"{PERIOD_END_CSV} not found, fitting SABR at the period ends")
        data = build_period_end_densities()
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    labels = []
    for label in data["label"]:
        if label not in labels:
            labels.append(label)
    fig, ax = plt.subplots(figsize=(6.3, 2.3))
    for i in range(len(labels)):
        piece = data[data["label"] == labels[i]].sort_values("strike")
        piece = piece[piece["strike"] <= 6400]
        p = piece["p4600"].iloc[0] * 100
        ax.plot(piece["strike"], piece["density"] * 1000, color=colors[i], linewidth=1.6,
                label=f"{labels[i]}, P(<4600) {p:.1f}%")
    ax.set_xlabel("SPX at expiry")
    ax.set_ylabel("density (x10$^{-3}$)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ax.set_title("Risk-neutral density at the end of each period (16 Apr expiry for P1-P2, 15 May for P3-P4)",
                 loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


WEEKEND_SNAPSHOTS = [
    ("Fri 4 Apr 16:14, last quote before the weekend (16 Apr expiry)", "2025-04-04 16:14", BLUE),
    ("Sun 6 Apr 20:20, evening open (15 May expiry)", "2025-04-06 20:20", ORANGE),
    ("Mon 7 Apr 09:43, SPX low of the sample (15 May expiry)", "2025-04-07 09:43", AQUA),
]


def weekend_figure(path):
    """Smile by delta just before and after the weekend, and at Monday's low.

    Delta puts the 16 Apr (~13 days) and 15 May (~40 days) expiries on the same
    footing, so the roll itself can be separated from the change in fear.
    """
    from smile_per_event import find_session, load_session, smile_at

    fig, ax = plt.subplots(figsize=(6.3, 2.3))
    for label, timestamp, color in WEEKEND_SNAPSHOTS:
        session, date = find_session(pd.Timestamp(timestamp))
        smile, row_time, forward = smile_at(load_session(session, date), pd.Timestamp(timestamp))
        keep = (smile["call_delta"] >= 0.05) & (smile["call_delta"] <= 0.95)
        smile = smile[keep].sort_values("call_delta")
        ax.plot(smile["call_delta"], smile["iv"] * 100, color=color, linewidth=1.6,
                label=f"{label}, F = {forward:,.0f}")
    ax.set_xlim(0.95, 0.05)
    ax.set_xticks([0.9, 0.75, 0.5, 0.25, 0.1])
    ax.set_xticklabels(["10P", "25P", "50", "25C", "10C"])
    ax.set_xlabel("delta")
    ax.set_ylabel("implied vol (%)")
    ax.set_ylim(top=ax.get_ylim()[1] + 3)
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    iv, kd, vix, es, densities, event_times = load_all()

    graphed = ["2025-04-02 16:07", "2025-04-04 06:00", "2025-04-04 07:21", "2025-04-07 10:13",
               "2025-04-07 10:35", "2025-04-07 11:14", "2025-04-09 07:00", "2025-04-09 13:18"]
    graphed_times = []
    for t in graphed:
        graphed_times.append(pd.Timestamp(t))

    overview_figure(iv, kd, vix, es, event_times, graphed_times, OUTPUT_FOLDER + "fig1_overview.png")

    # smiles are plotted against delta (set x_mode = "moneyness" for the K/F version)
    for x_mode, suffix in (("delta", ""),):
        event_figure(iv, kd, vix, es, densities,
                     [("2025-04-02 15:00", "2025-04-02 16:15"), ("2025-04-02 20:15", "2025-04-02 21:30")], 30,
                     [("16:07 Liberation Day", "2025-04-02 16:07", "smile_output/smile_2025-04-02_1607.csv",
                       "Liberation Day")],
                     OUTPUT_FOLDER + f"fig2_liberation_day{suffix}.png", 3.3, x_mode=x_mode)

        event_figure(iv, kd, vix, es, densities, [("2025-04-04 05:00", "2025-04-04 08:30")], 60,
                     [("06:00 WTO filing", "2025-04-04 06:00", "smile_output/smile_2025-04-04_0600.csv",
                       "China WTO filing"),
                      ("07:21 China +34%", "2025-04-04 07:21", "smile_output/smile_2025-04-04_0721.csv",
                       "China +34%")],
                     OUTPUT_FOLDER + f"fig3_china_retaliates{suffix}.png", 4.0, x_mode=x_mode)

        event_figure(iv, kd, vix, es, densities, [("2025-04-07 09:31", "2025-04-07 12:00")], 30,
                     [("10:13 false pause report", "2025-04-07 10:13", "smile_output/smile_2025-04-07_1013.csv",
                       "False pause report"),
                      ("10:35 WH denial", "2025-04-07 10:35", "smile_output/smile_2025-04-07_1035.csv",
                       "WH denial"),
                      ("11:14 ultimatum to China", "2025-04-07 11:14", "smile_output/smile_2025-04-07_1114.csv",
                       "Ultimatum to China")],
                     OUTPUT_FOLDER + f"fig4_7_april{suffix}.png", 5.0, x_mode=x_mode)

        event_figure(iv, kd, vix, es, densities, [("2025-04-09 06:00", "2025-04-09 08:30")], 60,
                     [("07:00 China 84%", "2025-04-09 07:00", "smile_output/smile_2025-04-09_0700.csv",
                       "China 84%")],
                     OUTPUT_FOLDER + f"fig5_china_84{suffix}.png", 3.3, x_mode=x_mode)

        event_figure(iv, kd, vix, es, densities, [("2025-04-09 12:15", "2025-04-09 16:15")], 60,
                     [("13:18 90-day pause", "2025-04-09 13:18", "smile_output/smile_2025-04-09_1318.csv",
                       "90-day pause")],
                     OUTPUT_FOLDER + f"fig6_pause{suffix}.png", 3.3, x_mode=x_mode)

    weekend_figure(OUTPUT_FOLDER + "fig_weekend_roll.png")
    period_end_figure(OUTPUT_FOLDER + "fig7_period_end_densities.png")
    print("saved figures to", OUTPUT_FOLDER)