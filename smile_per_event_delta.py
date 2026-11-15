"""
smile_per_event_delta.py - the same event smiles as smile_per_event.py, with delta
on the x-axis instead of moneyness (QF620 group project)

For every event in the events file:
    - smiles for every minute from -10 to +10 minutes (event_smiles in smile_per_event.py,
      which now also stores delta and call_delta for each strike)
    - a GIF with one frame per minute and a static PNG with all minutes overlaid

x-axis: call delta N(d1) for every strike, reversed so low strikes (puts) are on the
left, like the moneyness charts. A put with delta -0.25 sits at call delta 0.75, so
the ticks read 10P, 25P, 50, 25C, 10C.

Why delta: a fixed delta adjusts for the vol level and time to expiry, so smiles stay
comparable across the 6 April expiry roll (~15 -> ~39 days) where moneyness does not.
It also shows where the 4600 strike floor cuts the put wing (the smile stops early).

Output goes to smile_delta_output/ (CSV with the delta columns, GIF, PNG).
"""
import os

import pandas as pd

from smile_per_event import MINUTES_AFTER, MINUTES_BEFORE, FRAME_SECONDS, event_smiles, load_events

OUTPUT_FOLDER = "smile_delta_output/"
DELTA_RANGE = (0.05, 0.95)      # call delta; outside this the far wings are just noise
DELTA_TICKS = [0.9, 0.75, 0.5, 0.25, 0.1]
DELTA_LABELS = ["10P", "25P", "50", "25C", "10C"]

INK = "#0b0b0b"
MUTED = "#52514e"
BACKGROUND = "#fcfcfb"


def in_range(smile):
    keep = (smile["call_delta"] >= DELTA_RANGE[0]) & (smile["call_delta"] <= DELTA_RANGE[1])
    return smile[keep].sort_values("call_delta")


def style_axis(ax):
    ax.set_xlim(DELTA_RANGE[1], DELTA_RANGE[0])     # reversed: puts on the left
    ax.set_xticks(DELTA_TICKS)
    ax.set_xticklabels(DELTA_LABELS)
    ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlabel("delta", color=MUTED)
    ax.set_ylabel("implied vol (%)", color=MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9c8c3")
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED)
    ax.grid(color="#e8e7e3", linewidth=0.8)


def make_gif(smiles, label, output_path):
    """One frame per minute; grey line = first minute of the window, for reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    before_color = "#2a78d6"
    after_color = "#eb6834"
    reference_color = "#b4b2a9"

    minutes = sorted(smiles["minute"].unique())
    inside = in_range(smiles)
    y_low = inside["iv"].min() * 100.0
    y_high = inside["iv"].max() * 100.0
    pad = (y_high - y_low) * 0.08

    first = in_range(smiles[smiles["minute"] == minutes[0]])

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.plot(first["call_delta"], first["iv"] * 100.0, color=reference_color,
            linewidth=1.5, label=f"{minutes[0]:+d} min (reference)")
    line, = ax.plot([], [], color=before_color, linewidth=2.2, marker="o", markersize=3)
    style_axis(ax)
    ax.set_ylim(y_low - pad, y_high + pad)
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle(label, x=0.07, ha="left", fontsize=10, color=INK)
    title = ax.set_title("", loc="left", fontsize=11, color=INK)

    def draw(frame):
        m = minutes[frame]
        smile = in_range(smiles[smiles["minute"] == m])
        line.set_data(smile["call_delta"], smile["iv"] * 100.0)
        if m < 0:
            line.set_color(before_color)
        else:
            line.set_color(after_color)
        when = pd.Timestamp(smiles[smiles["minute"] == m]["time"].iloc[0])
        forward = smiles[smiles["minute"] == m]["forward"].iloc[0]
        title.set_text(f"{when:%Y-%m-%d %H:%M} ET   ({m:+d} min from event)   F = {forward:,.1f}")
        return line, title

    # hold the last frame a bit longer by repeating it
    frame_order = list(range(len(minutes))) + [len(minutes) - 1] * 4
    animation = FuncAnimation(fig, lambda i: draw(frame_order[i]), frames=len(frame_order), blit=False)
    animation.save(output_path, writer=PillowWriter(fps=1.0 / FRAME_SECONDS))
    plt.close(fig)
    return output_path


def make_overlay(smiles, label, output_path):
    """All minutes on one static chart (blue before, black at the event, orange after)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    before_ramp = LinearSegmentedColormap.from_list("before", ["#c6dbef", "#08306b"])
    after_ramp = LinearSegmentedColormap.from_list("after", ["#fdd0a2", "#7f2704"])

    minutes = sorted(smiles["minute"].unique())
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    for m in minutes:
        smile = in_range(smiles[smiles["minute"] == m])
        if m < 0:
            color = before_ramp((m + MINUTES_BEFORE) / max(MINUTES_BEFORE - 1, 1))
            width = 1.0
            z = 1
        elif m == 0:
            color = INK
            width = 2.0
            z = 3
        else:
            color = after_ramp((m - 1) / max(MINUTES_AFTER - 1, 1))
            width = 1.0
            z = 2
        ax.plot(smile["call_delta"], smile["iv"] * 100.0, color=color, linewidth=width, zorder=z)

    ax.plot([], [], color="#4a90d0", linewidth=2, label=f"-{MINUTES_BEFORE} to -1 min (light to dark)")
    ax.plot([], [], color=INK, linewidth=2, label="event minute")
    ax.plot([], [], color="#d9611c", linewidth=2, label=f"+1 to +{MINUTES_AFTER} min (light to dark)")
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax)
    ax.set_title(label, loc="left", fontsize=10, color=INK)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    events = load_events()

    # events at the same minute (e.g. two at 2025-04-09 00:01) share one chart
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

        name = f"smile_delta_{event_time:%Y-%m-%d_%H%M}"
        smiles.to_csv(OUTPUT_FOLDER + name + ".csv", index=False)
        make_gif(smiles, label, OUTPUT_FOLDER + name + ".gif")
        make_overlay(smiles, label, OUTPUT_FOLDER + name + ".png")
        print(f"saved {name}: {smiles['minute'].nunique()} minutes ({smiles['session'].iloc[0]})")