"""
event_densities.py - SABR fit + Breeden-Litzenberger density for every minute
around each graphed event (QF620 group project)

For every timestamp in event_minute_timestamps.csv (-10 to +10 min per event):
    1. load the market snapshot (same quote selection as sabr_calibration.py)
    2. fit SABR (beta = 0.7) with calibrate_snapshot
    3. SABR vol -> discounted Black call price on a 5-point strike grid,
       using the snapshot's own forward, T and discount factor
    4. Breeden-Litzenberger: q(K) = exp(rT) * d2C/dK2

The strike grid is kept inside the strikes SABR was fitted on, so nothing is
extrapolated.

Outputs (in event_density_output/):
    sabr_params_per_minute.csv   one row per timestamp: alpha, rho, nu, fit quality,
                                 density checks, P(S_T < 0.9F)
    densities_per_minute.csv     long format: event, minute, strike, density
    density_<event>.png          all minutes overlaid (blue = before, orange = after)
"""
import math
import os
import sys

import numpy as np
import pandas as pd

from breeden_litzenberger import (breeden_litzenberger, density_checks, prob_below,
                                  sabr_implied_vol)
from quote_selection import load_market_snapshot
from sabr_calibration import black_forward_price, calibrate_snapshot

TIMESTAMPS_CSV = "event_minute_timestamps.csv"
OUTPUT_FOLDER = "event_density_output/"
BETA = 0.7
STRIKE_STEP = 5.0


def density_for_snapshot(snapshot, result):
    """Call prices from the fitted SABR smile, then the BL density."""
    fitted_strikes = result.fitted_targets["strike"]
    k_min = float(fitted_strikes.min())
    k_max = float(fitted_strikes.max())
    strikes = np.arange(k_min, k_max + STRIKE_STEP, STRIKE_STEP)

    call_prices = []
    for k in strikes:
        vol = sabr_implied_vol(snapshot.forward, k, snapshot.maturity,
                               result.alpha, result.beta, result.rho, result.nu)
        price = black_forward_price(snapshot.forward, k, snapshot.maturity,
                                    snapshot.discount, vol, "call")
        call_prices.append(price)

    density_strikes, density = breeden_litzenberger(
        strikes, np.array(call_prices), snapshot.rate, snapshot.maturity)
    return density_strikes, density


def run_all(timestamps_csv=TIMESTAMPS_CSV):
    table = pd.read_csv(timestamps_csv, parse_dates=["event_time", "timestamp"])
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    param_rows = []
    density_rows = []
    for i in range(len(table)):
        row = table.iloc[i]
        event = row["event"]
        minute = int(row["minute"])
        timestamp = row["timestamp"]

        try:
            snapshot = load_market_snapshot(timestamp)
            result = calibrate_snapshot(snapshot, BETA)
            density_strikes, density = density_for_snapshot(snapshot, result)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            print(f"  skipped {event} {minute:+d}: {error}")
            continue

        checks = density_checks(density_strikes, density, snapshot.forward)
        tail_prob = prob_below(density_strikes, density, 0.9 * snapshot.forward)

        param_rows.append({
            "event": event,
            "event_time": row["event_time"],
            "minute": minute,
            "timestamp": timestamp,
            "session": snapshot.session,
            "spot": snapshot.spot,
            "spot_from_es": snapshot.spot_from_es,
            "forward": snapshot.forward,
            "T": snapshot.maturity,
            "alpha": result.alpha,
            "beta": result.beta,
            "rho": result.rho,
            "nu": result.nu,
            "price_rmse": result.price_rmse,
            "within_bid_ask_fraction": result.within_bid_ask / len(snapshot.targets),
            "optimizer_ok": result.success,
            "k_min": float(density_strikes[0]),
            "k_max": float(density_strikes[-1]),
            "total_mass": checks["total_mass"],
            "density_mean": checks["mean"],
            "negative_points": checks["negative_points"],
            "prob_below_90pct_F": tail_prob,
        })

        for k, value in zip(density_strikes, density):
            density_rows.append({"event": event, "minute": minute,
                                 "timestamp": timestamp, "strike": float(k),
                                 "density": float(value)})

        print(f"  {event:20s} {minute:+3d}  rho={result.rho:+.3f}  nu={result.nu:.3f}"
              f"  mass={checks['total_mass']:.3f}")

    params = pd.DataFrame(param_rows)
    densities = pd.DataFrame(density_rows)
    params.to_csv(OUTPUT_FOLDER + "sabr_params_per_minute.csv", index=False)
    densities.to_csv(OUTPUT_FOLDER + "densities_per_minute.csv", index=False)
    return params, densities


def plot_event(densities, params, event, output_path):
    """All minutes of one event overlaid; blue ramp before, orange ramp after."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    before_ramp = LinearSegmentedColormap.from_list("before", ["#c6dbef", "#08306b"])
    after_ramp = LinearSegmentedColormap.from_list("after", ["#fdd0a2", "#7f2704"])
    ink = "#0b0b0b"
    muted = "#52514e"
    background = "#fcfcfb"

    event_densities = densities[densities["event"] == event]
    event_params = params[params["event"] == event]
    minutes = sorted(event_densities["minute"].unique())

    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=background)
    ax.set_facecolor(background)

    for m in minutes:
        piece = event_densities[event_densities["minute"] == m]
        if m <= 0:
            color = before_ramp((m + 10) / 10.0)
        else:
            color = after_ramp((m - 1) / 9.0)
        width = 2.2 if m in (minutes[0], minutes[-1]) else 1.0
        label = None
        if m == minutes[0]:
            label = f"{m:+d} min"
        elif m == minutes[-1]:
            label = f"{m:+d} min"
        ax.plot(piece["strike"], piece["density"], color=color, linewidth=width, label=label)

    # forwards at the first and last minute
    first = event_params[event_params["minute"] == minutes[0]].iloc[0]
    last = event_params[event_params["minute"] == minutes[-1]].iloc[0]
    ax.axvline(first["forward"], color="#08306b", linewidth=0.8, linestyle="--")
    ax.axvline(last["forward"], color="#7f2704", linewidth=0.8, linestyle="--")

    event_time = pd.Timestamp(event_params["event_time"].iloc[0])
    ax.set_title(f"SPX risk-neutral density around {event} ({event_time:%d %b %H:%M} ET)",
                 loc="left", fontsize=11, color=ink)
    ax.set_xlabel("SPX level at expiry", color=muted)
    ax.set_ylabel("density q(K)", color=muted)
    ax.legend(frameon=False, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=muted)
    ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    csv_path = TIMESTAMPS_CSV
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    params, densities = run_all(csv_path)

    events = []
    for event in params["event"]:
        if event not in events:
            events.append(event)
    for event in events:
        name = event.lower().replace(" ", "_").replace("%", "pct").replace("+", "")
        path = OUTPUT_FOLDER + f"density_{name}.png"
        plot_event(densities, params, event, path)
        print("saved", path)