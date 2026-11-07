"""
qf620_rnd.py - risk-neutral density for SPX (QF620 group project)
"""
import math
 
import numpy as np
import pandas as pd

RISK_FREE_RATE = 0.043   # TODO: replace with 1-month T-bill, early April 2025
DIVIDEND_YIELD = 0.0
EXPIRY_TIME = "09:30:00"

def time_to_expiry(timestamp, expiry):
    """Years from the snapshot time to expiry (calendar time, ACT/365).
 
    timestamp : snapshot time, e.g. "2025-04-02 15:45:00" (ET)
    expiry    : expiry date as in the file names, e.g. "20250416"
    """
    expiry_date = pd.to_datetime(expiry, format="%Y%m%d")
    expiry_moment = pd.Timestamp(f"{expiry_date.date()} {EXPIRY_TIME}")
    snapshot = pd.Timestamp(timestamp)
 
    seconds_left = (expiry_moment - snapshot).total_seconds()
    if seconds_left <= 0:
        raise ValueError(f"Snapshot {snapshot} is not before expiry {expiry_moment}")
 
    return seconds_left / (365.0 * 24.0 * 3600.0)

def read_spot(assets_path, timestamp, max_gap_seconds=60):
    """Read the SPX spot at a chosen time from a <date>_assets.csv file.
 
    Uses the last row at or before the timestamp (rows are about 3 seconds apart).
 
    assets_path     : path to the assets csv, e.g. "data/RTH/2025-04-02_assets.csv"
    timestamp       : time to read, e.g. "2025-04-02 15:45:00" (ET)
    max_gap_seconds : refuse rows older than this, so a typo'd time doesn't
                      silently pick up a stale price
 
    Returns (spot, row_time) so you can see which row was actually used.
    """
    df = pd.read_csv(assets_path, index_col=0, parse_dates=True)
    df = df.sort_index()
 
    wanted = pd.Timestamp(timestamp)
    row_time = df.index.asof(wanted)
 
    if pd.isna(row_time):
        raise ValueError(f"{wanted} is before the first row ({df.index[0]}) in {assets_path}")
 
    gap = (wanted - row_time).total_seconds()
    if gap > max_gap_seconds:
        raise ValueError(f"Closest row before {wanted} is {row_time}, "
                         f"{gap:.0f}s earlier; check the time or the file")
 
    spot = float(df.loc[row_time, "SPX"])
    return spot, row_time

def norm_cdf(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
 
def black_scholes_call(S, K, T, r, q, vol):
    """European call price under Black-Scholes with continuous dividend yield.
 
    S   : spot price of SPX
    K   : strike
    T   : time to expiry in years
    r   : risk-free rate (continuous, annual)
    q   : dividend yield (continuous, annual)
    vol : implied volatility (annual, e.g. 0.25 for 25%)
    """
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T
    return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    """Black implied vol from SABR parameters (Hagan et al. 2002 formula).
 
    F     : forward price, F = S * exp((r - q) * T)
    K     : strike
    T     : time to expiry in years
    alpha : vol level
    beta  : CEV exponent (fixed, e.g. 1.0 or 0.5)
    rho   : correlation between price and vol (negative for SPX skew)
    nu    : vol of vol
    """
    one_minus_beta = 1.0 - beta
    log_fk = math.log(F / K)
    fk_power = (F * K) ** (one_minus_beta / 2.0)
 
    # z / x(z) term; at the money z -> 0 and the ratio -> 1
    z = (nu / alpha) * fk_power * log_fk
    if abs(z) < 1e-7:
        z_over_x = 1.0
    else:
        x = math.log((math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
        z_over_x = z / x
 
    denominator = fk_power * (
        1.0
        + (one_minus_beta ** 2 / 24.0) * log_fk ** 2
        + (one_minus_beta ** 4 / 1920.0) * log_fk ** 4
    )
 
    time_correction = 1.0 + (
        (one_minus_beta ** 2 / 24.0) * alpha * alpha / (fk_power * fk_power)
        + 0.25 * rho * beta * nu * alpha / fk_power
        + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu
    ) * T
 
    return (alpha / denominator) * z_over_x * time_correction

def breeden_litzenberger(strikes, call_prices, r, T):
    """Risk-neutral density from call prices on an evenly spaced strike grid.
 
    q(K) = exp(r * T) * d2C/dK2, using a central second difference.
    The first and last strikes have no neighbour on one side, so they are dropped.
 
    strikes     : evenly spaced strikes, e.g. 4600, 4605, 4610, ...
    call_prices : call price at each strike
    r           : risk-free rate (continuous, annual)
    T           : time to expiry in years
 
    Returns (density_strikes, density) as numpy arrays.
    """
    dK = strikes[1] - strikes[0]
    growth = math.exp(r * T)
 
    density_strikes = []
    density = []
    for i in range(1, len(strikes) - 1):
        second_diff = (call_prices[i + 1] - 2 * call_prices[i] + call_prices[i - 1]) / (dK * dK)
        density_strikes.append(strikes[i])
        density.append(growth * second_diff)
 
    return np.array(density_strikes), np.array(density)

def density_checks(density_strikes, density, forward):
    """Sanity checks on a density: negative points, total mass, mean vs forward."""
    dK = density_strikes[1] - density_strikes[0]
    negative_points = 0
    total_mass = 0.0
    weighted_sum = 0.0
    for k, value in zip(density_strikes, density):
        if value < 0:
            negative_points += 1
        total_mass += value * dK
        weighted_sum += k * value * dK
 
    mean = weighted_sum / total_mass if total_mass > 0 else float("nan")
    return {
        "negative_points": negative_points,
        "total_mass": total_mass,
        "mean": mean,
        "forward": forward,
    }

def make_density_function(density_strikes, density):
    """Turn the density grid into a function q(K) you can call at any strike.
 
    Between grid points it interpolates linearly; outside the grid it returns 0.
    """
    def q(K):
        if K < density_strikes[0] or K > density_strikes[-1]:
            return 0.0
        return float(np.interp(K, density_strikes, density))
    return q

def prob_below(density_strikes, density, level):
    """Risk-neutral probability that SPX ends below a level, P(S_T < level)."""
    dK = density_strikes[1] - density_strikes[0]
    prob = 0.0
    for k, value in zip(density_strikes, density):
        if k < level:
            prob += value * dK
    return prob
 
 
def plot_density(result, output_path="density.png", title=None):
    """Save a chart of one snapshot's density, with the forward marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    strikes = result["density_strikes"]
    density = result["density"]
    forward = result["forward"]
 
    line_color = "#2a78d6"
    ink = "#0b0b0b"
    muted = "#52514e"
    background = "#fcfcfb"
 
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=background)
    ax.set_facecolor(background)
 
    ax.plot(strikes, density, color=line_color, linewidth=2)
 
    peak = float(np.max(density))
    ax.axvline(forward, color=muted, linewidth=1, linestyle="--")
    ax.text(forward + 15, peak * 1.03, f"forward {forward:,.0f}", color=muted, fontsize=10)
 
    if title is None:
        title = f"SPX risk-neutral density, snapshot {result['timestamp']:%Y-%m-%d %H:%M}"
    ax.set_title(title, loc="left", fontsize=11, color=ink)
    ax.set_xlabel("SPX level at expiry", color=muted)
    ax.set_ylabel("density q(K)", color=muted)
 
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9c8c3")
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.tick_params(colors=muted)
    ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
    ax.set_ylim(0, peak * 1.12)
 
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

def run_snapshot(sabr_params, assets_path, timestamp, expiry,
                 k_min=4600.0, k_max=6100.0, dK=5.0,
                 r=RISK_FREE_RATE, q=DIVIDEND_YIELD):
    """Risk-neutral density for one snapshot, from already-fitted SABR parameters.
 
    sabr_params : dict with alpha, beta, rho, nu (from the SABR calibration)
    assets_path : the <date>_assets.csv for this snapshot's session
    timestamp   : snapshot time, e.g. "2025-04-02 15:45:00" (ET)
    expiry      : expiry from the option file names, e.g. "20250416"
    k_min/k_max : strike range; keep it inside the strikes SABR was fitted on
    dK          : strike step for the fine grid
    """
    # 1. market inputs for this snapshot
    spot, row_time = read_spot(assets_path, timestamp)
    T = time_to_expiry(timestamp, expiry)
    forward = spot * math.exp((r - q) * T)
 
    # 2. fine strike grid
    strikes = np.arange(k_min, k_max + dK, dK)
 
    # 3. SABR vol -> Black-Scholes call price at every strike
    vols = []
    call_prices = []
    for k in strikes:
        vol = sabr_implied_vol(forward, k, T,
                               sabr_params["alpha"], sabr_params["beta"],
                               sabr_params["rho"], sabr_params["nu"])
        vols.append(vol)
        call_prices.append(black_scholes_call(spot, k, T, r, q, vol))
 
    # 4. Breeden-Litzenberger
    density_strikes, density = breeden_litzenberger(strikes, call_prices, r, T)
 
    # 5. checks
    checks = density_checks(density_strikes, density, forward)
 
    return {
        "timestamp": row_time,
        "spot": spot,
        "T": T,
        "forward": forward,
        "strikes": strikes,
        "vols": np.array(vols),
        "call_prices": np.array(call_prices),
        "density_strikes": density_strikes,
        "density": density,
        "checks": checks,
    }

if __name__ == "__main__":
    # made-up SABR parameters until the real calibration is ready
    sabr_params = {"alpha": 0.22, "beta": 1.0, "rho": -0.7, "nu": 1.5}
 
    result = run_snapshot(
        sabr_params,
        assets_path="data/ETH/2025-04-01_assets.csv",
        timestamp="2025-04-02 09:00:00",
        expiry="20250417",
    )
 
    print("row used :", result["timestamp"])
    print("spot     :", result["spot"])
    print("T (days) :", round(result["T"] * 365, 2))
    print("forward  :", round(result["forward"], 2))
    print("checks   :", result["checks"])
 
    # the density itself: one value per strike on the grid
    print()
    print("strike    density")
    for k, value in zip(result["density_strikes"], result["density"]):
        if k % 100 == 0:
            print(f"{k:7.0f}   {value:.6f}")
 
    # or as a function you can call at any level
    q = make_density_function(result["density_strikes"], result["density"])
    print()
    print("q(5000) =", round(q(5000), 6))
    print("q(5592) =", round(q(5592), 6))
 
    # chart
    path = plot_density(result, "density_2025-04-02_0900.png",
                        title="SPX risk-neutral density, 2 Apr 09:00, 16 Apr expiry (made-up SABR params)")
    print()
    print("saved", path)