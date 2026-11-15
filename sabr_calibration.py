"""Fit SABR to OTM SPX option mid prices and one near-ATM straddle.

The straddle replaces the single OTM target at its strike, so neither leg is
counted twice. Hagan's Black volatility approximation is converted to a
discounted-forward Black price before computing each price residual.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import ndtr

from breeden_litzenberger import sabr_implied_vol
from quote_selection import DATA_DIR, MarketSnapshot, load_market_snapshot
from smile_per_event import load_events


OUTPUT_DIR = Path(__file__).resolve().parent / "calibration_output"
EVENTS_CSV = Path(__file__).resolve().parent / "2025 Liberation Day Tariff Main Events.csv"


@dataclass
class CalibrationResult:
    snapshot: MarketSnapshot
    alpha: float
    beta: float
    rho: float
    nu: float
    success: bool
    message: str
    price_rmse: float
    within_bid_ask: int
    fitted_targets: pd.DataFrame

    @property
    def parameters(self) -> dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta, "rho": self.rho, "nu": self.nu}


def black_forward_price(
    forward: float, strike: float, maturity: float, discount: float,
    volatility: float, side: str,
) -> float:
    """Discounted Black call, put, or call-plus-put straddle price."""
    if not all(math.isfinite(x) and x > 0 for x in
               (forward, strike, maturity, discount, volatility)):
        raise ValueError("Black price inputs must be positive and finite")
    stddev = volatility * math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * stddev * stddev) / stddev
    d2 = d1 - stddev
    call = discount * (forward * ndtr(d1) - strike * ndtr(d2))
    put = discount * (strike * ndtr(-d2) - forward * ndtr(-d1))
    if side == "call":
        return float(call)
    if side == "put":
        return float(put)
    if side == "straddle":
        return float(call + put)
    raise ValueError(f"unknown target side {side!r}")


def sabr_model_prices(
    parameters: Iterable[float], snapshot: MarketSnapshot, beta: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Hagan Black vols and corresponding prices for selected targets."""
    alpha, rho, nu = (float(value) for value in parameters)
    if not (alpha > 0 and nu > 0 and -1 < rho < 1 and 0 <= beta <= 1):
        raise ValueError("SABR parameters outside their domain")
    vols = []
    prices = []
    for row in snapshot.targets.itertuples(index=False):
        vol = sabr_implied_vol(
            snapshot.forward, row.strike, snapshot.maturity, alpha, beta, rho, nu
        )
        if not math.isfinite(vol) or vol <= 0:
            raise ValueError(f"invalid SABR volatility at strike {row.strike:g}")
        vols.append(vol)
        prices.append(black_forward_price(
            snapshot.forward, row.strike, snapshot.maturity, snapshot.discount,
            vol, row.kind,
        ))
    return np.asarray(vols), np.asarray(prices)


def price_residuals(
    parameters: Iterable[float], snapshot: MarketSnapshot, beta: float = 0.7,
) -> np.ndarray:
    """One unweighted model-minus-market price residual per target."""
    try:
        _, prices = sabr_model_prices(parameters, snapshot, beta)
        residuals = prices - snapshot.targets["mid"].to_numpy(dtype=float)
        if np.all(np.isfinite(residuals)):
            return residuals
    except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
        pass
    return np.full(len(snapshot.targets), 1e6)


def _starting_points(snapshot: MarketSnapshot, beta: float) -> list[tuple[float, float, float]]:
    """Deterministic starts scaled to the observed near-ATM straddle price."""
    straddle = snapshot.targets.loc[snapshot.targets["kind"] == "straddle", "mid"].iloc[0]
    atm_vol_estimate = straddle / (
        snapshot.discount * snapshot.forward * math.sqrt(2 * snapshot.maturity / math.pi)
    )
    atm_vol_estimate = float(np.clip(atm_vol_estimate, 0.03, 2.0))
    alpha = atm_vol_estimate * snapshot.forward ** (1 - beta)
    return [(0.7 * alpha, -0.7, 0.5),
            (alpha, -0.4, 1.5),
            (1.3 * alpha, 0.0, 3.0)]


def summarize_fit(
    snapshot: MarketSnapshot, parameters: Iterable[float], beta: float,
) -> tuple[pd.DataFrame, float, int]:
    """Attach model vols, prices, residuals, and bid-ask coverage to targets."""
    vols, prices = sabr_model_prices(parameters, snapshot, beta)
    fitted = snapshot.targets.copy()
    fitted["model_vol"] = vols
    fitted["model_price"] = prices
    fitted["price_error"] = prices - fitted["mid"].to_numpy(dtype=float)
    fitted["within_bid_ask"] = (prices >= fitted["bid"]) & (prices <= fitted["ask"])
    rmse = float(np.sqrt(np.mean(fitted["price_error"].to_numpy() ** 2)))
    return fitted, rmse, int(fitted["within_bid_ask"].sum())


def calibrate_snapshot(
    snapshot: MarketSnapshot, beta: float = 0.7,
    starts: Iterable[Iterable[float]] | None = None,
) -> CalibrationResult:
    """Fit alpha, rho, and nu using bounded multi-start price least squares."""
    if not 0 <= beta <= 1:
        raise ValueError("beta must lie between 0 and 1")
    starts = list(_starting_points(snapshot, beta) if starts is None else starts)
    if not starts:
        raise ValueError("at least one starting point is required")
    alpha_cap = max(100.0, 10.0 * max(float(start[0]) for start in starts))
    lower = np.array([1e-8, -0.999, 1e-6])
    upper = np.array([alpha_cap, 0.999, 10.0])

    fits = []
    for start in starts:
        x0 = np.asarray(start, dtype=float)
        if x0.shape != (3,) or not np.all((lower < x0) & (x0 < upper)):
            raise ValueError(f"starting point outside SABR bounds: {start}")
        fit = least_squares(
            price_residuals, x0, bounds=(lower, upper), args=(snapshot, beta),
            max_nfev=3000,
        )
        if np.all(np.isfinite(fit.fun)):
            fits.append(fit)
    if not fits:
        raise RuntimeError("all SABR optimizations failed")
    best = min(fits, key=lambda fit: float(np.dot(fit.fun, fit.fun)))
    fitted_targets, rmse, within = summarize_fit(snapshot, best.x, beta)
    return CalibrationResult(
        snapshot, float(best.x[0]), float(beta), float(best.x[1]),
        float(best.x[2]), bool(best.success), str(best.message), rmse, within,
        fitted_targets,
    )


def calibrate_times(
    timestamps: Iterable[pd.Timestamp | str],
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    beta: float = 0.7,
    max_age_seconds: float = 60,
) -> pd.DataFrame:
    """Calibrate requested times; save a summary and each quote-level fit."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for timestamp in dict.fromkeys(pd.Timestamp(t) for t in timestamps):
        record = {"requested_time": timestamp}
        try:
            snapshot = load_market_snapshot(timestamp, data_dir, max_age_seconds=max_age_seconds)
            result = calibrate_snapshot(snapshot, beta)
            name = f"fit_{snapshot.session}_{snapshot.requested_time:%Y-%m-%d_%H%M%S_%f}"
            result.fitted_targets.to_csv(output_dir / f"{name}_quotes.csv", index=False)
            record.update({
                "requested_time": snapshot.requested_time,
                "status": "ok" if result.success else "optimizer_warning",
                "session": snapshot.session,
                "file_date": snapshot.file_date,
                "last_trading_date": snapshot.last_trading_date,
                "settlement_time": snapshot.settlement_time,
                **{f"{name}_time": time for name, time in snapshot.row_times.items()},
                "spot": snapshot.spot,
                "spot_from_es": snapshot.spot_from_es,
                "forward": snapshot.forward,
                "T": snapshot.maturity,
                "r": snapshot.rate,
                "discount": snapshot.discount,
                "atm_strike": snapshot.atm_strike,
                "alpha": result.alpha,
                "beta": result.beta,
                "rho": result.rho,
                "nu": result.nu,
                "price_rmse": result.price_rmse,
                "within_bid_ask": result.within_bid_ask,
                "within_bid_ask_fraction": result.within_bid_ask / len(snapshot.targets),
                "target_count": len(snapshot.targets),
                "exclusions": str(snapshot.exclusions),
                "optimizer_message": result.message,
                "quote_fit_file": f"{name}_quotes.csv",
            })
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            record.update({"status": "error", "error": str(error)})
        records.append(record)
    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", action="append", help="New York time, repeatable")
    parser.add_argument("--events", action="store_true", help="Use unique timestamps in the event CSV")
    parser.add_argument("--event-offsets", default="0",
                        help="Comma-separated minutes around each event, e.g. -10,0,10")
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    times = [pd.Timestamp(value) for value in (args.timestamp or [])]
    if args.events:
        offsets = [int(value) for value in args.event_offsets.split(",")]
        for event_time in load_events(str(EVENTS_CSV))["time"].drop_duplicates():
            times.extend(pd.Timestamp(event_time) + pd.Timedelta(minutes=m) for m in offsets)
    if not times:
        parser.error("give --timestamp or --events")
    summary = calibrate_times(times, args.data_dir, args.output_dir, args.beta)
    columns = ["requested_time", "status", "alpha", "rho", "nu", "price_rmse", "error"]
    print(summary[[column for column in columns if column in summary]].to_string(index=False))
    print(f"Saved {args.output_dir / 'calibration_summary.csv'}")


if __name__ == "__main__":
    main()
