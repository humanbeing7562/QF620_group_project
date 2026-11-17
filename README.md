# QF620 Group Project: SPX options around the April 2025 tariff announcements

QF620 Stochastic Modelling in Finance, SMU MQF.

This repo studies how SPX, ES futures and SPX option prices reacted to the "Liberation Day" tariff news between 1 and 10 April 2025. Starting from minute-level option quotes, it builds:

- SPX and ES price paths per trading session
- a VIX-style index computed from our own option data, compared with Cboe VIX
- implied vol at key moneyness levels and at fixed deltas (25Δ / 35Δ puts and calls, ATM, ATMF)
- the implied vol smile every minute from 10 minutes before to 10 minutes after each event
- SABR fits and Breeden-Litzenberger risk-neutral densities around each event
- the final figures used in the report

## Folder layout

```
.
├── data/
│   ├── RTH/                       regular session files (09:31 to 16:15 ET)
│   └── ETH/                       overnight session files (20:15 to 09:15 ET next day)
├── 2025 Liberation Day Tariff Main Events.csv   event list (date, time ET, actor, description)
├── event_minute_timestamps.csv    one row per minute, -10 to +10 min, for each graphed event
├── period_end_densities.csv       cached SABR/BL densities at the end of each period
├── rates_cache.csv                cached Treasury yields used for the risk-free rate
│
├── risk_free_rate.py              risk-free rate shared by all scripts
├── key_strike_iv.py               parity forward, BS implied vol, IV at key moneyness levels
├── key_delta_iv.py                IV at fixed deltas, risk reversals and butterflies
├── spx_es_prices.py               SPX / ES price charts and per-session OHLC table
├── vix_index.py                   VIX-style index from option mids vs Cboe VIX
├── smile_per_event.py             smile per minute around each event (x-axis K/F)
├── smile_per_event_delta.py       same smiles with delta on the x-axis
├── quote_selection.py             quote cleaning and SABR calibration targets for one snapshot
├── sabr_calibration.py            SABR fit to option prices (command line tool)
├── breeden_litzenberger.py        SABR vol formula, BL density, density checks
├── event_densities.py             SABR + BL density for every minute around each event
├── report_figures.py              all report figures (fig1 to fig7)
├── sabr_calibration_walkthrough.ipynb   step-by-step notebook for one SABR fit
├── data.py                        older data helper (not used by the scripts above)
│
└── *_output/, report_figures/     generated outputs (committed)
```

## Data

Each session folder holds one set of files per date:

| File | Contents |
|---|---|
| `<date>_assets.csv` | SPX, ES_BID, ES_ASK, VIX, about every 3 seconds |
| `<date>_<call\|put>_<bid\|ask>_<expiry>.csv` | rows are timestamps, columns are strikes (4600 to 6600) |

- All times are New York time (ET), stored without a timezone.
- An ETH file is named after the date the session **starts**. A 03:00 quote on 2 Apr is in `data/ETH/2025-04-01_*`.
- The expiry in the file name is the last trading date. The options are AM-settled, so settlement is 09:30 ET the next trading day.
- Two expiries are used: **16 Apr 2025** for 1 to 4 Apr, and **15 May 2025** from the ETH session of 6 Apr onward. Levels before and after the 5–6 Apr weekend are not directly comparable because of this roll (about 12 to 39 days to expiry).

## Setup

Python 3.11 or newer (developed on 3.13).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pandas scipy matplotlib pillow jupyter
```

`pillow` is needed for the GIFs. `yfinance` is only needed if you switch the rate source in `risk_free_rate.py`.

**Run every script from the repo root.** The scripts use relative paths such as `data/` and `smile_output/`.

## How to run

The steps below go in order: later steps read the CSVs written by earlier ones. All outputs are already committed, so you only need to rerun the steps you change.

```bash
# 1. Risk-free rate (reads rates_cache.csv; --refresh downloads from FRED again)
python risk_free_rate.py

# 2. Market behaviour per session
python spx_es_prices.py          # -> spx_es_output/
python vix_index.py              # -> vix_output/
python key_strike_iv.py          # -> key_strike_output/
python key_delta_iv.py           # -> key_delta_output/

# 3. Smiles around each event
python smile_per_event.py        # -> smile_output/
python smile_per_event_delta.py  # -> smile_delta_output/

# 4. SABR + risk-neutral densities
python event_densities.py        # -> event_density_output/

# 5. Report figures
python report_figures.py         # -> report_figures/
```

`vix_index.py`, `key_strike_iv.py` and `key_delta_iv.py` work through every snapshot of every session, so they take a while.

### SABR calibration on its own

`sabr_calibration.py` fits SABR at any timestamp you give it and writes the results to `calibration_output/`:

```bash
# one or more timestamps (ET)
python sabr_calibration.py --timestamp "2025-04-02 09:31:24" --timestamp "2025-04-07 09:00:00"

# every event in the events CSV, at -10, 0 and +10 minutes
python sabr_calibration.py --events --event-offsets=-10,0,10

# other options: --beta (default 0.7), --data-dir, --output-dir
```

To see each step of one fit (quote checks, forward, target selection, residuals, fitted vs market prices), open `sabr_calibration_walkthrough.ipynb`.

## Method summary

**Forward.** Put-call parity at a strike with valid call and put mids. Black-Scholes then uses spot with an implied dividend yield q = r − ln(F/S)/T, so prices stay consistent with the market forward.

**Risk-free rate.** Average of the FRED 1-month Treasury yield (DGS1MO) over 1–10 Apr 2025, about 4.365%. Every script imports `RISK_FREE_RATE` from `risk_free_rate.py`.

**Time to expiry.** ACT/365 calendar time to 09:30 ET on the settlement date.

**Implied vol.** Taken from the OTM option at each strike: puts below F, calls at or above F. Quotes with a mid below 0.10 are treated as stuck at the minimum tick and dropped. Key levels are interpolated linearly in IV between neighbouring strikes. Points outside the 4600–6600 strike grid are left as NaN, not extrapolated.

**Delta levels.** Black-Scholes forward delta at each strike's own IV. 10Δ is not tracked because the 10Δ put needs a strike below 4600 for most of 4–10 Apr. 35Δ is tracked as well because the 25Δ put falls off the grid for much of 7–9 Apr. Delta is used throughout the report because it stays comparable across the expiry roll, which a fixed % of F does not.

**VIX-style index.** Cboe VIX formula on a single expiry (we only have one expiry per day, so there is no interpolation to 30 days). Model-free, uses market mids directly.

**SABR.** β fixed at 0.7. α, ρ, ν are fitted to OTM option mid prices plus one near-ATM straddle, which replaces the single OTM target at that strike. Hagan's Black vol approximation is converted to a discounted Black price before computing price residuals. Fits are checked against the bid-ask spread.

**Risk-neutral density.** Call prices from the fitted SABR smile on a 5-point strike grid, then Breeden-Litzenberger: q(K) = e^{rT} ∂²C/∂K². The grid stays inside the strikes used in the fit.

## Report figures

| Figure | Content |
|---|---|
| `fig1_overview.png` | 1–10 Apr: SPX & ES, VIX-style vs Cboe VIX, ATMF IV, 25Δ/35Δ risk reversals |
| `fig2_liberation_day.png` | 2 Apr 16:07, Liberation Day announcement |
| `fig3_china_retaliates.png` | 4 Apr 06:00 WTO filing and 07:21 China +34% |
| `fig4_7_april.png` | 7 Apr 10:13 false pause report, 10:35 denial, 11:14 ultimatum |
| `fig5_china_84.png` | 9 Apr 07:00 China raises tariff to 84% |
| `fig6_pause.png` | 9 Apr 13:18, 90-day pause |
| `fig7_period_end_densities.png` | risk-neutral densities at the end of each period, overlaid |
| `fig_weekend_roll.png` | smile by delta across the 5–6 Apr weekend and expiry roll |

The `*_delta.png` files in `report_figures/` are from an older run; the current script plots smiles against delta and writes the names without the suffix.

Each event figure has (a) SPX & ES, (b) VIX-style index vs Cboe VIX, (c) IV at 25/35Δ puts, ATMF and 35/25Δ calls on the left, and (d) the smile at every minute from −10 to +10 and (e) the BL density at −10 and +10 minutes on the right.

If `period_end_densities.csv` is missing, `report_figures.py` refits SABR at the period ends and recreates it.

## Known data issues

- If an option file is missing for a session, the scripts skip that session and print a message.
- In the ETH session of 6 Apr the SPX column is off; spot is rebuilt from the ES mid there.
- The cash SPX index lags after 16:00 and during fast moves.
- RTH on 10 Apr starts at 09:57.
- The ETH session starting 9 Apr is not part of the dataset by design, so the 10 Apr 00:01 and 06:44 events fall outside the sample.
- Event times on 7 Apr were corrected from 09:13/09:35 to 10:13/10:35 ET.

## Coding conventions

See `.claude/skills/python-function-standards/SKILL.md`: type hints on every parameter and return value, and a short 1–3 line docstring for every function.
