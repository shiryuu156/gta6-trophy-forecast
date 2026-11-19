# GTA VI Trophy Rarity Forecast

Forecasts every GTA VI PS5 trophy's Exophase earn rate at 2026-11-26 00:00 UTC,
as five quantiles per trophy, for the lowest pinball loss on the logit scale.

## Model

An Exophase earn rate is a ratio: `rate = earned / players`. Both parts grow
through launch week, and the rate's move is their difference.

Earn-rate curves saturate (a trophy's rate rises toward the share of players
who ever earn it), so a straight trend line extrapolated 4.5 days past a
2.5-day history overshoots badly — worst for common trophies already near
their ceiling. Instead the model estimates each trophy's **current drift**
(logit per hour) from its most recent distinct page states and extrapolates
it with a damping factor that decays with horizon, so the forecast approaches
an asymptote instead of running away on a line:

```
logit(r_target) = logit(r_last) + drift * H * TAU / (TAU + H)
```

with `H` the horizon in hours and `TAU = 36 h`. The same damped drift is
applied to `ln(earned)` and `ln(players)` separately and the ratio taken
(earned-count decomposition: keeps earned counts monotone, enforces
earned ≤ players, handles the 100% trophy naturally). The two medians are
averaged on the logit scale.

Because Exophase refreshes on its own schedule, consecutive reads often repeat
the same page. Only distinct page states carry trend information, so the fits
use distinct states stamped at their first appearance (the best estimate of
each refresh time). Per-trophy drifts are shrunk toward their tier's mean
drift and the global mean drift.

Spreads combine a random-walk base term (0.45 logit units per sqrt-day),
per-trophy residual volatility scaled by extrapolation distance, and a
staleness term for a pending refresh jump. When the whole history is one
frozen page (the practice cases), the median shifts down 0.35 logit — a
refresh is overdue on launch night and new players dilute every rate.

## Validation

A cohort-model backtest (`backtest.py`) simulates launch-week dynamics
(power-law player growth, per-trophy earning curves, 2–8 h refresh gaps,
2-decimal rounding) and scores the model against the house baseline over
20 simulated finals:

| seed | model | house baseline | model wins |
|------|-------|----------------|------------|
| 12345 | 0.282 | 0.349 | 19/20 |
| 999 | 0.280 | 0.345 | 17/20 |

Practice cases: output valid (55 rows, monotone quantiles in [0, 100]).
Against the known launch-night refresh (3,868 players at 04:30, rates down),
the model scores ~0.7–1.1 vs the house baseline's ~1.5. Runs are
deterministic (byte-identical) and take ~0.06 s per case (limit 8 s).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, no network.
