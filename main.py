"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

Forecasts each GTA VI PS5 trophy's Exophase earn rate at the case target,
as five quantiles per trophy, scored by pinball loss on the logit scale.

Why this shape
---------------
An earn rate is a ratio, earned / players, and both halves move through
launch week. A trophy's rate climbs toward the share of players who ever
earn it, so its logit drifts roughly linearly for a while and then bends
flat -- a straight line extrapolated 4.5 days past a 2.5-day history runs
away, worst for common trophies already near their ceiling. The fix used
here is to measure only the *current* drift and let its effect saturate
with horizon:

    logit(r_target) = logit(r_last) + drift * H * TAU / (TAU + H)

with H the horizon in hours and TAU = 36 h, so the total move is bounded
by drift * TAU no matter how far the target sits.

The same damped drift is run on ln(earned) and ln(players) separately and
the ratio taken: earned counts stay monotone, earned <= players is enforced
by construction, and the 100% trophy behaves. The two medians (ratio form
and direct logit form) are averaged 50/50 on the logit scale.

Exophase refreshes on its own clock, so consecutive reads often repeat the
same page verbatim. Only distinct page states carry trend information, so
every fit uses distinct states stamped at their first appearance -- the
best available estimate of each refresh time -- and only the most recent
36 hours of them.

Per-trophy drifts are noisy (rates print to 2 decimals; refreshes jump), so
each trophy's drift is pulled toward its tier's mean drift and the global
mean drift, the pull weighted by the fit's effective sample size.

Spreads (pinball loss rewards honest quantiles) combine a random-walk base
term, each trophy's own residual volatility scaled by how far the target
extrapolates past the history, and a staleness term: when the last distinct
state is old, a refresh jump is pending. When the whole history is a single
frozen page the median is shifted down (new players dilute every rate at
the overdue refresh) and the spread widened.

Deterministic closed-form least squares; standard library only; no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels, in the order each output row lists them.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those levels.
Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- model constants (see the docstring) ---
DAMP_TAU_HOURS = 36.0       # damping timescale for drift extrapolation
LOCAL_WINDOW_HOURS = 36.0   # only distinct states this recent feed the drifts
MIN_WINDOW_STATES = 3       # keep at least this many states even if older
SHRINK_STRENGTH = 4.0       # own drift weight = n_eff / (n_eff + SHRINK_STRENGTH)
BASE_VOL_PER_SQRT_DAY = 0.45  # random-walk spread, logit units per sqrt-day
FROZEN_MEDIAN_SHIFT = 0.35  # logit shift down when history is one frozen page
FROZEN_EXTRA_SPREAD = 0.45  # extra logit spread for a frozen history
STALE_SPREAD_PER_HOUR = 0.012
STALE_SPREAD_CAP = 0.35
SPREAD_FLOOR = 0.15
SPREAD_CEIL = 1.60


def clamp_rate(pct):
    """Clamp a percent into the range where a logit exists."""
    return min(99.99, max(0.01, pct))


def logit_of(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def pct_of_logit(lg):
    if lg >= 0.0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4_js(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_after_launch(moment):
    return (moment - LAUNCH).total_seconds() / 3600.0


def slope_fit(xs, ys):
    """Least squares for y ~ a + b*x. Returns (slope, resid_std, n_eff).

    Returns None when the x values cannot pin a slope down.
    """
    count = len(xs)
    if count < 2:
        return None
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    if count > 2:
        resid_var = sum((y - (intercept + slope * x)) ** 2
                        for x, y in zip(xs, ys)) / count
        resid_std = math.sqrt(max(resid_var, 0.0))
    else:
        resid_std = 0.0
    # Uniform weights here, so the effective sample size is the count.
    return slope, resid_std, float(count)


def collapse_reads(history):
    """Collapse consecutive identical page states, stamped at first sight.

    Returns a list of (hours_after_launch, players_tracked, rates) with one
    entry per distinct page state, oldest first.
    """
    snapshots = []
    previous_key = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == previous_key:
            continue
        snapshots.append((hours_after_launch(parse_ts(read["at"])),
                          read["players_tracked"], read["rates"]))
        previous_key = key
    return snapshots


def shrink_slope(own, tier_mean, global_mean, n_eff):
    """Pull a noisy per-trophy slope toward tier + global priors."""
    prior = 0.5 * tier_mean + 0.5 * global_mean
    own_weight = n_eff / (n_eff + SHRINK_STRENGTH)
    return own_weight * own + (1.0 - own_weight) * prior


def forecast_case(case, name):
    if case["levels"] != QUANTILES:
        raise ValueError(f"{name} asks for levels {case['levels']}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    trophies = case["trophies"]
    horizon_hours = case["horizon_minutes"] / 60.0
    horizon_days = case["horizon_minutes"] / 1440.0
    checkpoint_h = hours_after_launch(parse_ts(case["at"]))

    snaps = collapse_reads(case["history"])
    last_t, last_players, last_rates = snaps[-1]
    first_t = snaps[0][0]
    history_span_h = max(last_t - first_t, 1.0)
    stale_hours = max(checkpoint_h - last_t, 0.0)
    frozen = len(snaps) == 1

    # Local window: the most recent distinct states (at least MIN_WINDOW_STATES).
    window = [s for s in snaps if s[0] >= last_t - LOCAL_WINDOW_HOURS]
    if len(window) < MIN_WINDOW_STATES:
        window = snaps[-MIN_WINDOW_STATES:] if len(snaps) >= MIN_WINDOW_STATES else snaps
    window_t = [s[0] for s in window]

    # Damping: the drift's total effect over this horizon.
    # move = drift * horizon_hours * TAU / (TAU + horizon_hours)
    damp_factor = DAMP_TAU_HOURS / (DAMP_TAU_HOURS + horizon_hours)

    # --- players(t): damped drift of ln(players), never shrinking ---
    players_fit = slope_fit(window_t,
                            [math.log(max(s[1], 1)) for s in window])
    players_drift = max(players_fit[0], 0.0) if players_fit else 0.0
    players_target = max(
        math.exp(math.log(max(last_players, 1))
                 + players_drift * horizon_hours * damp_factor),
        float(last_players))

    # --- per-trophy fits on the window ---
    # For each trophy: slope of ln(earned) and of logit(rate), with residuals.
    earned_fits, rate_fits = [], []
    for idx in range(len(trophies)):
        ln_earned, logit_rates = [], []
        for _t, players, rates in window:
            rate = rates[idx]
            rate = 0.0 if rate is None else rate
            earned = rate / 100.0 * players
            ln_earned.append(math.log(max(earned, 0.0) + 0.5))
            logit_rates.append(logit_of(rate))
        earned_fits.append(slope_fit(window_t, ln_earned))
        rate_fits.append(slope_fit(window_t, logit_rates))

    def average_slope(fits, indices):
        slopes = [f[0] for j, f in enumerate(fits)
                  if f is not None and j in indices]
        return sum(slopes) / len(slopes) if slopes else 0.0

    all_idx = set(range(len(trophies)))
    global_earned = average_slope(earned_fits, all_idx)
    global_rate = average_slope(rate_fits, all_idx)
    tier_earned, tier_rate = {}, {}
    for tier in sorted({t["tier"] for t in trophies}):
        members = {j for j, t in enumerate(trophies) if t["tier"] == tier}
        tier_earned[tier] = average_slope(earned_fits, members)
        tier_rate[tier] = average_slope(rate_fits, members)

    # --- spread scaffolding, shared by every trophy ---
    base_spread = BASE_VOL_PER_SQRT_DAY * math.sqrt(horizon_days)
    if frozen:
        base_spread = math.sqrt(base_spread ** 2 + FROZEN_EXTRA_SPREAD ** 2)
    stale_spread = min(STALE_SPREAD_PER_HOUR * stale_hours, STALE_SPREAD_CAP)
    # How far the target extrapolates past the history, in days.
    extrap = math.sqrt(max(horizon_days / max(history_span_h / 24.0, 0.25), 1.0))

    rows = []
    for idx, trophy in enumerate(trophies):
        last_rate = last_rates[idx]
        last_rate = 0.0 if last_rate is None else last_rate
        last_logit = logit_of(last_rate)

        # --- ratio forecast: ln(earned) drift over ln(players) drift ---
        fit_e = earned_fits[idx]
        if fit_e is not None:
            slope_e, _resid_e, n_eff = fit_e
            slope_e = shrink_slope(slope_e, tier_earned[trophy["tier"]],
                                   global_earned, n_eff)
            slope_e = max(slope_e, 0.0)  # earned counts never shrink
            earned_last = math.log(max(last_rate / 100.0 * last_players, 0.0) + 0.5)
            earned_target = math.exp(earned_last
                                     + slope_e * horizon_hours * damp_factor) - 0.5
            earned_target = max(earned_target, 0.0)
            ratio_rate = 100.0 * min(earned_target, players_target) / players_target
        else:
            ratio_rate = last_rate

        # --- direct forecast: logit(rate) drift ---
        fit_r = rate_fits[idx]
        if fit_r is not None:
            slope_r, resid_r, n_eff = fit_r
            slope_r = shrink_slope(slope_r, tier_rate[trophy["tier"]],
                                   global_rate, n_eff)
            direct_rate = pct_of_logit(last_logit
                                       + slope_r * horizon_hours * damp_factor)
            resid = resid_r
        else:
            direct_rate = last_rate
            resid = 0.0

        # Ensemble on the logit scale; frozen history shifts the median down.
        median_logit = 0.5 * (logit_of(ratio_rate) + logit_of(direct_rate))
        if frozen:
            median_logit -= FROZEN_MEDIAN_SHIFT
        median = pct_of_logit(median_logit)

        # A trophy already at 100% stays (nearly) there.
        if last_rate >= 99.9:
            median = 99.95
            spread = 0.10
        else:
            spread = math.sqrt(base_spread ** 2
                               + (resid * extrap) ** 2
                               + stale_spread ** 2)
            spread = min(max(spread, SPREAD_FLOOR), SPREAD_CEIL)

        median_logit = logit_of(median)
        quantiles = []
        for z in Z_SCORES:
            q = pct_of_logit(median_logit + z * spread)
            quantiles.append(round4_js(min(100.0, max(0.0, q))))
        for k in range(1, 5):  # rounding must not break monotonicity
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
