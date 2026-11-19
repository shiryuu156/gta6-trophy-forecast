"""gta6-trophy-rarity-forecast-01 entry: damped local drift + earned-count
decomposition, with hierarchical shrinkage.

Model
-----
An Exophase earn rate is a ratio: rate_i(t) = earned_i(t) / players(t). Both
parts grow through launch week, and the rate's move is their difference.

Earn-rate curves saturate (a trophy's rate rises toward the share of players
who ever earn it), so a straight trend line extrapolated 4.5 days past a
2.5-day history overshoots badly -- worst for common trophies already near
their ceiling. Instead we estimate each trophy's CURRENT drift (logit per
hour) from its most recent distinct page states and extrapolate it with a
damping factor that decays with horizon: the total forecast move is bounded
by drift * TAU_DAMP, so the forecast approaches an asymptote instead of
running away on a line:

    logit(r_target) = logit(r_last) + drift * H * TAU / (TAU + H)

where H is the horizon in hours and TAU = 36 h.

The same damped drift is applied to ln(earned) and ln(players) separately
and the ratio taken (earned-count decomposition: keeps earned counts monotone,
enforces earned <= players, handles the 100% trophy naturally). The two
medians are averaged on the logit scale.

Because Exophase refreshes on its own schedule, consecutive reads often repeat
the same page. Only distinct page states carry trend information, so the fits
use distinct states stamped at their first appearance (the best estimate of
each refresh time).

Per-trophy drifts are noisy (rates print to 2 decimals; refreshes jump), so
each trophy's drift is shrunk toward its tier's mean drift and the global mean
drift, with the shrinkage weight set by the fit's effective sample size.

Spreads (pinball loss rewards honest quantiles):
  * base random-walk term, 0.55 logit units per sqrt-day of horizon;
  * per-trophy residual volatility from the logit fit, scaled by how far the
    target extrapolates past the history;
  * a staleness term: when the last distinct page state is old, a refresh
    jump is pending, so the spread widens with the stale time.
  * when the whole history is one frozen page (the practice cases), the median
    is shifted down 0.35 logit (a refresh is overdue on launch night and new
    players dilute every rate) and the spread is widened.

Everything is deterministic closed-form least squares; standard library only.
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

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Spread of a rate's move, in logit units per square-root day (base term).
# Tuned on a cohort-model backtest: 0.35 is best in-simulation, but the real
# world has shocks the simulation cannot see (weekend player surge, refresh
# blackouts), so 0.45 keeps a safety margin while still beating the house
# baseline's 0.5 -- our median is better because we model the trend.
SIGMA0 = 0.45
# Damping timescale for the drift extrapolation, in hours: the total forecast
# move is bounded by drift * TAU_DAMP no matter how long the horizon.
TAU_DAMP_H = 36.0
# Window for the local drift fit: only distinct states this recent (hours
# before the last state) are used, at least MIN_STATES states.
DRIFT_WINDOW_H = 36.0
MIN_STATES = 3
# Shrinkage strength: a trophy's own slope gets weight n_eff / (n_eff + K).
SHRINK_K = 4.0
# Median shift (logit) when the entire history is a single frozen page.
FROZEN_SHIFT = 0.35
# Extra spread (logit) when the entire history is a single frozen page.
FROZEN_SPREAD = 0.45
# Staleness spread: per hour the last distinct state has been unchanged.
STALE_PER_HOUR = 0.012
STALE_CAP = 0.35
# Hard caps on the total spread, in logit units.
SIGMA_MIN = 0.15
SIGMA_MAX = 1.60


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    p = clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg >= 0:
        e = math.exp(-lg)
        return 100.0 / (1.0 + e)
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt):
    return (dt - RELEASE).total_seconds() / 3600.0


def wls(xs, ys, ws):
    """Weighted least squares for y ~ a + b*x. Returns (a, b, resid_std, n_eff)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    if len(xs) > 2:
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        var = sum(w * r * r for r, w in zip(resid, ws)) / sw
        resid_std = math.sqrt(max(var, 0.0))
    else:
        resid_std = 0.0
    n_eff = sw * sw / sum(w * w for w in ws)
    return a, b, resid_std, n_eff


def distinct_states(history):
    """Collapse consecutive identical page states, stamped at first appearance.

    Returns a list of (hours_since_release, players, rates_tuple).
    """
    states = []
    prev_key = None
    for rec in history:
        if "h" in rec:  # backtest harness smuggles hours in
            h = rec["h"]
        else:
            h = hours_since_release(parse_time(rec["at"]))
        key = (rec["players_tracked"], tuple(rec["rates"]))
        if key != prev_key:
            states.append((h, rec["players_tracked"], rec["rates"]))
            prev_key = key
    return states


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    trophies = case["trophies"]
    n = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    h_target = hours_since_release(parse_time(case["target"]))
    x_target = math.log1p(h_target)

    states = distinct_states(case["history"])
    last_h, last_players, last_rates = states[-1]
    first_h = states[0][0]
    span_h = max(last_h - first_h, 1.0)
    stale_h = hours_since_release(parse_time(case["at"])) - last_h
    frozen = len(states) == 1
    horizon_h = case["horizon_minutes"] / 60.0

    # Local drift window: the most recent distinct states (at least MIN_STATES).
    win = [s for s in states if s[0] >= last_h - DRIFT_WINDOW_H]
    if len(win) < MIN_STATES:
        win = states[-MIN_STATES:] if len(states) >= MIN_STATES else states
    win_h = [s[0] for s in win]
    # Damping factor for the drift extrapolation over this horizon.
    damp = TAU_DAMP_H / (TAU_DAMP_H + horizon_h)

    def local_drift(ys):
        """WLS slope of ys on hours over the drift window. Returns
        (drift, resid_std, n_eff) or None if the window is degenerate."""
        if len(win) < 2:
            return None
        ws = [1.0] * len(win)
        fit = wls(win_h, ys, ws)
        if fit is None:
            return None
        _, b, resid, n_eff = fit
        return b, resid, n_eff

    # --- players(t): damped drift of ln P ---
    ln_p_win = [math.log(max(p, 1)) for _, p, _ in win]
    dp = local_drift(ln_p_win)
    if dp is not None:
        drift_p = dp[0]
    else:
        drift_p = 0.0
    # Players never shrink: clamp the drift at >= 0.
    drift_p = max(drift_p, 0.0)
    ln_p_target = math.log(max(last_players, 1)) + drift_p * horizon_h * damp
    p_target = max(math.exp(ln_p_target), last_players)

    # --- per-trophy drifts ---
    tier_of = [t["tier"] for t in trophies]
    drifts_e, drifts_l = [], []
    for i in range(n):
        # Earned counts (+0.5 smoothing) and logit rates over the window.
        ln_e = [math.log(max((r[i] if r[i] is not None else 0.0) / 100.0 * p, 0.0) + 0.5)
                for _, p, r in win]
        lg_r = [logit_pct(r[i] if r[i] is not None else 0.0) for _, _, r in win]
        drifts_e.append(local_drift(ln_e))
        drifts_l.append(local_drift(lg_r))

    def mean_drift(drifts, idxs):
        vals = [d[0] for j, d in enumerate(drifts) if d is not None and j in idxs]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    all_idx = set(range(n))
    global_de = mean_drift(drifts_e, all_idx)
    global_dl = mean_drift(drifts_l, all_idx)
    tier_de, tier_dl = {}, {}
    for tier in set(tier_of):
        idxs = {j for j, t in enumerate(tier_of) if t == tier}
        tier_de[tier] = mean_drift(drifts_e, idxs)
        tier_dl[tier] = mean_drift(drifts_l, idxs)

    # Staleness spread (a refresh jump is pending when the page is stale).
    stale_sigma = min(STALE_PER_HOUR * max(stale_h, 0.0), STALE_CAP)
    base_sigma = SIGMA0 * math.sqrt(horizon_days)
    if frozen:
        base_sigma = math.sqrt(base_sigma * base_sigma + FROZEN_SPREAD * FROZEN_SPREAD)
    # Extrapolation factor: how far past the history the target sits.
    extrap = math.sqrt(max(horizon_days / max(span_h / 24.0, 0.25), 1.0))

    rows = []
    for i, trophy in enumerate(trophies):
        last_r = last_rates[i]
        last_r = 0.0 if last_r is None else last_r

        lg_last = logit_pct(last_r)

        # --- earned-count forecast: damped drift of ln E, divided by P(t) ---
        de = drifts_e[i]
        if de is not None:
            drift_e, _, n_eff = de
            prior = 0.5 * tier_de[tier_of[i]] + 0.5 * global_de
            w_own = n_eff / (n_eff + SHRINK_K)
            drift_e = w_own * drift_e + (1.0 - w_own) * prior
            drift_e = max(drift_e, 0.0)  # earned counts never shrink
            ln_e_last = math.log(max(last_r / 100.0 * last_players, 0.0) + 0.5)
            e_target = math.exp(ln_e_last + drift_e * horizon_h * damp) - 0.5
            e_target = max(e_target, 0.0)
            r_earned = 100.0 * min(e_target, p_target) / p_target
        else:
            r_earned = last_r

        # --- logit-drift forecast ---
        dl = drifts_l[i]
        if dl is not None:
            drift_l, resid_l, n_eff = dl
            prior = 0.5 * tier_dl[tier_of[i]] + 0.5 * global_dl
            w_own = n_eff / (n_eff + SHRINK_K)
            drift_l = w_own * drift_l + (1.0 - w_own) * prior
            lg_target = lg_last + drift_l * horizon_h * damp
            r_logit = inv_logit_pct(lg_target)
            resid = resid_l
        else:
            r_logit = last_r
            resid = 0.0

        # Ensemble on the logit scale.
        lg_med = 0.5 * (logit_pct(r_earned) + logit_pct(r_logit))
        if frozen:
            lg_med -= FROZEN_SHIFT
        median = inv_logit_pct(lg_med)

        # The 100% trophy stays (nearly) 100%.
        if last_r >= 99.9:
            median = 99.95
            sigma = 0.10
        else:
            sigma = math.sqrt(base_sigma * base_sigma
                              + (resid * extrap) ** 2
                              + stale_sigma * stale_sigma)
            sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)

        lg_m = logit_pct(median)
        qs = []
        for z in Z:
            q = inv_logit_pct(lg_m + z * sigma)
            qs.append(round4(min(100.0, max(0.0, q))))
        # Keep quantiles non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})

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
