"""Backtest harness for the GTA VI trophy rarity forecast model.

Simulates a plausible launch-week Exophase recording with an explicit cohort
model: players join over time (power-law player growth), each cohort earns a
trophy with probability F_i(age) = r_inf_i * (1 - exp(-age / tau_i)), so
earned_i(t) = integral over cohorts. The page refreshes every ~2-8 hours
(stale reads between refreshes) and rates print to 2 decimals.

Scores the model's forecast against the synthetic truth at the target with the
contest's pinball loss on the logit scale, and compares against the house
baseline (persistence + 0.5*sqrt(d) spread).

Usage: python3 backtest.py [main.py]
"""

import importlib.util
import json
import math
import random
import statistics
import sys

Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    p = clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def pinball_loss(y, qs):
    ly = logit_pct(y)
    total = 0.0
    for tau, q in zip(TAUS, qs):
        d = ly - logit_pct(q)
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / 5.0


def simulate(rng, n_trophies=55, checkpoint_h=59.5, target_h=168.0, read_every_min=30):
    """Cohort-model simulation. Returns (reads, answer_rates)."""
    P0 = 638.0
    b = rng.uniform(0.55, 0.75)

    def players(h):
        return P0 * (1.0 + max(h, 0.0)) ** b

    def inflow(h):
        # dP/dh for the power law, per hour.
        return P0 * b * (1.0 + max(h, 0.0)) ** (b - 1.0)

    archetypes = [
        ("instant", 0.02, (0.99, 1.0), (0.1, 1.0)),
        ("early", 0.25, (0.35, 0.75), (2.0, 12.0)),
        ("mid", 0.35, (0.10, 0.40), (12.0, 60.0)),
        ("late", 0.20, (0.02, 0.12), (60.0, 200.0)),
        ("grind", 0.15, (0.002, 0.03), (150.0, 600.0)),
        ("rare", 0.03, (0.0005, 0.005), (300.0, 1000.0)),
    ]
    trophies = []
    for _ in range(n_trophies):
        r = rng.random()
        acc = 0.0
        for _name, share, r_inf_rng, tau_rng in archetypes:
            acc += share
            if r <= acc:
                trophies.append((rng.uniform(*r_inf_rng), rng.uniform(*tau_rng)))
                break
        else:
            trophies.append((0.2, 40.0))

    # Precompute earned counts on a fine hourly grid via cohort integration.
    # E_i(t) = sum over join-hours s<=t of inflow(s) * r_inf_i * (1-exp(-(t-s)/tau_i))
    grid_step = 1.0
    grid_max = target_h + 2
    n_grid = int(grid_max / grid_step) + 1
    # inflow per grid step (players joining in that hour)
    join = [inflow(g * grid_step) * grid_step for g in range(n_grid)]
    earned_grid = [[0.0] * n_grid for _ in range(n_trophies)]
    for i, (r_inf, tau) in enumerate(trophies):
        run = 0.0
        for g in range(n_grid):
            t = g * grid_step
            # New joiners in this step have age ~0 (earn ~0 immediately except
            # instant trophies); older cohorts age by one step.
            # Exact-ish update: E(t+1) = E(t) + sum_cohorts inflow(s)*r_inf*(F(age+1)-F(age))
            # We approximate with the survival form: unearned pool decays.
            # Simpler direct sum each step is O(n^2); use the recurrence:
            # U(t) = players who joined, are still "able", haven't earned.
            # dU/dt = inflow - U/tau ; E = r_inf * (cumulative joined - U)
            pass
        # Recurrence on the unearned-able pool U (exact for exponential F).
        U = 0.0
        cum_joined = 0.0
        for g in range(n_grid):
            cum_joined += join[g]
            # U evolves: U += join (new able players) then decays by exp(-dt/tau)
            U = (U + join[g]) * math.exp(-grid_step / tau)
            earned_grid[i][g] = r_inf * (cum_joined - U)

    def true_rate(i, h):
        g = min(int(h / grid_step), n_grid - 1)
        P = players(h)
        return 100.0 * earned_grid[i][g] / P if P > 0 else 0.0

    refresh_times = [0.0]
    t = 0.0
    while t < target_h + 2:
        t += rng.uniform(2.0, 8.0)
        refresh_times.append(t)

    def page_at(h):
        last = max(rt for rt in refresh_times if rt <= h)
        P = players(last)
        rates = [round(true_rate(i, last) + 1e-9, 2) for i in range(n_trophies)]
        return P, rates

    reads = []
    h = -0.1667
    while h <= checkpoint_h:
        P, rates = page_at(max(h, 0.0))
        reads.append({"h": h, "players": int(round(P)), "rates": rates})
        h += read_every_min / 60.0

    _, rates_ans = page_at(target_h)
    return reads, rates_ans


def load_model(path):
    spec = importlib.util.spec_from_file_location("entry_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_backtest(model_path, n_sims=20, seed=12345, verbose=True):
    model = load_model(model_path)
    rng = random.Random(seed)
    losses_model, losses_base, losses_persist = [], [], []
    cover_model, cover_base = [], []
    for s in range(n_sims):
        reads, answer = simulate(rng)
        case = {
            "id": "sim",
            "at": "2026-11-21T11:30:00Z",
            "horizon_minutes": (168.0 - 59.5) * 60,
            "target": "2026-11-26T00:00:00Z",
            "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
            "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                     "release_date": "2026-11-19", "source": "x"},
            "trophies": [{"id": i + 1, "name": f"T{i+1}", "description": "",
                         "tier": "bronze", "points": 15, "secret": False}
                        for i in range(55)],
            "history": [
                {"slot": "x", "at": "x", "players_tracked": r["players"],
                 "trophies_earned": 0, "rates": r["rates"], "h": r["h"]}
                for r in reads
            ],
            "notes": "",
        }
        fc = model.forecast_case(case, "sim")
        rows = {row["id"]: row["q"] for row in fc["trophies"]}
        lm, lb, lp, cm, cb = 0.0, 0.0, 0.0, 0, 0
        last_rates = reads[-1]["rates"]
        d_days = (168.0 - 59.5) / 24.0
        spread = 0.5 * math.sqrt(d_days)
        for i, y in enumerate(answer):
            q = rows[i + 1]
            lm += pinball_loss(y, q)
            cm += 1 if q[0] <= y <= q[4] else 0
            lg = logit_pct(last_rates[i] if last_rates[i] is not None else 0)
            qb = [100.0 / (1.0 + math.exp(-(lg + z * spread))) for z in Z]
            lb += pinball_loss(y, qb)
            cb += 1 if qb[0] <= y <= qb[4] else 0
            lp += abs(logit_pct(y) - lg)  # pure persistence, zero spread
        losses_model.append(lm / 55)
        losses_base.append(lb / 55)
        losses_persist.append(lp / 55)
        cover_model.append(cm / 55)
        cover_base.append(cb / 55)
        if verbose:
            print(f"sim {s:2d}: model {lm/55:.4f}  baseline {lb/55:.4f}  "
                  f"persist {lp/55:.4f}  cover {cm/55:.2f} vs {cb/55:.2f}")
    print()
    print(f"MODEL    mean {statistics.mean(losses_model):.4f}  median {statistics.median(losses_model):.4f}  cover {statistics.mean(cover_model):.2f}")
    print(f"BASELINE mean {statistics.mean(losses_base):.4f}  median {statistics.median(losses_base):.4f}  cover {statistics.mean(cover_base):.2f}")
    print(f"PERSIST  mean {statistics.mean(losses_persist):.4f}  (zero-spread median error)")
    wins = sum(1 for a, b in zip(losses_model, losses_base) if a < b)
    print(f"model beats baseline in {wins}/{n_sims} sims")
    return statistics.mean(losses_model), statistics.mean(losses_base)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "main.py"
    run_backtest(path)
