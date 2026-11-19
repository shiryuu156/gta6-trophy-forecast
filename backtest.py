"""Backtest + practice validation for the GTA VI trophy rarity forecast.

Two checks, both scored the way the house scores (pinball loss on the
logit scale, mean over trophies and quantiles):

1. Cohort-model backtest: simulates a plausible launch week (power-law
   player growth, per-trophy earning curves, 2-8 h Exophase refresh gaps,
   2-decimal rounding) and scores the model against the house baseline
   (persistence + 0.5*sqrt(days) spread) over simulated finals.

2. Practice-case validation: runs main.py on the kit's real practice cases
   (p00/p01), checks the output format the grader enforces (55 rows, one
   per trophy, monotone quantiles in [0, 100], <= 64 KiB), determinism
   (two runs byte-identical) and runtime (< 8 s per case). The true
   practice answers (the 04:30 / 05:00 UTC refreshed reads) are NOT in the
   kit -- only the server holds them -- so no local pinball loss can be
   computed on the practice cases; instead this reports how far the
   model's medians sit from the house baseline's on the same cases, as a
   sanity check that the frozen-history shift fires.

Usage:
    python3 backtest.py [main.py] [--practice KIT_INPUT_DIR]
"""

import importlib.util
import json
import math
import random
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


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


def stamp(hours_after_release):
    return (RELEASE + timedelta(hours=hours_after_release)).strftime("%Y-%m-%dT%H:%M:%SZ")


def simulate(rng, n_trophies=55, checkpoint_h=59.5, target_h=168.0, read_every_min=30):
    """Cohort-model simulation. Returns (reads, answer_rates)."""
    P0 = 638.0
    b = rng.uniform(0.55, 0.75)

    def players(h):
        return P0 * (1.0 + max(h, 0.0)) ** b

    def inflow(h):
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

    # Earned counts on an hourly grid via the unearned-able pool recurrence
    # (exact for exponential earning curves F(age) = r_inf*(1-exp(-age/tau))).
    grid_step = 1.0
    grid_max = target_h + 2
    n_grid = int(grid_max / grid_step) + 1
    join = [inflow(g * grid_step) * grid_step for g in range(n_grid)]
    earned_grid = [[0.0] * n_grid for _ in range(n_trophies)]
    for i, (r_inf, tau) in enumerate(trophies):
        U = 0.0
        cum_joined = 0.0
        for g in range(n_grid):
            cum_joined += join[g]
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


def baseline_rows(last_rates, horizon_days):
    spread = 0.5 * math.sqrt(horizon_days)
    rows = []
    for rate in last_rates:
        lg = logit_pct(rate if rate is not None else 0)
        rows.append([100.0 / (1.0 + math.exp(-(lg + z * spread))) for z in Z])
    return rows


def run_backtest(model_path, n_sims=20, seed=12345):
    model = load_model(model_path)
    rng = random.Random(seed)
    losses_model, losses_base, losses_persist = [], [], []
    cover_model, cover_base = [], []
    for s in range(n_sims):
        reads, answer = simulate(rng)
        case = {
            "id": "sim",
            "at": stamp(59.5),
            "horizon_minutes": (168.0 - 59.5) * 60,
            "target": stamp(168.0),
            "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
            "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                     "release_date": "2026-11-19", "source": "x"},
            "trophies": [{"id": i + 1, "name": f"T{i+1}", "description": "",
                          "tier": "bronze", "points": 15, "secret": False}
                         for i in range(55)],
            "history": [
                {"slot": "x", "at": stamp(r["h"]), "players_tracked": r["players"],
                 "trophies_earned": 0, "rates": r["rates"]}
                for r in reads
            ],
            "notes": "",
        }
        fc = model.forecast_case(case, "sim")
        rows = {row["id"]: row["q"] for row in fc["trophies"]}
        lm, lb, lp, cm, cb = 0.0, 0.0, 0.0, 0, 0
        last_rates = reads[-1]["rates"]
        d_days = (168.0 - 59.5) / 24.0
        base = baseline_rows(last_rates, d_days)
        for i, y in enumerate(answer):
            q = rows[i + 1]
            lm += pinball_loss(y, q)
            cm += 1 if q[0] <= y <= q[4] else 0
            lb += pinball_loss(y, base[i])
            cb += 1 if base[i][0] <= y <= base[i][4] else 0
            lp += abs(logit_pct(y) - logit_pct(last_rates[i]))
        losses_model.append(lm / 55)
        losses_base.append(lb / 55)
        losses_persist.append(lp / 55)
        cover_model.append(cm / 55)
        cover_base.append(cb / 55)
        print(f"sim {s:2d}: model {lm/55:.4f}  baseline {lb/55:.4f}  "
              f"persist {lp/55:.4f}  cover {cm/55:.2f} vs {cb/55:.2f}")
    print()
    print(f"MODEL    mean {statistics.mean(losses_model):.4f}  "
          f"median {statistics.median(losses_model):.4f}  cover {statistics.mean(cover_model):.2f}")
    print(f"BASELINE mean {statistics.mean(losses_base):.4f}  "
          f"median {statistics.median(losses_base):.4f}  cover {statistics.mean(cover_base):.2f}")
    print(f"PERSIST  mean {statistics.mean(losses_persist):.4f}  (zero-spread median error)")
    wins = sum(1 for a, b in zip(losses_model, losses_base) if a < b)
    print(f"model beats baseline in {wins}/{n_sims} sims (seed {seed})")
    return statistics.mean(losses_model), statistics.mean(losses_base)


def validate_practice(model_path, kit_input_dir):
    """Run main.py on the real practice cases; validate format, determinism,
    runtime, and compare medians against the house baseline."""
    kit_input = Path(kit_input_dir)
    cases = sorted(kit_input.glob("*.in"))
    if not cases:
        print("no practice cases found in", kit_input)
        return False
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        out1, out2 = Path(tmp) / "o1", Path(tmp) / "o2"
        env = dict(__import__("os").environ,
                   MAKEFUN_INPUT_DIR=str(kit_input),
                   MAKEFUN_OUTPUT_DIR=str(out1))
        t0 = time.perf_counter()
        r1 = subprocess.run([sys.executable, model_path], env=env,
                            capture_output=True, text=True)
        run1_s = time.perf_counter() - t0
        env["MAKEFUN_OUTPUT_DIR"] = str(out2)
        t0 = time.perf_counter()
        r2 = subprocess.run([sys.executable, model_path], env=env,
                            capture_output=True, text=True)
        run2_s = time.perf_counter() - t0
        if r1.returncode != 0 or r2.returncode != 0:
            print("main.py FAILED on practice cases:")
            print(r1.stdout, r1.stderr, r2.stdout, r2.stderr)
            return False
        for case_path in cases:
            name = case_path.stem
            case = json.loads(case_path.read_text())
            f1, f2 = (out1 / f"{name}.out"), (out2 / f"{name}.out")
            b1, b2 = f1.read_bytes(), f2.read_bytes()
            if b1 != b2:
                print(f"{name}: NON-DETERMINISTIC (two runs differ)")
                ok = False
            if len(b1) > 65536:
                print(f"{name}: output too large ({len(b1)} bytes)")
                ok = False
            fc = json.loads(b1)
            rows = fc["trophies"]
            assert set(fc) == {"trophies"}, f"{name}: extra keys {set(fc)}"
            assert len(rows) == len(case["trophies"]) == 55, f"{name}: {len(rows)} rows"
            assert sorted(r["id"] for r in rows) == sorted(t["id"] for t in case["trophies"])
            for r in rows:
                assert set(r) == {"id", "q"} and len(r["q"]) == 5
                assert all(0.0 <= q <= 100.0 for q in r["q"])
                assert all(r["q"][k] <= r["q"][k + 1] for k in range(4)), f"{name}: non-monotone {r}"
            # Median shift vs the house baseline on the same case.
            last_rates = case["history"][-1]["rates"]
            base = baseline_rows(last_rates, case["horizon_minutes"] / 1440.0)
            shifts = []
            for r, brow in zip(sorted(rows, key=lambda x: x["id"]), base):
                shifts.append(logit_pct(r["q"][2]) - logit_pct(brow[2]))
            mean_shift = statistics.mean(shifts)
            width = statistics.mean(
                logit_pct(r["q"][4]) - logit_pct(r["q"][0])
                for r in rows)
            base_width = statistics.mean(
                logit_pct(b[4]) - logit_pct(b[0]) for b in base)
            print(f"{name}: valid 55 rows, monotone, {len(b1)} bytes; "
                  f"median logit shift vs baseline {mean_shift:+.3f}; "
                  f"90% band width {width:.3f} logit vs baseline {base_width:.3f}")
    print(f"practice runs: {run1_s:.2f}s and {run2_s:.2f}s total "
          f"({run1_s/len(cases):.2f}s/case, limit 8s/case)")
    print("practice answers are server-side only; no local pinball loss on p00/p01")
    return ok


if __name__ == "__main__":
    args = sys.argv[1:]
    path = args[0] if args and not args[0].startswith("--") else "main.py"
    practice_dir = None
    if "--practice" in args:
        practice_dir = args[args.index("--practice") + 1]
    print("=== cohort backtest, seed 12345 ===")
    m1, b1 = run_backtest(path, seed=12345)
    print()
    print("=== cohort backtest, seed 999 ===")
    m2, b2 = run_backtest(path, seed=999)
    print()
    print(f"SUMMARY model {(m1+m2)/2:.4f} vs baseline {(b1+b2)/2:.4f} "
          f"(means across both seeds)")
    if practice_dir:
        print()
        print("=== practice-case validation ===")
        ok = validate_practice(path, practice_dir)
        sys.exit(0 if ok else 1)
