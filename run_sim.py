#!/usr/bin/env python3
"""
run_sim.py - Trinetra sonar obstacle-avoidance simulator, command line.

Single run (live window):
  python run_sim.py --scenario pillar_left
  python run_sim.py --scenario square_tank --mode STOP --speed 0.25 --seed 3
  python run_sim.py --scenario u_trap --stage VFH --realtime 4        # 4x real time
  python run_sim.py --scenario pillar_left --headless --save out/     # PNG summary, no window
  python run_sim.py --scenario noisy_pillar --log-pings pings.jsonl   # write pings for replay

Batch / Monte Carlo (no graphics, faster than real time):
  python run_sim.py --batch --seeds 10 --out results/
  python run_sim.py --batch --scenarios pillar_left,u_trap --modes AVOID --stages SIDESTEP,VFH --seeds 20

Other:
  python run_sim.py --list
  python run_sim.py --replay log12_events.csv --fixed-pose 0,0,90
  python run_sim.py --world worlds/my_tank.json --goal 8,3 --start 1,1,0   # your own world file
  python run_sim.py --config my_overrides.json ...                       # see config.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config as cfgmod  # noqa: E402
from controllers import Leg  # noqa: E402
from scenarios import all_scenarios, export_worlds  # noqa: E402

# default batch matrix: (scenario, mode, stage)
DEFAULT_MATRIX = [
    ("wall_0.1", "STOP", "SIDESTEP"), ("wall_0.2", "STOP", "SIDESTEP"), ("wall_0.3", "STOP", "SIDESTEP"),
    ("wall_0.3_noslow", "STOP", "SIDESTEP"), ("wall_0.2", "OFF", "SIDESTEP"),
    ("pillar_left", "AVOID", "SIDESTEP"), ("pillar_right", "AVOID", "SIDESTEP"),
    ("pillar_left", "AVOID", "VFH"), ("pillar_right", "AVOID", "VFH"),
    ("gap_wide", "AVOID", "SIDESTEP"), ("gap_narrow", "AVOID", "SIDESTEP"), ("gap_narrow", "AVOID", "VFH"),
    ("u_trap", "AVOID", "SIDESTEP"), ("u_trap", "AVOID", "VFH"),
    ("square_tank", "OFF", "SIDESTEP"), ("square_tank", "STOP", "SIDESTEP"), ("square_tank", "AVOID", "SIDESTEP"),
    ("square_tank_close", "STOP", "SIDESTEP"), ("square_tank_close", "AVOID", "SIDESTEP"),
    ("noisy_open", "AVOID", "SIDESTEP"), ("noisy_pillar", "AVOID", "SIDESTEP"), ("noisy_pillar", "AVOID", "VFH"),
    ("moving_crossing", "STOP", "SIDESTEP"), ("moving_crossing", "AVOID", "SIDESTEP"),
]


def custom_scenario(world_path: str, goal: str, start: str, cruise: float) -> dict:
    with open(world_path, "r", encoding="utf-8") as f:
        w = json.load(f)
    gx, gy = (float(v) for v in goal.split(","))
    sx, sy, sh = (float(v) for v in start.split(","))
    return dict(name=w.get("name", "custom"), world=w, legs=[Leg("WP", gx, gy)], start=(sx, sy, sh), cruise=cruise,
                mode="AVOID", stage="SIDESTEP", end="mission", timeout=600.0, overrides={}, desc="custom world")


def run_single(scen: dict, a) -> dict:
    from simulation import Simulation
    overrides = cfgmod.load_json(a.config)
    sim = Simulation(scen, mode=a.mode, stage=a.stage, speed=a.speed, seed=a.seed, overrides=overrides,
                     randomize=a.randomize, ping_log=a.log_pings)
    print(f"[SIM] {scen['name']}: {scen.get('desc', '')}")
    print(f"[SIM] mode {sim.mode} stage {sim.stage} cruise {sim.cruise} m/s seed {a.seed}; "
          f"sonar sector refresh {sim.sonar.sector_refresh_s:.2f} s")
    view = None
    on_frame = None
    if not a.headless:
        from viz import SimView
        view = SimView(sim, live=True, follow=a.follow)
        on_frame = view.update
    res = sim.run(on_frame=on_frame, frame_every=a.frame_every, realtime=a.realtime)
    for t, e in sim.events:
        print(f"   t={t:6.1f}s  {e}")
    print(json.dumps(res, indent=2))
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        from viz import summary_figure, SimView
        base = os.path.join(a.save, f"{scen['name']}_{sim.mode}_{sim.stage}_s{a.seed}")
        summary_figure(sim, base + "_summary.png")
        if view is None:
            v2 = SimView(sim, live=False)
            v2.save(base + "_final.png")
            v2.close()
        else:
            view.fig.savefig(base + "_final.png", dpi=90)
        print(f"[SIM] saved {base}_summary.png")
    if view is not None and not a.headless:
        view.plt.ioff()
        view.plt.show()
    return res


def _snapshot_hook(sim, base: str, max_snaps: int = 2):
    """Save live-view frames a few seconds after interesting avoidance events."""
    from viz import SimView
    state = {"view": None, "due": [], "n": 0, "seen": 0}
    keys = ("SIDESTEP", "BACKOFF", "WP_BLOCKED", "STOP_TRIGGER", "REJOIN", "COLLISION")

    def hook(s):
        ev = s.events[state["seen"]:]
        state["seen"] = len(s.events)
        for t, e in ev:
            if any(e.startswith(k) for k in keys) and len(state["due"]) + state["n"] < max_snaps:
                if not state["due"] or t > state["due"][-1] + 8:
                    state["due"].append(t + (6.0 if e.startswith("SIDESTEP") else 3.0))
        if state["due"] and s.t >= state["due"][0]:
            state["due"].pop(0)
            if state["view"] is None:
                state["view"] = SimView(s, live=False)
            state["n"] += 1
            state["view"].save(f"{base}_live{state['n']}_t{int(s.t)}.png")
        return True
    return hook, state


def run_batch(a) -> None:
    from simulation import Simulation
    from viz import summary_figure
    S = all_scenarios()
    overrides = cfgmod.load_json(a.config)
    matrix = DEFAULT_MATRIX
    if a.scenarios:
        names = a.scenarios.split(",")
        modes = a.modes.split(",") if a.modes else None
        stages = a.stages.split(",") if a.stages else ["SIDESTEP"]
        matrix = [(n, m, st) for n in names for m in (modes or [S[n]["mode"]]) for st in stages
                  if not (m != "AVOID" and st != stages[0])]
    out = a.out
    os.makedirs(out, exist_ok=True)
    rows: List[dict] = []
    baseline: Dict[tuple, float] = {}
    t0 = time.time()
    seeds = list(range(1, a.seeds + 1))
    for name, mode, stage in matrix:
        scen = S[name]
        for seed in seeds:
            rnd = seed > 1 or a.randomize          # seed 1 = nominal geometry, others randomised
            sim = Simulation(scen, mode=mode, stage=stage, seed=seed, overrides=overrides, randomize=rnd)
            hook = None
            if seed == 1 and not a.no_plots:
                base = os.path.join(out, f"{name}_{mode}{'_' + stage if mode == 'AVOID' else ''}")
                hook, _ = _snapshot_hook(sim, base)
            r = sim.run(on_frame=hook, frame_every=2)
            if seed == 1 and not a.no_plots:
                summary_figure(sim, f"{base}_summary.png")
            if scen.get("end", "mission") == "mission":
                key = (name, seed)
                if key not in baseline:
                    b = Simulation(scen, mode="OFF", seed=seed, overrides=overrides, randomize=rnd, empty_world=True)
                    baseline[key] = b.run()["time_s"]
                r["baseline_s"] = baseline[key]
                r["time_lost_s"] = round(r["time_s"] - baseline[key], 1) if r["completed"] else None
            rows.append(r)
        done = [x for x in rows if x["scenario"] == name and x["mode"] == mode and x["stage"] == stage]
        print(f"[BATCH] {name:16s} {mode:5s} {stage:8s} runs {len(done):2d}  collisions {sum(x['collisions'] for x in done)}"
              f"  min clr {min(x['min_clearance_m'] for x in done):5.2f}  false stops {sum(x['false_stops'] for x in done)}"
              f"  completed {sum(x['completed'] for x in done)}/{len(done)}   ({time.time() - t0:5.0f} s)", flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(out, "runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    table = summarize(rows, matrix)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(table)
    print(table)
    print(f"[BATCH] {len(rows)} runs in {time.time() - t0:.0f} s -> {out}/runs.csv, summary.md")


def _mean(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isinf(x))]
    return statistics.mean(v) if v else None


def summarize(rows: List[dict], matrix) -> str:
    hdr = ("| scenario | mode | stage | runs | completed | safe aborts | collisions | min clr [m] | mean min clr [m] | "
           "stop clr [m] | false stops | late | time lost [s] | stops | side-steps | back-offs | skipped legs |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for name, mode, stage in matrix:
        g = [r for r in rows if r["scenario"] == name and r["mode"] == mode and r["stage"] == stage]
        if not g:
            continue
        f = lambda v, d=2: "-" if v is None else (f"{v:.{d}f}")
        mc = [r["min_clearance_m"] for r in g]
        lines.append(
            f"| {name} | {mode} | {stage if mode == 'AVOID' else '-'} | {len(g)} | {sum(r['completed'] for r in g)}/{len(g)} | "
            f"{sum(r.get('aborted', False) for r in g)} | {sum(r['collisions'] for r in g)} | {f(min(mc) if not all(math.isinf(x) for x in mc) else None)} | "
            f"{f(_mean(mc))} | {f(_mean([r.get('stop_clearance_m') for r in g]))} | {sum(r['false_stops'] for r in g)} | "
            f"{sum(r['late'] for r in g)} | {f(_mean([r.get('time_lost_s') for r in g]), 0)} | "
            f"{f(_mean([r['stops'] for r in g]), 1)} | {f(_mean([r['sidesteps'] for r in g]), 1)} | "
            f"{f(_mean([r['backoffs'] for r in g]), 1)} | {f(_mean([r['skipped_legs'] for r in g]), 1)} |")
    return hdr + "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="pillar_left")
    ap.add_argument("--mode", choices=["OFF", "WARN", "STOP", "AVOID"], type=str.upper)
    ap.add_argument("--stage", choices=["SIDESTEP", "VFH"], type=str.upper)
    ap.add_argument("--speed", type=float, help="cruise speed m/s (overrides the scenario)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--randomize", action="store_true", help="jitter start pose / obstacles (Monte Carlo)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--follow", action="store_true", help="live view follows the vehicle")
    ap.add_argument("--realtime", type=float, default=0.0, help="0 = as fast as possible, 1 = real time, 4 = 4x")
    ap.add_argument("--frame-every", type=int, default=4, help="draw every N control ticks (20 Hz)")
    ap.add_argument("--save", help="folder for PNGs")
    ap.add_argument("--log-pings", help="write every ping as JSON lines (for replay.py)")
    ap.add_argument("--config", help="JSON overrides, e.g. {\"avoidance\": {\"stop_base_m\": 1.5}}")
    ap.add_argument("--world", help="custom world JSON (use with --goal/--start)")
    ap.add_argument("--goal", default="20,0")
    ap.add_argument("--start", default="0,0,0")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--scenarios", help="comma list for --batch (default: full matrix)")
    ap.add_argument("--modes", help="comma list for --batch")
    ap.add_argument("--stages", help="comma list for --batch")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--replay", help="logged pings (auv_depth events CSV or JSONL)")
    ap.add_argument("--pose-csv")
    ap.add_argument("--fixed-pose", default="0,0,0")
    ap.add_argument("--max-pings", type=int)
    a = ap.parse_args(argv)
    if a.headless or a.batch:
        import matplotlib
        matplotlib.use("Agg")
    if a.list:
        for k, s in all_scenarios().items():
            print(f"{k:18s} {s['mode']:5s} {s['desc']}")
        return
    if a.replay:
        from replay import main as rmain
        args = [a.replay, "--fixed-pose", a.fixed_pose, "--out", (a.out.rstrip('/') + "_replay.png")]
        if a.pose_csv:
            args += ["--pose-csv", a.pose_csv]
        if a.config:
            args += ["--config", a.config]
        if a.max_pings:
            args += ["--max-pings", str(a.max_pings)]
        rmain(args)
        return
    export_worlds()
    if a.batch:
        run_batch(a)
        return
    scen = custom_scenario(a.world, a.goal, a.start, a.speed or 0.2) if a.world else all_scenarios()[a.scenario]
    run_single(scen, a)


if __name__ == "__main__":
    main()
