"""
replay.py - run logged Ping360 pings through the SAME detection code (no simulator involved).

Supported inputs
  1. auv_depth.py incoming-events log  (log{n}_events.csv: TIMESTAMP, SOURCE, PAYLOAD_JSON)
     rows with SOURCE == "PING360_SCAN" -> payload {angle_grad, angle_deg, sample_period, samples}
  2. JSON-lines ping log written by this simulator (run_sim.py --log-pings pings.jsonl):
     one SonarPing.to_dict() per line, optional "pose_est": [x, y, yaw]
  3. any CSV/JSONL you write yourself with the same fields

Pose for each ping (in this order):
  --pose-csv file with columns t,x,y,yaw (ENU deg, same clock as the pings; e.g. exported DVL/AHRS
  dead reckoning from log{n}.csv), else "pose_est" inside the ping record, else a fixed pose
  (bench / tank test with the vehicle held still: --fixed-pose x,y,yaw).

  python replay.py log12_events.csv --fixed-pose 0,0,90 --out replay_log12.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Iterator, List, Optional, Tuple

import numpy as np

from detection import DetectionConfig, SonarObstacleMapper, SonarPing

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


def iter_pings(path: str) -> Iterator[Tuple[SonarPing, Optional[Tuple[float, float, float]]]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonl", ".json", ".ndjson"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                pose = tuple(d["pose_est"]) if d.get("pose_est") else None
                yield SonarPing.from_dict(d), pose
        return
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        if header and header[:2] == ["TIMESTAMP", "SOURCE"]:
            for row in rd:
                if len(row) < 3 or row[1] != "PING360_SCAN":
                    continue
                try:
                    p = json.loads(row[2])
                    t = float(row[0])
                    yield SonarPing.from_driver(int(p["angle_grad"]), int(p["sample_period"]), p["samples"], t), None
                except (ValueError, KeyError, json.JSONDecodeError):
                    continue
        else:
            cols = {k: i for i, k in enumerate(header or [])}
            for row in rd:
                d = {k: row[i] for k, i in cols.items()}
                d["samples"] = json.loads(d["samples"])
                yield SonarPing.from_dict(d), None


def load_pose_csv(path: str) -> List[Tuple[float, float, float, float]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append((float(r["t"]), float(r["x"]), float(r["y"]), float(r["yaw"])))
    return out


def replay(path: str, cfg: Optional[DetectionConfig] = None, pose_csv: Optional[str] = None,
           fixed_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0), out_png: Optional[str] = None,
           max_pings: Optional[int] = None) -> dict:
    cfg = cfg or DetectionConfig()
    mapper = SonarObstacleMapper(cfg)
    poses = load_pose_csv(pose_csv) if pose_csv else []
    pi = 0
    fan = []          # (x, y, intensity) for the sonar image
    hits, rngs = [], []
    confirmed = set()
    n = 0
    t_last = 0.0
    for ping, pose in iter_pings(path):
        if max_pings and n >= max_pings:
            break
        while pi < len(poses) and poses[pi][0] <= ping.t + 0.5:
            mapper.add_pose(*poses[pi]); pi += 1
        if not poses:
            p = pose or fixed_pose
            mapper.add_pose(ping.t, *p)
        beam = mapper.on_ping(ping)
        n += 1
        t_last = ping.t
        if beam.sonar_xy is not None and n % 1 == 0:
            r = (np.arange(0, ping.n, 6) + 0.5) * ping.range_m / ping.n
            a = math.radians(beam.beam_yaw_deg)
            fan.append(np.column_stack((beam.sonar_xy[0] + r * math.cos(a), beam.sonar_xy[1] + r * math.sin(a),
                                        ping.samples[::6])))
        if beam.hit_xy is not None:
            hits.append(beam.hit_xy); rngs.append((beam.rel_bearing_deg, beam.range_m))
        if n % 25 == 0:
            o = mapper.obstacles(ping.t)
            if len(o):
                confirmed.update((round(px, 3), round(py, 3)) for px, py in o)
    occ = np.asarray(sorted(confirmed)) if confirmed else np.zeros((0, 2))
    res = dict(pings=n, detections=len(hits), detection_rate=round(len(hits) / max(n, 1), 3),
               confirmed_cells=int(len(occ)))
    if out_png and n:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
        F = np.vstack(fan[-4000:]) if fan else np.zeros((0, 3))
        if len(F):
            ax[0].scatter(F[:, 0], F[:, 1], c=F[:, 2], s=1, cmap="inferno", vmin=0, vmax=255)
        if hits:
            H = np.asarray(hits)
            ax[0].scatter(H[:, 0], H[:, 1], s=4, c="cyan", label="first returns")
        if len(occ):
            ax[0].scatter(occ[:, 0], occ[:, 1], s=14, marker="s", facecolors="none", edgecolors="lime",
                          label="cells confirmed at any time")
        ax[0].set_aspect("equal"); ax[0].set_facecolor("black"); ax[0].legend(fontsize=8)
        ax[0].set_title(f"{os.path.basename(path)}: {n} pings, {len(hits)} detections")
        if rngs:
            R = np.asarray(rngs)
            ax[1].scatter(R[:, 0], R[:, 1], s=4)
        ax[1].set_xlabel("relative bearing [deg] (left +)"); ax[1].set_ylabel("first-return range [m]")
        ax[1].grid(alpha=0.3); ax[1].set_title("Detections per bearing")
        fig.tight_layout(); fig.savefig(out_png, dpi=90); plt.close(fig)
        res["figure"] = out_png
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Replay logged Ping360 pings through detection.py")
    ap.add_argument("log")
    ap.add_argument("--pose-csv")
    ap.add_argument("--fixed-pose", default="0,0,0", help="x,y,yaw for a vehicle held still")
    ap.add_argument("--config", help="JSON with a 'detection' section")
    ap.add_argument("--out", default="replay.png")
    ap.add_argument("--max-pings", type=int)
    a = ap.parse_args(argv)
    cfg = DetectionConfig()
    if a.config:
        with open(a.config) as f:
            for k, v in json.load(f).get("detection", {}).items():
                setattr(cfg, k, v)
    fp = tuple(float(v) for v in a.fixed_pose.split(","))
    print(json.dumps(replay(a.log, cfg, a.pose_csv, fp, a.out, a.max_pings), indent=2))


if __name__ == "__main__":
    main()
