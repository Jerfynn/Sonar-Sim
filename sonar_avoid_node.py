"""
sonar_avoid_node.py - glue for the Jetson backend (auv_depth.py). Imports ONLY detection.py and
avoidance.py (pure logic). Copy these three files next to auv_depth.py.

Threads
  Ping360Link thread  -> node.on_ping_raw(angle_grad, sample_period, samples)     (any rate)
  20 Hz nav loop      -> node.step(...) once per tick, then use the returned
                         speed setpoint (T1) and heading reference (T2). T3/depth untouched.

Minimal hook-up in auv_depth.py (see README.md for the full walk-through):

    from sonar_avoid_node import SonarAvoidNode
    AVOID = SonarAvoidNode(mode="STOP")          # OFF / WARN / STOP / AVOID

    # Ping360Link._process_scan(self, angle_grad, s_per, intensities, last_log):
    AVOID.on_ping_raw(angle_grad, s_per, intensities)

    # in the 20 Hz pattern step, after computing the pattern's own (v_ref, heading_ref):
    out = AVOID.step(now, pe, pn, yaw, vx, v_ref, heading_ref,
                     track=(leg_origin_e, leg_origin_n, leg_heading), goal=(goal_e, goal_n),
                     turning=(phase == "TURN"), active=True)
    v_ref, heading_ref = out.speed_ref, out.heading_ref      # -> your T1 speed loop / T2 heading hold
    if out.cmd.waypoint_blocked: <end the current leg as if reached>
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from avoidance import AvoidanceConfig, AvoidCommand, AvoidInput, ObstacleAvoidance, arbitrate
from detection import DetectionConfig, SonarObstacleMapper, SonarPing


@dataclass
class NodeOutput:
    speed_ref: float
    heading_ref: Optional[float]
    owner: str
    cmd: AvoidCommand


class SonarAvoidNode:
    def __init__(self, mode: str = "STOP", stage: str = "SIDESTEP",
                 det_cfg: Optional[DetectionConfig] = None, avoid_cfg: Optional[AvoidanceConfig] = None,
                 clock=time.monotonic):
        self.det_cfg = det_cfg or DetectionConfig()
        self.avoid_cfg = avoid_cfg or AvoidanceConfig()
        self.avoid_cfg.sonar_x_m = self.det_cfg.sonar_x_m
        self.avoid_cfg.point_margin_m = self.det_cfg.grid_res_m / 2
        self.mapper = SonarObstacleMapper(self.det_cfg)
        self.avoid = ObstacleAvoidance(self.avoid_cfg)
        self.avoid.set_mode(mode, stage)
        self.clock = clock
        self.lock = threading.Lock()
        self.last_cmd: Optional[AvoidCommand] = None

    def set_mode(self, mode: str, stage: Optional[str] = None) -> None:
        with self.lock:
            self.avoid.set_mode(mode, stage)

    # ---- Ping360 thread
    def on_ping_raw(self, angle_grad: int, sample_period: int, samples: Sequence[int], t: Optional[float] = None):
        ping = SonarPing.from_driver(angle_grad, sample_period, samples, self.clock() if t is None else t)
        with self.lock:
            return self.mapper.on_ping(ping)

    # ---- 20 Hz loop
    def step(self, t: float, x: float, y: float, yaw_deg: float, speed_mps: Optional[float],
             mission_speed: float, mission_heading: Optional[float],
             track: Optional[Tuple[float, float, float]] = None, goal: Optional[Tuple[float, float]] = None,
             turning: bool = False, active: bool = True) -> NodeOutput:
        with self.lock:
            self.mapper.add_pose(t, x, y, yaw_deg)
            pts = self.mapper.obstacles(t)
            cmd = self.avoid.step(AvoidInput(t, x, y, yaw_deg, speed_mps or 0.0, pts, mission_speed,
                                             mission_heading, track, goal, turning, active))
            self.last_cmd = cmd
        sp, hd, owner = arbitrate(mission_speed, mission_heading, cmd)
        return NodeOutput(sp, hd, owner, cmd)

    # ---- for the GCS (PySide6 sonar widget): small JSON-able dict
    def telemetry(self, t: Optional[float] = None, max_points: int = 150) -> dict:
        t = self.clock() if t is None else t
        with self.lock:
            c = self.last_cmd
            pts = self.mapper.obstacles(t)
            b = self.mapper.last_beam
        if len(pts) > max_points:
            pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(int)]
        return {
            "avoid_mode": self.avoid_cfg.mode, "avoid_state": c.state if c else "-", "avoid_zone": c.zone if c else "-",
            "obst_dist_m": None if (c is None or c.obstacle_dist_m is None) else round(c.obstacle_dist_m, 2),
            "stop_dist_m": round(c.stop_dist_m, 2) if c else None, "slow_dist_m": round(c.slow_dist_m, 2) if c else None,
            "avoid_msg": c.message if c else "",
            "sonar_last": None if b is None else {"angle_grad": b.angle_grad, "rel_deg": round(b.rel_bearing_deg, 1),
                                                   "range_m": None if b.range_m is None else round(b.range_m, 2)},
            "obst_pts": [[round(float(px), 2), round(float(py), 2)] for px, py in pts],
        }
