"""
simulation.py - the simulator loop (physics 100 Hz, control 20 Hz, Ping360 at its own rate) and
truth-based metrics.  Detection/avoidance only ever see what the real backend would see:
estimated (dead-reckoned) pose, DVL/AHRS-like data and Ping360 pings.
"""
from __future__ import annotations

import json
import math
import time
from typing import Callable, Dict, List, Optional

import numpy as np

import config as cfgmod
from avoidance import AvoidInput, ObstacleAvoidance, arbitrate
from controllers import DeadReckoner, HeadingHold, HeadingRef, Leg, Mission, SpeedHold
from detection import SonarObstacleMapper
from ping360_sim import Ping360Sim
from vehicle_model import Vehicle
from world import World


class Simulation:
    def __init__(self, scen: dict, mode: Optional[str] = None, stage: Optional[str] = None,
                 speed: Optional[float] = None, seed: int = 0, overrides: Optional[dict] = None,
                 empty_world: bool = False, randomize: bool = False, ping_log: Optional[str] = None):
        self.scen = scen
        ov = cfgmod.deep_merge(scen.get("overrides", {}), overrides or {})
        self.cfg = cfgmod.build(ov)
        self.mode = (mode or scen.get("mode", "AVOID")).upper()
        self.stage = (stage or scen.get("stage", "SIDESTEP")).upper()
        self.cfg["avoidance"].mode = self.mode
        self.cfg["avoidance"].stage = self.stage
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        sc = self.cfg["sim"]
        world = World(scen["world"])
        if randomize:
            world = world.jittered(self.rng, sc.obstacle_jitter_m)
        if empty_world:
            world = world.empty_copy()
        self.world = world
        x0, y0, h0 = scen["start"]
        if randomize:
            x0 += float(self.rng.normal(0, sc.start_jitter_m)); y0 += float(self.rng.normal(0, sc.start_jitter_m))
            h0 += float(self.rng.normal(0, sc.start_jitter_deg))
        self.vehicle = Vehicle(self.cfg["vehicle"], x0, y0, h0, self.rng)
        self.sonar = Ping360Sim(self.cfg["sonar_sim"], world, self.rng)
        self.mapper = SonarObstacleMapper(self.cfg["detection"])
        self.avoid = ObstacleAvoidance(self.cfg["avoidance"])
        cc = self.cfg["control"]
        self.nav = DeadReckoner(cc, x0, y0, h0)
        self.href = HeadingRef(cc, h0)
        self.hh = HeadingHold(cc)
        self.sh = SpeedHold(cc)
        legs = [l if isinstance(l, Leg) else Leg(**l) for l in scen["legs"]]
        self.mission = Mission(cc, legs, speed if speed is not None else scen.get("cruise"))
        self.cruise = self.mission.cruise
        self.t = 0.0
        self.t1 = self.t2 = cc.neutral_us
        self.t3 = cc.neutral_us                # never commanded by avoidance (2-D sim)
        self.dvl_latest = None
        self.dvl_latest_t = -1e9
        self.cmd = None
        self.mo = None
        self.owner = "MISSION"
        self.events: List[tuple] = []
        self.log: Dict[str, list] = {k: [] for k in (
            "t", "x", "y", "yaw", "ex", "ey", "u", "u_est", "t1", "t2", "state", "zone", "d_obs", "stop_d",
            "slow_d", "clear", "dvl_ok", "speed_ref")}
        self.min_clear = math.inf
        self.collisions = 0
        self.in_contact = False
        self.false_stops = 0
        self.late = 0
        self.in_late = False
        self.done = False
        self.end_reason = ""
        self.stopped_since = None
        self.collide_t = None
        self.ping_log = open(ping_log, "w", encoding="utf-8") if ping_log else None
        self.last_ping = None
        self.last_beam = None
        self.all_hits: List[tuple] = []
        self.first_stop = None

    # ------------------------------------------------------------------ truth helpers
    def _hull_clearance(self) -> float:
        a, b = self.vehicle.hull_segment()
        return self.world.capsule_clearance(a, b, self.vehicle.p.radius_m, self.t)

    def _truth_ahead(self, half_w: float) -> float:
        hx, hy = self.vehicle.point(self.cfg["sonar_sim"].sonar_x_m)
        return self.world.corridor_distance(hx, hy, self.vehicle.s.yaw, half_w, self.t)

    # ------------------------------------------------------------------ loop
    def physics_step(self, dt: float) -> None:
        v = self.vehicle
        prev = (v.s.x, v.s.y, v.s.yaw)
        v.step(self.t1, self.t2, dt)
        self.t += dt
        clr = self._hull_clearance()
        if clr < 0.0:
            v.stop_on_collision(prev[:2])
            v.s.yaw = prev[2]
            if not self.in_contact:
                self.collisions += 1
                self.events.append((self.t, "COLLISION"))
                self.collide_t = self.collide_t or self.t
            self.in_contact = True
            self.last_contact_t = self.t
            clr = 0.0
        elif clr > 0.05:
            self.in_contact = False
        if self.t > 1.0:
            self.min_clear = min(self.min_clear, clr)
        self.clear_now = clr
        d = v.dvl(self.t)
        if d is not None:
            self.dvl_latest, self.dvl_latest_t = d, self.t
        if self.sonar.due(self.t):
            ping = self.sonar.ping(self.t, v.s.x, v.s.y, v.s.yaw)
            self.last_beam = self.mapper.on_ping(ping)
            self.last_ping = ping
            if self.last_beam.hit_xy is not None:
                self.all_hits.append(self.last_beam.hit_xy)
            if self.ping_log:
                rec = ping.to_dict()
                rec["pose_est"] = [self.nav.x, self.nav.y, self.nav.yaw]
                rec["pose_true"] = [v.s.x, v.s.y, v.s.yaw]
                self.ping_log.write(json.dumps(rec) + "\n")

    def control_step(self, dt: float) -> None:
        v, cc = self.vehicle, self.cfg["control"]
        yaw_m, rate_m = v.ahrs()
        dvl = self.dvl_latest if (self.t - self.dvl_latest_t) < 0.3 else None
        self.nav.update(self.t, dt, yaw_m, rate_m, dvl, self.t1)
        self.mapper.add_pose(self.t, self.nav.x, self.nav.y, self.nav.yaw)
        mo = self.mission.step(self.t, self.nav.x, self.nav.y, self.nav.yaw)
        pts = self.mapper.obstacles(self.t)
        cmd = self.avoid.step(AvoidInput(
            t=self.t, x=self.nav.x, y=self.nav.y, yaw_deg=self.nav.yaw, speed_mps=self.nav.u, obstacles=pts,
            mission_speed=mo.speed, mission_heading=mo.heading, track=mo.track, goal=mo.goal,
            mission_turning=mo.turning, mission_active=not mo.done))
        for e in cmd.events:
            self.events.append((self.t, e))
            if e in ("STOP_TRIGGER", "WARN_STOP"):
                truth = self._truth_ahead(self.cfg["avoidance"].corridor_half_m + 0.6)
                if self.first_stop is None:
                    self.first_stop = (self.t, self.vehicle.s.x, self.vehicle.s.y, self.vehicle.s.u)
                if truth > cmd.slow_dist_m + self.cfg["sim"].false_stop_extra_m:
                    self.false_stops += 1
                    self.events.append((self.t, f"FALSE_STOP(truth {truth:.1f} m)"))
        if cmd.abort:
            self.done, self.end_reason = True, "aborted safely (turn blocked)"
        if cmd.waypoint_blocked:
            self.mission.skip_leg(self.nav.x, self.nav.y, self.t)
            mo = self.mission.step(self.t, self.nav.x, self.nav.y, self.nav.yaw)
        speed_ref, head_tgt, self.owner = arbitrate(mo.speed, mo.heading, cmd)
        if head_tgt is None:
            head_tgt = self.href.ref
        ref, ref_rate = self.href.step(head_tgt, dt)
        self.t2, _ = self.hh.step(ref, ref_rate, yaw_m, rate_m, dt)
        self.t1 = self.sh.step(speed_ref, self.nav.u, dt)
        # late detection (truth gets close while nothing is being done about it)
        if self.mode in ("STOP", "AVOID"):
            if v.s.u > 0.08 and cmd.state in ("TRACK",) and cmd.speed_cap is None:
                truth = self._truth_ahead(0.25)
                late = truth < self.cfg["sim"].late_frac * cmd.stop_dist_m
                if late and not self.in_late:
                    self.late += 1
                    self.events.append((self.t, f"LATE(truth {truth:.1f} m)"))
                self.in_late = late
            else:
                self.in_late = False
        self.cmd, self.mo, self.speed_ref = cmd, mo, speed_ref
        L = self.log
        for k, val in (("t", self.t), ("x", v.s.x), ("y", v.s.y), ("yaw", v.s.yaw), ("ex", self.nav.x),
                       ("ey", self.nav.y), ("u", v.s.u), ("u_est", self.nav.u), ("t1", self.t1), ("t2", self.t2),
                       ("state", cmd.state), ("zone", cmd.zone),
                       ("d_obs", np.nan if cmd.obstacle_dist_m is None else cmd.obstacle_dist_m),
                       ("stop_d", cmd.stop_dist_m), ("slow_d", cmd.slow_dist_m), ("clear", self.clear_now),
                       ("dvl_ok", self.nav.dvl_ok), ("speed_ref", speed_ref)):
            L[k].append(val)
        self._check_end(cmd, mo)

    def _check_end(self, cmd, mo) -> None:
        sc = self.cfg["sim"]
        end = self.scen.get("end", "mission")
        if self.done:
            return
        if mo.done:
            self.done, self.end_reason = True, "mission complete"
        elif self.done:
            return
        elif end == "stopped":
            if cmd.state in ("HOLD", "SCAN", "BRAKE") and abs(self.vehicle.s.u) < 0.03:
                self.stopped_since = self.stopped_since or self.t
                if self.t - self.stopped_since >= sc.stopped_hold_s:
                    self.done, self.end_reason = True, "stopped before obstacle"
            else:
                self.stopped_since = None
            if self.collide_t is not None and self.t - self.collide_t > 3.0:
                self.done, self.end_reason = True, "collided"
        if (not self.done and self.collide_t is not None and self.in_contact
                and self.t - self.collide_t > 20.0 and abs(self.vehicle.s.u) < 0.02):
            self.done, self.end_reason = True, "collided (stuck)"
        if not self.done and self.t >= self.scen.get("timeout", sc.timeout_s):
            self.done, self.end_reason = True, "timeout"

    def run(self, on_frame: Optional[Callable] = None, frame_every: int = 4, realtime: float = 0.0) -> dict:
        sc = self.cfg["sim"]
        n_phys = max(1, int(round(sc.dt_ctrl_s / sc.dt_phys_s)))
        k = 0
        wall0 = time.time()
        self.clear_now = self._hull_clearance()
        while not self.done:
            for _ in range(n_phys):
                self.physics_step(sc.dt_phys_s)
            self.control_step(sc.dt_ctrl_s)
            k += 1
            if on_frame is not None and k % frame_every == 0:
                if on_frame(self) is False:
                    self.done, self.end_reason = True, "closed"
            if realtime > 0:
                lag = self.t / realtime - (time.time() - wall0)
                if lag > 0:
                    time.sleep(lag)
        if on_frame is not None:
            on_frame(self)
        if self.ping_log:
            self.ping_log.close()
        return self.result()

    def result(self) -> dict:
        ev = [e for _, e in self.events]
        nose_clear = None
        if self.scen.get("end") == "stopped":
            nose_clear = round(float(self.log["clear"][-1]), 2) if self.log["clear"] else None
        brake = None
        if self.first_stop is not None and self.scen.get("end") == "stopped":
            brake = round(math.hypot(self.vehicle.s.x - self.first_stop[1], self.vehicle.s.y - self.first_stop[2]), 2)
        return dict(
            v_at_stop=None if self.first_stop is None else round(self.first_stop[3], 2), brake_dist_m=brake,
            scenario=self.scen["name"], mode=self.mode, stage=self.stage, seed=self.seed, cruise=self.cruise,
            end=self.end_reason, time_s=round(self.t, 1),
            completed=self.end_reason in ("mission complete", "stopped before obstacle"),
            aborted=self.end_reason.startswith("aborted"),
            collisions=self.collisions, min_clearance_m=round(float(self.min_clear), 2),
            stop_clearance_m=nose_clear, false_stops=self.false_stops, late=self.late,
            stops=ev.count("STOP_TRIGGER"), backoffs=ev.count("BACKOFF"),
            skipped_legs=ev.count("WP_BLOCKED") + ev.count("GIVE_UP"),
            sidesteps=sum(e.startswith("SIDESTEP") for e in ev), pings=self.mapper.n_pings,
            sector_refresh_s=round(self.sonar.sector_refresh_s, 2),
        )
