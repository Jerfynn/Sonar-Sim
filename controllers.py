"""
controllers.py - simple stand-ins for the backend controllers so the simulator can fly missions.

  DeadReckoner     DVL body velocity (lever-arm corrected) + AHRS heading -> x, y; model-based
                   speed hold-over during DVL dropouts (a held speed would make braking reverse!)
  HeadingRef       profiled heading reference (rate / decel limited, ~10 s per 90 deg)
  HeadingHold      T2: feed-forward of the reference rate + PD + I (cancels the yaw drift)
  SpeedHold        T1: drag feed-forward + PI, deadband compensated (brakes with reverse thrust)
  Mission          legs = absolute waypoints or relative pattern legs (heading + length, like
                   $SQUARE in auv_depth.py), LOS track following, turn-in-place at corners
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


@dataclass
class ControlConfig:
    neutral_us: int = 1500
    t1_deadband_us: int = 25
    t2_deadband_us: int = 25
    t1_fwd_max_us: int = 150          # SQ_T1_MAX_US in auv_depth.py
    t1_rev_max_us: int = 150
    t2_max_us: int = 110
    t1_dir: int = 1
    t2_dir: int = 1
    db_blend_us: float = 4.0          # smooth deadband compensation (like _db_smooth in auv_depth.py)
    # plant estimates for feed-forward
    k_fwd: float = 0.0018
    k_rev: float = 0.0025
    drag: float = 0.20
    yaw_gain_dps_per_us: float = 0.22
    # speed loop
    spd_kp_us: float = 260.0
    spd_ki_us: float = 60.0
    spd_i_max_us: float = 40.0
    # heading loop
    hdg_kp_us: float = 2.2
    hdg_kd_us: float = 1.6
    hdg_ki_us: float = 0.6
    hdg_i_max_us: float = 25.0
    turn_rate_dps: float = 9.0        # ~10 s per 90 deg
    turn_decel_dps2: float = 4.0
    turn_accel_dps2: float = 6.0
    # mission
    cruise_mps: float = 0.2
    turn_speed_mps: float = 0.0
    turn_start_deg: float = 25.0
    turn_done_deg: float = 5.0
    los_lookahead_m: float = 2.0
    intercept_max_deg: float = 40.0
    accept_m: float = 0.3
    # nav
    dvl_x_m: float = -0.38
    dvl_holdover_s: float = 3.0


def db_smooth(u: float, db: float, blend: float) -> float:
    """Effort beyond the ESC deadband -> pulse offset, with a linear blend near zero (no +-db jump)."""
    a = abs(u)
    if a >= blend:
        return math.copysign(db + a, u)
    return u * (db + blend) / blend


class DeadReckoner:
    def __init__(self, cfg: ControlConfig, x: float, y: float, yaw: float):
        self.c = cfg
        self.x, self.y, self.yaw = x, y, yaw
        self.u = 0.0
        self.v = 0.0
        self.rate = 0.0
        self.dvl_t = -1e9
        self.dvl_ok = True
        self.t = 0.0

    def _model_accel(self, t1: int) -> float:
        c = self.c
        e = (t1 - c.neutral_us) * c.t1_dir
        if abs(e) <= c.t1_deadband_us:
            return 0.0
        ex = abs(e) - c.t1_deadband_us
        return ex * c.k_fwd if e > 0 else -ex * c.k_rev

    def update(self, t: float, dt: float, yaw: float, rate_dps: float,
               dvl: Optional[Tuple[float, float, bool]], t1_cmd: int) -> None:
        self.yaw, self.rate = yaw, rate_dps
        if dvl is not None and dvl[2]:
            vx, vy_stbd, _ = dvl
            self.u = vx
            self.v = -vy_stbd - math.radians(rate_dps) * self.c.dvl_x_m   # lever arm -> turning point
            self.dvl_t = t
        else:
            # hold-over: propagate the surge model with the commanded T1
            self.u += (self._model_accel(t1_cmd) - self.c.drag * self.u * abs(self.u)) * dt
            self.v *= max(0.0, 1.0 - 1.5 * dt)
        self.dvl_ok = (t - self.dvl_t) <= 0.5
        ps = math.radians(yaw)
        self.x += (self.u * math.cos(ps) - self.v * math.sin(ps)) * dt
        self.y += (self.u * math.sin(ps) + self.v * math.cos(ps)) * dt
        self.t = t


class HeadingRef:
    """Rate / decel limited reference that walks towards the target heading."""

    def __init__(self, cfg: ControlConfig, yaw: float):
        self.c = cfg
        self.ref = yaw
        self.rate = 0.0

    def reset(self, yaw: float) -> None:
        self.ref, self.rate = yaw, 0.0

    def step(self, target: float, dt: float) -> Tuple[float, float]:
        c = self.c
        err = wrap180(target - self.ref)
        a = abs(err)
        rate_mag = min(c.turn_rate_dps, math.sqrt(2.0 * c.turn_decel_dps2 * a), 1.5 * a)
        want = math.copysign(rate_mag, err)
        dr = max(-c.turn_accel_dps2 * dt, min(c.turn_accel_dps2 * dt, want - self.rate))
        self.rate += dr
        step = self.rate * dt
        if abs(step) > a:
            self.ref, self.rate = target, 0.0
        else:
            self.ref = wrap180(self.ref + step)
        return self.ref, self.rate


class HeadingHold:
    def __init__(self, cfg: ControlConfig):
        self.c = cfg
        self.i = 0.0

    def step(self, ref: float, ref_rate: float, yaw: float, rate: float, dt: float) -> Tuple[int, float]:
        c = self.c
        err = wrap180(ref - yaw)
        if abs(err) < 20.0:
            self.i = max(-c.hdg_i_max_us, min(c.hdg_i_max_us, self.i + c.hdg_ki_us * err * dt))
        u = ref_rate / c.yaw_gain_dps_per_us + c.hdg_kp_us * err + c.hdg_kd_us * (ref_rate - rate) + self.i
        umax = c.t2_max_us - c.t2_deadband_us
        u = max(-umax, min(umax, u))
        off = db_smooth(u, c.t2_deadband_us, c.db_blend_us)
        return int(round(c.neutral_us + c.t2_dir * off)), err


class SpeedHold:
    def __init__(self, cfg: ControlConfig):
        self.c = cfg
        self.i = 0.0

    def step(self, v_ref: float, v: float, dt: float) -> int:
        c = self.c
        e = v_ref - v
        if abs(v_ref) < 1e-3 and abs(v) < 0.02:
            self.i = 0.0
            return c.neutral_us
        self.i = max(-c.spd_i_max_us, min(c.spd_i_max_us, self.i + c.spd_ki_us * e * dt))
        k = c.k_fwd if v_ref >= 0 else c.k_rev
        ff = c.drag * v_ref * abs(v_ref) / k
        u = ff + c.spd_kp_us * e + self.i
        u = max(-(c.t1_rev_max_us - c.t1_deadband_us), min(c.t1_fwd_max_us - c.t1_deadband_us, u))
        off = db_smooth(u, c.t1_deadband_us, c.db_blend_us)
        return int(round(c.neutral_us + c.t1_dir * off))


@dataclass
class Leg:
    kind: str                         # "WP" absolute waypoint, "REL" heading + length from current pos
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    length: float = 0.0


@dataclass
class MissionOutput:
    speed: float
    heading: Optional[float]
    track: Optional[Tuple[float, float, float]]
    goal: Optional[Tuple[float, float]]
    turning: bool
    done: bool
    leg: int
    phase: str


class Mission:
    def __init__(self, cfg: ControlConfig, legs: List[Leg], cruise: Optional[float] = None):
        self.c = cfg
        self.legs = legs
        self.cruise = cruise if cruise is not None else cfg.cruise_mps
        self.i = 0
        self.phase = "START"
        self.origin: Optional[Tuple[float, float]] = None
        self.goal: Optional[Tuple[float, float]] = None
        self.bearing = 0.0
        self.length = 0.0
        self.done = False
        self.skipped: List[int] = []
        self.leg_log: List[dict] = []

    @staticmethod
    def square(x0: float, y0: float, heading: float, side: float, turn: int = 1, sides: int = 4) -> List[Leg]:
        """Relative pattern legs, turn=+1 left (CCW), -1 right."""
        return [Leg("REL", heading=wrap180(heading + turn * 90.0 * k), length=side) for k in range(sides)]

    def _start_leg(self, x: float, y: float, t: float) -> None:
        if self.i >= len(self.legs):
            self.done = True
            self.phase = "DONE"
            return
        L = self.legs[self.i]
        self.origin = (x, y)
        if L.kind == "REL":
            self.bearing = L.heading
            b = math.radians(L.heading)
            self.goal = (x + L.length * math.cos(b), y + L.length * math.sin(b))
            self.length = L.length
        else:
            self.goal = (L.x, L.y)
            self.bearing = math.degrees(math.atan2(L.y - y, L.x - x))
            self.length = math.hypot(L.x - x, L.y - y)
        self.phase = "TURN"
        self.t_leg = t

    def skip_leg(self, x: float, y: float, t: float) -> None:
        if self.done:
            return
        self.skipped.append(self.i)
        self._finish_leg(x, y, t, blocked=True)

    def _finish_leg(self, x: float, y: float, t: float, blocked: bool = False) -> None:
        a = 0.0
        if self.origin is not None:
            b = math.radians(self.bearing)
            a = (x - self.origin[0]) * math.cos(b) + (y - self.origin[1]) * math.sin(b)
        self.leg_log.append(dict(leg=self.i, along=a, length=self.length, blocked=blocked, t=t))
        self.i += 1
        self._start_leg(x, y, t)

    def step(self, t: float, x: float, y: float, yaw: float) -> MissionOutput:
        c = self.c
        if self.phase == "START":
            self._start_leg(x, y, t)
        if self.done:
            return MissionOutput(0.0, None, None, None, False, True, self.i, "DONE")
        b = math.radians(self.bearing)
        ux, uy = math.cos(b), math.sin(b)
        dx, dy = x - self.origin[0], y - self.origin[1]
        along = dx * ux + dy * uy
        cross = -dx * uy + dy * ux          # left +
        if self.phase == "TURN":
            err = wrap180(self.bearing - yaw)
            if abs(err) <= c.turn_done_deg:
                self.phase = "TRACK"
            else:
                return MissionOutput(c.turn_speed_mps, self.bearing, (self.origin[0], self.origin[1], self.bearing),
                                     self.goal, True, False, self.i, "TURN")
        # TRACK: line of sight on the leg
        if along >= self.length - c.accept_m:
            self._finish_leg(x, y, t)
            return self.step(t, x, y, yaw)
        corr = math.degrees(math.atan2(cross, c.los_lookahead_m))
        corr = max(-c.intercept_max_deg, min(c.intercept_max_deg, corr))
        href = wrap180(self.bearing - corr)
        if abs(wrap180(href - yaw)) > c.turn_start_deg + 20:
            # far off heading (e.g. after an avoidance manoeuvre): turn first
            return MissionOutput(c.turn_speed_mps, href, (self.origin[0], self.origin[1], self.bearing),
                                 self.goal, True, False, self.i, "TURN")
        remaining = self.length - along
        spd = min(self.cruise, max(0.06, 0.25 * remaining))
        return MissionOutput(spd, href, (self.origin[0], self.origin[1], self.bearing), self.goal,
                             False, False, self.i, "TRACK")
