"""
vehicle_model.py - 2-D torpedo AUV (Trinetra) dynamics and sensors, fitted to the tank logs.

Thruster pulses in (T1 stern surge, T2 nose horizontal, T3 ignored in 2-D), pose/velocity out.
Reference point = turning point (the vehicle yaws about it; T2 is ~0.8 m ahead of it).

Surge : dv/dt = k*(|pwm-1500| - deadband) * sign  - drag * v|v|
        k = 0.0018 m/s^2/us forward, 0.0025 reverse, drag 0.20  (-> ~0.6 m/s at T1 1570)
Yaw   : rate_cmd = 0.22 deg/s/us beyond deadband (clamped), first-order lag 0.6 s,
        plus a constant drift -0.3..-0.8 deg/s (random per run) with a slow random walk.
DVL   : 0.38 m behind the turning point -> sees lateral v = v_y - 0.38*r ; noise; dropouts
        0.5-2.8 s (Poisson).  Reports (vx forward, vy STARBOARD) like auv_depth.py uw_vx_mps.
AHRS  : heading (ENU yaw, CCW+) + noise + small bias, gyro yaw rate + noise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


@dataclass
class VehicleParams:
    neutral_us: int = 1500
    deadband_us: int = 25
    pwm_min: int = 1100
    pwm_max: int = 1900
    k_fwd: float = 0.0018           # m/s^2 per us beyond deadband
    k_rev: float = 0.0025
    drag: float = 0.20              # dv/dt -= drag * v|v|
    thrust_tau_s: float = 0.25
    yaw_gain_dps_per_us: float = 0.22
    yaw_rate_max_dps: float = 12.0
    yaw_tau_s: float = 0.6
    yaw_drift_dps: Tuple[float, float] = (-0.8, -0.3)   # uniform range, per run
    yaw_drift_walk: float = 0.05    # deg/s per sqrt(s)
    t2_dir: int = 1                 # +1: T2 > 1500 turns CCW (yaw increases)
    t1_dir: int = 1                 # +1: T1 > 1500 forward
    length_m: float = 1.8
    radius_m: float = 0.10
    nose_m: float = 0.90            # nose ahead of the turning point
    tail_m: float = 0.90
    sway_damp: float = 1.5          # residual sway decays towards the current
    current_mps: Tuple[float, float] = (0.0, 0.0)   # world (East, North)
    # sensors
    dvl_x_m: float = -0.38
    dvl_noise_mps: float = 0.006
    dvl_dropout_mean_interval_s: float = 25.0
    dvl_dropout_s: Tuple[float, float] = (0.5, 2.8)
    dvl_rate_hz: float = 8.0
    ahrs_noise_deg: float = 0.25
    ahrs_bias_deg: float = 0.5
    gyro_noise_dps: float = 0.3


@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0        # deg ENU
    u: float = 0.0          # surge m/s
    v: float = 0.0          # sway (left +) at the turning point
    r_t2: float = 0.0       # deg/s from T2
    r: float = 0.0          # total yaw rate deg/s
    a_thr: float = 0.0      # surge thrust accel (lagged)


class Vehicle:
    def __init__(self, p: VehicleParams, x: float, y: float, yaw: float, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        self.s = VehicleState(x=x, y=y, yaw=yaw)
        self.drift = float(rng.uniform(*p.yaw_drift_dps))
        self.t = 0.0
        self.collided = False
        # DVL dropout process
        self.dvl_next_drop = self._next_drop(0.0)
        self.dvl_drop_until = -1.0
        self.dvl_last_t = -1.0
        self.dvl_last: Optional[Tuple[float, float, bool]] = None
        self.ahrs_bias = float(rng.normal(0, p.ahrs_bias_deg))

    def _next_drop(self, t: float) -> float:
        if self.p.dvl_dropout_mean_interval_s <= 0:
            return 1e12
        return t + float(self.rng.exponential(self.p.dvl_dropout_mean_interval_s))

    # ------------------------------------------------------------------ dynamics
    def thrust_accel(self, t1: int) -> float:
        p = self.p
        e = (t1 - p.neutral_us) * p.t1_dir
        if abs(e) <= p.deadband_us:
            return 0.0
        ex = abs(e) - p.deadband_us
        return ex * p.k_fwd if e > 0 else -ex * p.k_rev

    def yaw_cmd(self, t2: int) -> float:
        p = self.p
        e = (t2 - p.neutral_us) * p.t2_dir
        if abs(e) <= p.deadband_us:
            return 0.0
        r = math.copysign((abs(e) - p.deadband_us) * p.yaw_gain_dps_per_us, e)
        return max(-p.yaw_rate_max_dps, min(p.yaw_rate_max_dps, r))

    def step(self, t1: int, t2: int, dt: float) -> None:
        p, s = self.p, self.s
        t1 = int(min(max(t1, p.pwm_min), p.pwm_max))
        t2 = int(min(max(t2, p.pwm_min), p.pwm_max))
        s.a_thr += (self.thrust_accel(t1) - s.a_thr) * min(1.0, dt / p.thrust_tau_s)
        s.u += (s.a_thr - p.drag * s.u * abs(s.u)) * dt
        s.r_t2 += (self.yaw_cmd(t2) - s.r_t2) * min(1.0, dt / p.yaw_tau_s)
        self.drift += p.yaw_drift_walk * math.sqrt(dt) * float(self.rng.normal())
        self.drift = min(max(self.drift, -1.2), 0.0)
        s.r = s.r_t2 + self.drift
        s.yaw = wrap180(s.yaw + s.r * dt)
        s.v += (-p.sway_damp * s.v) * dt
        ps = math.radians(s.yaw)
        cx, cy = p.current_mps
        s.x += (s.u * math.cos(ps) - s.v * math.sin(ps) + cx) * dt
        s.y += (s.u * math.sin(ps) + s.v * math.cos(ps) + cy) * dt
        self.t += dt

    def stop_on_collision(self, prev_xy: Tuple[float, float]) -> None:
        self.s.x, self.s.y = prev_xy
        self.s.u = 0.0
        self.s.a_thr = 0.0
        self.collided = True

    # ------------------------------------------------------------------ geometry
    def hull_segment(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        p, s = self.p, self.s
        ps = math.radians(s.yaw)
        c, sn = math.cos(ps), math.sin(ps)
        a = p.nose_m - p.radius_m
        b = p.tail_m - p.radius_m
        return (s.x + a * c, s.y + a * sn), (s.x - b * c, s.y - b * sn)

    def point(self, bx: float, by: float = 0.0) -> Tuple[float, float]:
        s = self.s
        ps = math.radians(s.yaw)
        return s.x + bx * math.cos(ps) - by * math.sin(ps), s.y + bx * math.sin(ps) + by * math.cos(ps)

    # ------------------------------------------------------------------ sensors
    def ahrs(self) -> Tuple[float, float]:
        p = self.p
        return (wrap180(self.s.yaw + self.ahrs_bias + float(self.rng.normal(0, p.ahrs_noise_deg))),
                self.s.r + float(self.rng.normal(0, p.gyro_noise_dps)))

    def dvl(self, t: float) -> Optional[Tuple[float, float, bool]]:
        """New DVL sample (vx fwd, vy starboard, valid) at dvl_rate_hz, else None."""
        p, s = self.p, self.s
        if t - self.dvl_last_t < 1.0 / p.dvl_rate_hz:
            return None
        self.dvl_last_t = t
        if t >= self.dvl_next_drop and self.dvl_drop_until < t:
            self.dvl_drop_until = t + float(self.rng.uniform(*p.dvl_dropout_s))
            self.dvl_next_drop = self._next_drop(self.dvl_drop_until)
        if t < self.dvl_drop_until:
            return (0.0, 0.0, False)
        ps = math.radians(s.yaw)
        cx, cy = p.current_mps
        # water-referenced vs bottom-referenced: DVL measures over ground -> include current
        u_g = s.u + cx * math.cos(ps) + cy * math.sin(ps)
        v_g = s.v - cx * math.sin(ps) + cy * math.cos(ps)
        v_dvl_left = v_g + math.radians(s.r) * p.dvl_x_m
        return (u_g + float(self.rng.normal(0, p.dvl_noise_mps)),
                -v_dvl_left + float(self.rng.normal(0, p.dvl_noise_mps)), True)
