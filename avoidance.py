"""
avoidance.py - sonar obstacle avoidance, pure logic (numpy only, no simulator imports).

Input each control tick (20 Hz): estimated pose, forward speed, confirmed obstacle points
(world xy from detection.OccupancyGrid), and what the mission WANTS (speed + heading
setpoints, current leg line, goal).  Output: AvoidCommand with optional overrides for the
speed setpoint (-> T1) and the heading reference (-> T2 heading hold).  T3 / depth is never
touched - arbitrate() only ever returns speed and heading.

Modes
  OFF   : compute zones for display only
  WARN  : compute zones, raise WARN events, never override
  STOP  : SLOW cap (2.5-5 m), STOP brake + hold (< stop distance), back-off when too close.
          While stopped the mission may still TURN IN PLACE (this is what blocked the square in the
          tank log: heading was frozen in front of the wall).  A leg that stays blocked for
          blocked_skip_s is reported as waypoint_blocked so the pattern moves on.
  AVOID : STOP behaviour + stage 1 (stop-scan-turn side-step and rejoin) or
          stage 2 (VFH polar histogram steering), back-off/escape when trapped.

Frames: world ENU, yaw deg ENU (CCW +), body x forward / y LEFT.  Distances in the zone check
are measured from the sonar head (cfg.sonar_x_m ahead of the turning point).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

MODES = ("OFF", "WARN", "STOP", "AVOID")


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


@dataclass
class AvoidanceConfig:
    mode: str = "AVOID"
    stage: str = "SIDESTEP"          # SIDESTEP (stage 1) or VFH (stage 2)
    sonar_x_m: float = 0.80
    point_margin_m: float = 0.125    # half a grid cell
    # zones (cone AND corridor so tank walls beside the track do not trigger)
    cone_half_deg: float = 30.0
    corridor_half_m: float = 0.50    # hull radius 0.10 + 0.40 margin
    near_box_m: float = 1.0          # (display only) near field is handled by collide_half_m
    collide_half_m: float = 0.30     # at the nose the zone is only this wide (hull radius + 0.2), it widens
                                     # with the cone to corridor_half_m: a wall BESIDE the nose is not "ahead"
    # speed dependent STOP distance: base + v*react + v^2/(2*decel), clipped
    stop_base_m: float = 1.8
    react_time_s: float = 2.0        # sonar refresh + hit confirmation + controller
    brake_decel_mps2: float = 0.10
    stop_min_m: float = 2.0
    stop_max_m: float = 4.0
    slow_margin_m: float = 2.5       # SLOW zone = stop .. stop + margin
    slow_speed_mps: float = 0.10
    exit_hyst_m: float = 0.30
    clear_hold_s: float = 1.0
    # goal awareness: obstacle far enough beyond the goal -> only SLOW, no STOP
    goal_margin_m: float = 1.3
    skip_goal_within_m: float = 3.0  # AVOID: goal this close and behind the obstacle -> skip leg
    # braking / back-off
    brake_v_eps: float = 0.03
    brake_timeout_s: float = 10.0
    backoff_trigger_m: float = 1.3
    backoff_dist_m: float = 1.5
    backoff_speed_mps: float = 0.10
    backoff_timeout_s: float = 25.0
    swing_radius_m: float = 1.15     # nose swing (0.9 m) + margin: no turn in place inside this
    nose_m: float = 0.90
    tail_m: float = 0.90
    hull_radius_m: float = 0.10
    swing_margin_m: float = 0.15     # in-place turn refused if the swept hull comes closer than this
    turn_blocked_abort_s: float = 20.0
    max_backoff_total_m: float = 4.5 # reversing allowed per obstacle episode
    min_backoff_m: float = 0.3       # less own trail than this behind -> do not reverse at all
    side_guard_x_m: Tuple[float, float] = (-1.2, 1.2)   # body x band beside the hull (from turning point)
    side_guard_y_m: float = 0.95     # obstacle this close beside the hull -> no turning towards it
    # STOP mode
    blocked_skip_s: float = 12.0     # <= 0: hold forever
    # AVOID stage 1 (side-step)
    scan_time_s: float = 2.6         # >= one sector refresh
    sidestep_angles_deg: Tuple[float, ...] = (45.0, 60.0, 30.0, 75.0, 90.0)
    min_shift_m: float = 0.5
    max_shift_m: float = 4.0
    shift_step_m: float = 0.25
    plan_half_m: float = 0.85        # planning corridor (> corridor_half_m + DR/beam smear)
    cluster_link_m: float = 0.75
    pass_margin_m: float = 1.6       # drive this far beyond the obstacle before rejoining
    lookahead_m: float = 10.0
    rejoin_angle_deg: float = 40.0   # mission intercept angle used for the rejoin check
    maneuver_speed_mps: float = 0.15
    turn_tol_deg: float = 6.0
    turn_settle_s: float = 0.5
    turn_timeout_s: float = 30.0
    pass_los_m: float = 2.0
    max_retries: int = 3
    retry_reset_s: float = 15.0      # clear TRACK this long -> retry counter reset
    max_episode_s: float = 150.0     # one obstacle may cost at most this long, then the leg is skipped
    # AVOID stage 2 (VFH)
    vfh_window_m: float = 6.0
    vfh_sector_deg: float = 5.0
    vfh_robot_r_m: float = 0.60
    vfh_thr_high: float = 3.0
    vfh_thr_low: float = 1.5
    vfh_fov_deg: float = 65.0        # directions outside are unknown (turn first, then look)
    vfh_turn_first_deg: float = 25.0
    vfh_done_s: float = 2.0
    vfh_timeout_s: float = 90.0


@dataclass
class AvoidInput:
    t: float
    x: float
    y: float
    yaw_deg: float
    speed_mps: float
    obstacles: np.ndarray
    mission_speed: float = 0.0
    mission_heading: Optional[float] = None
    track: Optional[Tuple[float, float, float]] = None   # (ox, oy, bearing_deg)
    goal: Optional[Tuple[float, float]] = None
    mission_turning: bool = False
    mission_active: bool = True


@dataclass
class AvoidCommand:
    state: str
    zone: str
    speed_ref: Optional[float] = None
    speed_cap: Optional[float] = None
    heading_ref: Optional[float] = None
    waypoint_blocked: bool = False
    abort: bool = False               # pattern cannot continue safely (e.g. in-place turn blocked) -> surface/hold
    obstacle_dist_m: Optional[float] = None
    stop_dist_m: float = 0.0
    slow_dist_m: float = 0.0
    events: List[str] = field(default_factory=list)
    plan: Optional[dict] = None
    message: str = ""


@dataclass
class SidestepPlan:
    side: int                 # +1 left, -1 right
    theta_deg: float
    heading_out: float
    origin: Tuple[float, float]
    bearing: float
    c_target: float
    a_pass: float
    shift: float

    def as_dict(self) -> dict:
        return dict(side=self.side, theta=self.theta_deg, heading_out=self.heading_out, origin=self.origin,
                    bearing=self.bearing, c_target=self.c_target, a_pass=self.a_pass, shift=self.shift)


def _seg_dist(px: np.ndarray, py: np.ndarray, a: Tuple[float, float], b: Tuple[float, float]) -> np.ndarray:
    ax, ay = a; bx, by = b
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        return np.hypot(px - ax, py - ay)
    u = np.clip(((px - ax) * vx + (py - ay) * vy) / L2, 0.0, 1.0)
    return np.hypot(px - (ax + u * vx), py - (ay + u * vy))


class ObstacleAvoidance:
    def __init__(self, cfg: Optional[AvoidanceConfig] = None):
        self.cfg = cfg or AvoidanceConfig()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.state = "TRACK"
        self.zone = "CLEAR"
        self.t_state = 0.0
        self.heading_hold: Optional[float] = None
        self.retries = 0
        self.plan: Optional[SidestepPlan] = None
        self.clear_since: Optional[float] = None
        self.ok_since: Optional[float] = None
        self.backoff_start: Optional[Tuple[float, float]] = None
        self.drive_start: Optional[Tuple[float, float]] = None
        self.vfh_prev: Optional[float] = None
        self.vfh_blocked_prev: Optional[np.ndarray] = None
        self.vfh_ok_since: Optional[float] = None
        self.hold_since: Optional[float] = None
        self.skip_sent = False
        self.last_warn = False
        self.after_backoff = False
        self.episode_start: Optional[float] = None
        self.turn_blocked_since: Optional[float] = None
        self.turn_block_heading: Optional[float] = None
        self.turn_block_pts: Optional[np.ndarray] = None
        self.backoff_allow = 0.0
        self.trail_m = 0.0                 # free path driven forward along the current heading
        self.trail_heading: Optional[float] = None
        self.trail_last: Optional[Tuple[float, float]] = None
        self.backed_total = 0.0
        self.episode_track = None
        self._t = 0.0

    def set_mode(self, mode: str, stage: Optional[str] = None) -> None:
        mode = mode.upper()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.cfg.mode = mode
        if stage:
            self.cfg.stage = stage.upper()
        self.reset()

    def _go(self, state: str, t: float) -> None:
        if state == "BACKOFF" and self.state != "BACKOFF":
            self.backoff_allow = max(0.0, self.trail_m - 0.2)
        self.state = state
        self.t_state = t
        self.ok_since = None
        self.clear_since = None

    # ------------------------------------------------------------------ geometry helpers
    def stop_distance(self, v: float) -> Tuple[float, float]:
        c = self.cfg
        v = abs(v)
        d = c.stop_base_m + v * c.react_time_s + v * v / (2.0 * c.brake_decel_mps2)
        stop = min(max(d, c.stop_min_m), c.stop_max_m)
        return stop, stop + c.slow_margin_m

    def forward_clearance(self, pts: np.ndarray, x: float, y: float, yaw: float,
                          half_w: Optional[float] = None) -> Optional[float]:
        """Distance from the sonar head to the nearest obstacle point in cone AND corridor
        (half_w given: a straight strip of that half width = collision course)."""
        if pts is None or len(pts) == 0:
            return None
        c = self.cfg
        ps = math.radians(yaw)
        cs, sn = math.cos(ps), math.sin(ps)
        dx = pts[:, 0] - x; dy = pts[:, 1] - y
        bx = dx * cs + dy * sn - c.sonar_x_m
        by = -dx * sn + dy * cs
        tanc = math.tan(math.radians(c.cone_half_deg))
        lat = np.abs(by) - c.point_margin_m
        width = np.minimum(c.corridor_half_m, np.maximum(c.collide_half_m, np.maximum(bx, 0.0) * tanc))
        if half_w is not None:
            width = np.full_like(bx, half_w)
        m = (bx > -0.3) & (lat <= width)
        if not m.any():
            return None
        return float(max(bx[m].min() - c.point_margin_m, 0.0))

    def _swing_blocked(self, pts: np.ndarray, x: float, y: float, yaw: float) -> bool:
        if pts is None or len(pts) == 0:
            return False
        ps = math.radians(yaw)
        dx = pts[:, 0] - x; dy = pts[:, 1] - y
        bx = dx * math.cos(ps) + dy * math.sin(ps)
        d = np.hypot(dx, dy) - self.cfg.point_margin_m
        return bool(((d < self.cfg.swing_radius_m) & (bx > -0.3)).any())

    def side_guard(self, pts: np.ndarray, x: float, y: float, yaw: float, href: Optional[float]) -> Tuple[Optional[float], bool]:
        """The sonar only looks forward: an obstacle that is BESIDE the hull (seen earlier, kept in the
        grid) must not be hit by turning towards it (e.g. rejoining the track). Returns (heading, limited)."""
        if href is None or pts is None or len(pts) == 0:
            return href, False
        c = self.cfg
        err = wrap180(href - yaw)
        if abs(err) < 3.0:
            return href, False
        ps = math.radians(yaw)
        dx = pts[:, 0] - x; dy = pts[:, 1] - y
        bx = dx * math.cos(ps) + dy * math.sin(ps)
        by = -dx * math.sin(ps) + dy * math.cos(ps)
        side = (bx > c.side_guard_x_m[0]) & (bx < c.side_guard_x_m[1]) & (np.abs(by) - c.point_margin_m < c.side_guard_y_m)
        if (side & (np.sign(by) == np.sign(err))).any():       # turning (left = +) towards that side
            return wrap180(yaw + math.copysign(2.0, -err) * 0.0), True
        return href, False

    def _update_trail(self, inp: AvoidInput) -> None:
        """Track how far the vehicle has come straight along its present heading: reversing up to that
        distance is safe (it is its own, just-travelled path); anything further is blind."""
        if self.trail_heading is None or abs(wrap180(inp.yaw_deg - self.trail_heading)) > 20.0:
            self.trail_heading, self.trail_m = inp.yaw_deg, 0.0
            self.trail_last = (inp.x, inp.y)
            return
        if self.trail_last is not None:
            ps = math.radians(self.trail_heading)
            ds = (inp.x - self.trail_last[0]) * math.cos(ps) + (inp.y - self.trail_last[1]) * math.sin(ps)
            self.trail_m = max(0.0, self.trail_m + ds)
        self.trail_last = (inp.x, inp.y)

    def turn_sweep_blocked(self, pts: np.ndarray, x: float, y: float, yaw_from: float, yaw_to: float) -> bool:
        """In-place turn about the turning point: does the swept hull (nose AND tail) come closer than
        swing_margin_m to a remembered obstacle cell?  The tail is invisible to the forward sonar, but
        cells seen earlier (e.g. the tank wall alongside) are still in the grid."""
        if pts is None or len(pts) == 0:
            return False
        c = self.cfg
        dx = pts[:, 0] - x; dy = pts[:, 1] - y
        near = np.hypot(dx, dy) < max(c.nose_m, c.tail_m) + c.swing_margin_m + c.point_margin_m + 0.1
        if not near.any():
            return False
        dx, dy = dx[near], dy[near]
        dpsi = wrap180(yaw_to - yaw_from)
        n = max(2, int(abs(dpsi) / 8.0) + 2)
        lim = c.hull_radius_m + c.swing_margin_m + c.point_margin_m
        for k in range(n):
            ps = math.radians(yaw_from + dpsi * k / (n - 1))
            cs, sn = math.cos(ps), math.sin(ps)
            a = (c.nose_m - c.hull_radius_m) * cs, (c.nose_m - c.hull_radius_m) * sn
            b = (-(c.tail_m - c.hull_radius_m) * cs, -(c.tail_m - c.hull_radius_m) * sn)
            if (_seg_dist(dx, dy, a, b) < lim).any():
                return True
        return False

    @staticmethod
    def _track_frame(origin, bearing, px, py):
        b = math.radians(bearing)
        ux, uy = math.cos(b), math.sin(b)
        dx = np.asarray(px) - origin[0]; dy = np.asarray(py) - origin[1]
        return dx * ux + dy * uy, -dx * uy + dy * ux     # along, cross (left +)

    @staticmethod
    def _from_track(origin, bearing, a, cc):
        b = math.radians(bearing)
        ux, uy = math.cos(b), math.sin(b)
        return origin[0] + a * ux - cc * uy, origin[1] + a * uy + cc * ux

    def _goal_ahead(self, inp: AvoidInput) -> Optional[float]:
        """Goal distance along the current heading, measured from the sonar head."""
        if inp.goal is None:
            return None
        ps = math.radians(inp.yaw_deg)
        return (inp.goal[0] - inp.x) * math.cos(ps) + (inp.goal[1] - inp.y) * math.sin(ps) - self.cfg.sonar_x_m

    # ------------------------------------------------------------------ main step
    def step(self, inp: AvoidInput) -> AvoidCommand:
        c = self.cfg
        t = inp.t
        self._t = t
        pts = inp.obstacles if inp.obstacles is not None else np.zeros((0, 2))
        self._update_trail(inp)
        d = self.forward_clearance(pts, inp.x, inp.y, inp.yaw_deg)
        v_eff = max(abs(inp.speed_mps), min(max(inp.mission_speed, 0.0), c.slow_speed_mps))
        stop_d, slow_d = self.stop_distance(v_eff)
        g = self._goal_ahead(inp)
        exempt = d is not None and g is not None and g > -0.5 and (d - max(g, 0.0)) >= c.goal_margin_m
        # zone with hysteresis
        if d is None or d > slow_d:
            zone = "CLEAR"
        elif d > stop_d and not (self.zone == "STOP" and d <= stop_d + c.exit_hyst_m):
            zone = "SLOW"
        else:
            zone = "STOP"
        self.zone = zone
        cmd = AvoidCommand(state=self.state, zone=zone + ("*" if exempt and zone != "CLEAR" else ""),
                           obstacle_dist_m=d, stop_dist_m=stop_d, slow_dist_m=slow_d)
        mode = c.mode.upper()
        if mode == "OFF":
            cmd.state = "OFF"
            return cmd
        if mode == "WARN":
            w = zone == "STOP" and not exempt
            if w and not self.last_warn:
                cmd.events.append("WARN_STOP")
            self.last_warn = w
            cmd.state = "WARN"
            return cmd
        if not inp.mission_active:
            self._go("TRACK", t)
            cmd.state = "IDLE"
            return cmd

        stop_now = zone == "STOP" and not exempt
        if inp.track is not None and self.episode_track is not None and inp.track != self.episode_track:
            self.episode_start = None          # new mission leg -> new episode
            self.backed_total = 0.0
        if self.state != "TRACK" and self.episode_start is not None and t - self.episode_start > c.max_episode_s:
            self.retries = c.max_retries
            self.episode_start = None
            return self._fail(cmd, inp, f"obstacle not cleared in {c.max_episode_s:.0f} s")
        # distance needed to RESUME at the speed the mission would resume with (no stop/go cycling)
        v_res = max(inp.mission_speed, 0.0)
        if c.slow_margin_m > 0 and d is not None and d <= self.stop_distance(v_res)[1]:
            v_res = min(v_res, c.slow_speed_mps)
        resume_d = max(stop_d, self.stop_distance(v_res)[0]) + c.exit_hyst_m
        # back-off only for an obstacle on a collision course: a wall BESIDE the hull is not fixed by
        # reversing (the vehicle would only slide along it, blind, into whatever is behind)
        d_hit = self.forward_clearance(pts, inp.x, inp.y, inp.yaw_deg, half_w=c.collide_half_m)
        too_close = d_hit is not None and d_hit < c.backoff_trigger_m and not exempt
        st = self.state

        # global back-off trigger (already inside the STOP zone, e.g. obstacle appeared late)
        if too_close and st not in ("BACKOFF",) and self.trail_m - 0.2 < c.min_backoff_m:
            too_close = False                  # no known-free water behind: hold instead of reversing blind
            if st in ("TRACK", "PASS", "DRIVE_OUT", "VFH") or abs(inp.speed_mps) > c.brake_v_eps:
                if st == "TRACK":
                    cmd.events.append("STOP_TRIGGER")
                self.heading_hold = inp.yaw_deg
                self._go("BRAKE" if c.mode.upper() == "AVOID" else "BRAKE", t)
                st = "BRAKE"
        if too_close and st not in ("BACKOFF",):
            if st == "TRACK":
                cmd.events.append("STOP_TRIGGER")
            elif st in ("VFH", "TURN_OUT", "DRIVE_OUT", "TURN_BACK", "PASS"):
                self.retries += 1
            cmd.events.append("BACKOFF")
            self.heading_hold = inp.yaw_deg if st in ("TRACK", "PASS", "DRIVE_OUT", "VFH") else (self.heading_hold or inp.yaw_deg)
            self.backoff_start = (inp.x, inp.y)
            self._go("BACKOFF", t)
            st = "BACKOFF"

        if st == "TRACK":
            if zone == "CLEAR":
                self.clear_since = self.clear_since or t
                if t - self.clear_since > c.retry_reset_s:
                    self.retries = 0
                    self.episode_start = None
                    self.backed_total = 0.0
            else:
                self.clear_since = None
            if inp.mission_turning and abs(inp.speed_mps) < 0.08 and inp.mission_heading is not None:
                # mission turns in place (pattern corner): never brake/scan for that, but refuse a turn
                # whose swept hull (nose or TAIL) would touch a remembered obstacle.
                # cells beside / behind the hull are not refreshed by the forward sonar and would age out
                # of the grid: once a turn has been refused, keep the cells that blocked it (latched)
                chk = pts if self.turn_block_pts is None else (
                    np.vstack((pts, self.turn_block_pts)) if len(pts) else self.turn_block_pts)
                blocked = self.turn_sweep_blocked(chk, inp.x, inp.y, inp.yaw_deg, inp.mission_heading)
                if blocked:
                    cmd.speed_ref = 0.0
                    if self.turn_blocked_since is None:
                        self.turn_blocked_since = t
                        self.turn_block_heading = inp.yaw_deg
                        near = np.hypot(pts[:, 0] - inp.x, pts[:, 1] - inp.y) < 2.0 if len(pts) else None
                        self.turn_block_pts = pts[near].copy() if near is not None and near.any() else None
                        cmd.events.append("TURN_BLOCKED")
                    cmd.heading_ref = self.turn_block_heading
                    cmd.message = "in-place turn refused: hull/tail would hit a known obstacle"
                    if t - self.turn_blocked_since > c.turn_blocked_abort_s:
                        cmd.abort = True
                        cmd.events.append("ABORT")
                    cmd.state = "TRACK"
                    return cmd
                if stop_now:
                    cmd.speed_ref = 0.0
                    cmd.message = "turning in place, obstacle in cone"
                    cmd.state = "TRACK"
                    return cmd
            else:
                self.turn_blocked_since = None
                self.turn_block_pts = None
            if stop_now:
                cmd.events.append("STOP_TRIGGER")
                if self.episode_start is None:
                    self.episode_start, self.episode_track = t, inp.track
                self.heading_hold = inp.yaw_deg
                self._go("BRAKE", t)
                st = "BRAKE"
            else:
                if zone == "SLOW" or (zone == "STOP" and exempt):
                    cmd.speed_cap = c.slow_speed_mps
                h, lim = self.side_guard(pts, inp.x, inp.y, inp.yaw_deg, inp.mission_heading)
                if lim:
                    cmd.heading_ref = h
                    cmd.speed_cap = min(cmd.speed_cap or 9.0, c.maneuver_speed_mps)
                    cmd.state = "TRACK"
                    cmd.message = "side guard: obstacle beside hull, not turning towards it"
                    return cmd
                cmd.state = "TRACK"
                cmd.message = {"CLEAR": "clear", "SLOW": f"SLOW obstacle {d:.1f} m" if d is not None else ""}.get(zone, "slow (goal before obstacle)")
                return cmd

        if st == "BRAKE":
            cmd.speed_ref = 0.0
            cmd.heading_ref = self.heading_hold
            if not self.after_backoff and not stop_now and (d is None or d > resume_d):
                self.clear_since = self.clear_since or t
                if t - self.clear_since >= c.clear_hold_s:
                    cmd.events.append("RESUME")
                    self._go("TRACK", t)
                    cmd.speed_ref = cmd.heading_ref = None
                    cmd.state = "TRACK"
                    return cmd
            else:
                self.clear_since = None
            stopped = abs(inp.speed_mps) < c.brake_v_eps
            if stopped:
                self.ok_since = self.ok_since or t
            else:
                self.ok_since = None
            if (self.ok_since is not None and t - self.ok_since >= 0.4) or t - self.t_state > c.brake_timeout_s:
                self.after_backoff = False
                if mode == "STOP":
                    self.hold_since = t
                    self.skip_sent = False
                    self._go("HOLD", t)
                else:
                    self._go("SCAN", t)
            cmd.state = self.state
            cmd.message = f"BRAKE obstacle {d:.1f} m" if d is not None else "BRAKE"
            return cmd

        if st == "HOLD":                       # STOP mode
            cmd.speed_ref = 0.0
            swing = inp.mission_heading is not None and self.turn_sweep_blocked(
                pts, inp.x, inp.y, inp.yaw_deg, inp.mission_heading)
            if inp.mission_heading is not None and not swing:
                cmd.heading_ref = inp.mission_heading      # turning in place is allowed
            else:
                cmd.heading_ref = self.heading_hold
            if not stop_now and not inp.mission_turning and (d is None or d > resume_d or exempt):
                self.clear_since = self.clear_since or t
                if t - self.clear_since >= c.clear_hold_s:
                    cmd.events.append("RESUME")
                    self._go("TRACK", t)
                    cmd.speed_ref = cmd.heading_ref = None
                    cmd.state = "TRACK"
                    return cmd
            else:
                self.clear_since = None
            if inp.mission_turning:
                self.hold_since = t
                self.heading_hold = inp.yaw_deg
            if (c.blocked_skip_s > 0 and not self.skip_sent and not inp.mission_turning
                    and t - (self.hold_since or t) >= c.blocked_skip_s):
                cmd.waypoint_blocked = True
                cmd.events.append("WP_BLOCKED")
                self.skip_sent = True
                self.hold_since = t
            if inp.mission_turning:
                self.skip_sent = False
            cmd.state = "HOLD"
            cmd.message = f"STOPPED obstacle {d:.1f} m" if d is not None else "STOPPED"
            return cmd

        if st == "BACKOFF":
            if self.backed_total + (math.hypot(inp.x - self.backoff_start[0], inp.y - self.backoff_start[1])
                                    if self.backoff_start else 0.0) > c.max_backoff_total_m:
                self.backed_total = 0.0
                self.retries = c.max_retries
                self._go("BRAKE", t)
                return self._fail(cmd, inp, "back-off budget used (reversing is blind)")
            cmd.speed_ref = -c.backoff_speed_mps
            cmd.heading_ref = self.heading_hold
            bx, by = self.backoff_start or (inp.x, inp.y)
            moved = math.hypot(inp.x - bx, inp.y - by)
            limit = min(c.backoff_dist_m, self.backoff_allow)
            done_far = (d is None or d > stop_d) and moved >= 0.5 * limit
            if moved >= limit or t - self.t_state > c.backoff_timeout_s or done_far:
                cmd.events.append("BACKOFF_DONE")
                self.after_backoff = True
                self.backed_total += moved
                self._go("BRAKE", t)
                cmd.speed_ref = 0.0
            cmd.state = self.state
            cmd.message = f"BACK-OFF {moved:.1f}/{limit:.1f} m (own trail)"
            return cmd

        if st == "SCAN":
            cmd.speed_ref = 0.0
            cmd.heading_ref = self.heading_hold
            cmd.state = "SCAN"
            cmd.message = "SCAN (sonar sweep)"
            if t - self.t_state < c.scan_time_s:
                return cmd
            track = inp.track or (inp.x, inp.y, self.heading_hold if self.heading_hold is not None else inp.yaw_deg)
            if not stop_now and (d is None or d > resume_d) and not self.track_blocked(pts, inp, track):
                cmd.events.append("RESUME")
                self._go("TRACK", t)
                cmd.speed_ref = cmd.heading_ref = None
                cmd.state = "TRACK"
                return cmd
            # goal in front of / inside / just behind the blocking obstacle (tank wall at a pattern
            # corner): going around is pointless -> report the leg as blocked, mission moves on
            if inp.goal is not None and d is not None:
                gd = math.hypot(inp.goal[0] - inp.x, inp.goal[1] - inp.y)
                bc = self.blocking_extent(pts, inp.x, inp.y, track)
                goal_a = self._track_frame((track[0], track[1]), track[2], inp.goal[0], inp.goal[1])[0]
                goal_hidden = bc is not None and float(goal_a) <= bc[1] + c.goal_margin_m
                if goal_hidden or (gd <= c.skip_goal_within_m and (g is None or g >= d - c.goal_margin_m)):
                    cmd.waypoint_blocked = True
                    cmd.events.append("WP_BLOCKED")
                    self._go("TRACK", t)
                    cmd.state = "TRACK"
                    cmd.message = "leg end blocked -> next leg"
                    return cmd
            plan = None
            if c.stage.upper() == "VFH" and self.retries > 0:
                # VFH got trapped once (U-shape / local minimum): use the geometric side-step planner
                plan = self.plan_sidestep(pts, inp.x, inp.y, track, inp.goal)
                if plan is not None:
                    self.plan = plan
                    cmd.events.append(f"SIDESTEP_{'LEFT' if plan.side > 0 else 'RIGHT'}")
                    self._go("TURN_OUT", t)
                    st = "TURN_OUT"
            if plan is not None:
                pass
            elif c.stage.upper() == "VFH":
                self.vfh_prev = None
                self.vfh_blocked_prev = None
                self.vfh_ok_since = None
                self._go("VFH", t)
                st = "VFH"
            else:
                plan = self.plan_sidestep(pts, inp.x, inp.y, track, inp.goal)
                if plan is None:
                    return self._fail(cmd, inp, "no side-step found")
                self.plan = plan
                cmd.events.append(f"SIDESTEP_{'LEFT' if plan.side > 0 else 'RIGHT'}")
                self._go("TURN_OUT", t)
                st = "TURN_OUT"

        if st in ("TURN_OUT", "DRIVE_OUT", "TURN_BACK", "PASS"):
            return self._sidestep_step(cmd, inp, pts, d, stop_d, stop_now)

        if st == "VFH":
            return self._vfh_step(cmd, inp, pts, d, stop_d, slow_d)

        cmd.state = self.state
        return cmd

    def _fail(self, cmd: AvoidCommand, inp: AvoidInput, why: str) -> AvoidCommand:
        self.retries += 1
        if self.retries > self.cfg.max_retries:
            cmd.events.append("GIVE_UP")
            cmd.waypoint_blocked = True
            cmd.message = f"trapped ({why}) - leg skipped"
            self.retries = 0
            self._go("TRACK", self._t)
            cmd.state = "TRACK"
            cmd.speed_ref = 0.0
            cmd.heading_ref = self.heading_hold
            return cmd
        if self.trail_m - 0.2 < self.cfg.min_backoff_m:
            # nothing known-free behind: wait (scan again) instead of reversing blind
            cmd.message = f"{why} - no free trail behind, rescanning ({self.retries}/{self.cfg.max_retries})"
            self._go("SCAN", self._t)
            cmd.state = "SCAN"
            cmd.speed_ref = 0.0
            cmd.heading_ref = self.heading_hold
            return cmd
        cmd.events.append("BACKOFF")
        cmd.message = f"{why} - back off ({self.retries}/{self.cfg.max_retries})"
        self.backoff_start = (inp.x, inp.y)
        self._go("BACKOFF", self._t)
        cmd.state = "BACKOFF"
        cmd.speed_ref = -self.cfg.backoff_speed_mps
        cmd.heading_ref = self.heading_hold
        return cmd

    def track_blocked(self, pts: np.ndarray, inp: AvoidInput, track) -> bool:
        """Obstacle on the mission line ahead (between vehicle and goal / lookahead)?"""
        c = self.cfg
        if pts is None or len(pts) == 0:
            return False
        origin, bearing = (track[0], track[1]), track[2]
        A, C = self._track_frame(origin, bearing, pts[:, 0], pts[:, 1])
        av, cv = self._track_frame(origin, bearing, inp.x, inp.y)
        a_end = float(av) + c.lookahead_m
        if inp.goal is not None:
            ga, _ = self._track_frame(origin, bearing, inp.goal[0], inp.goal[1])
            a_end = min(a_end, float(ga) + c.goal_margin_m)
        m = (A > float(av)) & (A < a_end) & (np.abs(C - float(cv)) < c.corridor_half_m + c.point_margin_m)
        return bool(m.any())

    def blocking_extent(self, pts, x, y, track) -> Optional[Tuple[float, float]]:
        """(a_first, a_max) along the track of the obstacle cluster blocking the corridor ahead."""
        c = self.cfg
        if pts is None or len(pts) == 0:
            return None
        origin, bearing = (track[0], track[1]), track[2]
        A, C = self._track_frame(origin, bearing, pts[:, 0], pts[:, 1])
        av, cv = self._track_frame(origin, bearing, x, y)
        av, cv = float(av), float(cv)
        rel = (A > av - 1.5) & (A < av + c.lookahead_m) & (np.abs(C - cv) < c.max_shift_m + c.plan_half_m + 2.0)
        A, C = A[rel], C[rel]
        blk = (A > av) & (np.abs(C - cv) < c.plan_half_m)
        if not blk.any():
            return None
        seed = int(np.flatnonzero(blk)[np.argmin(A[blk])])
        in_c = self._cluster(np.column_stack((A, C)), seed)
        return float(A[seed]), float(A[in_c].max())

    def _cluster(self, P: np.ndarray, seed: int) -> np.ndarray:
        in_c = np.zeros(len(P), bool); in_c[seed] = True
        frontier = [seed]
        while frontier:
            dmat = np.hypot(P[:, 0][None, :] - P[frontier, 0][:, None], P[:, 1][None, :] - P[frontier, 1][:, None])
            new = (dmat <= self.cfg.cluster_link_m).any(axis=0) & ~in_c
            in_c |= new
            frontier = list(np.flatnonzero(new))
        return in_c

    # ------------------------------------------------------------------ stage 1: side-step
    def plan_sidestep(self, pts: np.ndarray, x: float, y: float, track: Tuple[float, float, float],
                      goal: Optional[Tuple[float, float]]) -> Optional[SidestepPlan]:
        c = self.cfg
        if pts is None or len(pts) == 0:
            return None
        origin, bearing = (track[0], track[1]), track[2]
        A, C = self._track_frame(origin, bearing, pts[:, 0], pts[:, 1])
        av, cv = self._track_frame(origin, bearing, x, y)
        av, cv = float(av), float(cv)
        rel = (A > av - 1.5) & (A < av + c.lookahead_m) & (np.abs(C - cv) < c.max_shift_m + c.plan_half_m + 2.0)
        if not rel.any():
            return None
        A, C = A[rel], C[rel]
        blk = (A > av) & (np.abs(C - cv) < c.plan_half_m)
        if not blk.any():
            blk = A > av
            if not blk.any():
                return None
        seed = int(np.flatnonzero(blk)[np.argmin(A[blk])])
        # cluster the blocking obstacle (connected points)
        in_c = self._cluster(np.column_stack((A, C)), seed)
        a_pass = float(A[in_c].max()) + c.pass_margin_m
        goal_side = 0
        if goal is not None:
            _, gc = self._track_frame(origin, bearing, goal[0], goal[1])
            goal_side = int(np.sign(float(gc) - cv)) if abs(float(gc) - cv) > 0.3 else 0
        best = None
        tan_rejoin = math.tan(math.radians(c.rejoin_angle_deg))
        for side in (1, -1):
            for s in np.arange(c.min_shift_m, c.max_shift_m + 1e-6, c.shift_step_m):
                ct = cv + side * s
                for th in c.sidestep_angles_deg:
                    tr = math.radians(th)
                    a_join = av + s / math.tan(tr) if th < 89.0 else av
                    if a_join > a_pass - 0.5:
                        continue
                    seg_out = ((av, cv), (a_join, ct))
                    a_back = a_pass + s / tan_rejoin
                    seg_par = ((a_join, ct), (a_pass, ct))
                    seg_back = ((a_pass, ct), (a_back, cv))
                    ok = True
                    for seg in (seg_out, seg_par, seg_back):
                        if (_seg_dist(A, C, *seg) < c.plan_half_m + c.point_margin_m).any():
                            ok = False
                            break
                    if ok:
                        cost = s + 0.4 * (th / 45.0 - 1.0) ** 2 + (0.6 if goal_side and goal_side != side else 0.0)
                        if best is None or cost < best[0]:
                            best = (cost, side, th, ct, s)
                        break
                if best is not None and best[1] == side:
                    break
        if best is None:
            return None
        _, side, th, ct, s = best
        return SidestepPlan(side, th, wrap180(bearing + side * th), origin, bearing, ct, a_pass, s)

    def _sidestep_step(self, cmd, inp, pts, d, stop_d, stop_now) -> AvoidCommand:
        c = self.cfg
        t = inp.t
        p = self.plan
        a, cc = self._track_frame(p.origin, p.bearing, inp.x, inp.y)
        a, cc = float(a), float(cc)
        cmd.plan = p.as_dict()
        st = self.state
        if st == "TURN_OUT":
            cmd.speed_ref = 0.0
            cmd.heading_ref = p.heading_out
            err = abs(wrap180(p.heading_out - inp.yaw_deg))
            self.ok_since = (self.ok_since or t) if err < c.turn_tol_deg else None
            if self.ok_since is not None and t - self.ok_since >= c.turn_settle_s:
                self.drive_start = (inp.x, inp.y)
                self._go("DRIVE_OUT", t)
            elif t - self.t_state > c.turn_timeout_s:
                return self._fail(cmd, inp, "turn timeout")
            cmd.message = f"TURN OUT {'LEFT' if p.side > 0 else 'RIGHT'} {p.theta_deg:.0f} deg"
        elif st == "DRIVE_OUT":
            remaining = p.side * (p.c_target - cc) / max(math.sin(math.radians(p.theta_deg)), 0.2)
            if d is not None and d < min(stop_d, max(remaining, 0.0) + 1.0):
                cmd.events.append("STOP_TRIGGER")
                self.heading_hold = inp.yaw_deg
                self.retries += 1
                self._go("BRAKE" if self.retries <= c.max_retries else "TRACK", t)
                cmd.speed_ref = 0.0
                cmd.heading_ref = self.heading_hold
                cmd.state = self.state
                return cmd
            cmd.speed_ref = c.maneuver_speed_mps
            cmd.heading_ref = p.heading_out
            moved = math.hypot(inp.x - self.drive_start[0], inp.y - self.drive_start[1])
            if remaining <= 0.05 or moved > p.shift / max(math.sin(math.radians(p.theta_deg)), 0.2) * 1.5 + 1.0:
                self._go("TURN_BACK", t)
            cmd.message = f"SIDE-STEP drive, {max(remaining, 0):.1f} m to offset line"
        elif st == "TURN_BACK":
            href = self._pass_heading(p, cc)
            cmd.speed_ref = 0.03
            cmd.heading_ref = href
            err = abs(wrap180(href - inp.yaw_deg))
            if err < 2 * c.turn_tol_deg:
                self._go("PASS", t)
            elif t - self.t_state > c.turn_timeout_s:
                self._go("PASS", t)
            cmd.message = "TURN BACK to track heading"
        elif st == "PASS":
            if stop_now:
                cmd.events.append("STOP_TRIGGER")
                self.heading_hold = inp.yaw_deg
                self.retries += 1
                self._go("BRAKE", t)
                cmd.speed_ref = 0.0
                cmd.heading_ref = self.heading_hold
                cmd.state = "BRAKE"
                return cmd
            cmd.speed_ref = c.maneuver_speed_mps
            cmd.heading_ref = self._pass_heading(p, cc)
            if a >= p.a_pass:
                cmd.events.append("REJOIN")
                self._go("TRACK", t)
                cmd.speed_ref = cmd.heading_ref = None
            cmd.message = f"PASS obstacle, {max(p.a_pass - a, 0):.1f} m to rejoin"
        cmd.state = self.state
        return cmd

    def _pass_heading(self, p: SidestepPlan, cc: float) -> float:
        corr = math.degrees(math.atan2(cc - p.c_target, self.cfg.pass_los_m))
        return wrap180(p.bearing - max(-30.0, min(30.0, corr)))

    # ------------------------------------------------------------------ stage 2: VFH
    def vfh_histogram(self, pts: np.ndarray, x: float, y: float, yaw: float) -> Tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        ns = int(round(360.0 / c.vfh_sector_deg))
        centers = -180.0 + (np.arange(ns) + 0.5) * c.vfh_sector_deg
        H = np.zeros(ns)
        if pts is not None and len(pts):
            dx = pts[:, 0] - x; dy = pts[:, 1] - y
            dist = np.hypot(dx, dy)
            m = (dist < c.vfh_window_m) & (dist > 0.05)
            if m.any():
                dist = dist[m]
                beta = (np.degrees(np.arctan2(dy[m], dx[m])) - yaw + 180.0) % 360.0 - 180.0
                mag = (c.vfh_window_m - dist) / c.vfh_window_m * 4.0
                gam = np.degrees(np.arcsin(np.clip(c.vfh_robot_r_m / np.maximum(dist, 1e-3), 0.0, 1.0)))
                diff = np.abs((centers[None, :] - beta[:, None] + 180.0) % 360.0 - 180.0)
                H = (mag[:, None] * (diff <= gam[:, None] + c.vfh_sector_deg / 2)).sum(axis=0)
        return centers, H

    def vfh_select(self, centers, H, goal_rel: float) -> Optional[float]:
        c = self.cfg
        prev = self.vfh_blocked_prev
        blocked = H > c.vfh_thr_high
        if prev is not None and len(prev) == len(H):
            blocked |= prev & (H > c.vfh_thr_low)
        self.vfh_blocked_prev = blocked
        free = ~blocked
        safe = free & np.roll(free, 1) & np.roll(free, -1)
        if not safe.any():
            return None
        cand = centers[safe]
        prev_rel = self.vfh_prev if self.vfh_prev is not None else 0.0
        cost = (5.0 * np.abs((cand - goal_rel + 180) % 360 - 180) + 2.0 * np.abs(cand)
                + 2.0 * np.abs((cand - prev_rel + 180) % 360 - 180)
                + np.where(np.abs(cand) > c.vfh_fov_deg, 150.0, 0.0))
        return float(cand[int(np.argmin(cost))])

    def _vfh_step(self, cmd, inp, pts, d, stop_d, slow_d) -> AvoidCommand:
        c = self.cfg
        t = inp.t
        goal_hdg = inp.mission_heading if inp.mission_heading is not None else (self.heading_hold or inp.yaw_deg)
        goal_rel = wrap180(goal_hdg - inp.yaw_deg)
        centers, H = self.vfh_histogram(pts, inp.x, inp.y, inp.yaw_deg)
        sel = self.vfh_select(centers, H, goal_rel)
        if sel is None:
            return self._fail(cmd, inp, "VFH: no free valley")
        self.vfh_prev = sel
        href = wrap180(inp.yaw_deg + sel)
        href, lim = self.side_guard(pts, inp.x, inp.y, inp.yaw_deg, href)
        if lim:
            sel = 0.0
        if abs(sel) > 15.0 and self._swing_blocked(pts, inp.x, inp.y, inp.yaw_deg):
            return self._fail(cmd, inp, "VFH: no room to turn")
        if abs(sel) > c.vfh_turn_first_deg or (d is not None and d < c.stop_min_m):
            spd = 0.0
        else:
            room = 99.0 if d is None else d - c.stop_min_m
            spd = c.maneuver_speed_mps * min(1.0, max(0.3, room / 2.0))
        cmd.speed_ref, cmd.heading_ref = spd, href
        cmd.plan = dict(vfh_centers=centers, vfh_H=H, vfh_sel=sel, yaw=inp.yaw_deg)
        # done when the mission direction itself is clear
        gi = int(np.argmin(np.abs((centers - goal_rel + 180) % 360 - 180)))
        d_goal = self.forward_clearance(pts, inp.x, inp.y, goal_hdg)
        _, side_lim = self.side_guard(pts, inp.x, inp.y, inp.yaw_deg, goal_hdg)
        goal_free = (not self.vfh_blocked_prev[gi] and (d_goal is None or d_goal > slow_d)
                     and abs(goal_rel) < c.vfh_fov_deg and not side_lim)
        if goal_free:
            self.vfh_ok_since = self.vfh_ok_since or t
            if t - self.vfh_ok_since >= c.vfh_done_s:
                cmd.events.append("REJOIN")
                self._go("TRACK", t)
                cmd.speed_ref = cmd.heading_ref = None
        else:
            self.vfh_ok_since = None
        if self.state == "VFH" and t - self.t_state > c.vfh_timeout_s:
            return self._fail(cmd, inp, "VFH timeout")
        cmd.state = self.state
        cmd.message = f"VFH steer {sel:+.0f} deg"
        return cmd


# ---------------------------------------------------------------------------------------------
# Arbiter: avoidance may override the T1 speed setpoint and the T2 heading reference ONLY.
# ---------------------------------------------------------------------------------------------
def arbitrate(mission_speed: float, mission_heading: Optional[float], cmd: AvoidCommand
              ) -> Tuple[float, Optional[float], str]:
    speed, heading, owner = mission_speed, mission_heading, "MISSION"
    if cmd.speed_ref is not None:
        speed, owner = cmd.speed_ref, "AVOID"
    elif cmd.speed_cap is not None and speed > cmd.speed_cap:
        speed, owner = cmd.speed_cap, "CAP"
    if cmd.heading_ref is not None:
        heading, owner = cmd.heading_ref, "AVOID"
    return speed, heading, owner


def cap_t1_pwm(t1_pwm: int, v_meas: Optional[float], cmd: AvoidCommand, neutral: int = 1500,
               deadband_us: int = 25, us_per_mps2: float = 1.0 / 0.0018, drag: float = 0.20,
               fwd_sign: int = 1) -> int:
    """For PWM-level pattern code (e.g. _square_step returns T1 pwm): apply the speed cap /
    brake directly on the pulse.  A full speed override should instead go through your T1 speed
    controller with cmd.speed_ref as the setpoint."""
    u = (t1_pwm - neutral) * fwd_sign
    if cmd.speed_ref is not None and cmd.speed_ref <= 0.0:
        v = v_meas or 0.0
        if v > 0.03:
            u = -(deadband_us + min(120.0, 250.0 * v))
        else:
            u = 0.0
    elif cmd.speed_cap is not None:
        cap = cmd.speed_cap
        u_cap = deadband_us + drag * cap * cap * us_per_mps2 + 10.0
        if v_meas is not None and v_meas > cap + 0.03:
            u_cap = 0.0
        u = min(u, u_cap)
    return int(round(neutral + fwd_sign * u))
