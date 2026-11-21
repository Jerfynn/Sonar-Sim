"""
detection.py - Ping360 obstacle detection, pure logic (numpy only, no simulator imports).

Pipeline per ping (runs unchanged on simulated or real Ping360 data):
  SonarPing (angle_grad, range_m, samples[])
    -> median filter (kills 1-2 sample spikes)
    -> adaptive threshold: max(absolute floor, k * ping median, SO-CFAR local noise)
    -> FIRST real return = first above-threshold run >= min_blob_m with peak >= peak_min
       (ghost/multipath echoes behind a wall are ignored because only the first return counts)
    -> world point using the vehicle pose AT PING TIME (PoseHistory interpolation)
    -> OccupancyGrid: hit counts, ray clearing in front of the return, aging

Frames (match auv_depth.py): world ENU metres (x = East, y = North), yaw in degrees ENU
(0 = East, counter-clockwise positive, as in _sq_unit()).  Body frame: x forward, y LEFT.

Ping360 angle convention: angle_grad 0..399 (400 per revolution).  `fwd_grad` is the
angle that points along the bow (200 for the 100..300 forward sector in auv_depth.py);
`angle_dir` = +1 if angle_grad increases clockwise seen from above (towards starboard).
Verify both once in the tank with a target on the bow and one on the starboard side.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

SOUND_SPEED_MPS = 1500.0
GRAD_TO_DEG = 0.9
PING360_TICK_S = 25e-9


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------------------------
# Ping container - same content as a real driver message
# --------------------------------------------------------------------------------------------
@dataclass
class SonarPing:
    angle_grad: int
    range_m: float
    samples: np.ndarray                  # uint8 intensities, sample i at (i + 0.5) * range_m / N
    t: float = 0.0                       # timestamp (s) of the ping (same clock as PoseHistory)
    sample_period: Optional[int] = None  # Ping360 ticks of 25 ns
    gain: int = 0
    transmit_duration: int = 0

    @property
    def n(self) -> int:
        return int(len(self.samples))

    @staticmethod
    def range_from_sample_period(sample_period: int, n_samples: int, c: float = SOUND_SPEED_MPS) -> float:
        return sample_period * PING360_TICK_S * n_samples * c / 2.0

    @staticmethod
    def sample_period_for_range(range_m: float, n_samples: int, c: float = SOUND_SPEED_MPS) -> int:
        return int(round(2.0 * range_m / (c * PING360_TICK_S * n_samples)))

    @classmethod
    def from_driver(cls, angle_grad: int, sample_period: int, samples: Sequence[int], t: float,
                    c: float = SOUND_SPEED_MPS, **kw) -> "SonarPing":
        """Build from brping / native-UDP fields (msg.angle, msg.sample_period, msg.data)."""
        s = np.frombuffer(bytes(samples), dtype=np.uint8) if isinstance(samples, (bytes, bytearray)) \
            else np.asarray(samples, dtype=np.uint8)
        return cls(int(angle_grad), cls.range_from_sample_period(sample_period, len(s), c), s, t,
                   sample_period=int(sample_period), **kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["samples"] = [int(v) for v in self.samples]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SonarPing":
        s = np.asarray(d["samples"], dtype=np.uint8)
        rng = d.get("range_m")
        if rng is None:
            rng = cls.range_from_sample_period(int(d["sample_period"]), len(s))
        return cls(int(d["angle_grad"]), float(rng), s, float(d.get("t", 0.0)),
                   d.get("sample_period"), int(d.get("gain", 0) or 0), int(d.get("transmit_duration", 0) or 0))


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------
@dataclass
class DetectionConfig:
    # sonar mounting / angle convention
    fwd_grad: float = 200.0
    angle_dir: int = 1
    sonar_x_m: float = 0.80          # head position ahead of the turning point (body x)
    sonar_y_m: float = 0.0           # body y (left +)
    # usable sector: relative bearings beyond this are ignored (tether, own hull, rear)
    valid_half_sector_deg: float = 100.0
    # per-ping detector
    blanking_m: float = 0.75
    max_use_range_m: float = 50.0
    median_samples: int = 5
    cfar_guard_m: float = 0.06
    cfar_train_m: float = 0.60
    cfar_alpha: float = 3.0          # x local noise (smallest-of leading/lagging window)
    global_alpha: float = 4.0        # x ping median (beyond blanking)
    abs_min: float = 60.0            # absolute floor (0..255 scale)
    peak_min: float = 90.0           # blob must reach this peak
    min_blob_m: float = 0.05         # first-return run length
    # occupancy grid (world-fixed, recentred on the vehicle)
    grid_size_m: float = 20.0
    grid_res_m: float = 0.25
    min_hits: float = 2.0
    hit_inc: float = 1.0
    max_count: float = 8.0
    age_s: float = 20.0
    clear_dec: float = 0.5           # decrement for cells a beam passed through without echo
    clear_margin_m: float = 0.40     # do not clear this close in front of the return
    recenter_dist_m: float = 4.0
    pose_max_gap_s: float = 0.6      # max extrapolation of the pose history
    fov_half_deg: float = 60.0       # scanned sector (+-) - cells outside it cannot be re-observed ...
    keep_near_m: float = 2.5         # ... so confirmed cells this close to the hull outside the FOV do not age


@dataclass
class BeamResult:
    t: float
    angle_grad: int
    rel_bearing_deg: float           # body frame, left +
    valid: bool                      # inside usable sector
    range_m: Optional[float]         # first return (None = open water)
    strength: float
    max_range_m: float
    sonar_xy: Optional[Tuple[float, float]] = None   # world position of the head at ping time
    beam_yaw_deg: Optional[float] = None
    hit_xy: Optional[Tuple[float, float]] = None


# --------------------------------------------------------------------------------------------
# Single-ping detector
# --------------------------------------------------------------------------------------------
class PingDetector:
    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        self.last_filtered: Optional[np.ndarray] = None
        self.last_threshold: Optional[np.ndarray] = None

    def rel_bearing(self, angle_grad: float) -> float:
        c = self.cfg
        return wrap180(-c.angle_dir * (angle_grad - c.fwd_grad) * GRAD_TO_DEG)

    @staticmethod
    def _median(x: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return x
        h = k // 2
        xp = np.pad(x, (h, h), mode="edge")
        return np.partition(np.lib.stride_tricks.sliding_window_view(xp, k), h, axis=1)[:, h]

    def first_return(self, samples: np.ndarray, range_m: float) -> Tuple[Optional[float], float]:
        """Returns (range of the leading edge of the first real echo or None, peak strength)."""
        c = self.cfg
        n = len(samples)
        if n < 16 or range_m <= 0:
            return None, 0.0
        dr = range_m / n
        x = self._median(samples.astype(np.float32), c.median_samples)
        start = min(n - 1, int(math.ceil(c.blanking_m / dr)))
        stop = min(n, int(c.max_use_range_m / dr))
        if stop - start < 8:
            return None, 0.0
        med = float(np.median(x[start:stop])) + 1.0
        guard = max(1, int(round(c.cfar_guard_m / dr)))
        train = max(2, int(round(c.cfar_train_m / dr)))
        cs = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
        idx = np.arange(n)
        # leading (closer) window [i-guard-train, i-guard)
        l0 = np.clip(idx - guard - train, 0, n); l1 = np.clip(idx - guard, 0, n)
        # lagging (farther) window (i+guard, i+guard+train]
        g0 = np.clip(idx + guard + 1, 0, n); g1 = np.clip(idx + guard + 1 + train, 0, n)
        ln = np.maximum(l1 - l0, 0); gn = np.maximum(g1 - g0, 0)
        lead = np.where(ln > 0, (cs[l1] - cs[l0]) / np.maximum(ln, 1), np.inf)
        lag = np.where(gn > 0, (cs[g1] - cs[g0]) / np.maximum(gn, 1), np.inf)
        local = np.minimum(lead, lag)
        local = np.where(np.isfinite(local), local, med)
        thr = np.maximum(np.maximum(c.abs_min, c.global_alpha * med), c.cfar_alpha * local)
        self.last_filtered, self.last_threshold = x, thr
        above = x >= thr
        above[:start] = False
        above[stop:] = False
        if not above.any():
            return None, 0.0
        d = np.diff(np.concatenate(([0], above.astype(np.int8), [0])))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        min_len = max(1, int(round(c.min_blob_m / dr)))
        for s0, e0 in zip(starts, ends):
            if e0 - s0 >= min_len:
                peak = float(x[s0:e0].max())
                if peak >= c.peak_min:
                    return (s0 + 0.5) * dr, peak
        return None, 0.0

    def process(self, ping: SonarPing, pose: Optional[Tuple[float, float, float]]) -> BeamResult:
        """pose = (x, y, yaw_deg) of the vehicle turning point at ping.t (None = unknown)."""
        c = self.cfg
        rel = self.rel_bearing(ping.angle_grad)
        valid = abs(rel) <= c.valid_half_sector_deg
        res = BeamResult(ping.t, ping.angle_grad, rel, valid, None, 0.0, ping.range_m)
        if not valid:
            return res
        rng, peak = self.first_return(np.asarray(ping.samples), ping.range_m)
        res.range_m, res.strength = rng, peak
        if pose is not None:
            x, y, yaw = pose
            ps = math.radians(yaw)
            cy, sy = math.cos(ps), math.sin(ps)
            hx = x + c.sonar_x_m * cy - c.sonar_y_m * sy
            hy = y + c.sonar_x_m * sy + c.sonar_y_m * cy
            byaw = yaw + rel
            res.sonar_xy, res.beam_yaw_deg = (hx, hy), byaw
            if rng is not None:
                b = math.radians(byaw)
                res.hit_xy = (hx + rng * math.cos(b), hy + rng * math.sin(b))
        return res


# --------------------------------------------------------------------------------------------
# Pose history (pose AT PING TIME)
# --------------------------------------------------------------------------------------------
class PoseHistory:
    def __init__(self, max_len: int = 400, max_gap_s: float = 0.6):
        self.buf: Deque[Tuple[float, float, float, float]] = deque(maxlen=max_len)
        self.max_gap_s = max_gap_s

    def add(self, t: float, x: float, y: float, yaw_deg: float) -> None:
        if self.buf and t <= self.buf[-1][0]:
            return
        self.buf.append((t, x, y, yaw_deg))

    def at(self, t: float) -> Optional[Tuple[float, float, float]]:
        if not self.buf:
            return None
        if t >= self.buf[-1][0]:
            t1, x1, y1, h1 = self.buf[-1]
            return (x1, y1, h1) if t - t1 <= self.max_gap_s else None
        if t <= self.buf[0][0]:
            t0, x0, y0, h0 = self.buf[0]
            return (x0, y0, h0) if t0 - t <= self.max_gap_s else None
        # binary search
        lo, hi = 0, len(self.buf) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.buf[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        t0, x0, y0, h0 = self.buf[lo]
        t1, x1, y1, h1 = self.buf[hi]
        a = (t - t0) / max(t1 - t0, 1e-9)
        return (x0 + a * (x1 - x0), y0 + a * (y1 - y0), wrap180(h0 + a * wrap180(h1 - h0)))


# --------------------------------------------------------------------------------------------
# Local occupancy grid
# --------------------------------------------------------------------------------------------
class OccupancyGrid:
    def __init__(self, cfg: DetectionConfig, cx: float = 0.0, cy: float = 0.0):
        self.cfg = cfg
        self.res = cfg.grid_res_m
        self.n = int(round(cfg.grid_size_m / self.res))
        self.counts = np.zeros((self.n, self.n), dtype=np.float32)   # [ix, iy]
        self.last_t = np.full((self.n, self.n), -1e9, dtype=np.float64)
        self.ox = math.floor((cx - cfg.grid_size_m / 2) / self.res) * self.res
        self.oy = math.floor((cy - cfg.grid_size_m / 2) / self.res) * self.res

    @property
    def center(self) -> Tuple[float, float]:
        h = self.n * self.res / 2
        return self.ox + h, self.oy + h

    def recenter(self, x: float, y: float) -> None:
        cx, cy = self.center
        if abs(x - cx) < self.cfg.recenter_dist_m and abs(y - cy) < self.cfg.recenter_dist_m:
            return
        sx = int(round((x - cx) / self.res))
        sy = int(round((y - cy) / self.res))
        self.counts = self._shift(self.counts, sx, sy, 0.0)
        self.last_t = self._shift(self.last_t, sx, sy, -1e9)
        self.ox += sx * self.res
        self.oy += sy * self.res

    @staticmethod
    def _shift(a: np.ndarray, sx: int, sy: int, fill) -> np.ndarray:
        out = np.full_like(a, fill)
        n = a.shape[0]
        if abs(sx) >= n or abs(sy) >= n:
            return out
        src_x = slice(max(sx, 0), n + min(sx, 0)); dst_x = slice(max(-sx, 0), n + min(-sx, 0))
        src_y = slice(max(sy, 0), n + min(sy, 0)); dst_y = slice(max(-sy, 0), n + min(-sy, 0))
        out[dst_x, dst_y] = a[src_x, src_y]
        return out

    def _idx(self, x, y):
        return (np.floor((np.asarray(x) - self.ox) / self.res).astype(int),
                np.floor((np.asarray(y) - self.oy) / self.res).astype(int))

    def update(self, beam: BeamResult) -> None:
        c = self.cfg
        if not beam.valid or beam.sonar_xy is None:
            return
        hx, hy = beam.sonar_xy
        b = math.radians(beam.beam_yaw_deg)
        ux, uy = math.cos(b), math.sin(b)
        # ray clearing (negative evidence) in front of the return
        r_clear = (beam.range_m - c.clear_margin_m) if beam.range_m is not None else beam.max_range_m
        r_clear = min(r_clear, c.grid_size_m)
        if r_clear > c.blanking_m:
            r = np.arange(c.blanking_m, r_clear, self.res * 0.5)
            ix, iy = self._idx(hx + r * ux, hy + r * uy)
            ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
            if ok.any():
                cells = np.unique(ix[ok] * self.n + iy[ok])
                cx_, cy_ = cells // self.n, cells % self.n
                self.counts[cx_, cy_] = np.maximum(self.counts[cx_, cy_] - c.clear_dec, 0.0)
        if beam.hit_xy is not None:
            ix, iy = self._idx(beam.hit_xy[0], beam.hit_xy[1])
            if 0 <= ix < self.n and 0 <= iy < self.n:
                if beam.t - self.last_t[ix, iy] > c.age_s:
                    self.counts[ix, iy] = 0.0
                self.counts[ix, iy] = min(self.counts[ix, iy] + c.hit_inc, c.max_count)
                self.last_t[ix, iy] = beam.t

    def keep_blind_cells(self, t: float, x: float, y: float, yaw_deg: float) -> None:
        """Confirmed cells near the hull that are OUTSIDE the scanned sector (beside / behind the vehicle)
        can neither be refreshed nor cleared by the sonar: keep them alive instead of letting them age out
        (they protect the tail during in-place turns and back-offs)."""
        c = self.cfg
        r = c.keep_near_m
        i0, j0 = self._idx(x - r, y - r)
        i1, j1 = self._idx(x + r, y + r)
        i0, j0 = max(int(i0), 0), max(int(j0), 0)
        i1, j1 = min(int(i1) + 1, self.n), min(int(j1) + 1, self.n)
        if i0 >= i1 or j0 >= j1:
            return
        sub = self.counts[i0:i1, j0:j1]
        live = (sub >= c.min_hits) & ((t - self.last_t[i0:i1, j0:j1]) <= c.age_s)
        if not live.any():
            return
        ii, jj = np.nonzero(live)
        cx = self.ox + (ii + i0 + 0.5) * self.res - x
        cy = self.oy + (jj + j0 + 0.5) * self.res - y
        rel = (np.degrees(np.arctan2(cy, cx)) - yaw_deg + 180.0) % 360.0 - 180.0
        blind = (np.abs(rel) > c.fov_half_deg - 5.0) & (np.hypot(cx, cy) <= r)
        if blind.any():
            self.last_t[ii[blind] + i0, jj[blind] + j0] = t

    def _mask(self, t: float, min_count: float) -> np.ndarray:
        return (self.counts >= min_count) & ((t - self.last_t) <= self.cfg.age_s)

    def occupied_points(self, t: float) -> np.ndarray:
        """World (x, y) centres of confirmed cells: count >= min_hits and not aged out. Shape (M, 2)."""
        ix, iy = np.nonzero(self._mask(t, self.cfg.min_hits))
        return np.column_stack((self.ox + (ix + 0.5) * self.res, self.oy + (iy + 0.5) * self.res))

    def candidate_points(self, t: float) -> np.ndarray:
        ix, iy = np.nonzero(self._mask(t, 1e-3) & (self.counts < self.cfg.min_hits))
        return np.column_stack((self.ox + (ix + 0.5) * self.res, self.oy + (iy + 0.5) * self.res))

    def image(self, t: float) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """For display: counts with aged cells zeroed, and (xmin, xmax, ymin, ymax)."""
        img = np.where((t - self.last_t) <= self.cfg.age_s, self.counts, 0.0)
        return img, (self.ox, self.ox + self.n * self.res, self.oy, self.oy + self.n * self.res)


# --------------------------------------------------------------------------------------------
# Convenience bundle: everything the backend needs on the sonar side
# --------------------------------------------------------------------------------------------
class SonarObstacleMapper:
    """Feed poses (20 Hz is enough) and pings (any rate, own thread OK if you hold a lock)."""

    def __init__(self, cfg: Optional[DetectionConfig] = None):
        self.cfg = cfg or DetectionConfig()
        self.detector = PingDetector(self.cfg)
        self.poses = PoseHistory(max_gap_s=self.cfg.pose_max_gap_s)
        self.grid: Optional[OccupancyGrid] = None
        self.recent_hits: Deque[Tuple[float, float, float]] = deque(maxlen=600)
        self.last_beam: Optional[BeamResult] = None
        self.n_pings = 0
        self.n_hits = 0

    def add_pose(self, t: float, x: float, y: float, yaw_deg: float) -> None:
        self.poses.add(t, x, y, yaw_deg)
        if self.grid is None:
            self.grid = OccupancyGrid(self.cfg, x, y)
        else:
            self.grid.recenter(x, y)
            self.grid.keep_blind_cells(t, x, y, yaw_deg)

    def on_ping(self, ping: SonarPing) -> BeamResult:
        pose = self.poses.at(ping.t)
        beam = self.detector.process(ping, pose)
        self.n_pings += 1
        if self.grid is not None and pose is not None:
            self.grid.update(beam)
        if beam.hit_xy is not None:
            self.n_hits += 1
            self.recent_hits.append((ping.t, beam.hit_xy[0], beam.hit_xy[1]))
        self.last_beam = beam
        return beam

    def obstacles(self, t: float) -> np.ndarray:
        if self.grid is None:
            return np.zeros((0, 2))
        return self.grid.occupied_points(t)
