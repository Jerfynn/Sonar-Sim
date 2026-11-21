"""
world.py - 2-D world for the simulator: wall segments, circular obstacles, moving obstacles.

JSON format (worlds/*.json):
{
  "name": "tank_10x6",
  "walls": [[x1, y1, x2, y2], ...],          # line segments (tank walls, U-shapes ...)
  "circles": [[x, y, r], ...],                # pillars / posts
  "moving": [{"r": 0.4, "path": [[x1, y1], [x2, y2]], "speed": 0.15, "phase": 0.0}],
  "reflectivity": {"wall": 1.0, "circle": 0.8, "moving": 0.6},
  "seabed_altitude_m": 1.5,                   # optional diffuse bottom return (25 deg vertical beam)
  "bounds": [xmin, xmax, ymin, ymax]          # optional, for plotting
}
Coordinates: ENU metres (x East, y North).
"""
from __future__ import annotations

import copy
import json
import math
from typing import List, Optional, Tuple

import numpy as np


class World:
    def __init__(self, d: dict):
        self.d = copy.deepcopy(d)
        self.name = d.get("name", "world")
        self.walls = np.asarray(d.get("walls", []), dtype=float).reshape(-1, 4)
        self.circles = np.asarray(d.get("circles", []), dtype=float).reshape(-1, 3)
        self.moving = list(d.get("moving", []))
        refl = d.get("reflectivity", {})
        self.refl_wall = float(refl.get("wall", 1.0))
        self.refl_circle = float(refl.get("circle", 0.8))
        self.refl_moving = float(refl.get("moving", 0.6))
        self.seabed_altitude_m = d.get("seabed_altitude_m")
        self.bounds = d.get("bounds")

    # ---------------------------------------------------------------- io
    @classmethod
    def load(cls, path: str) -> "World":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.d, f, indent=2)

    def empty_copy(self) -> "World":
        """Same world without any obstacle (baseline for 'time lost')."""
        d = copy.deepcopy(self.d)
        d["walls"], d["circles"], d["moving"] = [], [], []
        return World(d)

    def jittered(self, rng: np.random.Generator, sigma: float) -> "World":
        d = copy.deepcopy(self.d)
        if sigma > 0:
            d["circles"] = [[c[0] + rng.normal(0, sigma), c[1] + rng.normal(0, sigma), c[2]] for c in d.get("circles", [])]
        return World(d)

    # ---------------------------------------------------------------- obstacles
    def moving_at(self, t: float) -> np.ndarray:
        out = []
        for m in self.moving:
            p = np.asarray(m["path"], dtype=float)
            seg = np.hypot(*(np.diff(p, axis=0).T))
            L = float(seg.sum())
            if L <= 0:
                out.append([p[0, 0], p[0, 1], m["r"]])
                continue
            s = (m.get("speed", 0.1) * t + m.get("phase", 0.0) * L) % (2 * L)
            s = s if s <= L else 2 * L - s          # back and forth
            acc = 0.0
            for i, ls in enumerate(seg):
                if s <= acc + ls or i == len(seg) - 1:
                    a = (s - acc) / max(ls, 1e-9)
                    q = p[i] + a * (p[i + 1] - p[i])
                    out.append([q[0], q[1], m["r"]])
                    break
                acc += ls
        return np.asarray(out, dtype=float).reshape(-1, 3)

    def circles_at(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        mv = self.moving_at(t)
        circ = np.vstack((self.circles, mv)) if len(mv) else self.circles
        refl = np.concatenate((np.full(len(self.circles), self.refl_circle), np.full(len(mv), self.refl_moving)))
        return circ, refl

    # ---------------------------------------------------------------- ray casting
    def raycast(self, ox: float, oy: float, ang: np.ndarray, t: float, max_r: float
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rays from (ox, oy) at world angles ang (rad). Returns range (inf = none), |cos(incidence)|,
        reflectivity, each shape (K,)."""
        ang = np.asarray(ang, dtype=float)
        dx, dy = np.cos(ang), np.sin(ang)
        K = len(ang)
        best = np.full(K, np.inf)
        cosi = np.zeros(K)
        refl = np.zeros(K)
        if len(self.walls):
            x1, y1, x2, y2 = (self.walls[:, i][None, :] for i in range(4))
            ex, ey = x2 - x1, y2 - y1
            den = dx[:, None] * ey - dy[:, None] * ex
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = ((x1 - ox) * ey - (y1 - oy) * ex) / den
                uu = ((x1 - ox) * dy[:, None] - (y1 - oy) * dx[:, None]) / den
            ok = (np.abs(den) > 1e-12) & (tt > 1e-6) & (uu >= 0) & (uu <= 1)
            tt = np.where(ok, tt, np.inf)
            j = np.argmin(tt, axis=1)
            tmin = tt[np.arange(K), j]
            L = np.hypot(ex, ey)[0, j]
            nx, ny = -ey[0, j] / L, ex[0, j] / L
            ci = np.abs(dx * nx + dy * ny)
            upd = tmin < best
            best = np.where(upd, tmin, best)
            cosi = np.where(upd, ci, cosi)
            refl = np.where(upd, self.refl_wall, refl)
        circ, crefl = self.circles_at(t)
        if len(circ):
            cx, cy, r = circ[:, 0][None, :], circ[:, 1][None, :], circ[:, 2][None, :]
            fx, fy = ox - cx, oy - cy
            b = fx * dx[:, None] + fy * dy[:, None]
            cc = fx * fx + fy * fy - r * r
            disc = b * b - cc
            with np.errstate(invalid="ignore"):
                sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
            tt = -b - sq
            tt = np.where((disc >= 0) & (tt > 1e-6), tt, np.inf)
            j = np.argmin(tt, axis=1)
            tmin = tt[np.arange(K), j]
            hx, hy = ox + tmin * dx, oy + tmin * dy
            with np.errstate(invalid="ignore"):
                nx = (hx - circ[j, 0]) / circ[j, 2]
                ny = (hy - circ[j, 1]) / circ[j, 2]
                ci = np.abs(dx * nx + dy * ny)
            upd = tmin < best
            best = np.where(upd, tmin, best)
            cosi = np.where(upd, np.nan_to_num(ci), cosi)
            refl = np.where(upd, crefl[j], refl)
        best = np.where(best <= max_r, best, np.inf)
        return best, cosi, refl

    # ---------------------------------------------------------------- hull clearance (truth)
    def capsule_clearance(self, p1: Tuple[float, float], p2: Tuple[float, float], radius: float, t: float) -> float:
        """Min distance between a capsule (segment p1-p2, radius) and any obstacle. <0 = collision."""
        best = np.inf
        ax, ay = p1; bx, by = p2
        if len(self.walls):
            for w in self.walls:
                best = min(best, _seg_seg_dist(ax, ay, bx, by, *w))
        circ, _ = self.circles_at(t)
        for cx, cy, r in circ:
            best = min(best, _pt_seg_dist(cx, cy, ax, ay, bx, by) - r)
        return best - radius

    def corridor_distance(self, ox: float, oy: float, yaw_deg: float, half_w: float, t: float,
                          max_r: float = 30.0) -> float:
        """Truth: distance along heading from (ox, oy) to the first obstacle inside a corridor."""
        ps = math.radians(yaw_deg)
        if not len(self.walls) and not len(self.circles) and not self.moving:
            return math.inf
        # rotate the world into the heading frame and test the corridor strip directly
        c, s = math.cos(ps), math.sin(ps)
        best = math.inf
        if len(self.walls):
            x1 = (self.walls[:, 0] - ox) * c + (self.walls[:, 1] - oy) * s
            y1 = -(self.walls[:, 0] - ox) * s + (self.walls[:, 1] - oy) * c
            x2 = (self.walls[:, 2] - ox) * c + (self.walls[:, 3] - oy) * s
            y2 = -(self.walls[:, 2] - ox) * s + (self.walls[:, 3] - oy) * c
            for a, b, cc, d in zip(x1, y1, x2, y2):
                # clip the segment to the strip |y| <= half_w, take the min x >= 0
                ts = [0.0, 1.0]
                dy = d - b
                if abs(dy) < 1e-12:
                    if abs(b) > half_w:
                        continue
                else:
                    t0, t1 = sorted(((-half_w - b) / dy, (half_w - b) / dy))
                    ts = [max(0.0, t0), min(1.0, t1)]
                    if ts[0] > ts[1]:
                        continue
                xs = [a + tt * (cc - a) for tt in ts]
                lo, hi = min(xs), max(xs)
                if hi < 0:
                    continue
                best = min(best, max(lo, 0.0))
        circ, _ = self.circles_at(t)
        for cx, cy, r in circ:
            xc = (cx - ox) * c + (cy - oy) * s
            yc = -(cx - ox) * s + (cy - oy) * c
            dyy = max(0.0, abs(yc) - half_w)
            if dyy < r and xc + r > 0:
                best = min(best, max(xc - math.sqrt(r * r - dyy * dyy), 0.0))
        return best if best <= max_r else math.inf


def _pt_seg_dist(px, py, ax, ay, bx, by) -> float:
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    u = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    return math.hypot(px - (ax + u * vx), py - (ay + u * vy))


def _seg_seg_dist(ax, ay, bx, by, cx, cy, dx, dy) -> float:
    def ccw(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    d1 = ccw(ax, ay, bx, by, cx, cy); d2 = ccw(ax, ay, bx, by, dx, dy)
    d3 = ccw(cx, cy, dx, dy, ax, ay); d4 = ccw(cx, cy, dx, dy, bx, by)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return 0.0
    return min(_pt_seg_dist(ax, ay, cx, cy, dx, dy), _pt_seg_dist(bx, by, cx, cy, dx, dy),
               _pt_seg_dist(cx, cy, ax, ay, bx, by), _pt_seg_dist(dx, dy, ax, ay, bx, by))


def rect_walls(x0: float, y0: float, x1: float, y1: float) -> List[List[float]]:
    return [[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]]
