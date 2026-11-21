"""
viz.py - top-down visualisation (matplotlib).

  SimView(sim, live=True)   live window: world, vehicle, true + DR track, sweeping beam, raw hits,
                            occupancy grid, STOP/SLOW zones, side-step plan, mode/state text,
                            current ping A-scan with the adaptive threshold, speed/distance strip
  SimView.save(path)        same view to PNG (headless snapshots)
  summary_figure(sim, path) post-run plots for the report
"""
from __future__ import annotations

import math
from typing import Optional

import matplotlib
import numpy as np


def _use_backend(live: bool) -> None:
    if not live:
        matplotlib.use("Agg", force=True)


STATE_COLORS = {"TRACK": "#2b8cbe", "BRAKE": "#e31a1c", "HOLD": "#fb6a4a", "SCAN": "#fd8d3c", "BACKOFF": "#6a3d9a",
                "TURN_OUT": "#33a02c", "DRIVE_OUT": "#33a02c", "TURN_BACK": "#b2df8a", "PASS": "#1b9e77",
                "VFH": "#e7298a", "OFF": "#999999", "WARN": "#bbbb00", "IDLE": "#cccccc"}


def _zone_polygon(x, y, yaw, sonar_x, d0, d1, cone, corridor, collide_half):
    xs = np.linspace(max(d0, 0.0), d1, 30)
    tanc = math.tan(math.radians(cone))
    hw = np.minimum(corridor, np.maximum(collide_half, xs * tanc))
    bx = np.concatenate((xs, xs[::-1])) + sonar_x
    by = np.concatenate((hw, -hw[::-1]))
    ps = math.radians(yaw)
    return np.column_stack((x + bx * math.cos(ps) - by * math.sin(ps), y + bx * math.sin(ps) + by * math.cos(ps)))


def _capsule(x, y, yaw, nose, tail, r, n=10):
    ps = math.radians(yaw)
    a = np.linspace(-math.pi / 2, math.pi / 2, n)
    front = np.column_stack((nose - r + r * np.cos(a), r * np.sin(a)))
    back = np.column_stack((-tail + r - r * np.cos(a), -r * np.sin(a)))
    pts = np.vstack((front, back))
    c, s = math.cos(ps), math.sin(ps)
    return np.column_stack((x + pts[:, 0] * c - pts[:, 1] * s, y + pts[:, 0] * s + pts[:, 1] * c))


def draw_world(ax, world, t):
    from matplotlib.patches import Circle
    for x1, y1, x2, y2 in world.walls:
        ax.plot([x1, x2], [y1, y2], color="k", lw=3, solid_capstyle="butt", zorder=3)
    circ, _ = world.circles_at(t)
    for cx, cy, r in circ:
        ax.add_patch(Circle((cx, cy), r, fc="0.45", ec="k", zorder=3))


class SimView:
    def __init__(self, sim, live: bool = True, follow: bool = False):
        _use_backend(live)
        import matplotlib.pyplot as plt
        self.plt = plt
        self.sim = sim
        self.live = live
        self.follow = follow
        if live:
            plt.ion()
        self.fig = plt.figure(figsize=(14, 8.2))
        gs = self.fig.add_gridspec(3, 2, width_ratios=[2.1, 1], height_ratios=[1, 1, 1])
        self.ax = self.fig.add_subplot(gs[:, 0])
        self.ax_scan = self.fig.add_subplot(gs[0, 1])
        self.ax_ts = self.fig.add_subplot(gs[1, 1])
        self.ax_txt = self.fig.add_subplot(gs[2, 1])
        self.closed = False
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _on_close(self, _):
        self.closed = True

    def draw(self) -> None:
        sim = self.sim
        ax = self.ax
        ax.clear()
        t = sim.t
        v = sim.vehicle
        cfg_a, cfg_d = sim.cfg["avoidance"], sim.cfg["detection"]
        g = sim.mapper.grid
        if g is not None:
            img, ext = g.image(t)
            ax.imshow(np.ma.masked_where(img.T <= 0, img.T), origin="lower", extent=ext, cmap="Oranges",
                      vmin=0, vmax=cfg_d.max_count, alpha=0.55, zorder=1, interpolation="nearest")
            occ = g.occupied_points(t)
            if len(occ):
                ax.scatter(occ[:, 0], occ[:, 1], s=9, marker="s", c="r", zorder=4, label="confirmed cells")
        hits = [h for h in sim.mapper.recent_hits if t - h[0] < 3.0]
        if hits:
            h = np.asarray(hits)
            ax.scatter(h[:, 1], h[:, 2], s=6, c="orange", zorder=5, label="raw hits (3 s)")
        draw_world(ax, sim.world, t)
        L = sim.log
        if L["t"]:
            ax.plot(L["x"], L["y"], color="#1f78b4", lw=1.6, label="true track", zorder=6)
            ax.plot(L["ex"], L["ey"], color="#a6cee3", lw=1.0, ls="--", label="DR track", zorder=6)
        mo = sim.mo
        if mo is not None and mo.track is not None and mo.goal is not None:
            ax.plot([mo.track[0], mo.goal[0]], [mo.track[1], mo.goal[1]], ":", color="0.3", lw=1)
            ax.plot(*mo.goal, "kx", ms=9, mew=2, zorder=7)
        cmd = sim.cmd
        ex, ey, eyaw = sim.nav.x, sim.nav.y, sim.nav.yaw
        if cmd is not None:
            sp = _zone_polygon(ex, ey, eyaw, cfg_a.sonar_x_m, 0, cmd.stop_dist_m, cfg_a.cone_half_deg,
                               cfg_a.corridor_half_m, cfg_a.collide_half_m)
            sl = _zone_polygon(ex, ey, eyaw, cfg_a.sonar_x_m, cmd.stop_dist_m, cmd.slow_dist_m, cfg_a.cone_half_deg,
                               cfg_a.corridor_half_m, cfg_a.collide_half_m)
            zc = {"STOP": "red", "SLOW": "gold"}.get(cmd.zone.rstrip("*"), "limegreen")
            ax.fill(sp[:, 0], sp[:, 1], color="red", alpha=0.18 if zc != "red" else 0.40, zorder=2, lw=0)
            ax.fill(sl[:, 0], sl[:, 1], color="gold", alpha=0.18, zorder=2, lw=0)
            if cmd.plan and "c_target" in cmd.plan:
                p = cmd.plan
                b = math.radians(p["bearing"]); ux, uy = math.cos(b), math.sin(b)
                o = p["origin"]
                ox, oy = o[0] - p["c_target"] * uy, o[1] + p["c_target"] * ux
                aa = np.array([0.0, p["a_pass"]])
                ax.plot(ox + aa * ux, oy + aa * uy, color="green", lw=1, ls="-.", label="side-step line")
        # sonar sector and beam
        hx, hy = v.point(sim.cfg["sonar_sim"].sonar_x_m)
        sc = sim.cfg["sonar_sim"]
        if sc.sector_half_deg < 180:
            for s_ in (-1, 1):
                a = math.radians(v.s.yaw + s_ * sc.sector_half_deg)
                ax.plot([hx, hx + sim.sonar.range_m * math.cos(a)], [hy, hy + sim.sonar.range_m * math.sin(a)],
                        color="g", lw=0.5, alpha=0.4)
        if sim.last_beam is not None:
            b = sim.last_beam
            a = math.radians(v.s.yaw + b.rel_bearing_deg)
            r = b.range_m if b.range_m is not None else b.max_range_m
            ax.plot([hx, hx + r * math.cos(a)], [hy, hy + r * math.sin(a)], color="lime", lw=2, zorder=8)
        cap = _capsule(v.s.x, v.s.y, v.s.yaw, v.p.nose_m, v.p.tail_m, max(v.p.radius_m, 0.1))
        ax.fill(cap[:, 0], cap[:, 1], color="yellow", ec="k", zorder=9)
        ax.set_aspect("equal")
        bnd = sim.world.bounds
        if self.follow or not bnd:
            ax.set_xlim(v.s.x - 10, v.s.x + 10); ax.set_ylim(v.s.y - 7, v.s.y + 7)
        else:
            ax.set_xlim(bnd[0], bnd[1]); ax.set_ylim(bnd[2], bnd[3])
        ax.grid(alpha=0.3)
        ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
        ax.set_title(f"{sim.scen['name']}  |  mode {sim.mode}{'/' + sim.stage if sim.mode == 'AVOID' else ''}"
                     f"  |  t = {t:5.1f} s")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.7)
        self._draw_scan()
        self._draw_ts()
        self._draw_text()

    def _draw_scan(self):
        ax = self.ax_scan
        ax.clear()
        sim = self.sim
        p, det = sim.last_ping, sim.mapper.detector
        if p is None:
            return
        r = (np.arange(p.n) + 0.5) * p.range_m / p.n
        ax.plot(r, p.samples, color="0.6", lw=0.5, label="raw")
        if det.last_filtered is not None and len(det.last_filtered) == p.n:
            ax.plot(r, det.last_filtered, color="b", lw=0.8, label="median")
            ax.plot(r, det.last_threshold, color="r", lw=0.8, label="threshold")
        b = sim.last_beam
        if b is not None and b.range_m is not None:
            ax.axvline(b.range_m, color="g", lw=1.5, label="first return")
        ax.axvspan(0, sim.cfg["detection"].blanking_m, color="0.85")
        ax.set_ylim(0, 260); ax.set_xlim(0, p.range_m)
        ax.set_title(f"Ping angle {p.angle_grad} grad (rel {b.rel_bearing_deg:+.0f} deg)" if b else "Ping", fontsize=9)
        ax.set_xlabel("range [m]", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        ax.tick_params(labelsize=7)

    def _draw_ts(self):
        ax = self.ax_ts
        ax.clear()
        L = self.sim.log
        if not L["t"]:
            return
        t = np.asarray(L["t"])
        ax.plot(t, L["u"], label="speed", color="b")
        ax.plot(t, L["speed_ref"], label="speed ref", color="b", ls=":", lw=0.8)
        ax.set_ylabel("m/s", fontsize=8); ax.set_ylim(-0.2, 0.45)
        a2 = ax.twinx()
        a2.plot(t, L["d_obs"], color="k", lw=0.8, label="obstacle dist")
        a2.plot(t, L["stop_d"], color="r", lw=0.6, ls="--")
        a2.plot(t, L["slow_d"], color="orange", lw=0.6, ls="--")
        a2.set_ylim(0, 12); a2.set_ylabel("m", fontsize=8)
        ax.tick_params(labelsize=7); a2.tick_params(labelsize=7)
        ax.set_xlim(max(0, t[-1] - 60), t[-1] + 1)
        ax.legend(fontsize=6, loc="upper left")

    def _draw_text(self):
        ax = self.ax_txt
        ax.clear(); ax.axis("off")
        sim = self.sim
        c, mo = sim.cmd, sim.mo
        if c is None:
            return
        d = "-" if c.obstacle_dist_m is None else f"{c.obstacle_dist_m:.2f} m"
        zc = {"STOP": "red", "SLOW": "darkorange"}.get(c.zone.rstrip("*"), "green")
        lines = [
            f"MODE   {sim.mode}" + (f" / {sim.stage}" if sim.mode == "AVOID" else ""),
            f"STATE  {c.state}",
            f"ZONE   {c.zone}   obstacle {d}",
            f"stop {c.stop_dist_m:.2f} m   slow {c.slow_dist_m:.2f} m",
            f"speed {sim.vehicle.s.u:+.2f} (est {sim.nav.u:+.2f}) ref {sim.speed_ref:+.2f} m/s",
            f"T1 {sim.t1}  T2 {sim.t2}  T3 {sim.t3} (untouched)  owner {sim.owner}",
            f"mission leg {mo.leg + 1} {mo.phase}   DVL {'OK' if sim.nav.dvl_ok else 'DROPOUT'}",
            f"collisions {sim.collisions}  min clear {sim.min_clear:.2f} m  false stops {sim.false_stops}",
            f"{c.message}",
        ]
        for i, s in enumerate(lines):
            ax.text(0.0, 1.0 - i * 0.11, s, family="monospace", fontsize=9, va="top",
                    color=zc if i == 2 else "k", transform=ax.transAxes)

    def update(self, sim=None) -> bool:
        if self.closed:
            return False
        self.draw()
        if self.live:
            self.plt.pause(0.001)
        return True

    def save(self, path: str) -> None:
        self.draw()
        self.fig.savefig(path, dpi=90, bbox_inches="tight")

    def close(self):
        self.plt.close(self.fig)


def summary_figure(sim, path: str) -> None:
    import matplotlib.pyplot as plt
    L = sim.log
    t = np.asarray(L["t"])
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[:, 0])
    g = sim.mapper.grid
    if g is not None:
        img, ext = g.image(sim.t)
        ax.imshow(np.ma.masked_where(img.T <= 0, img.T), origin="lower", extent=ext, cmap="Oranges",
                  alpha=0.5, zorder=1, interpolation="nearest")
    if sim.all_hits:
        h = np.asarray(sim.all_hits)
        ax.scatter(h[:, 0], h[:, 1], s=1.5, c="orange", alpha=0.35, zorder=2, label="all sonar hits")
    draw_world(ax, sim.world, sim.t)
    states = L["state"]
    x, y = np.asarray(L["x"]), np.asarray(L["y"])
    for st in sorted(set(states)):
        m = np.array([s == st for s in states])
        ax.scatter(x[m], y[m], s=4, c=STATE_COLORS.get(st, "k"), label=st, zorder=5)
    ax.plot(L["ex"], L["ey"], color="0.4", lw=0.7, ls="--", label="DR estimate", zorder=4)
    v = sim.vehicle
    cap = _capsule(v.s.x, v.s.y, v.s.yaw, v.p.nose_m, v.p.tail_m, 0.1)
    ax.fill(cap[:, 0], cap[:, 1], color="yellow", ec="k", zorder=6)
    for te, e in sim.events:
        if e in ("STOP_TRIGGER", "BACKOFF", "WP_BLOCKED", "GIVE_UP", "COLLISION") or e.startswith("SIDESTEP"):
            i = min(np.searchsorted(t, te), len(t) - 1)
            ax.annotate(e.replace("_TRIGGER", "").replace("SIDESTEP_", "STEP "), (x[i], y[i]), fontsize=7,
                        xytext=(4, 6), textcoords="offset points", zorder=7,
                        color="red" if e == "COLLISION" else "k")
    bnd = sim.world.bounds
    if bnd:
        ax.set_xlim(bnd[0], bnd[1]); ax.set_ylim(bnd[2], bnd[3])
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    r = sim.result()
    ax.set_title(f"{sim.scen['name']}  mode {sim.mode}{'/' + sim.stage if sim.mode == 'AVOID' else ''}  "
                 f"v={sim.cruise} m/s  seed {sim.seed}\n{r['end']}, t={r['time_s']} s, collisions {r['collisions']}, "
                 f"min clearance {r['min_clearance_m']} m, false stops {r['false_stops']}", fontsize=10)
    ax.legend(fontsize=7, loc="best", markerscale=3, ncol=2)
    a1 = fig.add_subplot(gs[0, 1])
    a1.plot(t, L["u"], label="true speed"); a1.plot(t, L["u_est"], lw=0.7, label="DR speed")
    a1.plot(t, L["speed_ref"], ls=":", label="speed ref")
    dvl_bad = ~np.asarray(L["dvl_ok"], bool)
    if dvl_bad.any():
        a1.fill_between(t, -0.2, 0.45, where=dvl_bad, color="0.8", alpha=0.6, label="DVL dropout")
    a1.set_ylim(-0.2, 0.45); a1.set_ylabel("m/s"); a1.legend(fontsize=7, ncol=2); a1.grid(alpha=0.3)
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a2.plot(t, L["d_obs"], "k", lw=0.8, label="sonar obstacle dist (cone)")
    a2.plot(t, L["stop_d"], "r--", lw=0.8, label="STOP dist")
    a2.plot(t, L["slow_d"], "--", color="orange", lw=0.8, label="SLOW dist")
    a2.plot(t, L["clear"], "g", lw=0.8, label="true hull clearance")
    a2.set_ylim(0, 10); a2.set_ylabel("m"); a2.legend(fontsize=7); a2.grid(alpha=0.3)
    a3 = fig.add_subplot(gs[2, 1], sharex=a1)
    a3.plot(t, L["t1"], label="T1"); a3.plot(t, L["t2"], label="T2", lw=0.7)
    a3.set_ylabel("us"); a3.set_xlabel("t [s]"); a3.grid(alpha=0.3)
    ys = 1640
    prev = None
    for i, st in enumerate(states):
        if st != prev:
            j = i
            while j < len(states) and states[j] == st:
                j += 1
            a3.axvspan(t[i], t[min(j, len(t) - 1)], ymin=0.9, ymax=1.0, color=STATE_COLORS.get(st, "k"), lw=0)
            prev = st
    a3.set_ylim(1300, 1700)
    a3.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)
