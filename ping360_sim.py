"""
ping360_sim.py - simulated Blue Robotics Ping360 mechanical scanning sonar.

Output per ping: detection.SonarPing(angle_grad, range_m, samples[uint8 x N], t, sample_period,
gain, transmit_duration) - the same content the brping / native-UDP driver in auv_depth.py gives
(msg.angle, msg.sample_period, msg.data), so detection runs unchanged on real pings.

Modelled:
  * mechanical stepping, gradians (400/rev), sector ping-pong sweep or full circle
  * ping duration = 2R/c + motor/processing overhead  -> sector refresh ~1.5-3 s
  * beam 2 deg (horizontal) sampled with sub-rays and a gaussian beam pattern; 25 deg vertical
    beam gives a diffuse seabed band at altitude / sin(12.5 deg)
  * echo amplitude ~ reflectivity * cos(incidence)^0.7 * spreading/absorption, saturating at 255
  * pulse length from transmit duration, rough-surface tail behind each echo
  * Rayleigh speckle (background) + multiplicative speckle on echoes
  * near-field ringing inside the blanking range
  * occasional false spikes (1-3 samples), small clutter blobs (fish / bubbles)
  * weak multipath "ghost" echoes behind walls
  * tether returns in the rear sector (to be ignored by detection)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from detection import SonarPing, SOUND_SPEED_MPS, GRAD_TO_DEG


@dataclass
class Ping360SimConfig:
    range_m: float = 12.0
    n_samples: int = 1200
    gain: int = 1
    transmit_duration_us: int = 32
    fwd_grad: int = 200
    angle_dir: int = 1
    sonar_x_m: float = 0.80
    sector_half_deg: float = 60.0         # >= 180 -> full circle
    step_grad: int = 2
    ping_overhead_s: float = 0.018
    beam_width_deg: float = 2.0
    sub_rays: int = 7
    vertical_beam_deg: float = 25.0
    # amplitude model
    a0: float = 1800.0
    absorption_db_per_m: float = 0.05
    incidence_exp: float = 0.7
    tail_frac: float = 0.35
    tail_len_m: float = 0.15
    # defects
    noise_sigma: float = 6.0               # Rayleigh sigma of background (mean ~1.25 sigma)
    echo_speckle_k: float = 4.0            # gamma shape for multiplicative speckle (higher = less)
    ringing_amp: float = 200.0
    ringing_len_m: float = 0.15
    spike_rate: float = 0.08               # expected false spikes per ping
    clutter_prob: float = 0.02             # chance of a fish/bubble blob per ping
    ghost_prob: float = 0.35               # chance of a ghost echo when a wall is hit
    ghost_frac: Tuple[float, float] = (0.10, 0.25)
    tether_half_deg: float = 40.0          # rear sector (around 180 deg rel) with tether returns
    tether_range_m: Tuple[float, float] = (0.8, 3.0)
    seabed_amp: float = 35.0


class Ping360Sim:
    def __init__(self, cfg: Ping360SimConfig, world, rng: np.random.Generator):
        self.cfg = cfg
        self.world = world
        self.rng = rng
        self.sample_period = SonarPing.sample_period_for_range(cfg.range_m, cfg.n_samples)
        self.range_m = SonarPing.range_from_sample_period(self.sample_period, cfg.n_samples)
        self.dr = self.range_m / cfg.n_samples
        self.r_axis = (np.arange(cfg.n_samples) + 0.5) * self.dr
        if cfg.sector_half_deg >= 180.0:
            self.full = True
            self.a_min, self.a_max = 0, 399
        else:
            self.full = False
            h = int(round(cfg.sector_half_deg / GRAD_TO_DEG))
            self.a_min, self.a_max = cfg.fwd_grad - h, cfg.fwd_grad + h
        self.angle = self.a_min
        self.direction = 1
        self.ping_dt = 2 * self.range_m / SOUND_SPEED_MPS + cfg.ping_overhead_s
        self.next_t = 0.0
        sig = cfg.beam_width_deg / 2.3548
        self.sub_off = np.linspace(-1.5 * cfg.beam_width_deg / 2, 1.5 * cfg.beam_width_deg / 2, cfg.sub_rays)
        w = np.exp(-0.5 * (self.sub_off / sig) ** 2)
        self.sub_w = w / w.sum()
        pulse_m = SOUND_SPEED_MPS * cfg.transmit_duration_us * 1e-6 / 2
        self.pulse_sigma = max(pulse_m / 2.0, self.dr)

    @property
    def sector_refresh_s(self) -> float:
        n = (self.a_max - self.a_min) / self.cfg.step_grad
        return n * self.ping_dt

    def rel_deg(self, angle_grad: int) -> float:
        a = -self.cfg.angle_dir * (angle_grad - self.cfg.fwd_grad) * GRAD_TO_DEG
        return (a + 180.0) % 360.0 - 180.0

    def _advance(self) -> None:
        c = self.cfg
        if self.full:
            self.angle = (self.angle + c.step_grad) % 400
            return
        nxt = self.angle + self.direction * c.step_grad
        if nxt > self.a_max or nxt < self.a_min:
            self.direction *= -1
            nxt = self.angle + self.direction * c.step_grad
        self.angle = nxt

    def due(self, t: float) -> bool:
        return t >= self.next_t

    def ping(self, t: float, x: float, y: float, yaw_deg: float) -> SonarPing:
        """Ping at time t with the TRUE vehicle pose (turning point x, y, yaw)."""
        c, rng = self.cfg, self.rng
        ag = int(self.angle % 400)
        rel = self.rel_deg(ag)
        ps = math.radians(yaw_deg)
        hx, hy = x + c.sonar_x_m * math.cos(ps), y + c.sonar_x_m * math.sin(ps)
        n = c.n_samples
        r = self.r_axis
        # background: Rayleigh speckle + near-field ringing
        prof = rng.rayleigh(c.noise_sigma, n)
        prof += c.ringing_amp * np.exp(-r / c.ringing_len_m) * rng.uniform(0.6, 1.0, n)
        echo = np.zeros(n)
        angs = np.radians(yaw_deg + rel + self.sub_off)
        rng_hit, cosi, refl = self.world.raycast(hx, hy, angs, t, self.range_m)
        hit = np.isfinite(rng_hit)
        wall_hit_r = None
        if hit.any():
            for k in np.flatnonzero(hit):
                rk = rng_hit[k]
                amp = (c.a0 * refl[k] * max(cosi[k], 0.05) ** c.incidence_exp * self.sub_w[k] * len(self.sub_w)
                       / max(rk, 0.3) * 10 ** (-c.absorption_db_per_m * rk / 10))
                amp /= len(self.sub_w)
                i0 = int(rk / self.dr)
                lo, hi = max(0, i0 - 8), min(n, i0 + int(5 * c.tail_len_m / self.dr) + 8)
                seg = r[lo:hi]
                g = np.exp(-0.5 * ((seg - rk) / self.pulse_sigma) ** 2)
                tail = np.where(seg > rk, c.tail_frac * np.exp(-(seg - rk) / c.tail_len_m), 0.0)
                echo[lo:hi] += amp * np.maximum(g, tail)
            wall_hit_r = float(np.nanmin(rng_hit[hit]))
            echo *= rng.gamma(c.echo_speckle_k, 1.0 / c.echo_speckle_k, n)
            # multipath ghost behind the first return
            if rng.random() < c.ghost_prob:
                rg = wall_hit_r + float(rng.uniform(0.8, 2.5)) if rng.random() < 0.6 else 2.0 * wall_hit_r
                if rg < self.range_m - 0.1:
                    peak = float(echo.max()) * float(rng.uniform(*c.ghost_frac))
                    echo += peak * np.exp(-0.5 * ((r - rg) / (3 * self.pulse_sigma)) ** 2) * rng.gamma(3, 1 / 3, n)
        prof += echo
        # seabed band from the 25 deg vertical beam
        if self.world.seabed_altitude_m:
            rb = self.world.seabed_altitude_m / math.sin(math.radians(c.vertical_beam_deg / 2))
            band = c.seabed_amp * np.clip((r - rb) / 1.5, 0, 1) * np.exp(-np.clip(r - rb - 1.5, 0, None) / 4.0)
            if wall_hit_r is not None:
                band = np.where(r < wall_hit_r, band, band * 0.3)
            prof += band * rng.rayleigh(0.8, n)
        # tether in the rear sector
        if abs(abs(rel) - 180.0) <= c.tether_half_deg:
            for _ in range(int(rng.integers(1, 3))):
                rt = float(rng.uniform(*c.tether_range_m))
                prof += float(rng.uniform(120, 220)) * np.exp(-0.5 * ((r - rt) / 0.05) ** 2)
        # false spikes
        for _ in range(int(rng.poisson(c.spike_rate))):
            i = int(rng.integers(0, n - 3))
            w = int(rng.integers(1, 4))
            prof[i:i + w] += float(rng.uniform(150, 255))
        # clutter blob (fish / bubbles)
        if rng.random() < c.clutter_prob:
            rc = float(rng.uniform(0.8, self.range_m))
            wc = float(rng.uniform(0.03, 0.08))
            prof += float(rng.uniform(80, 200)) * np.exp(-0.5 * ((r - rc) / wc) ** 2)
        samples = np.clip(prof, 0, 255).astype(np.uint8)
        p = SonarPing(ag, self.range_m, samples, t, self.sample_period, c.gain, c.transmit_duration_us)
        self.next_t = t + self.ping_dt * float(rng.uniform(0.95, 1.08))
        self._advance()
        return p
