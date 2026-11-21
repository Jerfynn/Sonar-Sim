"""
scenarios.py - all test scenarios. Worlds are exported to worlds/*.json (python scenarios.py).

Each scenario: world dict, start pose (x, y, yaw ENU deg), mission legs, cruise speed, default
avoidance mode/stage, end condition ("mission" or "stopped"), config overrides.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from controllers import Leg, Mission
from world import rect_walls

HERE = os.path.dirname(os.path.abspath(__file__))

# Tank profile: shorter distances for a 10 x 6 m tank (open water defaults are too long)
TANK_AVOID = {"avoidance": {"stop_base_m": 1.0, "react_time_s": 1.5, "stop_min_m": 1.2, "slow_margin_m": 1.5,
                            "goal_margin_m": 1.0, "backoff_trigger_m": 0.8, "swing_radius_m": 1.05,
                            "skip_goal_within_m": 2.5},
              "sonar_sim": {"range_m": 10.0}}

NOISY = {"sonar_sim": {"noise_sigma": 9.0, "spike_rate": 0.4, "clutter_prob": 0.06, "ghost_prob": 0.6,
                       "echo_speckle_k": 2.5},
         "vehicle": {"dvl_dropout_mean_interval_s": 6.0, "dvl_noise_mps": 0.012, "ahrs_noise_deg": 0.5,
                     "current_mps": [0.0, 0.02]}}


def _w(name: str, walls=None, circles=None, moving=None, bounds=None, alt=1.5) -> dict:
    return {"name": name, "walls": walls or [], "circles": circles or [], "moving": moving or [],
            "reflectivity": {"wall": 1.0, "circle": 0.8, "moving": 0.6},
            "seabed_altitude_m": alt, "bounds": bounds}


WORLDS: Dict[str, dict] = {
    "open_wall": _w("open_wall", walls=[[15, -8, 15, 8]], bounds=[-2, 18, -8, 8]),
    "pillar_left": _w("pillar_left", circles=[[10, -0.25, 0.4]], bounds=[-2, 22, -6, 6]),
    "pillar_right": _w("pillar_right", circles=[[10, 0.25, 0.4]], bounds=[-2, 22, -6, 6]),
    "gap_wide": _w("gap_wide", circles=[[10, 1.35, 0.35], [10, -1.35, 0.35]], bounds=[-2, 22, -6, 6]),
    "gap_narrow": _w("gap_narrow", circles=[[10, 0.75, 0.35], [10, -0.75, 0.35]], bounds=[-2, 22, -6, 6]),
    "u_trap": _w("u_trap", walls=[[12, -1.5, 12, 1.5], [8, 1.5, 12, 1.5], [8, -1.5, 12, -1.5]],
                 bounds=[-2, 24, -7, 7]),
    "tank_10x6": _w("tank_10x6", walls=rect_walls(0, 0, 10, 6), bounds=[-0.5, 10.5, -0.5, 6.5], alt=1.0),
    "open_water": _w("open_water", bounds=[-2, 32, -6, 6], alt=1.0),
    "noisy_pillar": _w("noisy_pillar", circles=[[12, 0.2, 0.4]], bounds=[-2, 26, -6, 6], alt=1.0),
    "moving_crossing": _w("moving_crossing",
                          moving=[{"r": 0.4, "path": [[11, -7], [11, 7]], "speed": 0.12, "phase": 0.21}],
                          bounds=[-2, 24, -8, 8]),
}


def _scen(name, world, legs, start=(0.0, 0.0, 0.0), cruise=0.2, mode="AVOID", stage="SIDESTEP",
          end="mission", timeout=300.0, overrides=None, desc="") -> dict:
    return dict(name=name, world=WORLDS[world], world_name=world, legs=legs, start=start, cruise=cruise,
                mode=mode, stage=stage, end=end, timeout=timeout, overrides=overrides or {}, desc=desc)


def all_scenarios() -> Dict[str, dict]:
    S = {}
    for v in (0.1, 0.2, 0.3):
        S[f"wall_{v:.1f}"] = _scen(f"wall_{v:.1f}", "open_wall", [Leg("WP", 25.0, 0.0)], cruise=v, mode="STOP",
                                   end="stopped", timeout=320.0,
                                   desc=f"Wall straight ahead at {v} m/s - stopping distance")
    S["wall_0.3_noslow"] = _scen("wall_0.3_noslow", "open_wall", [Leg("WP", 25.0, 0.0)], cruise=0.3, mode="STOP",
                                 end="stopped", overrides={"avoidance": {"slow_margin_m": 0.0}},
                                 desc="Wall at 0.3 m/s with the SLOW zone disabled - pure speed-dependent STOP")
    S["pillar_left"] = _scen("pillar_left", "pillar_left", [Leg("WP", 20.0, 0.0)],
                             desc="Pillar slightly right of the track -> side-step LEFT, rejoin")
    S["pillar_right"] = _scen("pillar_right", "pillar_right", [Leg("WP", 20.0, 0.0)],
                              desc="Pillar slightly left of the track -> side-step RIGHT, rejoin")
    S["gap_wide"] = _scen("gap_wide", "gap_wide", [Leg("WP", 20.0, 0.0)],
                          desc="Two pillars, 2.0 m gap -> pass straight through")
    S["gap_narrow"] = _scen("gap_narrow", "gap_narrow", [Leg("WP", 20.0, 0.0)],
                            desc="Two pillars, 0.8 m gap -> too narrow, go around")
    S["u_trap"] = _scen("u_trap", "u_trap", [Leg("WP", 22.0, 0.0)], timeout=400.0,
                        desc="U-shape trap -> stop, back off, escape around, rejoin")
    S["square_tank"] = _scen("square_tank", "tank_10x6", Mission.square(2.5, 1.5, 0.0, 4.5, turn=+1),
                             start=(2.5, 1.5, 0.0), overrides=TANK_AVOID, timeout=400.0,
                             desc="4.5 m square (left turns) in a 10 x 6 m tank - legs end too close to the wall")
    S["square_tank_close"] = _scen("square_tank_close", "tank_10x6", Mission.square(2.5, 1.0, 0.0, 4.5, turn=+1),
                                   start=(2.5, 1.0, 0.0), overrides=TANK_AVOID, timeout=400.0,
                                   desc="Same square started 1.0 m from the wall: the first corner turn would "
                                        "swing the tail to ~0.1 m of it -> turn refused, pattern aborted safely")
    S["noisy_open"] = _scen("noisy_open", "open_water", [Leg("WP", 30.0, 0.0)],
                            overrides=_merge(NOISY, {"sonar_sim": {"sector_half_deg": 180.0, "range_m": 8.0}}),
                            desc="Open water, heavy noise, 360 deg scan (tether behind), DVL dropouts -> no false stops")
    S["noisy_pillar"] = _scen("noisy_pillar", "noisy_pillar", [Leg("WP", 24.0, 0.0)], overrides=NOISY,
                              desc="Heavy noise + DVL dropouts with a pillar on the track -> no miss")
    S["moving_crossing"] = _scen("moving_crossing", "moving_crossing", [Leg("WP", 22.0, 0.0)], mode="STOP",
                                 desc="Diver/ROV crossing the track -> stop, wait, resume when clear")
    return S


def _merge(a: dict, b: dict) -> dict:
    from config import deep_merge
    return deep_merge(a, b)


def export_worlds(folder: str = os.path.join(HERE, "worlds")) -> List[str]:
    os.makedirs(folder, exist_ok=True)
    out = []
    for k, w in WORLDS.items():
        p = os.path.join(folder, f"{k}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(w, f, indent=2)
        out.append(p)
    return out


if __name__ == "__main__":
    for p in export_worlds():
        print("wrote", p)
