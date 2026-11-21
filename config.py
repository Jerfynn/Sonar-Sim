"""
config.py - ONE place for every tunable parameter.

All parameters live in the dataclasses of the modules (so detection.py / avoidance.py stay
standalone for the Jetson).  This file merges them into one nested dict:

    {"vehicle": {...}, "sonar_sim": {...}, "detection": {...},
     "avoidance": {...}, "control": {...}, "sim": {...}}

  python config.py                  -> writes default_config.json (every parameter, defaults)
  run_sim.py --config my.json       -> partial overrides, e.g. {"avoidance": {"stop_base_m": 1.5}}

Shared values (sonar mount, Ping360 angle convention, thruster deadbands) are entered once in
"sonar_sim" / "vehicle" and copied to the other sections by sync().
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, asdict, fields, is_dataclass
from typing import Any, Dict, Optional

from vehicle_model import VehicleParams
from ping360_sim import Ping360SimConfig
from detection import DetectionConfig
from avoidance import AvoidanceConfig
from controllers import ControlConfig


@dataclass
class SimConfig:
    dt_phys_s: float = 0.01
    dt_ctrl_s: float = 0.05            # 20 Hz like the backend
    timeout_s: float = 300.0
    stopped_hold_s: float = 8.0
    start_jitter_m: float = 0.15       # Monte Carlo randomisation
    start_jitter_deg: float = 2.0
    obstacle_jitter_m: float = 0.15
    false_stop_extra_m: float = 1.0    # STOP is "false" if truth is clear to slow_dist + this
    late_frac: float = 0.5             # "late" if truth gets inside late_frac * stop_dist unmanaged


SECTIONS = {
    "vehicle": VehicleParams,
    "sonar_sim": Ping360SimConfig,
    "detection": DetectionConfig,
    "avoidance": AvoidanceConfig,
    "control": ControlConfig,
    "sim": SimConfig,
}


def _coerce(cls, key: str, value: Any) -> Any:
    for f in fields(cls):
        if f.name == key:
            default = getattr(cls(), key)
            if isinstance(default, tuple) and isinstance(value, list):
                return tuple(value)
            return value
    raise KeyError(f"unknown parameter {cls.__name__}.{key}")


def deep_merge(base: dict, over: Optional[dict]) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def build(overrides: Optional[Dict[str, dict]] = None) -> Dict[str, Any]:
    overrides = overrides or {}
    for sec in overrides:
        if sec not in SECTIONS:
            raise KeyError(f"unknown config section '{sec}' (valid: {list(SECTIONS)})")
    cfgs = {}
    for sec, cls in SECTIONS.items():
        kw = {k: _coerce(cls, k, v) for k, v in overrides.get(sec, {}).items()}
        cfgs[sec] = cls(**kw)
    sync(cfgs, overrides)
    return cfgs


def sync(cfgs: Dict[str, Any], overrides: Optional[dict] = None) -> None:
    """Copy shared values so they are entered once."""
    s, d, a, v, c = cfgs["sonar_sim"], cfgs["detection"], cfgs["avoidance"], cfgs["vehicle"], cfgs["control"]
    ov = overrides or {}
    for k in ("fwd_grad", "angle_dir", "sonar_x_m"):
        if k not in ov.get("detection", {}):
            setattr(d, k, getattr(s, k))
    if "sonar_x_m" not in ov.get("avoidance", {}):
        a.sonar_x_m = s.sonar_x_m
    a.point_margin_m = d.grid_res_m / 2.0
    for k_a, k_v in (("nose_m", "nose_m"), ("tail_m", "tail_m"), ("hull_radius_m", "radius_m")):
        if k_a not in ov.get("avoidance", {}):
            setattr(a, k_a, getattr(v, k_v))
    for k_c, k_v in (("t1_deadband_us", "deadband_us"), ("t2_deadband_us", "deadband_us"),
                     ("t1_dir", "t1_dir"), ("t2_dir", "t2_dir"), ("dvl_x_m", "dvl_x_m")):
        if k_c not in ov.get("control", {}):
            setattr(c, k_c, getattr(v, k_v))


def as_dict(cfgs: Dict[str, Any]) -> Dict[str, dict]:
    return {k: asdict(v) for k, v in cfgs.items()}


def load_json(path: Optional[str]) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    with open("default_config.json", "w", encoding="utf-8") as f:
        json.dump(as_dict(build()), f, indent=2)
    print("wrote default_config.json")
