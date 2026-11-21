# Ping360 sonar obstacle-avoidance simulator

2-D top-down simulator for the torpedo AUV with a Blue Robotics Ping360 scanning sonar, plus the
**detection** and **avoidance** logic that is meant to be dropped into the Jetson backend
(`auv_depth.py`, 20 Hz loop) unchanged.

Python 3.9+, `numpy`, `matplotlib` only.

```
pip install numpy matplotlib
python run_sim.py --list                                   # scenarios
python run_sim.py --scenario pillar_left                   # live window
python run_sim.py --scenario square_tank --mode STOP       # OFF / WARN / STOP / AVOID
python run_sim.py --scenario u_trap --stage VFH --realtime 4
python run_sim.py --scenario noisy_pillar --headless --save out/
python run_sim.py --batch --seeds 10 --out results/        # Monte Carlo, metrics table
python run_sim.py --replay log16_events.csv --fixed-pose 0,0,90   # real pings through detection
python config.py                                           # writes default_config.json (every parameter)
```

## Files

| file | role | imports simulator? |
|---|---|---|
| `detection.py` | `SonarPing`, CFAR first-return detector, `PoseHistory` (pose at ping time), `OccupancyGrid`, `SonarObstacleMapper` | **no** (numpy) |
| `avoidance.py` | zones, OFF/WARN/STOP/AVOID state machine, side-step planner, VFH, side guard, `arbitrate()`, `cap_t1_pwm()` | **no** (numpy) |
| `sonar_avoid_node.py` | thread-safe glue for `auv_depth.py` + GCS telemetry dict | **no** |
| `world.py` | walls / circles / moving obstacles, JSON load/save, ray cast, hull clearance | – |
| `vehicle_model.py` | dynamics fitted to the tank logs, DVL (lever arm, dropouts), AHRS | – |
| `ping360_sim.py` | simulated Ping360 producing driver-shaped pings | – |
| `controllers.py` | DR navigation, profiled heading ref, heading hold (T2), speed hold (T1), missions / square | – |
| `simulation.py` | 100 Hz physics / 20 Hz control / Ping360 loop, truth metrics | – |
| `viz.py` | live view, PNG snapshots, summary figures | – |
| `scenarios.py`, `worlds/*.json` | all scenarios and world files | – |
| `config.py`, `default_config.json` | **one config section** for every parameter | – |
| `replay.py` | logged pings (your `log{n}_events.csv` or JSONL) through `detection.py` | **no simulator** |
| `run_sim.py` | CLI | – |

## Conventions (same as auv_depth.py)

* World ENU metres (x East, y North). **Yaw = ENU degrees, CCW positive** (like `_sq_unit()`), so a
  right turn decreases yaw. Body frame: x forward, y **left**. DVL velocity in the simulator is
  reported as (forward, starboard) like `uw_vx_mps`.
* Reference point = turning point. Sonar head `sonar_x_m = 0.80` ahead of it, DVL 0.38 m behind,
  nose 0.9 m, tail 0.9 m.
* Ping360: `angle_grad` 0..399. `fwd_grad = 200` (centre of your 100..300 sector), `angle_dir = +1`
  (grad increasing = clockwise seen from above = starboard). **Check both once in the tank** with a
  target on the bow and one to starboard — if the starboard target appears on the left, set
  `angle_dir = -1`.
* Thrusters: T1 > 1500 forward, T2 > 1500 nose left (matches your `SimulationWorker`), deadband ±25 µs.
  Avoidance only ever outputs a **speed setpoint (T1)** and a **heading reference (T2)**. It has no T3
  output at all, so it cannot touch depth.

## Vehicle model (vehicle_model.py)

Values from your tank-log fit: surge 0.0018 m/s² per µs beyond the deadband forward, 0.0025 reverse, drag
`dv/dt -= 0.20 v|v|` (→ 0.6 m/s at 1570), yaw 0.22 °/s per µs, lag 0.6 s, clamp 12 °/s, random drift
−0.3…−0.8 °/s with a slow random walk. DVL lateral reading `v − 0.38·r`, noise, Poisson dropouts of 0.5–2.8 s.
AHRS heading noise + bias.

The DR navigator (`controllers.DeadReckoner`) removes the DVL lever arm. During dropouts it propagates the
**surge model with the commanded T1** instead of holding the last speed. A held speed would be wrong exactly when
braking: the speed loop keeps seeing "0.2 m/s" and keeps reversing. Worth copying into the backend
hold-over.

## Simulated Ping360 (ping360_sim.py)

* Sector ping-pong ±60° (or full circle), step 2 grad, ping time `2R/c + 18 ms` → **2.3 s sector
  refresh at 12 m**. `sample_period` computed like the real device (25 ns ticks), 1200 samples.
* Beam 2° × 25°: 7 sub-rays with a gaussian beam pattern (gives realistic edge smear on pillars); the
  25° vertical beam gives a weak diffuse **seabed band** at `altitude / sin 12.5°`.
* Amplitude ∝ reflectivity · cos(incidence)^0.7 / r · absorption, saturates at 255. Pulse length from
  the transmit duration, rough-surface tail.
* Defects: Rayleigh speckle, multiplicative echo speckle, near-field ringing inside the blanking range,
  1–3-sample false spikes, fish/bubble blobs, weak multipath **ghosts behind walls**, **tether returns in
  the rear sector**.
* Output `SonarPing(angle_grad, range_m, samples[uint8], t, sample_period, gain, transmit_duration)` —
  the same content as `msg.angle / msg.sample_period / msg.data`.

## Detection (detection.py)

Per ping:
1. Sector gate: beams with |relative bearing| > `valid_half_sector_deg` (100°) are ignored. This removes
   the tether and the vehicle's own stern.
2. 5-sample median filter. This removes 1–2-sample spikes.
3. Adaptive threshold = max(`abs_min`, `global_alpha` × ping median, `cfar_alpha` × SO-CFAR local noise).
   SO-CFAR (smallest of the leading/lagging windows) keeps the leading edge of a wall detectable.
4. **First real return**: first above-threshold run ≥ `min_blob_m` with peak ≥ `peak_min`. Ghosts behind a
   wall are never used. The leading edge is used as the range, which is the conservative choice.
5. World point from the **DR pose interpolated at the ping timestamp** (`PoseHistory`).
6. `OccupancyGrid` 20 × 20 m, 0.25 m cells, recentred on the vehicle. A hit adds +1. Cells the beam
   passed through without an echo get −0.5 (ray clearing), so moving obstacles disappear. A cell is
   confirmed at ≥ 2 hits, and confirmations age out after 20 s.

## Avoidance (avoidance.py)

**Zones** are a ±30° cone **AND** a ±0.5 m corridor (hull radius 0.1 + 0.4 margin), measured from the sonar head.
Right at the nose the zone is only ±0.3 m (`collide_half_m`) and widens with the cone. The corridor keeps tank walls
*beside* the track from triggering anything.

```
STOP  if d < stop  = clip(1.8 + v·2.0 + v²/(2·0.10), 2.0, 4.0)       # base + reaction + braking
SLOW  if d < stop + 2.5     -> speed cap 0.10 m/s
CLEAR otherwise
```

* **Goal-aware**: if the leg ends at least `goal_margin_m` before the obstacle, only SLOW applies. For
  example, a pattern corner in front of a tank wall: the vehicle may arrive at the corner and turn there.
* **Resume** requires the distance needed at the speed the mission would resume with. This prevents stop/go
  cycling when the stop distance shrinks as the vehicle slows.

| mode | behaviour |
|---|---|
| OFF | zones computed for display only |
| WARN | `WARN_STOP` events, never overrides |
| STOP | SLOW cap, BRAKE → HOLD. **While holding, the mission may still turn in place** unless the nose-swing circle (1.15 m) is occupied. A leg still blocked after `blocked_skip_s` (12 s) is reported as `waypoint_blocked` and the pattern moves on. Back-off if an obstacle is inside 1.3 m. |
| AVOID / SIDESTEP | BRAKE → SCAN (one sector sweep, 2.6 s) → plan → TURN_OUT θ → DRIVE_OUT d → TURN_BACK → PASS on the offset line → hand back to the mission, whose LOS rejoins the original track. If the leg's end is inside or just behind the blocking obstacle, the leg is skipped instead. If no plan is found: back off 1.5 m and rescan (U-shape). Give up and skip the leg after 3 retries or 150 s. |
| AVOID / VFH | polar histogram (5° sectors, 6 m window, enlarged by 0.6 m) steers toward the free sector nearest the mission's LOS heading. Directions outside ±65° count as unknown, so it turns first and looks before driving. If trapped once, it falls back to the side-step planner. |

**Side-step planner**: obstacle cells are projected into the track frame. The planner clusters the blocking
obstacle and searches side ∈ {left, right}, shift 0.5…4 m and θ ∈ {45, 60, 30, 75, 90}°. It checks the
out-leg, the parallel leg past the obstacle (+1.6 m) and the 40° rejoin leg against all cells with a
0.85 m clearance. The cost is the shift, plus a turn-angle penalty, plus a penalty for the side away from the goal.

**Side guard** (every state): the sonar only looks forward. An obstacle remembered in the grid that is now
*beside* the hull must not be hit by turning toward it, for example when rejoining the track. Turning toward
cells within 0.95 m of the hull side is blocked until they are behind the vehicle. This fixed the one
collision mechanism found during development (VFH rejoining into a pillar).

**Stern protection** (the forward sonar cannot see beside or behind the hull):
* **In-place turns are checked against the swept hull**, nose *and* tail, using remembered cells. A turn that
  would bring the hull within 0.15 m of a known obstacle is refused (`TURN_BLOCKED`). The heading is held, and
  after 20 s the command carries `abort=True` so the pattern can stop or surface safely. Once a turn is refused,
  the blocking cells are latched until the turn is resolved.
* **Confirmed cells within 2.5 m of the hull and outside the scanned sector don't age out.** The sonar can neither
  refresh nor clear them, and they are exactly the cells that protect the tail.
* **Back-off only for an obstacle on a collision course** (within ±0.3 m of the centreline). A wall *beside* the hull is
  not fixed by reversing.
* **Back-off never reverses further than the vehicle's own free trail**, i.e. the distance it has just driven
  forward on the current heading. With no trail behind it, the vehicle holds and rescans instead of reversing blind.
  There is also a 4.5 m budget per obstacle.

**Arbiter** `arbitrate(mission_speed, mission_heading, cmd)`:
* An override replaces the mission setpoint, and a cap limits it.
* Heading override → heading-hold reference.
* Nothing else is passed through.

`cap_t1_pwm()` is for code that only has a PWM.

## Scenarios

`wall_0.1/0.2/0.3` (STOP, stopping distance), `wall_0.3_noslow` (pure speed-dependent STOP),
`pillar_left/right`, `gap_wide` (2.0 m), `gap_narrow` (0.8 m), `u_trap`, `square_tank` (4.5 m square, left
turns, 10 × 6 m tank, start 1.5 m from the wall, tank distance profile), `square_tank_close` (same, start 1.0 m:
corner turn refused), `noisy_open` (360° scan, tether behind, heavy noise, DVL
dropouts every ~6 s), `noisy_pillar`, `moving_crossing` (diver/ROV crossing the track).

## Metrics (batch)

* **collisions**: truth hull (capsule) contacts.
* **min clr**: minimum hull clearance to any obstacle.
* **stop clr**: final hull clearance in the wall tests.
* **false stops**: STOP triggers when the truth shows nothing within the slow distance + 1 m in a corridor 0.6 m wider than the avoidance corridor.
* **late**: truth obstacle inside 50 % of the stop distance while the vehicle is still tracking at > 0.08 m/s with no cap active.
* **time lost**: mission time minus the same mission in an empty world (same seed).

Seed 1 uses the nominal geometry. Other seeds jitter the start (±0.15 m, ±2°) and the obstacles (±0.15 m).
Each seed also draws a different yaw drift, DVL dropout timing and sonar noise.

Results: `results/summary.md`, `results/runs.csv`, `results/*_summary.png`, `results/*_live*.png`.

## Integration into auv_depth.py

Copy `detection.py`, `avoidance.py`, `sonar_avoid_node.py` next to `auv_depth.py`.

```python
from sonar_avoid_node import SonarAvoidNode
AVOID = SonarAvoidNode(mode="STOP")            # start with STOP in the tank, AVOID later

# 1) Ping360Link._process_scan(self, angle_grad, s_per, intensities, last_log):
AVOID.on_ping_raw(angle_grad, s_per, intensities)       # timestamps with time.monotonic()

# 2) In the 20 Hz pattern steps (_square_step / rect / lawn): you already compute a speed
#    reference for _speed_t1(st, v_ref, vx, dt) and a heading reference for _sq_t2_pwm(ref, ...).
#    Pass both through the node BEFORE they are used:
out = AVOID.step(now, pe, pn, yaw, vx, v_ref, ref,
                 track=(leg_origin_e, leg_origin_n, leg_heading),   # current leg line, ENU yaw
                 goal=(corner_e, corner_n),
                 turning=(phase == "TURN"), active=True)
v_ref, ref = out.speed_ref, out.heading_ref
t1_u = self._speed_t1(sq, v_ref, vx, dt)                          # T1 from the (possibly overridden) setpoint
if out.cmd.waypoint_blocked:
    ...  # finish the current LEG as if its distance was reached -> next TURN
if out.cmd.abort:
    self._pattern_abort()   # in-place turn would hit a known obstacle: stop the pattern safely
# During TURN your profiled turn owns T2: keep it, but use out.speed_ref for T1. If out.heading_ref
# differs from the turn target (swing circle blocked), hold the heading instead of turning.

# 3) GCS: merge AVOID.telemetry() into json_packet() - state, zone, distances, message,
#    last beam and up to 150 confirmed obstacle points for the PySide6 sonar widget.
# 4) A $AVOID,<OFF|WARN|STOP|AVOID>[,SIDESTEP|VFH] command -> AVOID.set_mode(...).
```

* **Positions**: the node needs the same DR position (pe, pn) your patterns use. If position is
  invalid, keep calling `step()` with the last value. Detection still works for range, but the side-step
  quality drops. Use STOP mode without a DVL.
* **`--simulate`**: your `SimulationWorker` already runs the same dynamics. To get sonar data on the desktop,
  create `World(json)` + `Ping360Sim(cfg, world, rng)` in that worker and call `AVOID.on_ping_raw(...)`
  from it with the simulated pose. Also push the pings to the GCS sonar widget. Then the GCS, the backend
  and the avoidance can be tested end to end.
* Defaults are open-water values. For the tank use the `TANK_AVOID` profile in `scenarios.py`
  (STOP ≈ 1.5 m, goal margin 1.0 m).

## Real-data replay

```
python replay.py logs/log16_events.csv --fixed-pose 0,0,90 --out log16_replay.png
python replay.py logs/log16_events.csv --pose-csv log16_pose.csv     # t,x,y,yaw (ENU) from your DR
python replay.py pings.jsonl                                         # from run_sim.py --log-pings
```
It reads `PING360_SCAN` rows (angle_grad, sample_period, samples). The figure shows the sonar fan, first
returns, cells confirmed at any time and range-vs-bearing. Use `--config` to try detection parameters.

## Tuning from tank logs

Do these in order. Each step uses a log you can record in one tank session.

1. **Angle convention**: hold the vehicle still. Put a target (pole) 3 m on the bow, then 3 m to starboard.
   Replay with `--fixed-pose 0,0,0`: the bow target must be at bearing 0, the starboard one at −90 (left is +).
   Fix `fwd_grad` / `angle_dir`.
2. **Blanking**: plot a few raw pings (the A-scan panel shows the same). Set `blanking_m` just past the end of the
   ringing, typically 0.5–0.8 m.
3. **Noise floor / thresholds**: record 1–2 min pointing into open water. After the median filter, take the 99.9th
   percentile of samples beyond the blanking range. Set `abs_min` 10–20 above it, and `peak_min` well below the
   weakest wanted echo (a pillar at max range). Replay the open-water log: **confirmed cells must be 0**. If not,
   raise `global_alpha` / `cfar_alpha` or `min_hits`.
4. **`min_blob_m`**: the narrowest real echo (thin pole at short range) versus the widest spike. In practice
   0.04–0.08 m at 1 cm per sample.
5. **Sector gate**: replay a full-circle log with the tether out. Set `valid_half_sector_deg` so the tether
   arc is excluded.
6. **Grid timing**: time to confirm is roughly one to two sector refreshes (2.3 s at 12 m, ±60°). `age_s` must exceed the
   time an obstacle stays out of the sonar sector while you pass it. 20 s at 0.15 m/s is fine. Shorter ages
   make moving obstacles clear faster.
7. **Braking**: from your speed logs, step T1 to the brake value at 0.2 and 0.3 m/s. Take the deceleration you
   get *with your speed loop* and put it in `brake_decel_mps2` (the simulator brakes at about 0.1–0.13 m/s²).
   `react_time_s` ≈ sector refresh + 0.5 s.
8. **Distances**: `stop_base_m` = the clearance you want to keep from the sonar head, plus a margin for DR heading
   error (about 2° × range). `corridor_half_m` = hull radius + lateral DR error over ~5 m + margin.
9. **Turning**: `swing_radius_m` ≥ nose length + margin. Also keep pattern origins ≥ 1.3 m from walls, because the
   **tail** is not protected by the forward sonar (see results).
10. Re-run `python run_sim.py --batch` with your overrides in a JSON (`--config`) and check the table stays at 0
    collisions / 0 false stops.

## Hard rule found in simulation: keep ≥ 1.0 m from walls when turning

Closer than about 0.7 m, a wall beside the nose lies **inside the 0.75 m blanking range at the edge of the ±60°
sector**, so the sonar never registers it. An in-place turn there swings the 0.9 m tail into a wall the vehicle
has never seen. In `square_tank_close` the turn check refuses the corner turn for starts at 0.8–1.3 m. The one
randomised start at 0.62 m collides. Put pattern origins ≥ 1.3 m from tank walls. In the real tank, also
lower `blanking_m` as far as the ringing allows (see tuning step 2).

## Known limitations

* **Blind spots**: the forward sonar cannot see behind or beside the stern. The mitigations are listed under
  "Stern protection" above, but anything never seen (a wall inside the blanking range, an object that arrived from
  behind) stays invisible.
* **2-D only**: depth (T3) is untouched by design.
* **Snapshot planning**: moving obstacles are handled by STOP and wait, plus ray clearing. The side-step plan is
  made from a snapshot.
* **DR dependence**: the side-step relies on DR position for the offset line. With long DVL dropouts it degrades
  gracefully to stop-and-wait.
* **VFH** is the weaker stage in traps and passes closer; SIDESTEP is the recommended default.
