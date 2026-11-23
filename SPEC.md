# Microduck velocity-walking task: reproduction spec (step 1)

Written 2026-11-04. Nothing in here is implemented yet. Every item cites file:line in the sources below.

## Source aliases

All paths are under `~/microduck/upstream/`.

| alias | path | what |
|---|---|---|
| **CFG** | `bundle441/wandb_config.yaml` | **Ground truth.** The fully resolved env and agent config saved by wandb for the training run behind `alpha_walking.onnx` (HF `pollen-robotics/microduck-training/bundles/velocity2/441tzs6d_3750`) |
| **BUNDLE** | `bundle441/bundle.json` | provenance of that checkpoint |
| **V2** | `rl_cb8561f6/src/mjlab_microduck/tasks/microduck_velocity2_env_cfg.py` | task factory (Velocity2) |
| **V** | `rl_cb8561f6/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` | base microduck velocity factory that V2 builds on |
| **MDP** | `rl_cb8561f6/src/mjlab_microduck/tasks/mdp.py` | microduck obs, reward, curriculum and DR functions |
| **RC** | `rl_cb8561f6/src/mjlab_microduck/robot/microduck_constants.py` | robot entity and actuator cfg |
| **ACT** | `rl_cb8561f6/src/mjlab_microduck/actuator/friction_dr_bam.py` | BAM subclasses (friction DR, backlash encoder) |
| **XML** | `rl_cb8561f6/src/mjlab_microduck/robot/microduck/robot_walk_backlash.xml` | MJCF; meshes in `.../microduck/assets/` (86 files) |
| **BAMJ** | `deps/bam/bam/mjlab.py` | BAM mjlab actuator (commit 62bd8ce) |
| **BAMA** | `deps/bam/bam/actuator.py` | BAM control law and torque equation |
| **XL** | `deps/bam/bam/dynamixel/actuator.py` | XL330 constants |
| **M6** | `deps/bam/bam/params/xl330/m6.json` | identified XL330 m6 parameters |
| **ML** | `deps/src/mjlab/` | mjlab 1.3.0 wheel source |
| **RSL** | `deps/src/rsl_rl/` | rsl-rl-lib 5.0.1 wheel source |

`rl_cb8561f6` is a git worktree of pollen-robotics/microduck_rl at `cb8561f6`.

---

## 0. Target and provenance

- **Task:** `Mjlab-Velocity2-Rough-Backlash-MicroDuck` (BUNDLE `task_id`). It is registered with `make_microduck_velocity2_env_cfg(rough=True)`, the Velocity2 RL cfg and `MICRODUCK_WALK_BACKLASH_ROBOT_CFG` (`rl_cb8561f6/src/mjlab_microduck/tasks/__init__.py:357`).
- **Reference policy:** `alpha_walking.onnx` in HF `pollen-robotics/microduck-policies` (sha256 `e36332d3…`).
  - It is bit-exact with run `441tzs6d` @ `model_3750.pt` (BUNDLE `onnx[0].verified`).
  - Training command: `train … --env.scene.num-envs 4096 --agent.max-iterations 10000`. Pollen kept the iteration-3750 checkpoint, which the run name calls "BEST".
  - That is 3750 × 24 × 4096 = **368.6 M control steps**, or 1.47 B physics steps, on one RTX 4090 (CFG:1-60 `_wandb`).
- **Why not `velstand.onnx`:** the manifest marks it as the current default walk policy. It was built in stages: it was warm-started from `alpha_walking`, then put through repeated behavior-cloning and fine-tuning stages (`microduck_rl/docs/velstand_policy.md:20-30`, `manifest.json`). A single from-scratch PPO run can't reproduce it. **`alpha_walking.onnx` is the parity reference and the training target.**
- **Code version:**
  - The run's commit `2f1f76bc` is "not on a remote" (BUNDLE `code`) and isn't fetchable.
  - The nearest public commit is `cb8561f6`, 45 s before the run started (10:23:36 vs 10:24:21 +0200).
  - So CFG decides every parameter value, and the code supplies semantics only. Every numeric value below was checked against CFG.
- **Pinned dependencies** (same at `cb8561f6/uv.lock` and develop): mjlab 1.3.0, rsl-rl-lib 5.0.1, mujoco 3.10.0, mujoco-warp 3.8.1, warp-lang 1.12.0, torch 2.9.1, better-actuator-models (BAM) git `62bd8ce`.

## 1. Simulation and control rate

- **Rate:** physics dt **0.005 s**, decimation **4**, giving **50 Hz** control. An episode is 20 s, i.e. 1000 control steps (CFG:290-291, 2357; `max_episode_length = ceil(20/0.02)`, ML `envs/manager_based_rl_env.py:111`).
- **MuJoCo options** (CFG:2338-2365):
  - integrator `implicitfast`, solver `newton`, cone `pyramidal`
  - iterations **30**, ls_iterations **50**, tolerance 1e-8, ls_tolerance 0.01
  - impratio 1, ccd_iterations 50, gravity (0, 0, −9.81)
  - These override the XML's own `<option>` (dt 0.002, 100 iterations; compiled XML dump in §2).
  - `nconmax 200`, `njmax 1500` and `ls_parallel` are MJWarp buffer and kernel options with no CPU equivalent. The rough-terrain origin of 30/50/200 is V:780-790.
- **Step order** (ML `envs/manager_based_rl_env.py:412-457`). The CPU environment must reproduce it exactly:
  1. `process_action(a)`: store the raw action and shift the history (ML `managers/action_manager.py:152-175`).
  2. Four times: `apply_action` → per actuator, `get_command` → `apply_delay` → `compute` → write ctrl (ML `entity/entity.py:1237-1239`) → `mj_step` → sensor update (contact air-time tracking runs **every substep**, ML `sensor/contact_sensor.py:304-309, 378-418`).
  3. `episode_length += 1`, `common_step_counter += 1`.
  4. Terminations, then rewards. **No `mj_forward` before these**, so xpos, xquat, cvel, site vel and sensordata are one substep stale (the forward pass inside the last `mj_step` ran before integration). This is deliberate (`:381-395`, `:427-433`).
  5. Reset any done envs (`_reset_idx`, `:543-582`): curriculum → sim and scene reset → reset-mode events → manager resets.
  6. `mj_forward` for all envs (`:447`).
  7. Command manager compute (resample and standing-zeroing), then interval events (pushes), then `sense()`, then observations (`:449-457`).
  - **TO VERIFY in step 2:** which mjlab `EntityData` properties read qpos/qvel directly versus derived fields (`ML entity/data.py`). That determines exactly which reward inputs are stale.

## 2. Model (MJCF and meshes)

- **Robot file:** `get_walk_backlash_spec()` loads XML (RC:27, 65-66; CFG:2091). Includes: `joints_properties.xml`, `sensors.xml`, `additional.xml`, meshes under `assets/`. `add_backlash.py` and `config_mjcf_walk_backlash.json` generate it.
- **Compiled with mujoco 3.10.0** (`~/microduck/tools/inspect_mjcf.py`):
  - 16 bodies, nq 35, nv 34, 75 geoms; total mass **0.7372 kg**.
  - A freejoint on `trunk_base`, plus 14 servo hinges. Each servo hinge has an unactuated `passive_<joint>_backlash` hinge in series: range ±0.01745 rad (±1°), armature 0.001, damping 0.01, no frictionloss.
  - Servo joints in the XML have armature 0.0018, damping 0.053 and frictionloss 0.0048, but BAM overrides all three (§3).
  - Sensors: `imu_ang_vel` (gyro, site `imu`), `imu_lin_vel` (velocimeter), `imu_accel`, `orientation` (framequat) and `root_angmom` (subtreeangmom of `trunk_base`).
- **Joint, actuator, action and obs order** (XML actuator order; matches the ONNX `joint_names` metadata):
  `left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle, neck_pitch, head_pitch, head_yaw, head_roll, right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle`.
- **Servo joint ranges (rad):**
  - hip_yaw: L [−0.4363, 0.5236], R [−0.5236, 0.4363]
  - hip_roll ±0.384
  - hip_pitch, knee, ankle, head_pitch ±1.5708
  - neck_pitch [−1.5708, 1.0472]
  - head_yaw ±2.9671
  - head_roll ±0.4363
- **Default pose (HOME)** (RC:73-97, 154-157; CFG:2058-2072). Backlash joints start at 0. The ONNX `default_joint_pos` metadata matches.

  | joint | L | R |
  |---|---|---|
  | hip_yaw | 0 | 0 |
  | hip_roll | −0.0873 | +0.0873 |
  | hip_pitch | −0.4579 | +0.4579 |
  | knee | −0.0049 | +0.0049 |
  | ankle | +0.4530 | −0.4530 |

  neck_pitch 0.3491, head_pitch 0.3491, head_yaw 0, head_roll 0.
- **Collision cfg applied by mjlab** (RC:99-104; CFG:2034-2052):
  - Every geom matching `.*_collision` gets contype = conaffinity = 1 and condim 1. The exceptions are `left_foot_collision` and `right_foot_collision` (geoms 26 and 72, bodies `ankle_left` and `ankle_right`): condim 3, priority 1, sliding friction 1.0.
  - `disable_other_geoms: true` turns collision off for every other geom.
- **Soft joint limits:** factor 0.9 (CFG:2032), used by `dof_pos_limits` (§7).
- **BAM `edit_spec` changes** (BAMJ:234-293):
  - The 14 position actuators become motors with gear 1 and `forcerange` ±(max vin)·kt/R = ±8.2·0.366013/2.811392 = **±1.0675 Nm**.
  - Servo joints get `armature` **0.00180774** (M6), damping 0 and frictionloss 0 (rewritten every substep, §3).
  - Joint friction constraints are stiffened: `solref_friction` (−5e4, −2e2), `solimp_friction` (0.99, 0.9999, 0.001, 0.5, 2.0) (BAMJ:221-232; `stiff_frictionloss: true`, CFG:2019).
- **Terrain** (CFG:2188-2317; V:104-148, 768-790):
  - `terrain_type: generator`, **5 × 5 tiles of 8 × 8 m**, `curriculum: false`, difficulty U(0, 1) per tile, seed null (random), border 20 m wide and 1 m high, `max_init_terrain_level 5`.
  - Sub-terrains (proportion):
    - `flat` 0.25
    - `pyramid_stairs` 0.25 (step height 0–0.015 m scaled by difficulty, step width 0.15, platform 2, border 1)
    - `random_grid` 0.30 (cell 0.45 m, height 0–0.010, platform 1.5, border 0.25)
    - `pyramid_slope` 0.20 (heightfield, rise/run 0.03–0.10, platform on top 2 m, horizontal scale 0.1, vertical scale 0.001)
  - Every terrain geom gets solref (0.04, 1) and solimp (0.85, 0.95, 0.001, 0.5, 2.0) (`_soften_terrain_contacts`, V:151-169; CFG:2187).
  - **Discrepancy with the code:**
    - V:106-110 declares 10 × 20 tiles with no `curriculum` override, but the run used 5 × 5 with `curriculum: false`. Those are exactly the play overrides at V:792-795.
    - V:774 assigns the module-global `MICRODUCK_ROUGH_TERRAINS_CFG` by reference, and registration builds the play cfg right after the train cfg (`tasks/__init__.py:378-381`, `_make_cfg(play=True)`), so the play overrides mutate the object the training cfg points to.
    - **We reproduce what ran: 5 × 5, no curriculum.**
  - Generator internals: ML `terrains/terrain_generator.py:130-340`, `primitive_terrains.py`, `heightfield_terrains.py`.

## 3. Actuator: BAM m6, XL330, with backlash, delay and voltage DR

Classes: `BacklashEncoderBamActuator` → `FrictionDRBamActuator` → `bam.mjlab.BamActuator` (RC:120-139, 206-214; ACT:28-112). The steps below run **every physics substep**, i.e. 200 Hz.

1. **Encoder reads through the backlash:** `q_enc = q_servo + q_backlash`. Velocity stays motor-side, `q̇ = q̇_servo` (ACT:64-104).
2. **Command delay:**
   - The position target is delayed by a lag ∈ {3, 4, 5, 6} physics steps (15–30 ms).
   - The lag is redrawn uniformly **every physics step**, independently per env (update_period 0, hold_prob 0) (RC:130-131; CFG:2009-2013; ML `actuator/actuator.py:270-302`; ML `utils/buffers/delay_buffer.py:229-303`).
   - On reset the buffer is cleared, and the next append backfills it (delay_buffer:71-78, 194-209).
   - The lag is clamped to the available history (delay_buffer:240-245).
3. **Firmware P law** (BAMA:220-269; XL:79-99):
   - `duty = (q_target − q_enc) · kp · error_gain`
   - kp = 200 (`kp_fw`, RC:124). There is no kp DR (V:12).
   - `error_gain = (4096/2π)/(256·885) = 2.87745e-3`.
   - **Current limiter:** `XL330Actuator` sets `max_current = 1.75` A (XL:98). `BamActuator` overrides only `vin` and `kp` (BAMJ:186-189), so in this BAM version the limiter is **active**: duty is clamped to `[(kt·q̇ − R·1.75)/vin, (kt·q̇ + R·1.75)/vin]` (BAMA:258-264). Pollen's CPU script says otherwise (§12 R3).
   - duty is then clamped to ±1, and `V = vin · duty`.
4. **Motor torque:** `τ = kt·V/R − kt²·q̇/R`, with kt **0.36601350** and R **2.81139235** (M6; BAMA:271-292). MuJoCo then clamps it to `forcerange`.
5. **Supply voltage:**
   - `vin_env ~ U(6.5, 8.2)` and sag gain `g_env ~ U(0, 0.2)`, both drawn per env **at startup** and held across resets (BAMJ:335-351; RC:126-128).
   - Effective `vin = max(6.0, vin_env − g_env · Σ_j |τ_prev,j|)`, where `τ_prev` is the previous substep's **unclamped** BAM torque for this env (BAMJ:601-607, 671-673). It is zeroed at env reset (BAMJ:382-390).
6. **Friction budget** (BAMJ:424-496, 630-669). These values are written into MuJoCo's own `dof_frictionloss` (per env) and `dof_damping`, so the constraint solver applies them implicitly.
   - `τ_m = qfrc_actuator[dof]` from the previous solve.
   - `τ_e = −qfrc_bias + qfrc_constraint − qfrc_frictionDOF`, all from the previous solve. `qfrc_frictionDOF` is the scatter of `efc_force` rows of type `mjCNSTR_FRICTION_DOF` (BAMJ:541-574).
   - `s = exp(−(|q̇|/2.89037)^8.68326)`.
   - `fl = 0.00477118 + s·0.00467635 + |τ_e·8.5159e-6 − τ_m·0.266786| + s·|τ_e·0.0807793 − τ_m·1.07229e-5| + s·q`
   - where `q = 0.00490257·τ_e²` if `|τ_m| > |τ_e|`, else `0.00997247·τ_m²`.
   - `dof_frictionloss = friction_scale_env · fl` (ACT:37-47); `dof_damping = 0.00535967` (viscous).
   - BAM's `q_offset` is **not** applied on this path.
7. **Friction DR:** `friction_scale_env ~ U(0.9, 1.1)`, redrawn at every reset (MDP:3212-3240; CFG:768-830).

Implication: the actuator needs feedback between substeps and per-substep writes to model fields. **`mujoco.rollout` can't express either**, so step 2 needs a custom multithreaded C++ batch stepper over the MuJoCo 3.10.0 C API.

## 4. Action space

- 14-D raw policy output `a`, in the actuator order of §2.
- Joint target = `default_pos + 1.0 · a`: scale 1.0, `use_default_offset` (CFG:63-73; V:258-260; ML `envs/mdp/actions/actions.py:154-163`).
- **No clipping:** `clip: null` and runner `clip_actions: null` (CFG train_cfg).
- During training, the env gets the *sampled* Gaussian action. The ONNX outputs the mean.
- On reset, `action`, `prev_action` and `prev_prev_action` are zeroed (ML `managers/action_manager.py:140-150`).

## 5. Observations

**Actor, 61-D**, in this order. Confirmed three ways: dict construction order (ML `tasks/velocity/velocity_env_cfg.py:76-110`; V:577-581, 602-607, 653-658, 732-740), CFG, and the ONNX metadata `observation_names`.

| # | term | dim | value | noise (uniform, additive) | delay (control steps) | cite |
|---|---|---|---|---|---|---|
| 1 | base_ang_vel | 3 | `R_mis · ω_root` (root link angular velocity, body frame) | ±0.03 | lag ∈ {0, 1}, redrawn every 64 steps with a random per-env phase | CFG:1017-1034; V:609-611, 632, 641-643; MDP:3652-3660 |
| 2 | projected_gravity | 3 | `R_mis · g_b` (unit gravity in the root frame) | ±0.01 | same as above | CFG:1223-1240; V:613-615, 633, 644-647; MDP:3641-3649 |
| 3 | joint_pos | 14 | `(q_servo + bias) + q_backlash − q_default` | ±0.001 | 0 | CFG:1080-1151; V:665-678; MDP:6143-6160 |
| 4 | joint_vel | 14 | `q̇_servo + q̇_backlash` | ±0.25 | exactly 1 | CFG:1152-1222; V:653-658; MDP:6163-6177 |
| 5 | actions | 14 | last raw action | – | 0 | CFG:1004-1016; ML `envs/mdp/observations.py:80` |
| 6 | command | 3 | twist (vx, vy, ωz) | – | 0 | CFG:1050-1064 |
| 7 | head_command | 4 | (neck_pitch, head_pitch, head_yaw, head_roll) deltas | – | 0 | CFG:1065-1079; V:705-713 |
| 8 | body_command | 6 | (x, y, z, roll, pitch, yaw) | – | 0 | CFG:1035-1049; V:718-728 |

Details for the table:
- **`R_mis`** (IMU mounting error), rows 1–2: a per-env constant rotation. Axis = normalized N(0, I); angle ~ U(0, 6°). It is drawn once per process, never at reset (MDP:3618-3638).
- **`bias`** (encoder calibration error), row 3: per env and per joint, U(±0.015), drawn at startup (CFG:364-426; ML `envs/mdp/dr/joint.py:209`, `entity/data.py:371`).
- **Processing order:** compute → noise → clip → scale → delay (ML `managers/observation_manager.py:21, 330-350`). Every `scale` and `clip` is null. Delay buffers reset with the episode (`:248-251`).
- **Normalization:** rsl_rl `EmpiricalNormalization`, `(x − μ)/(σ + 0.01)`, baked into the ONNX as the `Sub` and `Div` nodes before the MLP (RSL `modules/normalization.py:46-66`).
- **ONNX I/O:** input `obs` [1, 61], output `actions` [1, 14].

**Critic, 76-D, no noise** (`enable_corruption: false`, CFG:1244). Order:
- `base_lin_vel` 3: root, body frame.
- `base_ang_vel` 3: `imu_ang_vel` sensor.
- `projected_gravity` 3.
- `joint_pos` 14: through the backlash, unbiased.
- `joint_vel` 14.
- `actions` 14.
- `command` 3.
- `foot_height` 2.
- `foot_air_time` 2.
- `foot_contact` 2.
- `foot_contact_forces` 6: `sign·log1p|f|` of the global net force.
- `head_command` 4.
- `body_command` 6.

Sources: CFG:1241-1557; ML `tasks/velocity/velocity_env_cfg.py:112-135`; V:585-588, 623-629; ML `tasks/velocity/mdp/observations.py:14-47`.

**Sensors behind the critic and rewards** (CFG:2096-2186):
- `feet_ground_contact`: both foot geoms vs the terrain body; fields found and force; `netforce` reduce; 1 slot; air-time tracking on.
- `self_collision`: `trunk_base` subtree vs itself; found only.
- `foot_height_scan`: per foot site, a center ray plus a 2-sample ring at radius 0.04 m, yaw-aligned, pointing down, max 1 m, geom group 0 only, parent body excluded.
  - Height = site z − hit z, minimum over rays. A backface hit gives 0. A miss gives `max_distance`, or `clamp(site z, 0, 1)` if every ray misses (ML `sensor/terrain_height_sensor.py:57-90`).

## 6. Commands

**twist** (CFG:110-139; V:685-693; V2:143-149; MDP:4505-4533; ML `tasks/velocity/mdp/velocity_command.py:75-138`; ML `managers/command_manager.py:99-123`):
- Resampled every U(3, 8) s per env, and at reset. Sampling in order:
  1. vx ~ U(−0.4, 0.4), vy ~ U(−0.3, 0.3), ωz ~ U(−1, 1).
  2. A heading target is drawn, but `rel_heading_envs: 0`, so it is never used.
  3. Standing with probability `rel_standing_envs` (curriculum, §10).
  4. World-frame with probability 0.
  5. Forward-only with probability 0.2: vx = max(|vx|, 0.3), vy = ωz = 0.
  6. Turn-in-place with probability 0.15: vx = vy = 0, ωz = ±U(0.4, 1.0), and the env is unmarked as standing.
- Every control step, standing envs have their command forced to 0.
- `init_velocity_prob` 0.

**head_pose:** 4-D, each dim uniform within the current curriculum range, resampled every U(2, 5) s (CFG:95-109; MDP:5061-5112). `zero_command_prob` 0.

**body_pose:** 6-D, uniform ±0.005 m for xyz and ±0.05 rad for rpy, resampled every U(2, 5) s (CFG:76-94). Its reward weight is 0, so it is just a live input.

## 7. Rewards

Per step, `r = Σ_i w_i · f_i · dt` with dt = 0.02 (`scale_rewards_by_dt: true`, CFG:2002; ML `managers/reward_manager.py:39`). Terms with weight 0 are **skipped entirely** (`:122`).

- "active" means `‖c_xy‖ + |c_z| > 0.01`.
- `v` and `ω` are the root link velocities in the body frame; `c` is the twist command.

| term | weight | f | cite |
|---|---|---|---|
| track_linear_velocity | 2.0 | exp(−(‖c_xy − v_xy‖² + v_z²)/0.1) | CFG:1937-1942; V2:55-56; ML `tasks/velocity/mdp/rewards.py:27-44` |
| track_angular_velocity | 2.0 | exp(−((c_z − ω_z)² + ω_x² + ω_y²)/0.5) | CFG:1931-1936; V2:57-58; :47-64 |
| upright | 2.0 | exp(−‖g_xy in the trunk_base frame‖²/0.05) | CFG:1943-2001; V2:67-68; :67-110 |
| pose | 1.0 | exp(−mean_j((q_j − q0_j)²/σ_j²)), over the 10 leg servo joints (servo side only) | CFG:1847-1925; V:178-194, 264-276; :385-466 |
| air_time | 3.0 | Σ_feet 1[0.125 < t_air < 0.3] · active | CFG:1562-1570; V2:74-77; :209-238 |
| foot_clearance | −2.0 | Σ_feet abs(h_foot − 0.02) · ‖v_foot,xy‖ (world), · active | CFG:1701-1763; :241-268 |
| foot_swing_height | −0.25 | Σ_feet at first contact (peak_h/0.02 − 1)², · active | CFG:1826-1834; :271-323 |
| foot_slip | −0.1 | Σ_feet ‖v_foot,xy‖² · in_contact, · active | CFG:1764-1825; V2:84; :326-353 |
| body_ang_vel | −0.05 | ‖ω_trunk,world,xy‖² | CFG:1576-1633; :184-193 |
| angular_momentum | −0.02 | ‖L_root subtree‖² (`root_angmom` sensor) | CFG:1571-1575; :196-206 |
| dof_pos_limits | −1.0 | Σ over the 14 servo joints of soft-limit violation | CFG:1643-1700; ML `envs/mdp/rewards.py:81-96` |
| action_rate_l2 | −0.1 → −1.0 (§10) | ‖a_t − a_{t−1}‖² (raw actions) | CFG:1559-1561; ML `envs/mdp/rewards.py:58-65` |
| self_collisions | −1.0 | `found` count of trunk_base subtree self-contacts (no force history) | CFG:1926-1930; :162-181 |
| head_pose_tracking | 2.0 | mean over 4 head joints of exp(−((q_servo + q_bl − q0) − c_head)²/0.25) | CFG:1841-1846; MDP:5128-5192 |
| head_pose_bias | 0 → 3 (§10) | −mean over 4 head joints of abs(EMA(err)); α = 0.02/1.0; EMA zeroed when episode_length ≤ 1; not updated while weight is 0 | CFG:1835-1840; V2:116-132; MDP:5229-5276 |
| body_pose_tracking | 0 | disabled | CFG:1634-1642 |

Reward-term notes:
- **pose std (σ_j):** standing regime when ‖c_xy‖ + |c_z| < 0.01, otherwise walking. The running regime (≥ 1.5) is unreachable.
  - Standing: hip_yaw 0.1, hip_roll 0.05, hip_pitch 0.15, knee 0.15, ankle 0.1.
  - Walking: hip_yaw 0.3, hip_roll 0.05, hip_pitch 0.4, knee 0.4, ankle 0.25.
- **foot_swing_height:** peak height is tracked only while the foot is in the air, then reset at landing. First contact means contact time < dt + 1e-8 (ML `sensor/contact_sensor.py:311-320`).
- **Removed from the base recipe:** soft_landing, com_height_target, stillness_at_zero_command, neck_action_rate_l2, joint_torques_l2 (V2:70-71, 80, 103-104).

## 8. Terminations

From CFG:2366-2383.

| term | condition | type |
|---|---|---|
| fell_over | acos(−g_b,z) > 1.2217 rad (70°) | terminal (ML `envs/mdp/terminations.py:24-32`) |
| nan_state | any non-finite joint pos/vel or root pos/quat/vel | terminal (MDP:963-1000) |
| out_of_terrain_bounds | abs(x) or abs(y) > 0.5 · 5 · 8 − 0.3 = 19.7 m | time-out, bootstrapped (ML `tasks/velocity/mdp/terminations.py:32-65`) |
| time_out | episode_length ≥ 1000 | time-out, bootstrapped (ML `envs/mdp/terminations.py:19-21`) |

## 9. Events and domain randomization

"add" and "scale" operations always start from the compile-time default, so they don't accumulate across resets (ML `envs/mdp/dr/_core.py:110-112`; V:453-457).

**Startup (once per env):**
- `foot_friction`: both foot geoms, **abs** U(0.7, 1.3), one value shared by both feet, sliding axis only (CFG:433-499; ML `envs/mdp/dr/geom.py:117-145`).
- `encoder_bias`: U(±0.015) per joint (CFG:364-426).
- `randomize_mass_inertia`: `pseudo_inertia` on `trunk_base`, α ~ U(ln 0.95/2, ln 1.05/2) = U(−0.02565, 0.02440). Mass and inertia scale by e^{2α} ∈ [0.95, 1.05]; CoM unchanged (CFG:831-894; V:497-514; ML `envs/mdp/dr/body.py:415-541`).
- `base_com`: `body_names: []` (CFG:293-363, 310). **Probably resolves to no bodies, making it a no-op.** TO VERIFY (ML `managers/scene_entity_config.py:155-190`).
- `expand_bam_friction_fields`: no-op marker (MDP:3198-3209).
- BAM `vin` and sag gain (§3), and the IMU misalignment `R_mis` (§5).

**Reset (every episode):**
- `reset_base`:
  - root position = env origin + (U(±0.5), U(±0.5), U(0.12, 0.13)); yaw U(±3.14); roll and pitch 0; velocity 0 (CFG:901-920; V:433; ML `envs/mdp/events.py:81-178`).
  - The env origin comes from the terrain-level/tile assignment. TO VERIFY with curriculum off (ML `terrains/terrain_entity.py`).
- `reset_robot_joints`: HOME pose plus offset 0, velocity 0 (CFG:921-987).
- `randomize_armature`: armature × U(0.9, 1.1) for every joint matching `.*`, presumably including the backlash hinges. TO VERIFY. Default servo armature is 0.0018077 (CFG:569-633; V:545-557).
- `randomize_com`: `trunk_base` ipos += U(±r) per axis; r from §10, 0.003 → 0.015 (CFG:634-698).
- `randomize_head_com`: ipos += U(±r) for `neck`, `neck_pitch`, `yaw_roll_motion`, `bottom_head_shell`, and `bearing_roll` (which is actually the right hip-yaw link, kept by mistake, V:37-51); r 0.003 → 0.010 (CFG:699-767).
- `randomize_joint_friction`: BAM `friction_scale` U(0.9, 1.1) (CFG:768-830).
- `reset_action_history`: touches microduck-only caches that this config never creates, so effectively a no-op (CFG:895-900; MDP:229-262).

**Interval:**
- `push_robot`: every U(3, 6) s per env, root world linear velocity x, y += U(±0.3) (CFG:500-568; ML `envs/mdp/events.py:321-336`).

## 10. Curricula

- Stage boundaries are measured in `common_step_counter` (control steps per env; iteration = step/24). A stage applies once the counter **exceeds** its step; `pose_command_range_curriculum` uses ≥ instead (MDP:3442-3461, 3295-3332, 3464-3501, 5471-5498).
- Curricula are evaluated inside `_reset_idx` (ML `envs/manager_based_rl_env.py:544`).
- Every curriculum reaches its final stage by iteration 2000.

| curriculum | stages (iteration → value) | cite |
|---|---|---|
| action_rate_l2 weight | 0 → −0.1, 500 → −0.2, 750 → −0.4, 1000 → −0.6, 1250 → −0.8, 1500 → −1.0 | CFG:141-157; V2:154-167 |
| head_pose_bias weight | 0 → 0, 600 → 1, 1000 → 2, 1500 → 3 | CFG:201-213; V2:176-187 |
| standing env fraction | 0 → 0.02, 500 → 0.05, 750 → 0.10, 1000 → 0.15, 1500 → 0.20, 2000 → 0.25 | CFG:269-285; V:843-856 |
| head_pose ranges (neck, head_pitch, yaw, roll) | 0: ±(.05, .05, .07, .015); 500: ±(.17, .17, .21, .047); 1000: ±(.39, .39, .49, .11); 1500: ±(.72, .72, .91, .20); 2000: ±(1.1, 1.1, 1.4, .31) | CFG:214-268; V:893-906 |
| body_pose ranges | constant | CFG:158-176 |
| trunk CoM range r | 0 → 0.003, 500 → 0.005, 1000 → 0.010, 1500 → 0.015 | CFG:177-189; V:927-946 |
| head CoM range r | 0 → 0.003, 500 → 0.005, 1000 → 0.010 | CFG:190-200; V:948-962 |
| terrain_levels_vel | at reset: move up a level if distance walked > 4 m; move down if distance < 0.5 · ‖c_xy‖ · 20 s. Behavior with generator `curriculum: false` is TO VERIFY | CFG:286-289; ML `tasks/velocity/mdp/curriculums.py:25-58` |

## 11. PPO (rsl-rl-lib 5.0.1)

Sources: CFG train_cfg (lines after 2404); V:972-1009; V2:194-198.

**Networks:**
- Actor: MLP 61 → 512 → 256 → 128 → 14, ELU, observation normalizer.
- Critic: MLP 76 → 512 → 256 → 128 → 1, ELU, own observation normalizer.
- Gaussian policy. The std is a raw learnable parameter of shape (14,), initialized to 1.0, with no exp and no clamp (RSL `modules/distribution.py:156-176`).

**Rollout:**
- 24 steps per env × 4096 envs = **98,304 samples per iteration**.
- Normalizers update on every observation passed to `process_env_step` (RSL `algorithms/ppo.py:152-158`; `modules/normalization.py:51-66`).

**Returns and advantages:**
- γ 0.99, λ 0.95.
- Time-out bootstrapping: `r += γ · V(s_t) · timeout` (ppo.py:174-179).
- GAE (ppo.py:187-206).
- Advantages are normalized **once over the whole batch** (`normalize_advantage_per_mini_batch: false`, ppo.py:207-209).

**Update:**
- 5 epochs × 4 minibatches of 24,576.
- **One `randperm` shared by all epochs** (RSL `storage/rollout_storage.py:222-244`).
- Loss = clipped surrogate (ε 0.2) + 1.0 · clipped value loss (ε 0.2) − 0.01 · entropy (ppo.py:296-313).
- A single Adam optimizer over actor and critic parameters, lr 1e-3 (ppo.py:115-118).
- Adaptive lr, **checked every minibatch before its gradient step**: KL(old‖new) mean > 0.02 → lr / 1.5 (floor 1e-5); KL < 0.005 → lr × 1.5 (cap 1e-2) (ppo.py:268-294).
- Gradient norm clipped to 1.0 **separately** for actor and critic (ppo.py:374-375).

**Other:**
- No symmetry: `ENABLE_SYMMETRY False` (V:7), `symmetry_cfg null`. No RND.
- seed 42, checkpoint every 250 iterations.

## 12. Known discrepancies and risks

- **R1. Unpublished commit.** The training commit `2f1f76bc` isn't public. Every parameter comes from CFG; semantics come from `cb8561f6` plus the pinned libraries. Function bodies could still differ from what actually ran.
- **R2. Terrain.** It used play-mode values through the global-mutation leak (§2). The tile layout is random (seed null), so only its distribution can be matched.
- **R3. Current limiter.** It is active in BAM `62bd8ce` (§3.3). But Pollen's CPU rehearsal script asserts "training runs WITHOUT the firmware current limiter" (`microduck_rl/scripts/infer_policy.py:40`, develop).
  - The limiter caps stall torque near kt·1.75 ≈ 0.64 Nm, versus about 0.96 Nm from PWM saturation alone. That is a large difference.
  - **The parity gate will measure both settings.**
- **R4. MJWarp (float32, GPU) vs MuJoCo C (float64, CPU).**
  - Neither has noslip.
  - Warp's contact buffer (`nconmax` 200) can drop contacts; CPU never does.
  - Small numerical differences in friction and contact are expected. That's why stiff_frictionloss exists.
- **R5. Stale derived quantities** at reward and termination time (§1, step 4) have to be replicated rather than "fixed".
- **R6. BAM can't run inside `mujoco.rollout`** (§3). That requires a C++ stepper with per-env `mjModel` copies. Per-env copies also cover the per-env DR fields (friction, armature, ipos, mass and inertia, frictionloss).
- **R7. TO VERIFY in step 2,** each against the cited mjlab file:
  - which `EntityData` fields are derived vs qpos/qvel
  - the `soft_joint_pos_limits` formula
  - whether `base_com` with empty `body_names` is a no-op
  - whether armature DR touches the backlash hinges
  - terrain-level updates and env origins with `curriculum: false`
  - stair, grid and slope geometry builders
  - ray-cast miss and backface semantics
  - contact sensor slot semantics on CPU MuJoCo
  - Entity reset of joint targets
  - **Plan:** export mjlab's compiled `MjModel` once offline and diff it field-by-field against ours. That's offline tooling only, never runtime; see the open question in the report.

## 13. Budget arithmetic for step 6

- Pollen reached the selected policy at **368.6 M control steps**.
- A 10 h budget equals that sample count only if the Mac sustains **≥ 10.2 k control steps/s** (≥ 41 k physics steps/s) end to end, *including* inference and the update.
- At throughput R, 10 h buys 36,000 · R samples.
- R is measured in step 5. Tracking quality at earlier iterations isn't public (wandb run is private), so "Pollen-level" gets defined in step 3 as the measured metrics of `alpha_walking.onnx` in our env.

## 14. Parity gate results (step 3, 2026-11-04)

Code: `duckmac/parity_yaw.py`, `tests/test_bam.py`, `tests/test_ray.py`; data in `reports/`.

**The original verdict was wrong.** "Tracking near zero-action level" came from the metric, not the env: instantaneous velocity error includes gait sway, and Pollen measured `alpha_walking` itself under-tracking, so that metric cannot separate a walking policy from a standing one. Fixed-command probes show it walking (feet alternate at about 2 Hz).

**Reference:** Pollen's own measurements of this policy in mjlab: `ANCHOR_YAW_REMAP` (develop `microduck_velstand_env_cfg.py:382-399`, 100 points, 8 envs per cell) and `docs/velstand_policy.md:186` (vx 0.4 -> 0.17). Their calibration script is not in the repo, so terrain, DR and head commands in their protocol are assumptions (flat, backlash, head 0 here).

**Result (flat, head 0):** correlation 0.988, but my env moves more than theirs: yaw bias +0.38 rad/s (RMSE 0.44), vx 0.22 at a 0.4 command vs 0.17. Falls: 1 in 1648 envs over 10 s.

**Ruled out:** current limiter (no effect here), obs noise, encoder bias and IMU misalignment, foot friction 0.7-1.3, supply voltage, terrain softening (feet have priority 1, so it does not reach foot contacts), robot model (August vs develop, walk vs allcollisions), actuator parameters (August vs develop identical), C++ BAM (matches Pollen's BAM classes to 4e-12 over 150 control steps; the limiter branch was not exercised).

**Most sensitive parameter:** servo friction. x2 removes the yaw bias but causes falls and collapses vx, so it is not a simple scale error.

**Superseded by the direct test below:** the gap to the table is not an env bug.

### Direct comparison with real mjlab (Warp on CPU, `.venv-ref`, offline only)
`tools/ref_*.py`; Pollen's task from `rl_cb8561f6`, unmodified apart from seed and test commands. Warp CPU, not CUDA.

- **Yaw sweep, flat:** real mjlab bias +0.387 (mine +0.381), vx at 0.4: 0.229 (mine 0.220), mine vs mjlab correlation 0.990. So `ANCHOR_YAW_REMAP` and the 0.17 m/s figure came from a different, unknown protocol, and are not valid gates for this env.
- **Training distribution, rough, 256 envs, 20 s, final curricula** (margins fixed before results: tracking 10%, falls 2x, significant reward terms 10%):

| metric | mjlab | mine (2 seeds) |
|---|---|---|
| falls/min | 0.293 | 0.223 / 0.328 |
| lin err m/s | 0.179 | 0.171 / 0.186 |
| yaw err rad/s | 0.398 | 0.373 / 0.407 |
| total reward | 7.207 | 7.263 / 7.177 |

  All large reward terms within 2%. All 137 obs channels: std within 5%, mean within 0.07 std.
- **GATE: PASS, with one known exception.** `foot_swing_height` is -0.0024 vs -0.0098. mjlab logs 2x the landings, 42% with peak height exactly 0; real landings agree (0.051 vs 0.046 per env-step, peak 0.0156 vs 0.0167 m). Most likely stance-foot contact flicker in Warp (inferred, not observed per substep). About 0.1% of total reward. Not replicated.
- **Current limiter:** real mjlab with BAM 62bd8ce has it active; duckmac defaults to on. The C++ limiter branch has not been exercised against BAM.
- **Remaining limits:** code is `cb8561f6`, not the unpublished training commit; Warp ran on CPU, not CUDA.

### R7 status
- Closed: EntityData derived vs qpos fields (`entity/data.py`); soft limits (`entity/entity.py:687-699`); `base_com` empty names is a no-op; armature DR includes the backlash hinges (`.*` over `joint_v_adr`); terrain levels keep updating with `curriculum: false`, and 4 proportions vs 5 columns gives an even column split; terrain builders (vendored verbatim); joint targets zeroed on reset (`entity/data.py:216-225`).
- Contact `found` differs between C and Warp at landing level (see above). Ray and height behaviour: `foot_clearance` matches within 1% and critic foot-height obs match, so no open issue.

## 15. Tuning measurements (step 5, 2026-11-04)

Short single runs, so differences of a few percent are noise. Data: `reports/bench_env.json`.

- **Threads (4096 envs):** 5: 7.9k, 8: 9.3k, 11: 9.0k env-steps/s. Default 8.
- **Env count (11 threads):** 512: 9.3k, 1024: 9.7k, 2048: 8.8k, 4096: 9.0k, 8192: 10.1k. Essentially flat, so keep Pollen's 4096.
- **Wall time:** simulation 92%, policy 6%, Python managers 2%. Simulation is the only bottleneck.
- **PPO update (4096 envs):** CPU 2.55 s, MPS 0.75 s. Collection about 10.5-11.5 s on both. Iteration about 12 s on MPS.
- **Memory:** 2.0 GB peak in training, under the 8 GB cap.
- **Budget:** about 8.2k control steps/s, so 10 h is about 3000 iterations (295 M samples, 80% of Pollen's 368.6 M); iteration 3750 needs about 12.5 h. Not measured under sustained thermal load.
