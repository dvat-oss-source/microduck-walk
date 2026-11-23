"""Batched CPU environment for Mjlab-Velocity2-Rough-Backlash-MicroDuck (run 441tzs6d).

Physics + BAM actuator run in the C++ stepper (duckmac/csrc/batch.cpp). This file
replicates mjlab's managers in numpy, following the step order of
ML envs/manager_based_rl_env.py:372-457 (SPEC §1). Every term cites its source.
No mjlab / torch / JAX at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from duckmac._batch import Batch
from duckmac.scene import HOME, JOINTS, TERRAIN_COLS, TERRAIN_ROWS, TERRAIN_SIZE, build_scene

STEP_DT = 0.02
PHYS_DT = 0.005
DECIM = 4
MAX_EP_LEN = 1000  # ceil(20 / 0.02)
F32 = np.float32

# Leg joints for the pose reward (CFG pose asset_cfg regex), per-joint std (CFG:1847-1925).
_POSE_STD_STAND = {"hip_yaw": 0.1, "hip_roll": 0.05, "hip_pitch": 0.15, "knee": 0.15, "ankle": 0.1}
_POSE_STD_WALK = {"hip_yaw": 0.3, "hip_roll": 0.05, "hip_pitch": 0.4, "knee": 0.4, "ankle": 0.25}
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")  # MDP:123 _NECK_JOINT_PATTERNS
HEAD_COM_BODIES = ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft", "bearing_roll")  # CFG:699-767

# Curricula, in common_step_counter units (CFG:141-289).
ACTION_RATE_STAGES = [(0, -0.1), (12000, -0.2), (18000, -0.4), (24000, -0.6), (30000, -0.8), (36000, -1.0)]
HEAD_BIAS_STAGES = [(0, 0.0), (14400, 1.0), (24000, 2.0), (36000, 3.0)]
STANDING_STAGES = [(0, 0.02), (12000, 0.05), (18000, 0.10), (24000, 0.15), (36000, 0.20), (48000, 0.25)]
COM_STAGES = [(0, 0.003), (12000, 0.005), (24000, 0.010), (36000, 0.015)]
HEAD_COM_STAGES = [(0, 0.003), (12000, 0.005), (24000, 0.010)]
HEAD_POSE_STAGES = [
    (0, (0.05, 0.05, 0.07, 0.015)),
    (12000, (0.17, 0.17, 0.21, 0.047)),
    (24000, (0.39, 0.39, 0.49, 0.11)),
    (36000, (0.72, 0.72, 0.91, 0.20)),
    (48000, (1.1, 1.1, 1.4, 0.31)),
]
BODY_POSE_RANGE = np.array([0.005, 0.005, 0.005, 0.05, 0.05, 0.05])

ACTOR_OBS_NAMES = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command",
                   "head_command", "body_command")


# ── quaternion helpers (w, x, y, z), batched ──────────────────────────────────
def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w = q[..., :1]
    xyz = q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w = q[..., :1]
    xyz = q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v - w * t + np.cross(xyz, t)


def quat_from_angle_axis(angle: np.ndarray, axis: np.ndarray) -> np.ndarray:
    h = 0.5 * angle[:, None]
    return np.concatenate([np.cos(h), axis * np.sin(h)], axis=-1)


def quat_from_euler_xyz(roll, pitch, yaw):
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    return np.stack([cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp,
                     cy * cr * sp + sy * sr * cp, sy * cr * cp - cy * sr * sp], axis=-1)


def quat_mul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def _stage(stages, counter, ge=False):
    val = stages[0][1]
    for step, v in stages:
        if (counter >= step) if ge else (counter > step):
            val = v
    return val


class _DelayBuffer:
    """ML utils/buffers/delay_buffer.py + circular_buffer.py (per-env, hold_prob 0)."""

    def __init__(self, n, dim, min_lag, max_lag, period, rng):
        self.n, self.min_lag, self.max_lag, self.period, self.rng = n, min_lag, max_lag, period, rng
        self.size = max_lag + 1 if max_lag > 0 else 1
        self.buf = np.zeros((self.size, n, dim))
        self.ptr = -1
        self.pushes = np.zeros(n, np.int64)
        self.lags = np.zeros(n, np.int64)
        self.step_count = np.zeros(n, np.int64)
        self.phase = rng.integers(0, period, n) if period > 0 else np.zeros(n, np.int64)

    def reset(self, ids):
        self.pushes[ids] = 0
        self.buf[:, ids] = 0.0
        self.lags[ids] = 0
        self.step_count[ids] = 0
        if self.period > 0:
            self.phase[ids] = self.rng.integers(0, self.period, self.n)[ids]

    def __call__(self, x):
        self.ptr = (self.ptr + 1) % self.size
        fresh = self.pushes == 0
        if fresh.any():
            self.buf[:, fresh] = x[fresh]
        self.buf[self.ptr] = x
        self.pushes += 1
        if self.period > 0:
            upd = (self.step_count + self.phase) % self.period == 0
        else:
            upd = np.ones(self.n, bool)
        cand = self.rng.integers(self.min_lag, self.max_lag + 1, self.n)
        self.lags = np.where(upd, cand, self.lags)
        self.step_count += 1
        cur_len = np.minimum(self.pushes, self.size)
        lag = np.clip(np.minimum(self.lags, cur_len - 1), 0, None)
        idx = (self.ptr - lag) % self.size
        return self.buf[idx, np.arange(self.n)]


@dataclass
class EnvCfg:
    num_envs: int = 4096
    seed: int = 42
    terrain_seed: int = 0
    num_threads: int = 11
    current_limit: bool = True  # SPEC R3: active in BAM 62bd8ce
    # mjlab runs one extra BAM compute for all envs whenever any env reset in the
    # step ("auto"). "always" / "never" force it (diagnostics, small-batch eval).
    extra_bam_pass: str = "auto"
    # Override curricula with a fixed common_step_counter (eval at final stages).
    curriculum_step_override: int | None = None
    # Fixed twist command (eval); None = sample like training.
    # A (3,) command for all envs or an (N, 3) array of per-env commands.
    fixed_command: tuple[float, float, float] | np.ndarray | None = None
    zero_head_body: bool = False  # diagnostics: head and body commands held at 0
    flat_terrain: bool = False  # diagnostics: every tile flat
    obs_noise: float = 1.0  # diagnostics: scales actor observation noise
    soften_terrain: bool = True  # diagnostics: False keeps default terrain contact params
    robot_xml: str | None = None  # diagnostics: alternative robot MJCF
    disable_pushes: bool = False
    # Reward experiments (defaults = Pollen's CFG values).
    track_lin_std: float = 0.2  # was 0.316; tighter so standing still stops paying
    air_time_weight: float = 5.0  # was 3.0; encourage stepping
    action_rate_delay: float = 1.0  # multiplies ACTION_RATE_STAGES step thresholds (2 = ramp twice as late)


class MicroduckEnv:
    def __init__(self, cfg: EnvCfg = EnvCfg()):
        self.cfg = cfg
        N = self.num_envs = cfg.num_envs
        self.rng = np.random.default_rng(cfg.seed)
        sc = build_scene(cfg.terrain_seed, flat=cfg.flat_terrain, soften=cfg.soften_terrain, robot_xml=cfg.robot_xml)
        self.model = m = sc.model
        self.terrain_origins = sc.terrain_origins
        self.b = Batch(m._address, N, cfg.num_threads, cfg.seed, cfg.current_limit)
        self.A = self.b.arrays()
        idx = self.b.indices()
        self.servo_q = np.array(idx["servo_qadr"])
        self.servo_d = np.array(idx["servo_dadr"])
        self.bl_q = np.array(idx["backlash_qadr"])
        self.bl_d = np.array(idx["backlash_dadr"])
        self.trunk = idx["trunk_body"]
        names = [m.joint(m.actuator_trnid[a, 0]).name.split("/", 1)[1] for a in range(m.nu)]
        if tuple(names) != JOINTS:
            raise RuntimeError(f"actuator order {names} != {JOINTS}")
        self.default_servo = np.array([HOME[j] for j in JOINTS])
        self.hinge_q = np.arange(7, m.nq)  # 28 non-free joints (servo + backlash)
        self.hinge_d = np.arange(6, m.nv)
        self.default_hinge = np.zeros(m.nq - 7)
        self.default_hinge[self.servo_q - 7] = self.default_servo
        # Soft joint limits (ML entity/entity.py:685-700), factor 0.9.
        rng_ = np.array([m.jnt_range[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        mid, half = rng_.mean(1), 0.5 * (rng_[:, 1] - rng_[:, 0]) * 0.9
        self.soft_lo, self.soft_hi = mid - half, mid + half
        self.leg = np.array([i for i, j in enumerate(JOINTS) if "neck" not in j and "head" not in j])
        key = lambda j: next(k for k in _POSE_STD_STAND if k in j)
        self.pose_std_stand = np.array([_POSE_STD_STAND[key(JOINTS[i])] for i in self.leg])
        self.pose_std_walk = np.array([_POSE_STD_WALK[key(JOINTS[i])] for i in self.leg])
        self.head = np.array([JOINTS.index(j) for j in HEAD_JOINTS])
        self.head_com_bodies = np.array([m.body("robot/" + n).id for n in HEAD_COM_BODIES])
        self.default_arm = m.dof_armature.copy()
        self.default_ipos = m.body_ipos.copy()
        self.default_mass = m.body_mass.copy()
        self.default_inertia = m.body_inertia.copy()

        # ── terrain origins (ML terrains/terrain_entity.py:321-361); curriculum off ─
        max_init = min(5, TERRAIN_ROWS - 1)
        self.terrain_levels = self.rng.integers(0, max_init + 1, N)
        # proportions has 4 entries != 5 cols → even split by column (terrain_entity.py:350-356)
        self.terrain_types = np.floor(np.arange(N) / (N / TERRAIN_COLS)).astype(np.int64)
        self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types].copy()
        self.bound = max(0.0, 0.5 * TERRAIN_ROWS * TERRAIN_SIZE[0] - 0.3)  # out_of_terrain_bounds margin 0.3

        # ── BAM actuator initialize (BAMJ:335-351): vin, sag gain per env, constant ─
        self.A["vin_env"][:] = self.rng.uniform(6.5, 8.2, N)
        self.A["vin_gain"][:] = self.rng.uniform(0.0, 0.2, N)

        # ── startup events (CFG events mode=startup) ───────────────────────────
        self.encoder_bias = self.rng.uniform(-0.015, 0.015, (N, 14))
        fr = self.rng.uniform(0.7, 1.3, N)  # shared_random over both feet, axis 0, abs
        self.A["foot_friction"][:, :, 0] = fr[:, None]
        alpha = self.rng.uniform(-0.02564664719377529, 0.024395082084716024, N)  # pseudo_inertia, alpha only
        s = np.exp(2.0 * alpha)
        self.A["body_mass"][:, self.trunk] = self.default_mass[self.trunk] * s
        self.A["body_inertia"][:, self.trunk] = self.default_inertia[self.trunk] * s[:, None]
        # base_com: body_names [] → no-op (SPEC §9).
        self.b.recompute_const(np.arange(N))
        # IMU misalignment (MDP:3618-3638): per env, sampled once.
        axis = self.rng.standard_normal((N, 3))
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-8
        ang = self.rng.uniform(0, 1, N) * math.radians(6.0)
        self.imu_q = quat_from_angle_axis(ang, axis)

        # ── manager state ─────────────────────────────────────────────────────
        self.common_step_counter = 0
        self.episode_length = np.zeros(N, np.int64)
        self.action = np.zeros((N, 14))
        self.prev_action = np.zeros((N, 14))
        self.cmd = np.zeros((N, 3))
        self.cmd_time_left = np.zeros(N)
        self.is_standing = np.zeros(N, bool)
        self.head_cmd = np.zeros((N, 4))
        self.head_time_left = np.zeros(N)
        self.body_cmd = np.zeros((N, 6))
        self.body_time_left = np.zeros(N)
        self.push_time_left = np.zeros(N)
        self.rel_standing = STANDING_STAGES[0][1]
        self.head_ranges = np.array(HEAD_POSE_STAGES[0][1])
        self.com_r = COM_STAGES[0][1]
        self.head_com_r = HEAD_COM_STAGES[0][1]
        self.w_action_rate = ACTION_RATE_STAGES[0][1]
        self.w_head_bias = 0.0
        # contact air-time (ML sensor/contact_sensor.py:378-418), float32 like MJWarp
        self.sim_time = np.zeros(N, F32)
        self.last_time = np.zeros(N, F32)
        self.cur_air = np.zeros((N, 2), F32)
        self.last_air = np.zeros((N, 2), F32)
        self.cur_contact = np.zeros((N, 2), F32)
        self.last_contact = np.zeros((N, 2), F32)
        self.peak_heights = np.zeros((N, 2))
        self.head_bias_ema = np.zeros((N, 4))
        self.heights_obs = np.zeros((N, 2))  # last sense()
        # Initial data = mj_forward(qpos0) (ML sim/sim.py:193-195): root at qpos0.
        self.A["S_root_pos"][:] = m.qpos0[:3]
        self.A["S_root_quat"][:] = m.qpos0[3:7]
        d_ = lambda dim, lo, hi, per: _DelayBuffer(N, dim, lo, hi, per, self.rng)
        self.dl_angvel = d_(3, 0, 1, 64)
        self.dl_grav = d_(3, 0, 1, 64)
        self.dl_jvel = d_(14, 1, 1, 0)
        self.reward_terms: dict[str, np.ndarray] = {}
        self.extras: dict = {}

    # ── public API ────────────────────────────────────────────────────────────
    def reset(self):
        """ML envs/manager_based_rl_env.py:351-370."""
        ids = np.arange(self.num_envs)
        self._reset_idx(ids)
        self._write_data_to_sim(np.ones(self.num_envs, np.uint8))
        self.b.forward_sense()
        self._commands_compute(0.0)
        self.heights_obs = self.A["heights"].copy()
        self._contact_obs_src = "F"
        return self._observations()

    def randomize_episode_lengths(self):
        """rsl_rl OnPolicyRunner.learn(init_at_random_ep_len=True) (mjlab train.py:174)."""
        self.episode_length = self.rng.integers(0, MAX_EP_LEN, self.num_envs)

    def step(self, action: np.ndarray):
        A, N = self.A, self.num_envs
        action = np.asarray(action, dtype=np.float64)
        # process_action (ML managers/action_manager.py:152-175); scale 1, default offset, no clip.
        self.prev_action = self.action
        self.action = action.copy()
        A["targets"][:] = self.default_servo + action
        self.b.step()
        for k in range(DECIM):  # scene.update after each substep: air-time tracking
            self.sim_time = (self.sim_time + F32(PHYS_DT)).astype(F32)
            self._air_time_update(A["found_sub"][:, k, :])
        self.episode_length += 1
        self.common_step_counter += 1

        terminated, truncated = self._terminations()
        reset = terminated | truncated
        rew = self._rewards()

        rids = np.nonzero(reset)[0]
        any_reset = len(rids) > 0
        if any_reset:
            self._reset_idx(rids)
        mode = self.cfg.extra_bam_pass
        if mode == "always" or (mode == "auto" and any_reset):
            mask = np.zeros(N, np.uint8)
            mask[rids] = 1
            self._write_data_to_sim(mask)
        self.b.forward_sense()
        self._commands_compute(STEP_DT)
        if not self.cfg.disable_pushes:
            self._push_interval(STEP_DT)
        self.heights_obs = A["heights"].copy()
        # Contact-sensor cache (ML sensor/sensor.py:96-151): computed at termination time
        # from the last mj_step and only invalidated by update()/reset(); a reset of any
        # env invalidates it, so obs see the post-forward sensordata only then.
        self._contact_obs_src = "F" if any_reset else "S"
        obs = self._observations()
        self.extras = {"time_outs": truncated, "terminated": terminated}
        return obs, rew, reset, self.extras

    # ── internals ─────────────────────────────────────────────────────────────
    def _write_data_to_sim(self, reset_mask: np.ndarray):
        self.b.bam_extra(reset_mask)

    def _air_time_update(self, found):
        elapsed = (self.sim_time - self.last_time)[:, None]
        is_c = found > 0
        first_c = (self.cur_air > 0) & is_c
        first_d = (self.cur_contact > 0) & ~is_c
        self.last_air = np.where(first_c, self.cur_air + elapsed, self.last_air).astype(F32)
        self.cur_air = np.where(~is_c, self.cur_air + elapsed, F32(0)).astype(F32)
        self.last_contact = np.where(first_d, self.cur_contact + elapsed, self.last_contact).astype(F32)
        self.cur_contact = np.where(is_c, self.cur_contact + elapsed, F32(0)).astype(F32)
        self.last_time = self.sim_time.copy()

    def _terminations(self):
        A = self.A
        q = A["S_root_quat"]
        g = quat_apply_inverse(q, np.array([0.0, 0.0, -1.0]))
        with np.errstate(invalid="ignore"):
            fell = np.abs(np.arccos(-g[:, 2])) > 1.2217304763960306  # bad_orientation
        qp, qv = A["qpos"], A["qvel"]
        bad = ~np.isfinite(qp[:, 7:]).all(1) | ~np.isfinite(qv[:, 6:]).all(1)
        for k in ("S_root_pos", "S_root_quat", "S_root_linvel_w", "S_root_angvel_w"):
            bad |= ~np.isfinite(A[k]).all(1)
        bad |= ~np.isfinite(A["S_feet_force"].reshape(self.num_envs, -1)).all(1)  # MDP:1000-1010
        bad |= A["nan_flag"] > 0
        xy = A["S_root_pos"][:, :2]
        with np.errstate(invalid="ignore"):
            oob = (np.abs(xy[:, 0]) > self.bound) | (np.abs(xy[:, 1]) > self.bound)
        tout = self.episode_length >= MAX_EP_LEN
        return fell | bad, oob | tout

    def _rewards(self):
        A, N = self.A, self.num_envs
        cmd = self.cmd
        q = A["S_root_quat"]
        v_b = quat_apply_inverse(q, A["S_root_linvel_w"])
        w_b = quat_apply_inverse(q, A["S_root_angvel_w"])
        cmd_norm = np.linalg.norm(cmd[:, :2], axis=1) + np.abs(cmd[:, 2])
        active = (cmd_norm > 0.01).astype(np.float64)
        qp = A["qpos"]
        jp = qp[:, self.servo_q]
        jp_out = jp + qp[:, self.bl_q]
        found = A["S_feet_found"]
        foot_v = np.linalg.norm(A["S_foot_linvel_w"][:, :, :2], axis=-1)
        t = {}
        t["track_linear_velocity"] = (2.0, np.exp(-(np.sum((cmd[:, :2] - v_b[:, :2]) ** 2, 1) + v_b[:, 2] ** 2)
                                                   / self.cfg.track_lin_std ** 2))
        t["track_angular_velocity"] = (2.0, np.exp(-((cmd[:, 2] - w_b[:, 2]) ** 2 + np.sum(w_b[:, :2] ** 2, 1))
                                                    / 0.7071067811865476 ** 2))
        g = quat_apply_inverse(q, np.array([0.0, 0.0, -1.0]))
        t["upright"] = (2.0, np.exp(-np.sum(g[:, :2] ** 2, 1) / 0.22360679774997896 ** 2))
        std = np.where((cmd_norm < 0.01)[:, None], self.pose_std_stand, self.pose_std_walk)
        err = (jp[:, self.leg] - self.default_servo[self.leg]) ** 2
        t["pose"] = (1.0, np.exp(-np.mean(err / std ** 2, 1)))
        in_range = (self.cur_air > 0.125) & (self.cur_air < 0.3)
        t["air_time"] = (self.cfg.air_time_weight, in_range.sum(1).astype(np.float64) * active)
        t["foot_clearance"] = (-2.0, np.sum(np.abs(self.heights_obs - 0.02) * foot_v, 1) * active)
        in_air = found == 0
        self.peak_heights = np.where(in_air, np.maximum(self.peak_heights, self.heights_obs), self.peak_heights)
        first_contact = (self.cur_contact > 0) & (self.cur_contact < STEP_DT + 1e-8)
        e_sw = self.peak_heights / 0.02 - 1.0
        t["foot_swing_height"] = (-0.25, np.sum(e_sw ** 2 * first_contact, 1) * active)
        self.peak_heights = np.where(first_contact, 0.0, self.peak_heights)
        t["foot_slip"] = (-0.1, np.sum(foot_v ** 2 * (found > 0), 1) * active)
        t["body_ang_vel"] = (-0.05, np.sum(A["S_root_angvel_w"][:, :2] ** 2, 1))
        t["angular_momentum"] = (-0.02, np.sum(A["S_angmom"] ** 2, 1))
        out = np.clip(-(jp - self.soft_lo), 0, None) + np.clip(jp - self.soft_hi, 0, None)
        t["dof_pos_limits"] = (-1.0, out.sum(1))
        t["action_rate_l2"] = (self.w_action_rate, np.sum((self.action - self.prev_action) ** 2, 1))
        t["self_collisions"] = (-1.0, A["S_self_found"].copy())
        herr = (jp_out[:, self.head] - self.default_servo[self.head]) - self.head_cmd
        t["head_pose_tracking"] = (2.0, np.mean(np.exp(-(herr / 0.5) ** 2), 1))
        if self.w_head_bias != 0.0:  # weight-0 terms are skipped entirely (reward_manager.py:122)
            self.head_bias_ema[self.episode_length <= 1] = 0.0
            a = min(1.0, STEP_DT / 1.0)
            self.head_bias_ema = (1 - a) * self.head_bias_ema + a * herr
            t["head_pose_bias"] = (self.w_head_bias, -np.mean(np.abs(self.head_bias_ema), 1))
        rew = np.zeros(N)
        self.reward_terms = {}
        for k, (w, f) in t.items():
            if w == 0.0:
                continue
            v = np.nan_to_num(f * w * STEP_DT, nan=0.0, posinf=0.0, neginf=0.0)
            rew += v
            self.reward_terms[k] = v
        return rew

    def _curricula(self, ids):
        c = self.common_step_counter if self.cfg.curriculum_step_override is None else self.cfg.curriculum_step_override
        self.w_action_rate = _stage([(s * self.cfg.action_rate_delay, v) for s, v in ACTION_RATE_STAGES], c)
        self.w_head_bias = _stage(HEAD_BIAS_STAGES, c)
        self.rel_standing = _stage(STANDING_STAGES, c)
        self.com_r = _stage(COM_STAGES, c)
        self.head_com_r = _stage(HEAD_COM_STAGES, c)
        self.head_ranges = np.array(_stage(HEAD_POSE_STAGES, c, ge=True))
        # terrain_levels_vel (ML tasks/velocity/mdp/curriculums.py:25-58)
        dist = np.linalg.norm(self.A["S_root_pos"][ids, :2] - self.env_origins[ids, :2], axis=1)
        up = dist > TERRAIN_SIZE[0] / 2
        down = dist < np.linalg.norm(self.cmd[ids, :2], axis=1) * (MAX_EP_LEN * STEP_DT) * 0.5
        down &= ~up
        lv = self.terrain_levels[ids] + up.astype(np.int64) - down.astype(np.int64)
        lv = np.where(lv >= TERRAIN_ROWS, self.rng.integers(0, TERRAIN_ROWS, len(ids)), np.clip(lv, 0, None))
        self.terrain_levels[ids] = lv
        self.env_origins[ids] = self.terrain_origins[lv, self.terrain_types[ids]]

    def _reset_idx(self, ids):
        """ML envs/manager_based_rl_env.py:543-582."""
        A, n = self.A, len(ids)
        self._curricula(ids)
        # sim.reset → mjwarp reset_data: qpos0, zero vel/ctrl/warmstart/time.
        A["qpos"][ids] = self.model.qpos0
        A["qvel"][ids] = 0.0
        A["qacc_warmstart"][ids] = 0.0
        A["ctrl"][ids] = 0.0
        A["time"][ids] = 0.0
        self.sim_time[ids] = 0.0
        # scene.reset: actuator delay + prev torque; contact air-time state.
        self.b.reset_actuators(ids)
        for arr in (self.cur_air, self.last_air, self.cur_contact, self.last_contact):
            arr[ids] = 0.0
        self.last_time[ids] = self.sim_time[ids]
        # reset events
        r = self.rng
        pos = self.env_origins[ids] + np.c_[r.uniform(-0.5, 0.5, n), r.uniform(-0.5, 0.5, n), r.uniform(0.12, 0.13, n)]
        yaw = r.uniform(-3.14, 3.14, n)
        quat = quat_mul(np.array([1.0, 0, 0, 0]), quat_from_euler_xyz(np.zeros(n), np.zeros(n), yaw))
        A["qpos"][ids, 0:3] = pos
        A["qpos"][ids, 3:7] = quat
        A["qvel"][ids, 0:6] = 0.0
        A["qpos"][ids, 7:] = self.default_hinge  # reset_joints_by_offset, range 0 (within soft limits)
        A["qvel"][ids, 6:] = 0.0
        A["dof_armature"][ids[:, None], self.hinge_d] = self.default_arm[self.hinge_d] * r.uniform(0.9, 1.1, (n, len(self.hinge_d)))
        A["body_ipos"][ids, self.trunk] = self.default_ipos[self.trunk] + r.uniform(-self.com_r, self.com_r, (n, 3))
        hb = self.head_com_bodies
        A["body_ipos"][ids[:, None], hb] = self.default_ipos[hb] + r.uniform(-self.head_com_r, self.head_com_r, (n, len(hb), 3))
        A["friction_scale"][ids] = r.uniform(0.9, 1.1, n)
        self.b.recompute_const(ids)
        # observation manager reset (delay buffers)
        for dl in (self.dl_angvel, self.dl_grav, self.dl_jvel):
            dl.reset(ids)
        self.action[ids] = 0.0
        self.prev_action[ids] = 0.0
        # command manager reset: resample all terms
        self._resample_twist(ids)
        self._resample_head(ids)
        self._resample_body(ids)
        # event manager reset: interval time
        self.push_time_left[ids] = r.uniform(3.0, 6.0, n)
        self.episode_length[ids] = 0

    def _resample_twist(self, ids):
        """VelocityCommandCommandOnly._resample_command (MDP:4505-4533; velocity_command.py:75-114)."""
        r, n = self.rng, len(ids)
        if n == 0:
            return
        self.cmd_time_left[ids] = r.uniform(3.0, 8.0, n)
        if self.cfg.fixed_command is not None:
            fc = np.asarray(self.cfg.fixed_command, dtype=np.float64)
            self.cmd[ids] = fc[ids] if fc.ndim == 2 else fc
            self.is_standing[ids] = False
            return
        c = np.c_[r.uniform(-0.4, 0.4, n), r.uniform(-0.3, 0.3, n), r.uniform(-1.0, 1.0, n)]
        r.uniform(-math.pi, math.pi, n)  # heading target (unused, rel_heading_envs 0)
        r.uniform(0, 1, n)  # is_heading_env draw
        standing = r.uniform(0, 1, n) <= self.rel_standing
        r.uniform(0, 1, n)  # is_world_env draw (prob 0)
        fwd = r.uniform(0, 1, n) <= 0.2
        c[fwd, 0] = np.maximum(np.abs(c[fwd, 0]), 0.3)
        c[fwd, 1:] = 0.0
        turn = r.uniform(0, 1, n) < 0.15
        nt = int(turn.sum())
        if nt:
            sign = np.where(r.uniform(0, 1, nt) < 0.5, -1.0, 1.0)
            c[turn, 0:2] = 0.0
            c[turn, 2] = sign * r.uniform(0.4, 1.0, nt)
            standing[turn] = False
        self.cmd[ids] = c
        self.is_standing[ids] = standing

    def _resample_head(self, ids):
        n = len(ids)
        self.head_time_left[ids] = self.rng.uniform(2.0, 5.0, n)
        self.head_cmd[ids] = self.rng.uniform(-self.head_ranges, self.head_ranges, (n, 4))
        if self.cfg.zero_head_body:
            self.head_cmd[ids] = 0.0

    def _resample_body(self, ids):
        n = len(ids)
        self.body_time_left[ids] = self.rng.uniform(2.0, 5.0, n)
        self.body_cmd[ids] = self.rng.uniform(-BODY_POSE_RANGE, BODY_POSE_RANGE, (n, 6))
        if self.cfg.zero_head_body:
            self.body_cmd[ids] = 0.0

    def _commands_compute(self, dt):
        """ML managers/command_manager.py:109-115 per term."""
        self.cmd_time_left -= dt
        self._resample_twist(np.nonzero(self.cmd_time_left <= 0.0)[0])
        self.cmd[self.is_standing] = 0.0  # _update_command: standing envs
        self.head_time_left -= dt
        self._resample_head(np.nonzero(self.head_time_left <= 0.0)[0])
        self.body_time_left -= dt
        self._resample_body(np.nonzero(self.body_time_left <= 0.0)[0])

    def _push_interval(self, dt):
        """ML managers/event_manager.py:264-287 + events.py:321-336 (push_by_setting_velocity)."""
        self.push_time_left -= dt
        ids = np.nonzero(self.push_time_left < 1e-6)[0]
        if len(ids) == 0:
            return
        self.push_time_left[ids] = self.rng.uniform(3.0, 6.0, len(ids))
        A = self.A
        lin = A["F_root_linvel_w"][ids].copy()
        ang = A["F_root_angvel_w"][ids].copy()
        lin[:, 0] += self.rng.uniform(-0.3, 0.3, len(ids))
        lin[:, 1] += self.rng.uniform(-0.3, 0.3, len(ids))
        quat = A["qpos"][ids, 3:7]
        A["qvel"][ids, 0:3] = lin
        A["qvel"][ids, 3:6] = quat_apply_inverse(quat, ang)

    def _observations(self):
        A, N = self.A, self.num_envs
        r = self.rng
        q = A["F_root_quat"]
        ang_b = quat_apply_inverse(q, A["F_root_angvel_w"])
        lin_b = quat_apply_inverse(q, A["F_root_linvel_w"])
        g_b = quat_apply_inverse(q, np.array([0.0, 0.0, -1.0]))
        qp, qv = A["qpos"], A["qvel"]
        jpos_out = qp[:, self.servo_q] + qp[:, self.bl_q] - self.default_servo
        jvel = qv[:, self.servo_d] + qv[:, self.bl_d]
        # actor (noise → delay; ML managers/observation_manager.py:330-350)
        ns = self.cfg.obs_noise
        a_ang = self.dl_angvel(quat_apply(self.imu_q, ang_b) + ns * r.uniform(-0.03, 0.03, (N, 3)))
        a_grav = self.dl_grav(quat_apply(self.imu_q, g_b) + ns * r.uniform(-0.01, 0.01, (N, 3)))
        a_jpos = jpos_out + self.encoder_bias + ns * r.uniform(-0.001, 0.001, (N, 14))
        a_jvel = self.dl_jvel(jvel + ns * r.uniform(-0.25, 0.25, (N, 14)))
        actor = np.concatenate([a_ang, a_grav, a_jpos, a_jvel, self.action, self.cmd, self.head_cmd,
                                self.body_cmd], axis=1)
        src = self._contact_obs_src
        found = A[f"{src}_feet_found"]
        force = A[f"{src}_feet_force"].reshape(N, 6)
        fin = lambda x: np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        critic = np.concatenate([
            lin_b, A["F_gyro"], g_b, jpos_out, jvel, self.action, self.cmd,
            fin(self.heights_obs), fin(self.cur_air.astype(np.float64)), (found > 0).astype(np.float64),
            fin(np.sign(force) * np.log1p(np.abs(force))), self.head_cmd, self.body_cmd,
        ], axis=1)
        return actor.astype(np.float32), critic.astype(np.float32)
