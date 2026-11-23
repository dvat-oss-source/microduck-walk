"""Watch a policy walk in a live 3D window, on the same physics as training.

    PYTHONPATH=. .venv/bin/mjpython -m duckmac.play                       # Pollen's alpha_walking
    PYTHONPATH=. .venv/bin/mjpython -m duckmac.play --policy runs/main/model_3750.pt --rough

Keys (click the window first): up/down = forward speed, left/right = turn,
comma/period = sidestep, enter = stop. Mouse: drag to orbit, scroll to zoom.
macOS needs mjpython for the window. --headless runs the same loop without one.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from duckmac.env import EnvCfg, MicroduckEnv, quat_apply_inverse
from duckmac.onnx_mlp import NumpyOnnxMLP

ROOT = Path(__file__).resolve().parents[1]
K_ENTER, K_RIGHT, K_LEFT, K_DOWN, K_UP, K_COMMA, K_PERIOD = 257, 262, 263, 264, 265, 44, 46
LIMITS = np.array([0.4, 0.3, 1.0])  # training command ranges (SPEC §6)


class CheckpointPolicy:
    """rsl_rl checkpoint: deterministic actor mean, (x - mean) / (std + 0.01) -> MLP with ELU."""

    def __init__(self, path):
        import torch

        sd = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
        g = lambda k: sd[k].double().numpy()
        self.mean, self.std = g("obs_normalizer._mean"), g("obs_normalizer._std")
        self.layers = [(g(f"mlp.{i}.weight"), g(f"mlp.{i}.bias")) for i in (0, 2, 4, 6)]

    def __call__(self, obs):
        x = (obs.astype(np.float64) - self.mean) / (self.std + 0.01)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < 3:
                x = np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1)
        return x.astype(np.float32)


def load_policy(path):
    return CheckpointPolicy(path) if str(path).endswith(".pt") else NumpyOnnxMLP(path)


class Session:
    def __init__(self, a):
        self.policy = load_policy(a.policy)
        self.cmd = np.array([a.vx, a.vy, a.wz], dtype=np.float64)
        self.env = MicroduckEnv(EnvCfg(
            num_envs=1, num_threads=1, seed=a.seed, fixed_command=self.cmd, zero_head_body=True,
            flat_terrain=not a.rough, disable_pushes=True, curriculum_step_override=10**9))
        self.obs, _ = self.env.reset()
        self.model = self.env.model
        self.data = mujoco.MjData(self.model)
        self.falls = 0

    def key(self, code):
        step = {K_UP: (0, 0.05), K_DOWN: (0, -0.05), K_COMMA: (1, 0.05), K_PERIOD: (1, -0.05),
                K_LEFT: (2, 0.1), K_RIGHT: (2, -0.1)}.get(code)
        if code == K_ENTER:
            self.cmd[:] = 0
        elif step:
            self.cmd[step[0]] = np.clip(self.cmd[step[0]] + step[1], -LIMITS[step[0]], LIMITS[step[0]])

    def step(self):
        env = self.env
        env.cmd[0] = self.cmd
        env.episode_length[:] = 0  # no 20 s time-out teleports while watching
        (self.obs, _), _, _, ex = env.step(self.policy(self.obs))
        self.falls += int(ex["terminated"].sum())

    def sync(self):
        self.data.qpos[:] = self.env.A["qpos"][0]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # the tracking camera follows subtree_com

    def velocity(self):
        A = self.env.A
        q = A["F_root_quat"]
        return quat_apply_inverse(q, A["F_root_linvel_w"])[0], quat_apply_inverse(q, A["F_root_angvel_w"])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(ROOT / "upstream/policies/alpha_walking.onnx"))
    ap.add_argument("--rough", action="store_true", help="training terrain instead of flat")
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=3)  # each seed is a different robot; 3 walks straight
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long; 0 = until the window closes")
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args()
    s = Session(a)

    if a.headless:
        vs = []
        for k in range(int((a.seconds or 10) / 0.02)):
            s.step()
            s.sync()
            if k > 100:
                v, w = s.velocity()
                vs.append((v[0], v[1], w[2]))
        print("mean vx vy wz", np.round(np.mean(vs, 0), 3), "falls", s.falls)
        return

    import mujoco.viewer

    print(__doc__)
    trunk = s.model.body("robot/trunk_base").id
    with mujoco.viewer.launch_passive(s.model, s.data, key_callback=s.key, show_left_ui=False, show_right_ui=False) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = trunk
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 0.7, -15, 135
        t0 = time.perf_counter()
        k = 0
        while v.is_running() and (not a.seconds or k * 0.02 < a.seconds):
            s.step()
            with v.lock():
                s.sync()
            v.sync()
            k += 1
            if k % 25 == 0:
                vel, w = s.velocity()
                print(f"\rcommand vx {s.cmd[0]:+.2f} vy {s.cmd[1]:+.2f} turn {s.cmd[2]:+.2f} | "
                      f"actual vx {vel[0]:+.2f} vy {vel[1]:+.2f} turn {w[2]:+.2f} | falls {s.falls}   ", end="", flush=True)
            lag = t0 + k * 0.02 - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
        print()


if __name__ == "__main__":
    main()
