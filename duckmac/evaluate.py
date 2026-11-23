"""Step 3 parity gate: run a policy in the CPU env and report tracking error and falls/min.

Metrics (per env-step, 50 Hz), on the training command distribution with curricula at
their final stage (curriculum_step_override), plus fixed-command probes:
  lin_err   = ||c_xy - v_xy||   (root link, body frame, m/s)
  yaw_err   = |c_z - w_z|       (rad/s)
  falls/min = terminated (fell_over | nan) per simulated env-minute
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from duckmac.env import EnvCfg, MicroduckEnv, quat_apply_inverse
from duckmac.onnx_mlp import NumpyOnnxMLP, OnnxPolicy  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


class ZeroPolicy:
    def __call__(self, obs):
        return np.zeros((obs.shape[0], 14), np.float32)


def run(policy, cfg: EnvCfg, seconds: float, warmup_s: float = 0.0, label: str = "") -> dict:
    env = MicroduckEnv(cfg)
    obs, _ = env.reset()
    env.randomize_episode_lengths() if cfg.fixed_command is None else None
    n_steps = int(seconds / 0.02)
    n_warm = int(warmup_s / 0.02)
    N = cfg.num_envs
    lin_err, yaw_err, moving = [], [], []
    falls = 0
    timeouts = 0
    t0 = time.time()
    t_pol = 0.0
    for i in range(n_warm + n_steps):
        cmd_before = env.cmd.copy()  # the command the rewards of this step see
        tp = time.time()
        act = policy(obs)
        t_pol += time.time() - tp
        (obs, _), rew, done, ex = env.step(act)
        if i < n_warm:
            continue
        A = env.A
        q = A["S_root_quat"]
        v_b = quat_apply_inverse(q, A["S_root_linvel_w"])
        w_b = quat_apply_inverse(q, A["S_root_angvel_w"])
        c = cmd_before
        ok = ~done | ex["time_outs"]
        lin_err.append(np.where(ok, np.linalg.norm(c[:, :2] - v_b[:, :2], axis=1), np.nan))
        yaw_err.append(np.where(ok, np.abs(c[:, 2] - w_b[:, 2]), np.nan))
        moving.append(np.linalg.norm(c[:, :2], axis=1) + np.abs(c[:, 2]) > 0.01)
        falls += int(ex["terminated"].sum())
        timeouts += int(ex["time_outs"].sum())
    wall = time.time() - t0
    lin = np.array(lin_err)
    yaw = np.array(yaw_err)
    mov = np.array(moving)
    env_minutes = N * n_steps * 0.02 / 60.0
    res = {
        "label": label,
        "num_envs": N,
        "seconds": seconds,
        "current_limit": cfg.current_limit,
        "fixed_command": cfg.fixed_command,
        "lin_err_mean": float(np.nanmean(lin)),
        "lin_err_moving": float(np.nanmean(np.where(mov, lin, np.nan))),
        "yaw_err_mean": float(np.nanmean(yaw)),
        "yaw_err_moving": float(np.nanmean(np.where(mov, yaw, np.nan))),
        "falls_per_min": falls / env_minutes,
        "falls": falls,
        "env_minutes": env_minutes,
        "wall_s": wall,
        "policy_s": t_pol,
        "env_steps_per_s": N * (n_warm + n_steps) / wall,
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(ROOT / "upstream/policies/alpha_walking.onnx"))
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--threads", type=int, default=11)
    ap.add_argument("--current-limit", choices=["on", "off", "both"], default="both")
    ap.add_argument("--zero", action="store_true", help="zero-action baseline")
    ap.add_argument("--probes", action="store_true", help="also run fixed-command probes")
    ap.add_argument("--out", default=str(ROOT / "reports/parity.json"))
    args = ap.parse_args()
    pol = ZeroPolicy() if args.zero else NumpyOnnxMLP(args.policy)
    lims = {"on": [True], "off": [False], "both": [True, False]}[args.current_limit]
    results = []
    for lim in lims:
        base = dict(num_envs=args.envs, num_threads=args.threads, seed=7, current_limit=lim,
                    curriculum_step_override=10**9)
        r = run(pol, EnvCfg(**base), args.seconds, label=f"train-dist limiter={'on' if lim else 'off'}")
        print(json.dumps(r))
        results.append(r)
        if args.probes:
            for cmd in [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (-0.3, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.8)]:
                r = run(pol, EnvCfg(**base, fixed_command=cmd), min(args.seconds, 20.0), warmup_s=2.0,
                        label=f"probe {cmd} limiter={'on' if lim else 'off'}")
                print(json.dumps(r))
                results.append(r)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
