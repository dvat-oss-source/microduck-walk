"""Parity against Pollen's own measurements of alpha_walking (run 441tzs6d @ 3750).

Reference: ANCHOR_YAW_REMAP in microduck_velstand_env_cfg.py:382-399 (develop): feeding
|wz| = table[lin][target] makes the policy turn at `target` while walking at `lin`,
measured in mjlab/MJWarp with backlash, 8 envs per cell, 0 falls. Its calibration script
is not in the repo, so its exact protocol (terrain, DR, head commands) is assumed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from duckmac.env import EnvCfg, MicroduckEnv, quat_apply_inverse
from duckmac.onnx_mlp import NumpyOnnxMLP
from duckmac.yaw_table import LIN, NEG, POS, TARGETS

ROOT = Path(__file__).resolve().parents[1]
def sweep(policy, per_cell=16, seconds=10.0, settle=2.0, post=None, **cfg_kw):
    cells = [(sgn, i, j) for sgn in (1, -1) for i in range(5) for j in range(10)]
    cmds = np.array([[LIN[i], 0.0, sgn * (POS if sgn > 0 else NEG)[i][j]] for sgn, i, j in cells])
    cmd = np.repeat(cmds, per_cell, axis=0)
    # Extra probes: straight walking at 0.2/0.3/0.4 m/s (doc: 0.4 -> 0.17).
    extra = np.array([[v, 0.0, 0.0] for v in (0.2, 0.3, 0.4)])
    cmd = np.concatenate([cmd, np.repeat(extra, per_cell, axis=0)])
    N = len(cmd)
    env = MicroduckEnv(EnvCfg(num_envs=N, fixed_command=cmd, curriculum_step_override=10**9,
                              disable_pushes=True, **cfg_kw))
    obs, _ = env.reset()
    if post is not None:
        post(env)
    acc = np.zeros((N, 2))
    cnt = np.zeros(N)
    falls = 0
    alive = np.ones(N, bool)
    n = int(seconds / 0.02)
    for k in range(n):
        (obs, _), _, _, ex = env.step(policy(obs))
        falls += int(ex["terminated"].sum())
        alive &= ~ex["terminated"]
        if k >= settle / 0.02:
            q = env.A["S_root_quat"]
            v = quat_apply_inverse(q, env.A["S_root_linvel_w"])
            w = quat_apply_inverse(q, env.A["S_root_angvel_w"])
            acc[alive, 0] += w[alive, 2]
            acc[alive, 1] += v[alive, 0]
            cnt[alive] += 1
    mean = acc / np.maximum(cnt, 1)[:, None]
    nc = len(cells)
    wz = mean[: nc * per_cell, 0].reshape(nc, per_cell).mean(1)
    tgt = np.array([sgn * TARGETS[j] for sgn, i, j in cells])
    lin_idx = np.array([i for _, i, _ in cells])
    straight = mean[nc * per_cell:, 1].reshape(3, per_cell).mean(1)
    err = wz - tgt
    rows = {f"lin={LIN[i]}": {"rmse": float(np.sqrt(np.mean(err[lin_idx == i] ** 2))),
                              "bias": float(np.mean((wz * np.sign(tgt) - np.abs(tgt))[lin_idx == i]))} for i in range(5)}
    return {"cfg": {k: v for k, v in cfg_kw.items()}, "falls": falls, "num_envs": N,
            "rmse_all": float(np.sqrt(np.mean(err ** 2))), "bias_all": float(np.mean(wz * np.sign(tgt) - np.abs(tgt))),
            "corr": float(np.corrcoef(wz, tgt)[0, 1]), "by_lin": rows,
            "straight_vx_0.2_0.3_0.4": straight.tolist(), "achieved": wz.tolist(), "target": tgt.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(ROOT / "upstream/policies/alpha_walking.onnx"))
    ap.add_argument("--out", default=str(ROOT / "reports/parity_yaw.json"))
    a = ap.parse_args()
    pol = NumpyOnnxMLP(a.policy)
    out = []
    for name, kw in [("flat, head 0, limiter on", dict(flat_terrain=True, zero_head_body=True, current_limit=True)),
                     ("flat, head 0, limiter off", dict(flat_terrain=True, zero_head_body=True, current_limit=False)),
                     ("rough, head 0, limiter on", dict(flat_terrain=False, zero_head_body=True, current_limit=True))]:
        r = sweep(pol, seed=11, **kw)
        r["label"] = name
        out.append(r)
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                     if k in ("falls", "rmse_all", "bias_all", "corr")},
              "straight", np.round(r["straight_vx_0.2_0.3_0.4"], 3),
              {k: round(v["rmse"], 3) for k, v in r["by_lin"].items()})
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
