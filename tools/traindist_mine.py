"""Same measurement as tools/ref_traindist_mjlab.py, in duckmac."""
import json, sys
import numpy as np
from duckmac.env import EnvCfg, MicroduckEnv, quat_apply_inverse
from duckmac.onnx_mlp import NumpyOnnxMLP

N = int(sys.argv[1]); seconds = float(sys.argv[2]); out = sys.argv[3]; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 7
pol = NumpyOnnxMLP("upstream/policies/alpha_walking.onnx")
env = MicroduckEnv(EnvCfg(num_envs=N, seed=seed, num_threads=11, curriculum_step_override=10**9))
actor, critic = env.reset()
rs = {}; lin = yaw = 0.0; c = 0; falls = touts = 0
so = {k: [0.0, 0.0] for k in ("actor", "critic")}
n = int(seconds / 0.02)
for k in range(n):
    cb = env.cmd.copy()
    (actor, critic), _, done, ex = env.step(pol(actor))
    ok = ~done; falls += int(ex["terminated"].sum()); touts += int(ex["time_outs"].sum())
    for t, v in env.reward_terms.items():
        rs[t] = rs.get(t, 0.0) + float(v.mean()) / 0.02
    q = env.A["F_root_quat"]; v = quat_apply_inverse(q, env.A["F_root_linvel_w"]); w = quat_apply_inverse(q, env.A["F_root_angvel_w"])
    lin += np.linalg.norm(cb[ok, :2] - v[ok, :2], axis=1).sum(); yaw += np.abs(cb[ok, 2] - w[ok, 2]).sum(); c += ok.sum()
    for g, x in (("actor", actor), ("critic", critic)):
        x = x.astype(np.float64); so[g][0] = so[g][0] + x.sum(0); so[g][1] = so[g][1] + (x * x).sum(0)
T = n * N
res = dict(num_envs=N, seconds=seconds, falls=falls, timeouts=touts, falls_per_min=falls / (T * 0.02 / 60), lin_err=lin / c, yaw_err=yaw / c,
           rewards={k: v / n for k, v in rs.items()},
           obs={k: dict(mean=(s[0] / T).tolist(), std=np.sqrt(np.maximum(s[1] / T - (s[0] / T) ** 2, 0)).tolist()) for k, s in so.items()})
json.dump(res, open(out, "w"), indent=1)
