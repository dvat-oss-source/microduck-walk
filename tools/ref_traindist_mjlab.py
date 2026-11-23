"""OFFLINE reference only (.venv-ref): alpha_walking in real mjlab, training distribution, final curricula."""
import json, sys, time
sys.path.insert(0, "upstream/rl_cb8561f6/src")
import numpy as np, torch
import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from duckmac.onnx_mlp import NumpyOnnxMLP

N = int(sys.argv[1]); seconds = float(sys.argv[2]); out = sys.argv[3]
pol = NumpyOnnxMLP("upstream/policies/alpha_walking.onnx")
cfg = load_env_cfg("Mjlab-Velocity2-Rough-Backlash-MicroDuck")
cfg.scene.num_envs = N
g = cfg.scene.terrain.terrain_generator
g.num_rows = g.num_cols = 5; g.curriculum = False; g.seed = 0
env = ManagerBasedRlEnv(cfg, device="cpu")
env.common_step_counter = 10**9
obs, _ = env.reset()
tw = env.command_manager.get_term("twist"); robot = env.scene["robot"]; rm = env.reward_manager
names = list(rm._term_names)
rs = np.zeros(len(names)); lin = yaw = 0.0; c = 0; falls = touts = 0
so = {k: [0.0, 0.0] for k in ("actor", "critic")}
t0 = time.time(); n = int(seconds / 0.02)
for k in range(n):
    cb = tw.command.clone().numpy()
    a = torch.from_numpy(pol(obs["actor"].numpy()))
    obs, _, term, trunc, _ = env.step(a)
    done = (term | trunc).numpy(); ok = ~done
    falls += int(term.sum()); touts += int(trunc.sum())
    rs += rm._step_reward.numpy().mean(0)
    v = robot.data.root_link_lin_vel_b.numpy(); w = robot.data.root_link_ang_vel_b.numpy()
    lin += np.linalg.norm(cb[ok, :2] - v[ok, :2], axis=1).sum(); yaw += np.abs(cb[ok, 2] - w[ok, 2]).sum(); c += ok.sum()
    for g_ in so:
        x = obs[g_].numpy().astype(np.float64); so[g_][0] = so[g_][0] + x.sum(0); so[g_][1] = so[g_][1] + (x * x).sum(0)
    if k % 100 == 0: print(k, round(time.time() - t0), "falls", falls, flush=True)
T = n * N
res = dict(num_envs=N, seconds=seconds, falls=falls, timeouts=touts, falls_per_min=falls / (T * 0.02 / 60), lin_err=lin / c, yaw_err=yaw / c,
           rewards=dict(zip(names, (rs / n).tolist())), wall_s=time.time() - t0,
           obs={k: dict(mean=(s[0] / T).tolist(), std=np.sqrt(np.maximum(s[1] / T - (s[0] / T) ** 2, 0)).tolist()) for k, s in so.items()})
json.dump(res, open(out, "w"), indent=1); print("DONE", flush=True)
