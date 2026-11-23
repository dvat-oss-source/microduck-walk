"""OFFLINE reference only (.venv-ref): the parity_yaw sweep in Pollen's real mjlab env on Warp CPU.

Same protocol as duckmac.parity_yaw.sweep: flat tiles, head/body commands 0, no pushes,
curricula at final stage, fixed per-env twist, mean body rates over upright envs after 2 s.
"""
import json, sys, time
sys.path.insert(0, "upstream/rl_cb8561f6/src")
import numpy as np, torch
import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from duckmac.onnx_mlp import NumpyOnnxMLP
from duckmac.yaw_table import LIN, NEG, POS, TARGETS

per_cell = int(sys.argv[1]); seconds = float(sys.argv[2]); out = sys.argv[3]
flat = (sys.argv[4] if len(sys.argv) > 4 else "flat") == "flat"
pol = NumpyOnnxMLP("upstream/policies/alpha_walking.onnx")
cells = [(s, i, j) for s in (1, -1) for i in range(5) for j in range(10)]
cmds = np.array([[LIN[i], 0.0, s * (POS if s > 0 else NEG)[i][j]] for s, i, j in cells])
cmd = np.concatenate([np.repeat(cmds, per_cell, 0), np.repeat(np.array([[v, 0, 0.] for v in (.2, .3, .4)]), per_cell, 0)])
N = len(cmd); CMD = torch.tensor(cmd, dtype=torch.float32)

cfg = load_env_cfg("Mjlab-Velocity2-Rough-Backlash-MicroDuck")
cfg.scene.num_envs = N
g = cfg.scene.terrain.terrain_generator
g.num_rows = g.num_cols = 5; g.curriculum = False; g.seed = 0
if flat:
    g.sub_terrains = {"flat": g.sub_terrains["flat"]}
cfg.events.pop("push_robot")
env = ManagerBasedRlEnv(cfg, device="cpu")
env.common_step_counter = 10**9
tw = env.command_manager.get_term("twist")
def rs(ids):
    tw.vel_command_b[ids] = CMD[ids]; tw.is_standing_env[ids] = False; tw.is_forward_env[ids] = False
tw._resample_command = rs
for n in ("head_pose", "body_pose"):
    env.command_manager.get_term(n)._resample_command = lambda ids: None
obs, _ = env.reset()
robot = env.scene["robot"]
acc = np.zeros((N, 2)); cnt = np.zeros(N); alive = np.ones(N, bool); falls = 0
t0 = time.time(); n = int(seconds / 0.02)
for k in range(n):
    with torch.no_grad():
        a = torch.from_numpy(pol(obs["actor"].numpy()))
    obs, _, term, trunc, _ = env.step(a)
    # After env.step derived fields are fresh (forward ran); non-reset envs only are counted.
    tm = term.numpy(); falls += int(tm.sum()); alive &= ~tm
    if k >= 2.0 / 0.02:
        w = robot.data.root_link_ang_vel_b.numpy(); v = robot.data.root_link_lin_vel_b.numpy()
        acc[alive, 0] += w[alive, 2]; acc[alive, 1] += v[alive, 0]; cnt[alive] += 1
    if k % 50 == 0:
        print(k, round(time.time() - t0, 1), "falls", falls, flush=True)
mean = acc / np.maximum(cnt, 1)[:, None]
nc = len(cells)
wz = mean[: nc * per_cell, 0].reshape(nc, per_cell).mean(1)
tgt = np.array([s * TARGETS[j] for s, i, j in cells])
li = np.array([i for _, i, _ in cells])
err = wz - tgt
res = dict(label=f"mjlab warp-cpu {'flat' if flat else 'rough'}", per_cell=per_cell, num_envs=N, falls=falls, wall_s=time.time() - t0,
           rmse_all=float(np.sqrt((err**2).mean())), bias_all=float((wz*np.sign(tgt)-abs(tgt)).mean()), corr=float(np.corrcoef(wz, tgt)[0, 1]),
           by_lin={f"lin={LIN[i]}": float((wz*np.sign(tgt)-abs(tgt))[li == i].mean()) for i in range(5)},
           straight=mean[nc*per_cell:, 1].reshape(3, per_cell).mean(1).tolist(), achieved=wz.tolist(), target=tgt.tolist())
json.dump(res, open(out, "w"), indent=1)
print({k: v for k, v in res.items() if k not in ("achieved", "target")}, flush=True)
