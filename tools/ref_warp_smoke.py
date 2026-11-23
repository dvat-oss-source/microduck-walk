"""OFFLINE reference only (.venv-ref): step Pollen's real mjlab task on Warp CPU."""
import sys, time
sys.path.insert(0, "upstream/rl_cb8561f6/src")
t0 = time.time()
import torch
import mjlab_microduck.tasks  # registers tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
print("imports", round(time.time() - t0, 1), flush=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
cfg = load_env_cfg("Mjlab-Velocity2-Rough-Backlash-MicroDuck")
cfg.scene.num_envs = N
g = cfg.scene.terrain.terrain_generator
print("terrain", g.num_rows, g.num_cols, "curriculum", g.curriculum, flush=True)
t = time.time()
env = ManagerBasedRlEnv(cfg, device="cpu")
print("env built", round(time.time() - t, 1), flush=True)
t = time.time(); obs, _ = env.reset(); print("reset", round(time.time() - t, 1), {k: tuple(v.shape) for k, v in obs.items()}, flush=True)
a = torch.zeros(N, 14)
for i in range(6):
    t = time.time(); env.step(a); print("step", i, round(time.time() - t, 2), flush=True)
