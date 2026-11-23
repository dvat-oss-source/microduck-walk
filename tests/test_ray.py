import numpy as np, mujoco, time
from duckmac.scene import build_scene
from duckmac._batch import Batch
s = build_scene(0); m = s.model
b = Batch(m._address, 8, 2, 0, True)
print(b.indices())
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
rng = np.random.default_rng(0)
P = np.c_[rng.uniform(-21, 21, 20000), rng.uniform(-21, 21, 20000), rng.uniform(-0.05, 0.4, 20000)]
t = time.time(); dist, nrm = b.ray_down(P); tg = time.time() - t
gg = np.zeros(6, np.uint8); gg[0] = 1
ref = np.zeros(len(P)); refn = np.zeros((len(P), 3)); gid = np.zeros(1, np.int32); n3 = np.zeros(3)
t = time.time()
for i, p in enumerate(P):
    ref[i] = mujoco.mj_ray(m, d, p, np.array([0, 0, -1.0]), gg, 1, -1, gid, n3); refn[i] = n3
tr = time.time() - t
print(f"grid {tg/len(P)*1e6:.2f} us/ray, mj_ray {tr/len(P)*1e6:.2f} us/ray")
print("max |dist diff|", np.abs(dist - ref).max(), "max |normal diff| (hits)", np.abs(nrm - refn)[ref >= 0].max(),
      "misses", (ref < 0).sum(), "mismatch hit/miss", ((ref < 0) != (dist < 0)).sum())
