"""OFFLINE (.venv-ref): log foot landings (peak height, first contact) in real mjlab."""
import json, sys
sys.path.insert(0, "upstream/rl_cb8561f6/src")
import numpy as np, torch
import mjlab_microduck.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from duckmac.onnx_mlp import NumpyOnnxMLP
N=int(sys.argv[1]); seconds=float(sys.argv[2]); out=sys.argv[3]
pol=NumpyOnnxMLP("upstream/policies/alpha_walking.onnx")
cfg=load_env_cfg("Mjlab-Velocity2-Rough-Backlash-MicroDuck"); cfg.scene.num_envs=N
g=cfg.scene.terrain.terrain_generator; g.num_rows=g.num_cols=5; g.curriculum=False; g.seed=0
env=ManagerBasedRlEnv(cfg,device="cpu"); env.common_step_counter=10**9
obs,_=env.reset()
rm=env.reward_manager; idx=rm._term_names.index("foot_swing_height"); tc=rm._term_cfgs[idx]; f=tc.func
cs=env.scene["feet_ground_contact"]; hs=env.scene["foot_height_scan"]; L=[]; H=[]
class W:
    def __call__(self,env_,**p):
        h=hs.data.heights.clone(); ia=cs.data.found==0
        ph=torch.where(ia,torch.maximum(f.peak_heights,h),f.peak_heights)
        fc=cs.compute_first_contact(dt=f.step_dt)
        e,ft=torch.nonzero(fc,as_tuple=True)
        for a,b in zip(e.tolist(),ft.tolist()):
            L.append((float(ph[a,b]),int(env_.episode_length_buf[a]),float(h[a,b])))
        H.append([float(h.mean()),float(h.max()),float(h.min())])
        return f(env_,**p)
    def reset(self,env_ids=None): return f.reset(env_ids=env_ids) if hasattr(f,"reset") else None
tc.func=W()
for k in range(int(seconds/0.02)):
    obs,*_=env.step(torch.from_numpy(pol(obs["actor"].numpy())))
    if k%100==0: print(k,len(L),flush=True)
json.dump(dict(landings=L,heights=H,steps=int(seconds/0.02),N=N),open(out,"w")); print("DONE",flush=True)
