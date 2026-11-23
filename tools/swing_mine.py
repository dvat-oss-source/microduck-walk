import json, sys
import numpy as np
from duckmac.env import EnvCfg, MicroduckEnv, STEP_DT
from duckmac.onnx_mlp import NumpyOnnxMLP
N=int(sys.argv[1]); seconds=float(sys.argv[2]); out=sys.argv[3]
pol=NumpyOnnxMLP("upstream/policies/alpha_walking.onnx")
env=MicroduckEnv(EnvCfg(num_envs=N,seed=7,num_threads=11,curriculum_step_override=10**9))
L=[];H=[]
orig=env._rewards
def wrapped():
    h=env.heights_obs; ia=env.A["S_feet_found"]==0
    ph=np.where(ia,np.maximum(env.peak_heights,h),env.peak_heights)
    fc=(env.cur_contact>0)&(env.cur_contact<STEP_DT+1e-8)
    for a,b in zip(*np.nonzero(fc)): L.append((float(ph[a,b]),int(env.episode_length[a]),float(h[a,b])))
    H.append([float(h.mean()),float(h.max()),float(h.min())])
    return orig()
env._rewards=wrapped
actor,_=env.reset()
for k in range(int(seconds/0.02)):
    (actor,_),*_=env.step(pol(actor))
json.dump(dict(landings=L,heights=H,steps=int(seconds/0.02),N=N),open(out,"w"))
