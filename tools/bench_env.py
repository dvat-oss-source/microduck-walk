import sys, time, json, resource
import numpy as np
from duckmac.env import EnvCfg, MicroduckEnv
from duckmac.onnx_mlp import NumpyOnnxMLP
pol=NumpyOnnxMLP("upstream/policies/alpha_walking.onnx")
res=[]
for N,T in [(4096,5),(4096,8),(4096,11),(512,11),(1024,11),(2048,11),(8192,11)]:
    env=MicroduckEnv(EnvCfg(num_envs=N,seed=1,num_threads=T,curriculum_step_override=10**9))
    a,_=env.reset(); env.randomize_episode_lengths()
    for _ in range(25): (a,_),*_=env.step(pol(a))
    tb=[0.0]; ob=env.b.step; of=env.b.forward_sense
    class B:
        def __getattr__(s,k): return getattr(ob.__self__,k)
        def step(s):
            t=time.perf_counter(); ob(); tb[0]+=time.perf_counter()-t
        def forward_sense(s):
            t=time.perf_counter(); of(); tb[0]+=time.perf_counter()-t
    env.b=B()
    n=max(20,int(40*4096/N/4)); tp=0.0; t0=time.perf_counter()
    for _ in range(n):
        t=time.perf_counter(); act=pol(a); tp+=time.perf_counter()-t
        (a,_),*_=env.step(act)
    w=time.perf_counter()-t0
    r=dict(N=N,threads=T,steps_per_s=N*n/w,sim_frac=tb[0]/w,policy_frac=tp/w,python_frac=1-(tb[0]+tp)/w,maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30)
    print(json.dumps({k:round(v,3) for k,v in r.items()}),flush=True); res.append(r); del env
json.dump(res,open("reports/bench_env.json","w"),indent=1)
