"""Train with rsl_rl using the PPO settings of Pollen's run 441tzs6d (bundle441 wandb config)."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import yaml
from rsl_rl.runners import OnPolicyRunner

from duckmac.env import EnvCfg
from duckmac.rsl_env import RslMicroduckEnv

ROOT = Path(__file__).resolve().parents[1]


def train_cfg(seed: int) -> dict:
    """CFG train_cfg, plus the class names mjlab's dataclasses add (ML rl/config.py)."""
    raw = yaml.safe_load((ROOT / "upstream/bundle441/wandb_config.yaml").read_text())["train_cfg"]["value"]
    c = copy.deepcopy(raw)
    c["actor"]["class_name"] = "MLPModel"
    c["critic"]["class_name"] = "MLPModel"
    c["actor"]["distribution_cfg"]["class_name"] = "GaussianDistribution"
    c["algorithm"]["class_name"] = "PPO"
    c["logger"] = "tensorboard"  # Pollen used wandb; no account needed here
    c["upload_model"] = False
    c["seed"] = seed
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "mps"], default="mps")  # update 3.4x faster than cpu
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)  # measured best, reports/bench_env.json
    ap.add_argument("--iterations", type=int, default=3750)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--current-limit", choices=["on", "off"], default="on")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--track-lin-std", type=float, default=0.2)
    ap.add_argument("--air-time-weight", type=float, default=5.0)
    ap.add_argument("--action-rate-delay", type=float, default=1.0)
    ap.add_argument("--resume", default=None, help="checkpoint path")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    run_dir = Path(a.run_dir or ROOT / "runs" / time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(a), indent=1))
    env = RslMicroduckEnv(EnvCfg(num_envs=a.envs, seed=a.seed, num_threads=a.threads,
                                 current_limit=a.current_limit == "on",
                                 track_lin_std=a.track_lin_std, air_time_weight=a.air_time_weight,
                                 action_rate_delay=a.action_rate_delay))
    runner = OnPolicyRunner(env, train_cfg(a.seed), log_dir=str(run_dir), device=a.device)
    if a.resume:
        runner.load(a.resume, map_location=a.device)
        # Curricula key off common_step_counter (24 per iteration); restore it. It is not in the checkpoint.
        env.env.common_step_counter = (runner.current_learning_iteration + 1) * 24
        print(f"resumed at iteration {runner.current_learning_iteration}, step counter {env.env.common_step_counter}")
    t = time.time()
    runner.learn(a.iterations, init_at_random_ep_len=not a.resume)
    dt = time.time() - t
    print(f"TOTAL {a.iterations} iters, {a.envs} envs, {a.device}: {dt:.1f}s, "
          f"{a.iterations * 24 * a.envs / dt:.0f} control steps/s")


if __name__ == "__main__":
    main()
