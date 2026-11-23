"""rsl_rl VecEnv adapter for MicroduckEnv (mirrors ML rl/vecenv_wrapper.py)."""

from __future__ import annotations

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from duckmac.env import MAX_EP_LEN, EnvCfg, MicroduckEnv


class RslMicroduckEnv(VecEnv):
    def __init__(self, cfg: EnvCfg):
        self.env = MicroduckEnv(cfg)
        self.num_envs = cfg.num_envs
        self.num_actions = 14
        self.max_episode_length = MAX_EP_LEN
        self.device = "cpu"  # physics lives on the CPU; the runner moves tensors to its device
        self.cfg = {"env": repr(cfg)}
        self._obs = self._pack(self.env.reset())

    def _pack(self, obs) -> TensorDict:
        actor, critic = obs
        return TensorDict(
            {"actor": torch.from_numpy(actor), "critic": torch.from_numpy(critic)}, batch_size=[self.num_envs]
        )

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return torch.from_numpy(self.env.episode_length)

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length = value.detach().cpu().numpy().astype(np.int64)

    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions: torch.Tensor):
        obs, rew, done, ex = self.env.step(actions.detach().cpu().numpy())
        self._obs = self._pack(obs)
        log = {f"Reward/{k}": float(v.mean()) / 0.02 for k, v in self.env.reward_terms.items()}
        extras = {"time_outs": torch.from_numpy(ex["time_outs"]), "log": log}
        return (self._obs, torch.from_numpy(rew.astype(np.float32)), torch.from_numpy(done.astype(np.int64)), extras)
