"""Soft Actor-Critic implementation specialized for the base-yaw task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.models import MLP, ResidualPolicy


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, device: torch.device) -> None:
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs_buf[idx] = next_obs
        self.dones[idx] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "obs": to_tensor(self.obs_buf[idxs], self.device),
            "actions": to_tensor(self.actions[idxs], self.device),
            "rewards": to_tensor(self.rewards[idxs], self.device),
            "next_obs": to_tensor(self.next_obs_buf[idxs], self.device),
            "dones": to_tensor(self.dones[idxs], self.device),
        }
        return batch


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.net = MLP(obs_dim + action_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.net = MLP(obs_dim, hidden_dims, 2 * action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_logstd = self.net(obs)
        mu, log_std = mu_logstd.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)


@dataclass
class SACConfig:
    gamma: float = 0.995
    tau: float = 0.005
    alpha: float = 0.2
    lr: float = 3e-4
    hidden_dims: Tuple[int, int] = (256, 256)
    target_entropy: float | None = None
    device: str = "cpu"


class SACAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: SACConfig) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(config.device)
        self.target_entropy = config.target_entropy or -float(action_dim)

        self.policy = GaussianPolicy(obs_dim, action_dim, config.hidden_dims).to(self.device)
        self.q1 = SoftQNetwork(obs_dim, action_dim, config.hidden_dims).to(self.device)
        self.q2 = SoftQNetwork(obs_dim, action_dim, config.hidden_dims).to(self.device)
        self.q1_target = SoftQNetwork(obs_dim, action_dim, config.hidden_dims).to(self.device)
        self.q2_target = SoftQNetwork(obs_dim, action_dim, config.hidden_dims).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=config.lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=config.lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=config.lr)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.lr)
        self.gamma = config.gamma
        self.tau = config.tau

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        obs_tensor = to_tensor(obs, self.device).unsqueeze(0)
        if eval_mode:
            with torch.no_grad():
                mu, _ = self.policy.forward(obs_tensor)
                action = torch.tanh(mu)
        else:
            with torch.no_grad():
                action, _, _ = self.policy.sample(obs_tensor)
        return action.cpu().numpy()[0]

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(batch["next_obs"])
            q1_next = self.q1_target(batch["next_obs"], next_action)
            q2_next = self.q2_target(batch["next_obs"], next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = batch["rewards"] + (1 - batch["dones"]) * self.gamma * min_q_next

        q1 = self.q1(batch["obs"], batch["actions"])
        q2 = self.q2(batch["obs"], batch["actions"])
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        action_new, log_prob, _ = self.policy.sample(batch["obs"])
        q_new = torch.min(self.q1(batch["obs"], action_new), self.q2(batch["obs"], action_new))
        policy_loss = (self.alpha * log_prob - q_new).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.q1_target, self.q1)
        self._soft_update(self.q2_target, self.q2)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha.item(),
        }

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for tgt_param, src_param in zip(target.parameters(), source.parameters()):
            tgt_param.data.mul_(1 - self.tau)
            tgt_param.data.add_(self.tau * src_param.data)

    def save_policy(self, path: str, obs_scale: np.ndarray, residual_limit: float, hidden_dims=(256, 256)) -> None:
        deployment_policy = ResidualPolicy(self.obs_dim, self.action_dim, hidden_dims)
        deployment_sd = deployment_policy.state_dict()
        source_sd = self.policy.state_dict()
        for key in deployment_sd.keys():
            src_key = key.replace("backbone.", "net.", 1)
            tensor = source_sd[src_key]
            if key.endswith("net.4.weight") and tensor.shape[0] == 2 * self.action_dim:
                tensor = tensor[: self.action_dim].clone()
            if key.endswith("net.4.bias") and tensor.shape[0] == 2 * self.action_dim:
                tensor = tensor[: self.action_dim].clone()
            deployment_sd[key] = tensor
        checkpoint = {
            "policy_state_dict": deployment_sd,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dims": hidden_dims,
            "obs_scale": obs_scale,
            "residual_limit": residual_limit,
        }
        torch.save(checkpoint, path)
