from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, log_std_bounds: Tuple[float, float] = (-5.0, 2.0)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)
        self.log_std_bounds = log_std_bounds

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mean = self.mean(x)
        log_std = self.log_std(x)
        min_log_std, max_log_std = self.log_std_bounds
        log_std = torch.tanh(log_std)
        log_std = min_log_std + 0.5 * (max_log_std - min_log_std) * (log_std + 1)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        pre_tanh = mean + std * noise
        action = torch.tanh(pre_tanh)
        log_prob = (-0.5 * ((pre_tanh - mean) / (std + 1e-6)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = _mlp(obs_dim + act_dim, hidden_dim, 1)
        self.q2 = _mlp(obs_dim + act_dim, hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([obs, action], dim=-1)
        return self.q1(xu), self.q2(xu)


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: float) -> None:
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=torch.as_tensor(self.obs[idx], device=device),
            next_obs=torch.as_tensor(self.next_obs[idx], device=device),
            act=torch.as_tensor(self.acts[idx], device=device),
            rew=torch.as_tensor(self.rews[idx], device=device),
            done=torch.as_tensor(self.dones[idx], device=device),
        )
        return batch


@dataclass
class SACConfig:
    obs_dim: int
    act_dim: int
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden_dim: int = 256
    init_temperature: float = 0.2
    target_entropy: Optional[float] = None


class SACAgent:
    def __init__(self, config: SACConfig, device: Optional[torch.device] = None):
        self.cfg = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(config.obs_dim, config.act_dim, hidden_dim=config.hidden_dim).to(self.device)
        self.critic = Critic(config.obs_dim, config.act_dim, hidden_dim=config.hidden_dim).to(self.device)
        self.critic_target = Critic(config.obs_dim, config.act_dim, hidden_dim=config.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.log_alpha = torch.tensor(np.log(config.init_temperature), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)

        if config.target_entropy is None:
            self.target_entropy = -float(config.act_dim)
        else:
            self.target_entropy = config.target_entropy


    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(obs_t)
                action = torch.tanh(mu)
            else:
                action, _, _ = self.actor.sample(obs_t)
        return action.cpu().numpy()[0]

    def update(self, buffer: ReplayBuffer, batch_size: int) -> Dict[str, float]:
        batch = buffer.sample(batch_size, self.device)
        obs, act, rew, next_obs, done = batch["obs"], batch["act"], batch["rew"], batch["next_obs"], batch["done"]

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_v = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = rew + (1.0 - done) * self.cfg.gamma * target_v

        cur_q1, cur_q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(cur_q1, target_q) + F.mse_loss(cur_q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        action_sample, log_prob, _ = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, action_sample)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_prob - min_q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.mul_(1 - self.cfg.tau)
                target_param.data.add_(self.cfg.tau * param.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            path,
        )

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        payload = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))