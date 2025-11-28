import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=(256, 256), act=nn.ReLU):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers += [nn.Linear(last, output_dim)]
        self.net = nn.Sequential(*layers)

    def _last_linear(self):
        for module in reversed(self.net):
            if isinstance(module, nn.Linear):
                return module
        raise RuntimeError("MLP network missing Linear layer.")

    def zero_last_layer(self):
        last = self._last_linear()
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        self.body = MLP(state_dim, action_dim, hidden)
        self.tanh = nn.Tanh()
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(f"Actor using device: {device}")
        device = "cpu"
        self.device = torch.device(device)
        self.to(self.device)
        self.body.zero_last_layer()

    def forward(self, state):
        return self.tanh(self.body(state))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden)
        self.q2 = MLP(state_dim + action_dim, 1, hidden)
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


@dataclass
class TD3Config:
    state_dim: int
    action_dim: int
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden: Tuple[int, int] = (256, 256)


class TD3:
    def __init__(self, cfg: TD3Config):
        self.cfg = cfg
        self.actor = Actor(cfg.state_dim, cfg.action_dim, cfg.hidden)
        self.actor_target = Actor(cfg.state_dim, cfg.action_dim, cfg.hidden)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(cfg.state_dim, cfg.action_dim, cfg.hidden)
        self.critic_target = Critic(cfg.state_dim, cfg.action_dim, cfg.hidden)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.total_it = 0

    @torch.no_grad()
    def act(self, state_np: np.ndarray, noise_std: float = 0.0, noise_mask: Optional[np.ndarray] = None) -> np.ndarray:
        state = torch.tensor(state_np, dtype=torch.float32, device=self.actor.device).unsqueeze(0)
        action = self.actor(state).squeeze(0).cpu().numpy()
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=action.shape).astype(np.float32)
            if noise_mask is not None:
                noise = noise * noise_mask
            action = np.clip(action + noise, -1.0, 1.0)
        return action.astype(np.float32)

    def train_step(self, batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[float, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones = batch

        device = self.actor.device
        s = torch.tensor(states, dtype=torch.float32, device=device)
        a = torch.tensor(actions, dtype=torch.float32, device=device)
        r = torch.tensor(rewards, dtype=torch.float32, device=device).view(-1, 1)
        s2 = torch.tensor(next_states, dtype=torch.float32, device=device)
        d = torch.tensor(dones, dtype=torch.float32, device=device).view(-1, 1)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.cfg.policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            a2 = (self.actor_target(s2) + noise).clamp(-1.0, 1.0)
            q1_t, q2_t = self.critic_target(s2, a2)
            q_t = torch.min(q1_t, q2_t)
            y = r + self.cfg.gamma * (1.0 - d) * q_t

        q1, q2 = self.critic(s, a)
        critic_loss = torch.nn.functional.mse_loss(q1, y) + torch.nn.functional.mse_loss(q2, y)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_opt.step()

        actor_loss = torch.tensor(0.0, device=device)
        if self.total_it % self.cfg.policy_freq == 0:
            a_pi = self.actor(s)
            actor_loss = -self.critic.q1_only(s, a_pi).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.actor_opt.step()

            # Soft update targets
            with torch.no_grad():
                for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                    pt.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
                for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                    pt.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return float(critic_loss.item()), float(actor_loss.item())
