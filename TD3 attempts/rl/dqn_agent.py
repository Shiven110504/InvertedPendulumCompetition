import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


class DQN:
    def __init__(
        self,
        env,
        replay_buffer,
        online_model: nn.Module,
        target_model: nn.Module,
        optimizer: optim.Optimizer,
        warmup_batches: int,
        update_freq: int,
        epochs: int = 1,
        gamma: float = 0.99,
        demo_env=None,
        device: Optional[torch.device] = None,
    ):
        self.env = env
        self.replay_buffer = replay_buffer
        self.online_model = online_model
        self.target_model = target_model
        self.optimizer = optimizer
        self.warmup_batches = warmup_batches
        self.update_freq = update_freq
        self.epochs = epochs
        self.gamma = gamma

        self.device = device or getattr(self.online_model, "device", None) or next(online_model.parameters()).device
        self.online_model.to(self.device)
        self.target_model.to(self.device)
        self.demo_env = demo_env if demo_env is not None else env

    @torch.no_grad()
    def choose_action_egreedy(self, state, eps: float) -> int:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.online_model(state_t).squeeze(0).detach().cpu().numpy()
        if np.random.rand() > eps:
            return int(np.argmax(q))
        return int(np.random.randint(self.env.action_space.n))

    @torch.no_grad()
    def choose_action_greedy(self, state) -> int:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.online_model(state_t).squeeze(0).detach().cpu().numpy()
        return int(np.argmax(q))

    def interaction_step(self, state, eps: float):
        action = self.choose_action_egreedy(state, eps)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        self.replay_buffer.store((state, action, reward, next_state, done))
        return next_state, reward, terminated, truncated, info

    def learn(self) -> float:
        states, actions, rewards, next_states, terminals = self.replay_buffer.draw_samples()
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).view(-1, 1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        terminals_t = torch.tensor(terminals, dtype=torch.float32, device=self.device).view(-1, 1)

        with torch.no_grad():
            q_next = self.target_model(next_states_t).max(1, keepdim=True)[0]
            y = rewards_t + self.gamma * q_next * (1.0 - terminals_t)

        q = self.online_model(states_t).gather(1, actions_t)

        loss = torch.nn.functional.mse_loss(q, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_model.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def soft_update_weights(self, tau: float = 1.0):
        for tgt, src in zip(self.target_model.parameters(), self.online_model.parameters()):
            tgt.data.copy_(tau * src.data + (1.0 - tau) * tgt.data)

    @torch.no_grad()
    def demo(self, render: bool = False, max_steps: int = 1000) -> float:
        env = self.demo_env if render else self.env
        state, _ = env.reset()
        total = 0.0
        for _ in range(max_steps):
            action = self.choose_action_greedy(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        return total

