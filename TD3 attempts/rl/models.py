import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCQ(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=(256, 128), activation_function=F.relu):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_function = activation_function
        self.input_layer = nn.Linear(self.input_dim, hidden_dim[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
        self.output_layer = nn.Linear(hidden_dim[-1], self.output_dim)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, state):
        x = self.activation_function(self.input_layer(state))
        for layer in self.hidden_layers:
            x = self.activation_function(layer(x))
        return self.output_layer(x)


class ReplayBuffer:
    def __init__(self, max_size=50000, batch_size=256):
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)
        self.states = np.empty(shape=(self.max_size,), dtype=object)
        self.actions = np.empty(shape=(self.max_size,), dtype=object)
        self.rewards = np.empty(shape=(self.max_size,), dtype=object)
        self.next_states = np.empty(shape=(self.max_size,), dtype=object)
        self.terminals = np.empty(shape=(self.max_size,), dtype=object)
        self.idx = 0
        self.size = 0

    def store(self, experience):
        s, a, r, next_s, terminal = experience
        self.states[self.idx] = s
        self.actions[self.idx] = a
        self.rewards[self.idx] = r
        self.next_states[self.idx] = next_s
        self.terminals[self.idx] = terminal
        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def draw_samples(self, batch_size=None):
        b = self.batch_size if batch_size is None else int(batch_size)
        idxs = np.random.choice(self.size, b, replace=False)
        experiences = (
            np.vstack(self.states[idxs]),
            np.vstack(self.actions[idxs]),
            np.vstack(self.rewards[idxs]),
            np.vstack(self.next_states[idxs]),
            np.vstack(self.terminals[idxs]),
        )
        return experiences

    def __len__(self):
        return self.size

