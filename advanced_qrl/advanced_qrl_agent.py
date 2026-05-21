import numpy as np
import torch
import torch.nn as nn

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import gymnasium as gym

class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self):
        self.policy = Policy()

    def quantum_features(self, state):
        qc = QuantumCircuit(4)

        for i in range(min(4, len(state))):
            qc.ry(float(state[i]), i)

        sv = Statevector.from_instruction(qc)

        probs = np.abs(sv.data[:4]) ** 2

        return torch.tensor(probs, dtype=torch.float32)

    def action(self, state):
        features = self.quantum_features(state)

        logits = self.policy(features)

        probs = torch.softmax(logits, dim=-1)

        return torch.multinomial(probs, 1).item()

    def train(self, episodes=10):
        env = gym.make("CartPole-v1")

        print("🚀 Quantum RL")

        for ep in range(episodes):
            state, _ = env.reset()

            done = False

            total = 0

            while not done:
                act = self.action(state)

                state, reward, terminated, truncated, _ = env.step(act)

                done = terminated or truncated

                total += reward

            print(
                f"episode={ep} "
                f"reward={total}"
            )

if __name__ == "__main__":
    Agent().train()
