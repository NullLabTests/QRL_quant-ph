"""
PennyLane Hybrid QRL Agent
==========================
Differentiable quantum policies using PennyLane + Qiskit backend.
Fully compatible with existing Qiskit modules.
"""

import pennylane as qml
import torch
import torch.nn as nn
import gymnasium as gym

dev = qml.device("qiskit.aer", wires=4, shots=1024)


class PennyLaneQRLPolicy(nn.Module):
    def __init__(self, n_qubits=4, n_actions=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.q_weights = nn.Parameter(torch.randn(2, n_qubits) * 0.1)

    @qml.qnode(dev, interface="torch")
    def quantum_policy(self, inputs, weights):
        # Encode classical state into quantum circuit
        for i in range(self.n_qubits):
            qml.RY(inputs[i % len(inputs)], wires=i)

        # Variational entangling layer
        qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))

        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, state):
        state = torch.tensor(
            state[:self.n_qubits],
            dtype=torch.float32
        )

        q_features = self.quantum_policy(state, self.q_weights)
        q_features = torch.stack(q_features)

        logits = q_features[:self.n_actions]

        return torch.softmax(logits, dim=0)


def demo_train(episodes=80):
    env = gym.make("CartPole-v1")

    policy = PennyLaneQRLPolicy()

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=0.02
    )

    print("🚀 Training PennyLane Hybrid QRL Agent...")

    for ep in range(episodes):
        state, _ = env.reset()

        total_reward = 0
        done = False

        while not done:
            action_probs = policy(state)

            action = torch.multinomial(
                action_probs,
                1
            ).item()

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            # Simple REINFORCE-style placeholder loss
            loss = -torch.log(action_probs[action]) * reward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward

        if ep % 20 == 0:
            print(f"Episode {ep:03d} | Reward: {total_reward:5.1f}")

    print("✅ PennyLane QRL training finished!")


if __name__ == "__main__":
    demo_train()
