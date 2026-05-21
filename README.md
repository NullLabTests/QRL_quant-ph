# QRL_quant-ph

**Advanced Quantum Reinforcement Learning Framework**  
for NISQ devices, hybrid quantum-classical AI, and AGI-oriented experimentation.

<div align="center">
  <img src="https://raw.githubusercontent.com/NullLabTests/NullLabTests/main/logo.png" alt="NullLabTests" width="280" />
</div>

---

## Features

- Grover-style amplitude amplification
- Variational Deep Q-Learning (original Qiskit)
- Fully quantum agent (TicTacToe)
- **Advanced QRL with learned gadgets + VQC-KAN hybrids** (see `4-AdvancedQRLforAGI/`)
- **Official PennyLane support** for differentiable quantum policies

---

## Installation

git clone git@github.com:NullLabTests/QRL_quant-ph.git
cd QRL_quant-ph

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

---

## Quick Start

### Main Advanced Agent (recommended)

python -m 4-AdvancedQRLforAGI.advanced_qrl_agent --env CartPole-v1

### PennyLane Differentiable QRL Demo

cd pennylane_support
python pennylane_qrl_agent.py

---

## PennyLane Support

Full PennyLane integration added for differentiable quantum circuits and hybrid QRL.

### Features

- Differentiable quantum policies (PyTorch compatible)
- Easy backend switching (simulator or real hardware)
- Seamless compatibility with existing Qiskit modules

---

## Original Modules (preserved)

1-GroverEnhancement/
2-QNNDeepQLearning/
3-QuantumTicTacToe/

---

Built for the @qiskit and @PennyLaneAI communities.

**Nullipotent Quantum AI Labs** —  
*“The nullipotency inherent in the pluripotent idempotence of omnipotent functions.”*

