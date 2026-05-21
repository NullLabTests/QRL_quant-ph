# QRL_quant-ph

Advanced Quantum Reinforcement Learning Framework.

## Install

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

## Quick Start

python3 advanced_qrl/advanced_qrl_agent.py

---

## PennyLane Support

Added full PennyLane integration for differentiable quantum circuits and hybrid QRL.

### Run the demo

```bash
cd pennylane_support
python pennylane_qrl_agent.py
```

### Features

- Differentiable quantum policies (PyTorch compatible)
- Easy backend switching (simulator or real hardware)
- Compatible with existing Qiskit workflows
- Hybrid quantum-classical RL experimentation

