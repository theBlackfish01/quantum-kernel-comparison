# Deep Quantum Kernel Learning

A modular research framework for **Deep Quantum Kernel Learning (DKL)**. This project explores how **trainable** quantum kernels can outperform static ones by optimizing the data embedding to maximize class separability (Kernel-Target Alignment).

## Key Features

1.  **Deep Kernel Learning**: Implements a generic optimization loop that trains the parameters of the Quantum Ansatz via Gradient Descent (maximizing alignment) *before* the SVM classification step.
2.  **Hybrid Backends**: Run on PennyLane's `default.qubit` or integrate with **Qiskit** (`qiskit.aer`).
3.  **Real-World Data**: Support for synthetic (`Double Cake`, `Moons`) and real-world (`MNIST` Digits) datasets, with automatic PCA dimensionality reduction.
4.  **Modular Architecture**: Clean separation of `Data`, `Models`, `Training`, and `Evaluation`.

## Project Structure

```text
├── config.yaml             # Experiment configuration
├── config_qiskit.yaml      # Qiskit backend configuration
├── main.py                 # Entry point
├── src/
│   ├── data/               # Loaders for Moons, Digits, etc.
│   ├── models/             # Trainable Quantum Kernels
│   ├── training/           # KernelOptimizer & SVCTrainer
│   └── evaluation/         # KTA Metrics & Plotting
```

## Installation

```bash
pip install pennylane scikit-learn matplotlib pyyaml qiskit pennylane-qiskit
```

## Usage

**1. Standard Benchmark (Deep Kernel Learning on Moons):**
```bash
python main.py
```
*Output*: Trains the kernel (increasing alignment from ~0.32 to ~0.35) and achieves high classification accuracy.

**2. Using Qiskit Backend:**
```bash
python main.py --config config_qiskit.yaml
```

## Configuration (`config.yaml`)

Control the experiment without changing code:

```yaml
data:
  type: "moons"        # Options: double_cake, moons, digits
  n_samples: 40        # Sample size
  
model:
  type: "quantum_kernel"
  dev_name: "default.qubit"  # or "qiskit.aer"

optimization:
  optimize_kernel: true      # Enable Deep Kernel Learning
  steps: 15
  learning_rate: 0.2
```
