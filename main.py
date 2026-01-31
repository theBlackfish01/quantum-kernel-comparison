# -*- coding: utf-8 -*-
import yaml
import argparse
import sys
import matplotlib.pyplot as plt
from src.data import DataManager, DataConfig
from src.models import QuantumKernel
from src.training import SVCTrainer, KernelOptimizer
from src.evaluation import plot_decision_boundaries, calculate_kta

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Quantum Kernel Benchmarking Framework")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file {args.config} not found.")
        sys.exit(1)

    # 1. Init Data
    print("Initializing Data Module...")
    data_conf = DataConfig(
        type=config['data']['type'],
        num_sectors=config['data'].get('num_sectors', 3), # Default 3 if not present
        test_size=config['data'].get('test_size', 0.3),
        random_state=config['data'].get('random_state', 42),
        use_scaler=config['data'].get('use_scaler', False),
        n_samples=config['data'].get('n_samples', 100)
    )
    dm = DataManager(data_conf)
    X_train, X_test, y_train, y_test = dm.load_data()
    print(f"Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 2. Init Model
    print(f"Initializing Model ({config['model']['type']})...")
    if config['model']['type'] == "quantum_kernel":
        kernel = QuantumKernel(
            num_wires=config['model']['wires'],
            num_layers=config['model']['layers'],
            dev_name=config['model']['dev_name']
        )
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")

    # 2.5 Optional: Deep Kernel Learning (Optimization)
    if config.get('optimization', {}).get('optimize_kernel', False):
        print("--- Deep Kernel Learning Detected ---")
        opt_conf = config['optimization']
        # Note: We must use a subset or the full training set for kernel alignment.
        # Computing KTA on large datasets is expensive (N^2), so for larger sets we might want to sample.
        # For Moons (N=200), using full X_train is fine.
        kernel_opt = KernelOptimizer(kernel, learning_rate=opt_conf.get('learning_rate', 0.1), steps=opt_conf.get('steps', 10))
        
        print("Calculating Initial KTA...")
        init_kta = calculate_kta(kernel, X_train, y_train)
        print(f"Initial KTA: {init_kta:.4f}")
        
        print("Optimizing Kernel Parameters...")
        kernel_opt.optimize(X_train, y_train)
        
        print("Calculating Final KTA...")
        final_kta = calculate_kta(kernel, X_train, y_train)
        print(f"Final KTA: {final_kta:.4f}")
        print("-------------------------------------")

    # 3. Init Trainer
    print("Initializing Trainer...")
    trainer = SVCTrainer(kernel)

    # 4. Train
    print("Training model (this may take a while)...")
    trainer.train(X_train, y_train)
    print("Training complete.")

    # 5. Evaluate
    print("Evaluating...")
    results = trainer.evaluate(X_test, y_test)
    print(f"Test Accuracy: {results['accuracy']:.4f}")

    # 6. Advanced Metrics (KTA) - calculated on Train set for this example
    print("Calculating Kernel-Target Alignment (KTA)...")
    try:
        kta = calculate_kta(kernel, X_train, y_train)
        print(f"KTA Score (Train): {kta:.4f}")
    except Exception as e:
        print(f"KTA calculation failed: {e}")

    # 7. Plotting
    print("Generating Decision Boundary Plot...")
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_decision_boundaries(trainer, X_train, y_train, ax) 
    plt.title(f"Quantum Kernel SVM (Acc: {results['accuracy']:.2f})")
    plt.tight_layout()
    output_plot = "decision_boundary.png"
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    plt.close()

if __name__ == "__main__":
    main()
