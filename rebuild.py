
code = r'''import yaml
import argparse
import sys
import matplotlib.pyplot as plt
from src.data import DataManager, DataConfig
from src.models import QuantumKernel
from src.training import SVCTrainer
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
        num_sectors=config['data']['num_sectors'],
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        use_scaler=config['data']['use_scaler']
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
'''

with open('run_benchmark.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("run_benchmark.py created successfully.")
