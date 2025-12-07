"""Export experiment results to CSV for documentation."""

import os
import pandas as pd
from src.config import RESULTS_DIR, EXPERIMENTS_SUMMARY_PATH


def export_results_to_csv(all_results: list):
    """
    Export experiment results to CSV file.

    Args:
        all_results: List of result dictionaries from experiments
    """
    # Prepare data for DataFrame
    data = []
    for result in all_results:
        row = {
            "Experiment": result["experiment_name"],
            "Model Type": result["model_type"],
            "Train Accuracy": f"{result['final_train_accuracy']:.4f}",
            "Val Accuracy": f"{result['final_val_accuracy']:.4f}",
            "Best Val Accuracy": f"{result['best_val_accuracy']:.4f}",
            "Overfit Gap": f"{result['overfit_gap']:.4f}",
            "Learning Rate": result["hyperparameters"]["learning_rate"],
            "Batch Size": result["hyperparameters"]["batch_size"],
            "Epochs": result["hyperparameters"]["epochs"],
            "Dropout": result["hyperparameters"]["dropout_rate"],
            "BatchNorm": result["hyperparameters"]["use_batchnorm"],
            "Augmentation": result["hyperparameters"]["use_augmentation"],
            "Run ID": result["run_id"]
        }
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save to CSV
    df.to_csv(EXPERIMENTS_SUMMARY_PATH, index=False)
    print(f"Results exported to {EXPERIMENTS_SUMMARY_PATH}")

    # Also print the table
    print("\n" + "="*80)
    print("EXPERIMENTS SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    return df


def create_summary_table(all_results: list) -> str:
    """
    Create a formatted summary table string.

    Args:
        all_results: List of result dictionaries

    Returns:
        Formatted table string
    """
    header = f"{'#':<3} {'Experiment':<30} {'Train Acc':<12} {'Val Acc':<12} {'Gap':<10} {'Diagnosis':<20}"
    separator = "-" * 90

    lines = [header, separator]

    diagnoses = {
        "exp1_baseline_cnn": "Overfitting",
        "exp2_cnn_regularization": "Improved",
        "exp3_cnn_augmentation": "Best generalization",
        "exp4_hyperparameter_tuning": "Optimized",
        "exp5_mlp_comparison": "Underfitting"
    }

    for i, result in enumerate(all_results, 1):
        name = result["experiment_name"]
        train_acc = result["final_train_accuracy"]
        val_acc = result["final_val_accuracy"]
        gap = result["overfit_gap"]
        diagnosis = diagnoses.get(name, "")

        line = f"{i:<3} {name:<30} {train_acc:.4f}       {val_acc:.4f}       {gap:.4f}     {diagnosis:<20}"
        lines.append(line)

    lines.append(separator)

    return "\n".join(lines)
