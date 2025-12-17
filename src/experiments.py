"""Run all 5 experiments for MLOps project - Progressive Optimization."""

import mlflow
from src.config import (
    setup_mlflow,
    MLFLOW_EXPERIMENT_NAME,
    promote_model_to_production,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE
)
from src.train import train_model


def run_all_experiments():
    """
    Run all 5 experiments demonstrating progressive optimization.

    Experiment Flow:
    1. Tiny CNN (Underfit) - 1 conv layer, 8 filters → shows underfitting
    2. Bigger CNN - 2 conv layers, 32→64 filters → fixes underfitting, shows overfitting
    3. + BatchNorm - adds BatchNorm → reduces overfitting
    4. + Dropout - adds Dropout → further reduces overfitting
    5. + Augmentation - adds data augmentation → best generalization
    """
    # Setup MLflow with local SQLite backend
    mlflow_client = setup_mlflow()
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    all_results = []
    best_result = None
    smallest_gap = float('inf')

    # ============================================================
    # Experiment 1: Tiny CNN (Underfitting Baseline)
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: Tiny CNN (Underfitting Baseline)")
    print("="*60)

    results1, _ = train_model(
        experiment_name="exp1_tiny_cnn_underfit",
        channels=[8],
        kernel_size=3,
        use_batchnorm=False,
        dropout_rate=0.0,
        use_augmentation=False,
        learning_rate=DEFAULT_LEARNING_RATE,
        batch_size=DEFAULT_BATCH_SIZE,
        epochs=DEFAULT_EPOCHS,
        description="Tiny CNN (1 layer, 8 filters) - demonstrates underfitting",
        save_model=True
    )
    all_results.append(results1)
    if abs(results1["overfit_gap"]) < smallest_gap:
        smallest_gap = abs(results1["overfit_gap"])
        best_result = results1

    # ============================================================
    # Experiment 2: Bigger CNN (Fix Underfitting)
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: Bigger CNN (Fix Underfitting)")
    print("="*60)

    results2, _ = train_model(
        experiment_name="exp2_bigger_cnn",
        channels=[32, 64],
        kernel_size=3,
        use_batchnorm=False,
        dropout_rate=0.0,
        use_augmentation=False,
        learning_rate=DEFAULT_LEARNING_RATE,
        batch_size=DEFAULT_BATCH_SIZE,
        epochs=DEFAULT_EPOCHS,
        description="Bigger CNN (2 layers, 32→64) - fixes underfitting, shows overfitting",
        save_model=True
    )
    all_results.append(results2)
    if abs(results2["overfit_gap"]) < smallest_gap:
        smallest_gap = abs(results2["overfit_gap"])
        best_result = results2

    # ============================================================
    # Experiment 3: + BatchNorm
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: + BatchNorm")
    print("="*60)

    results3, _ = train_model(
        experiment_name="exp3_batchnorm",
        channels=[32, 64],
        kernel_size=3,
        use_batchnorm=True,
        dropout_rate=0.0,
        use_augmentation=False,
        learning_rate=DEFAULT_LEARNING_RATE,
        batch_size=DEFAULT_BATCH_SIZE,
        epochs=DEFAULT_EPOCHS,
        description="Add BatchNorm - reduces overfitting",
        save_model=True
    )
    all_results.append(results3)
    if abs(results3["overfit_gap"]) < smallest_gap:
        smallest_gap = abs(results3["overfit_gap"])
        best_result = results3

    # ============================================================
    # Experiment 4: + Dropout
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 4: + Dropout")
    print("="*60)

    results4, _ = train_model(
        experiment_name="exp4_dropout",
        channels=[32, 64],
        kernel_size=3,
        use_batchnorm=True,
        dropout_rate=0.5,
        use_augmentation=False,
        learning_rate=DEFAULT_LEARNING_RATE,
        batch_size=DEFAULT_BATCH_SIZE,
        epochs=DEFAULT_EPOCHS,
        description="Add Dropout(0.5) - further reduces overfitting",
        save_model=True
    )
    all_results.append(results4)
    if abs(results4["overfit_gap"]) < smallest_gap:
        smallest_gap = abs(results4["overfit_gap"])
        best_result = results4

    # ============================================================
    # Experiment 5: + Data Augmentation
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 5: + Data Augmentation")
    print("="*60)

    results5, _ = train_model(
        experiment_name="exp5_augmentation",
        channels=[32, 64],
        kernel_size=3,
        use_batchnorm=True,
        dropout_rate=0.5,
        use_augmentation=True,
        learning_rate=DEFAULT_LEARNING_RATE,
        batch_size=DEFAULT_BATCH_SIZE,
        epochs=DEFAULT_EPOCHS,
        description="Add Augmentation - best generalization",
        save_model=True
    )
    all_results.append(results5)
    if abs(results5["overfit_gap"]) < smallest_gap:
        smallest_gap = abs(results5["overfit_gap"])
        best_result = results5

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENTS SUMMARY - Progressive Optimization")
    print("="*60)
    print(f"{'Experiment':<30} {'Channels':<12} {'Train Acc':<10} {'Val Acc':<10} {'Gap':<8}")
    print("-"*70)

    for result in all_results:
        name = result["experiment_name"]
        channels = str(result["channels"])
        train_acc = result["final_train_accuracy"]
        val_acc = result["final_val_accuracy"]
        gap = result["overfit_gap"]
        print(f"{name:<30} {channels:<12} {train_acc:.4f}     {val_acc:.4f}     {gap:+.4f}")

    print("-"*70)
    print(f"\nBest Model: {best_result['experiment_name']}")
    print(f"Smallest Train-Val Gap: {best_result['overfit_gap']:+.4f}")
    print(f"Validation Accuracy: {best_result['best_val_accuracy']:.4f}")
    print(f"Run ID: {best_result['run_id']}")

    # Register and promote best model to Production
    print("\n" + "="*60)
    print("REGISTERING BEST MODEL TO PRODUCTION")
    print("="*60)
    promote_model_to_production(best_result['run_id'])

    return all_results, best_result


if __name__ == "__main__":
    run_all_experiments()
