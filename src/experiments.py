"""Run all 5 experiments for MLOps project."""

import mlflow
from src.config import setup_mlflow, DAGSHUB_REPO_NAME
from src.train import train_model, save_model_info


def run_all_experiments():
    """
    Run all 5 experiments and track with MLflow.

    Experiments:
    1. Baseline CNN - No regularization
    2. CNN + Regularization (Dropout + BatchNorm)
    3. CNN + Data Augmentation
    4. Hyperparameter Tuning
    5. Simple MLP Comparison
    """
    # Setup MLflow
    mlflow_client = setup_mlflow()
    mlflow.set_experiment(DAGSHUB_REPO_NAME)

    all_results = []
    best_result = None
    best_val_acc = 0.0

    # ============================================================
    # Experiment 1: Baseline CNN
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: Baseline CNN")
    print("="*60)

    results1, _ = train_model(
        experiment_name="exp1_baseline_cnn",
        model_type="cnn",
        use_batchnorm=False,
        dropout_rate=0.0,
        use_augmentation=False,
        learning_rate=0.001,
        batch_size=64,
        epochs=10,
        description="Baseline CNN without regularization - observe overfitting",
        save_model=True
    )
    all_results.append(results1)
    if results1["best_val_accuracy"] > best_val_acc:
        best_val_acc = results1["best_val_accuracy"]
        best_result = results1

    # ============================================================
    # Experiment 2: CNN + Regularization
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: CNN + Regularization (Dropout + BatchNorm)")
    print("="*60)

    results2, _ = train_model(
        experiment_name="exp2_cnn_regularization",
        model_type="cnn",
        use_batchnorm=True,
        dropout_rate=0.5,
        use_augmentation=False,
        learning_rate=0.001,
        batch_size=64,
        epochs=10,
        description="CNN with BatchNorm and Dropout(0.5) - reduce overfitting",
        save_model=True
    )
    all_results.append(results2)
    if results2["best_val_accuracy"] > best_val_acc:
        best_val_acc = results2["best_val_accuracy"]
        best_result = results2

    # ============================================================
    # Experiment 3: CNN + Data Augmentation
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: CNN + Data Augmentation")
    print("="*60)

    results3, _ = train_model(
        experiment_name="exp3_cnn_augmentation",
        model_type="cnn",
        use_batchnorm=True,
        dropout_rate=0.5,
        use_augmentation=True,
        learning_rate=0.001,
        batch_size=64,
        epochs=10,
        description="CNN with regularization and data augmentation - best generalization",
        save_model=True
    )
    all_results.append(results3)
    if results3["best_val_accuracy"] > best_val_acc:
        best_val_acc = results3["best_val_accuracy"]
        best_result = results3

    # ============================================================
    # Experiment 4: Hyperparameter Tuning
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 4: Hyperparameter Tuning")
    print("="*60)

    # Try different learning rates and epochs
    results4, _ = train_model(
        experiment_name="exp4_hyperparameter_tuning",
        model_type="cnn",
        use_batchnorm=True,
        dropout_rate=0.5,
        use_augmentation=True,
        learning_rate=0.001,  # Can experiment with 0.01, 0.0001
        batch_size=64,        # Can experiment with 32, 128
        epochs=15,            # Increased epochs
        description="Hyperparameter tuning - optimized settings",
        save_model=True
    )
    all_results.append(results4)
    if results4["best_val_accuracy"] > best_val_acc:
        best_val_acc = results4["best_val_accuracy"]
        best_result = results4

    # ============================================================
    # Experiment 5: Simple MLP (Underfitting Demo)
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 5: Simple MLP (Underfitting Comparison)")
    print("="*60)

    results5, _ = train_model(
        experiment_name="exp5_mlp_comparison",
        model_type="mlp",
        use_batchnorm=False,
        dropout_rate=0.0,
        use_augmentation=False,
        learning_rate=0.001,
        batch_size=64,
        epochs=10,
        description="Simple MLP - demonstrates underfitting on image data",
        save_model=False  # Don't save MLP as best model
    )
    all_results.append(results5)
    # Note: We don't update best_result for MLP since it's just for comparison

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENTS SUMMARY")
    print("="*60)
    print(f"{'Experiment':<35} {'Train Acc':<12} {'Val Acc':<12} {'Gap':<10}")
    print("-"*60)

    for result in all_results:
        name = result["experiment_name"]
        train_acc = result["final_train_accuracy"]
        val_acc = result["final_val_accuracy"]
        gap = result["overfit_gap"]
        print(f"{name:<35} {train_acc:.4f}       {val_acc:.4f}       {gap:.4f}")

    print("-"*60)
    print(f"\nBest Model: {best_result['experiment_name']}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # Save best model info
    save_model_info(best_result)

    return all_results, best_result


if __name__ == "__main__":
    run_all_experiments()
