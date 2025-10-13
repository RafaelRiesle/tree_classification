import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from models.ensemble_models.ensemble_utils.ensemble_model_manager import (
    EnsembleModelManager,
)
from models.ensemble_models.pipelines.pipeline_generic import GenericPipeline
from general_utils.utility_functions import (
    load_data,
) 

# ---------------- Pfade ----------------

BASE_DIR = Path(__file__).parents[4]
TRAIN_PATH = BASE_DIR / "data/processed/trainset.csv"
TEST_PATH = BASE_DIR / "data/processed/testset.csv"
VAL_PATH = BASE_DIR / "data/processed/valset.csv"

# ---------------- Hilfsfunktion: Balkendiagramm ----------------
def plot_metrics(metrics: dict):
    """Säulendiagramm für Train/Test/Validation Accuracy"""
    plt.figure(figsize=(6, 4))
    plt.bar(metrics.keys(), metrics.values(), color=["skyblue", "orange", "green"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Train / Test / Validation Accuracy")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


# ---------------- Evaluation eines Modells ----------------
def evaluate_ensemble_model(run_id: str, pipeline: GenericPipeline):
    """
    Evaluate an ensemble model on Train, Test, and Validation sets.
    """
    ensemble = EnsembleModelManager()
    bm = ensemble.get_model_by_id(run_id)

    # Trainiertes Modell laden
    trained_model = ensemble.load_model_by_id(run_id)

    # Train, Test & Validation Daten laden
    train_df, test_df, val_df = load_data(TRAIN_PATH, TEST_PATH, VAL_PATH)

    # Pipeline auf Train fitten und Test/Validation transformieren
    X_train, y_train = pipeline.pipeline.fit(train_df)
    X_test, y_test = pipeline.pipeline.transform(test_df)
    X_val, y_val = pipeline.pipeline.transform(val_df)

    metrics = {}

    # Train Accuracy
    y_pred_train = trained_model.predict(X_train)
    metrics["Train"] = EnsembleModelManager.compute_metrics(y_train, y_pred_train)[
        "accuracy"
    ]

    # Test Accuracy
    y_pred_test = trained_model.predict(X_test)
    metrics["Test"] = EnsembleModelManager.compute_metrics(y_test, y_pred_test)[
        "accuracy"
    ]

    # Validation Accuracy
    y_pred_val = trained_model.predict(X_val)
    metrics["Validation"] = EnsembleModelManager.compute_metrics(y_val, y_pred_val)[
        "accuracy"
    ]

    # Ergebnisse anzeigen
    for k, v in metrics.items():
        print(f"Accuracy ({k}): {v:.4f}")

    # Plotten
    plot_metrics(metrics)


# ---------------- Gesamte Evaluation durchführen ----------------
def run_ensemble_evaluation():
    # Pipeline initialisieren
    pipeline = GenericPipeline(target_col="species")

    # Ensemble Manager initialisieren
    ensemble = EnsembleModelManager()

    # Bestes Modell auswählen
    best_run_id, best_model, _ = ensemble.get_best_model(metric="accuracy")

    print(f"Evaluating best model (run_id={best_run_id})...\n")
    evaluate_ensemble_model(best_run_id, pipeline)


if __name__ == "__main__":
    run_ensemble_evaluation()
