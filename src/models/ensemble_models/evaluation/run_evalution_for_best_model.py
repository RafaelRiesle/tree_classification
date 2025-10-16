import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from models.ensemble_models.ensemble_utils.ensemble_model_manager import EnsembleModelManager
from models.ensemble_models.pipelines.pipeline_generic import GenericPipeline
from general_utils.utility_functions import load_data
import json
import seaborn as sns


# === Pfade ===
BASE_DIR = Path(__file__).parents[4]
TRAIN_PATH = BASE_DIR / "data/processed/trainset.csv"
TEST_PATH = BASE_DIR / "data/processed/testset.csv"
VAL_PATH = BASE_DIR / "data/processed/valset.csv"
RESULTS_DIR = BASE_DIR / "data/ensemble_training/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# === Plot-Funktion für Accuracy ===
def plot_metrics(metrics: dict, save_path: Path):
    plt.figure(figsize=(6, 4))
    plt.bar(metrics.keys(), metrics.values(), color=["skyblue", "orange", "green"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Train / Test / Validation Accuracy")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# === Hauptfunktion für Evaluation ===
def evaluate_and_save_model(run_id: str, pipeline: GenericPipeline, ensemble_manager: EnsembleModelManager):
    """
    Evaluiert ein Modell nach run_id, speichert Metriken, Confusion-Matrizen und Feature-Importances.
    """
    # Ergebnisordner für dieses Modell
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Modell laden
    model = ensemble_manager.load_model_by_id(run_id)

    # Daten laden
    train_df, test_df, val_df = load_data(TRAIN_PATH, TEST_PATH, VAL_PATH)
    X_train, y_train = pipeline.pipeline.fit(train_df)
    X_test, y_test = pipeline.pipeline.transform(test_df)
    X_val, y_val = pipeline.pipeline.transform(val_df)

    # Vorhersagen
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_val = model.predict(X_val)

    # Metriken berechnen
    metrics = {
        "Train": EnsembleModelManager.compute_metrics(y_train, y_pred_train)["accuracy"],
        "Test": EnsembleModelManager.compute_metrics(y_test, y_pred_test)["accuracy"],
        "Validation": EnsembleModelManager.compute_metrics(y_val, y_pred_val)["accuracy"],
    }

    # Metriken speichern
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    # Plot für Accuracy speichern
    plot_file = run_dir / f"accuracy_{run_id}.png"
    plot_metrics(metrics, plot_file)

    # Confusion Matrices speichern
    for name, (y_true, y_pred) in zip(
        ["train", "test", "validation"],
        [(y_train, y_pred_train), (y_test, y_pred_test), (y_val, y_pred_val)],
    ):
        cm = EnsembleModelManager.compute_metrics(y_true, y_pred)["confusion_matrix"]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name.capitalize()} Confusion Matrix - {run_id}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(run_dir / f"{name}_confusion_matrix.png")
        plt.close()

    # === Feature Importances Plot ===
    if hasattr(model, "feature_importances_"):
        feature_importances = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": model.feature_importances_,
        }).sort_values(by="Importance", ascending=False)

        top_features = feature_importances.head(15)

        plt.figure(figsize=(8, 6))
        sns.barplot(
            data=top_features,
            y="Feature",
            x="Importance",
            hue="Feature",  # <- das hier ist neu
            dodge=False,
            palette="viridis",
            legend=False
        )

        plt.title(f"Top 15 Feature Importances - {run_id}")
        plt.xlabel("Importance")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(run_dir / f"top15_feature_importances_{run_id}.png")
        plt.close()

    print(f"Evaluation von run_id={run_id} abgeschlossen. Ergebnisse gespeichert unter {run_dir}")


# === Wrapper für das beste Modell ===
def run_evaluation_for_best_model():
    """
    Läuft die Evaluation für das beste Modell im Ensemble.
    """
    pipeline = GenericPipeline(target_col="species")
    ensemble_manager = EnsembleModelManager()
    best_run_id, _, _ = ensemble_manager.get_best_model(metric="accuracy")
    print(f"Evaluating best model: {best_run_id}")
    evaluate_and_save_model(best_run_id, pipeline, ensemble_manager)


if __name__ == "__main__":
    run_evaluation_for_best_model()
