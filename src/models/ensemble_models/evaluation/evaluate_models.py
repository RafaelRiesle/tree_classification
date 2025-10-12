from models.ensemble_models.ensemble_utils.ensemble_model_manager import (
    EnsembleModelManager,
)
from models.ensemble_models.evaluation.evaluation_utils import (
    evaluate_and_report,
    get_best_model,
    print_model_summary,
)
from tabulate import tabulate


# TODO validerungsdatensatz ebenfalls mit einbinden um modell auf besten modell anzuwenden
def run_evaluation():
    """
    Finds and evaluates the best model based on a chosen metric.
    """
    print("\nStarting Ensemble Model Evaluation...\n")

    ensemble = EnsembleModelManager()
    best_run_id, best_model, df_sorted = get_best_model(ensemble, metric="accuracy")

    # Print model summary
    print_model_summary(best_run_id, best_model)

    # Evaluate and report
    print("\nEvaluating best model...\n")
    evaluate_and_report(ensemble, best_run_id, top_n=20)

    # Show top 5 models
    print("\n" + "=" * 60)
    print("TOP 5 MODELS BY ACCURACY")
    print("=" * 60)
    top5 = df_sorted.head(5)[
        ["run_id", "model", "accuracy", "f1_macro", "precision_macro", "recall_macro"]
    ]
    print(tabulate(top5, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))


if __name__ == "__main__":
    run_evaluation()
