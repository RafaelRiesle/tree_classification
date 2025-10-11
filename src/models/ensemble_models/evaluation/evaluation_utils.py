from tabulate import tabulate


def evaluate_and_report(baseline_manager, run_id, top_n=20):
    """
    Evaluate a baseline model by run_id and print metrics, confusion matrix, and feature importances.

    Args:
        baseline_manager (EnsembleModelManager): The manager containing baseline models
        run_id (str): The ID of the model run to evaluate
        top_n (int): Number of top features to display
    """
    try:
        exp = baseline_manager.get_model_by_id(run_id)
    except ValueError as e:
        print(e)
        return

    print(
        f"\nEvaluating model: {exp['model']} with hyperparameters: {exp['hyperparams']}\n"
    )

    # Confusion matrix
    baseline_manager.plot_confusion_matrix(run_id)

    # Classification report
    baseline_manager.print_classification_report(run_id)

    # Feature importances (if available)
    try:
        baseline_manager.plot_feature_importances(run_id, top_n=top_n)
    except Exception:
        print("Feature importances not available for this model")


def get_best_model(baseline_manager, metric="accuracy", ascending=False):
    """
    Returns the best model based on a given metric.

    Args:
        baseline_manager (EnsembleModelManager): The manager containing baseline models
        metric (str): The metric used for ranking (e.g. "accuracy", "f1_macro")
        ascending (bool): False = higher is better (default), True = lower is better

    Returns:
        best_run_id (str): ID of the best run
        best_model (dict): Model details from the manager
        df_sorted (pd.DataFrame): Full results sorted by the metric
    """
    df_results = baseline_manager.load_models()

    if metric not in df_results.columns:
        raise ValueError(
            f"Metric '{metric}' not found. Available columns: {list(df_results.columns)}"
        )

    df_sorted = df_results.sort_values(metric, ascending=ascending).reset_index(
        drop=True
    )
    best_row = df_sorted.iloc[0]

    best_run_id = best_row["run_id"]
    best_model = baseline_manager.get_model_by_id(best_run_id)

    return best_run_id, best_model, df_sorted


def print_model_summary(best_run_id, best_model):
    """
    Print a concise summary of the best model and its parameters.
    """
    print("\n" + "=" * 60)
    print("BEST MODEL SUMMARY")
    print("=" * 60)

    model_info = [
        ["Run ID", best_run_id],
        ["Model", best_model.get("model")],
        ["Timestamp", best_model.get("timestamp")],
        ["Number of Features", len(best_model.get("features", []))],
    ]
    print(tabulate(model_info, tablefmt="fancy_grid"))

    print("\nHyperparameters:")
    params = best_model.get("hyperparams", {})
    if params:
        print(
            tabulate(
                params.items(), headers=["Parameter", "Value"], tablefmt="fancy_grid"
            )
        )
    else:
        print("No hyperparameters found.")

    print("\n" + "=" * 60)
