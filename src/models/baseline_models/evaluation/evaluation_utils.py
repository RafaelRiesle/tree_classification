def evaluate_and_report(baseline_manager, run_id, top_n=20):
    exp = baseline_manager.get_baseline_model_by_id(run_id)
    print("Evaluating model:", exp["model"], exp["hyperparams"])

    baseline_manager.plot_confusion_matrix(run_id)
    baseline_manager.print_classification_report(run_id)

    try:
        baseline_manager.plot_feature_importances(run_id, top_n=top_n)
    except Exception:
        print("Feature importances not available for this model")


def get_best_model(baseline_manager, metric="accuracy", ascending=False):
    """
    Returns the best model based on a given metric.

    Args:
        baseline_manager: Instance of BaselineModelManager
        metric (str): The metric used for ranking (e.g. "accuracy", "f1_score")
        ascending (bool): False = higher is better (default), True = lower is better

    Returns:
        best_run_id (int): ID of the best run
        best_model (dict): Model details from the manager
        df_results (pd.DataFrame): Full results sorted by the metric
    """
    df_results = baseline_manager.load_baseline_models()

    if metric not in df_results.columns:
        raise ValueError(
            f"Metric '{metric}' not found. Available columns: {list(df_results.columns)}"
        )

    df_sorted = df_results.sort_values(metric, ascending=ascending).reset_index(
        drop=True
    )
    best_row = df_sorted.iloc[0]

    best_run_id = best_row["run_id"]
    best_model = baseline_manager.get_baseline_model_by_id(best_run_id)

    return best_run_id, best_model, df_sorted
