from baseline_utils.baseline_model_manager import BaselineModelManager
from evaluation.evaluation_utils import evaluate_and_report, get_best_model

baseline = BaselineModelManager()

best_run_id, best_model, df_sorted = get_best_model(baseline, metric="accuracy")

print("Best model based on accuracy:")
print("Run ID:", best_run_id)
#print("Details:", best_model)

evaluate_and_report(baseline, best_run_id, top_n=20)
