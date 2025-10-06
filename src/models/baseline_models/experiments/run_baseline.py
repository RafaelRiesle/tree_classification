from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from pipelines.pipeline_generic import GenericPipeline
from baseline_utils.train_test_split import DatasetSplitLoader
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "raw_trainset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "baseline_training"


loader = DatasetSplitLoader(DATA_PATH, OUTPUT_DIR)
df = loader.load_and_prepare()
splits = loader.create_splits(train_ratio=0.7, test_ratio=0.3, validation_ratio=0.0)
train_df = splits["train"]
test_df = splits["test"]

models = [
    (RandomForestClassifier, {"n_estimators": 2, "max_depth": 2}),
    # (LogisticRegression, {"max_iter": 200}),
    # (SVC, {"kernel": "linear"})
]

pipeline = GenericPipeline(target_col="species")
df_results = pipeline.run(train_df, test_df, models)
