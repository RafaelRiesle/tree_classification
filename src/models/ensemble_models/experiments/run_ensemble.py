from pathlib import Path
from general_utils.utility_functions import load_data
from models.ensemble_models.pipelines.pipeline_generic import GenericPipeline
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb


BASE_DIR = Path(__file__).parents[4]
TRAIN_PATH = BASE_DIR / "data/processed/trainset.csv"
TEST_PATH = BASE_DIR / "data/processed/testset.csv"
VAL_PATH = BASE_DIR / "data/processed/valset.csv"


def define_models():
    return [
        (
            RandomForestClassifier,
            {"n_estimators": [100, 200],
              "max_depth": [10, 20],
            "min_samples_split": [1, 2],
              "max_features": ["log2", "sqrt"]}
        ),
        (
            xgb.XGBClassifier,
            {
                "n_estimators": [300, 500],
                "learning_rate": [0.05, 0.1],
                "max_depth": [6, 10],
            },
        ),
    ]


def train_models(train_df, test_df, models, target_col="species"):
    pipeline = GenericPipeline(target_col=target_col)
    pipeline.run(train_df, test_df, models)
    print("Training finished!")


def run_ensemble(train_path=TRAIN_PATH, test_path=TEST_PATH, val_path=VAL_PATH):
    train_df, test_df, val_df = load_data(train_path, test_path, val_path)
    models = define_models()

    for model_class, params in models:
        print(f"\nStarting training for model: {model_class.__name__}")
        pipeline = GenericPipeline()
        pipeline.run(train_df, test_df, [(model_class, params)], val_df=val_df)
        print(f"Training completed for {model_class.__name__}.\n")

    print("All models have been trained!")



if __name__ == "__main__":
    run_ensemble(train_path=TRAIN_PATH, test_path=TEST_PATH, val_path=VAL_PATH)
