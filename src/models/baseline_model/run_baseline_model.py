import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

from pipelines.preprocessing.run_preprocessing_pipeline import run_preprocessing_pipeline
from pipelines.processing.processing_pipeline import ProcessingPipeline
from pipelines.processing.features.spectral_indices import CalculateIndices
from pipelines.processing.features.basic_features import BasicFeatures
from pipelines.processing.features.temporal_features import TemporalFeatures
from pipelines.processing.processing_steps.interpolation import Interpolation

from models.baseline_model.baseline_model_utils import drop_unwanted_columns, split_into_X_y, evaluate_model
from models.baseline_model.calculate_keyfigures import StatisticalFeatures
from general_utils.constants import spectral_bands, indices


bands_and_indices = spectral_bands + indices

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "../../data/raw/raw_trainset.csv"
SPLITS_PATH = BASE_DIR / "../../data/raw/splits"
PREPROCESSED_PATH = BASE_DIR / "../../data/preprocessed"

TOP_LEVEL_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = TOP_LEVEL_DIR / "data" / "baseline_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Run preprocessing
run_preprocessing_pipeline(
    data_path=DATA_PATH,
    splits_output_path=SPLITS_PATH,
    preprocessed_output_path=PREPROCESSED_PATH,
    sample_size=None,
    remove_outliers=False,
    force_split_creation=False,
)

PATH_TRAIN = PREPROCESSED_PATH / "trainset.csv"
PATH_TEST = PREPROCESSED_PATH / "testset.csv"

# Define processing pipeline
steps = [
    BasicFeatures(on=True),
    Interpolation(on=True),
    CalculateIndices(on=True),
    TemporalFeatures(on=True),
]

print("Running processing pipeline for training data...")
pipeline_train = ProcessingPipeline(path=PATH_TRAIN, steps=steps)
df_train = pipeline_train.run()

print("Running processing pipeline for test data...")
pipeline_test = ProcessingPipeline(path=PATH_TEST, steps=steps)
df_test = pipeline_test.run()

df_train = drop_unwanted_columns(df_train)
df_test = drop_unwanted_columns(df_test)

# Group by id and calculate keyfigures 
sf = StatisticalFeatures()
df_train = sf.calculate_keyfigures_per_id(df_train, bands_and_indices)
df_test = sf.calculate_keyfigures_per_id(df_test, bands_and_indices)

df_train.to_csv(OUTPUT_DIR / "df_train.csv", index=False)
df_test.to_csv(OUTPUT_DIR / "df_test.csv", index=False)

# Encode labels
le = LabelEncoder()
df_train["species_encoded"] = le.fit_transform(df_train["species"])

X_train, y_train, X_test = split_into_X_y(df_train, df_test)

# Train model
xgb_baseline_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    random_state=42,
    eval_metric="mlogloss",
    objective="multi:softprob",
    num_class=len(le.classes_),
)

print("Training model...")
xgb_baseline_model.fit(X_train, y_train)

# Save model
model_path = OUTPUT_DIR / "baseline_xgb_model.joblib"
joblib.dump(xgb_baseline_model, model_path)
xgb_baseline_model.get_booster().save_model(OUTPUT_DIR / "baseline_xgb_model.json")

print("\nModel training complete.")
print(f"Model saved at: {model_path}\n")

evaluate_model(xgb_baseline_model, X_test, df_test, le)