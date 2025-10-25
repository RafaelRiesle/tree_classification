import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

from pipelines.processing.processing_pipeline import ProcessingPipeline
from pipelines.processing.features.spectral_indices import CalculateIndices
from pipelines.processing.features.basic_features import BasicFeatures
from pipelines.processing.features.temporal_features import TemporalFeatures
from pipelines.processing.processing_steps.interpolation import Interpolation

from models.baseline_model.calculate_keyfigures import StatisticalFeatures
from general_utils.constants import spectral_bands, indices

bands_and_indices = spectral_bands + indices 

from pathlib import Path
from pipelines.preprocessing.run_preprocessing_pipeline import run_preprocessing_pipeline

# === Paths (adjust if needed) ===
BASE_DIR = Path(__file__).parents[1]
DATA_PATH = BASE_DIR / "data/raw/raw_trainset.csv"
SPLITS_PATH = BASE_DIR / "data/raw/splits"
PREPROCESSED_PATH = BASE_DIR / "data/preprocessed"

# === Run preprocessing ===
run_preprocessing_pipeline(
    data_path=DATA_PATH,
    splits_output_path=SPLITS_PATH,
    preprocessed_output_path=PREPROCESSED_PATH,
    sample_size=None,
    remove_outliers=False,
    force_split_creation=False,
)


train_df = pd.read_csv(PREPROCESSED_PATH / "trainset.csv", parse_dates=["time"])
test_df  = pd.read_csv(PREPROCESSED_PATH / "testset.csv", parse_dates=["time"])
val_df   = pd.read_csv(PREPROCESSED_PATH / "valset.csv", parse_dates=["time"])

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Val shape:", val_df.shape)


# PATH_TRAIN = DATA_DIR / "trainset.csv"
# PATH_TEST = DATA_DIR / "testset.csv"

# # Define pipeline steps
# steps = [
#     BasicFeatures(on=True),
#     Interpolation(on=True),
#     CalculateIndices(on=True),
#     TemporalFeatures(on=True),
# ]

# print("Running processing pipeline for training data...")
# pipeline_train = ProcessingPipeline(path=PATH_TRAIN, steps=steps)
# df_train = pipeline_train.run()

# print("Running processing pipeline for test data...")
# pipeline_test = ProcessingPipeline(path=PATH_TEST, steps=steps)
# df_test = pipeline_test.run()

# # Drop unwanted columns
# drop_cols = ["disturbance_year", "is_disturbed", "date_diff"]
# df_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
# df_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

# # Calculate key figures per ID
# sf = StatisticalFeatures()
# df_train = sf.calculate_keyfigures_per_id(df_train, bands_and_indices)
# df_test = sf.calculate_keyfigures_per_id(df_test, bands_and_indices)

# # Encode labels
# le = LabelEncoder()
# df_train["species_encoded"] = le.fit_transform(df_train["species"])

# X_train = df_train.drop(columns=["id", "species", "species_encoded"])
# y_train = df_train["species_encoded"]

# X_test = df_test.drop(columns=["id", "species"], errors="ignore")
# X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# X_train = X_train.fillna(0)
# X_test = X_test.fillna(0)

# # Train model
# xgb_baseline_model = xgb.XGBClassifier(
#     n_estimators=200,
#     learning_rate=0.1,
#     max_depth=10,
#     random_state=42,
#     eval_metric="mlogloss",
#     use_label_encoder=False,
#     objective="multi:softprob",
#     num_class=len(le.classes_),
# )

# print("Training model...")
# xgb_baseline_model.fit(X_train, y_train)

# # ✅ Save trained model
# model_path = OUTPUT_DIR / "baseline_xgb_model.joblib"
# joblib.dump(xgb_baseline_model, model_path)

# # ✅ Print confirmation message
# print("\n✅ Model training complete.")
# print(f"💾 Model saved in: {model_path}\n")

# # Predict on test set
# print("Predicting on test data...")
# y_pred = xgb_baseline_model.predict(X_test)

# # Evaluate if test labels exist
# if "species" in df_test.columns:
#     y_test_true = le.transform(df_test["species"])
#     print("\nConfusion Matrix:")
#     print(confusion_matrix(y_test_true, y_pred))

#     print("\nClassification Report:")
#     print(classification_report(y_test_true, y_pred, target_names=le.classes_))
# else:
#     print("No species labels found in test data — only predictions generated.")
