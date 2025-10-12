import torch
import pandas as pd
import random
from pathlib import Path
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler

from general_utils.utility_functions import load_data
from models.lstm.lstm_utils.padded_species_dataset import PaddedSpeciesDataset
from models.lstm.lstm_utils.support_function import df_to_sequences

pl.seed_everything(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("../../../../data/processed")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
VAL_PATH = DATA_DIR / "val.csv"


def load_datasets(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    val_path: Path = VAL_PATH,
):
    train_df, test_df, val_df = load_data(
        train_path, test_path, val_path
    )
    return train_df, test_df, val_df


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list,
):
    scaler = StandardScaler()
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])
    val_df[feature_columns] = scaler.transform(val_df[feature_columns])

    return train_df, val_df, test_df, scaler


def encode_labels(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_column: str,
):
    le = LabelEncoder()
    le.fit(train_df[label_column])
    for df in (train_df, test_df, val_df):
        df[label_column] = le.transform(df[label_column])
    return train_df, test_df, val_df, le, len(le.classes_)


def create_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list,
    label_column: str,
):
    train_sequences = df_to_sequences(train_df, feature_columns, label_column)
    test_sequences = df_to_sequences(test_df, feature_columns, label_column)
    val_sequences = df_to_sequences(val_df, feature_columns, label_column)
    max_len = max(len(X) for X, _ in train_sequences)
    train_dataset = PaddedSpeciesDataset(train_sequences, max_len=max_len)
    test_dataset = PaddedSpeciesDataset(test_sequences, max_len=max_len)
    val_dataset = PaddedSpeciesDataset(val_sequences, max_len=max_len)
    return (
        train_sequences,
        val_sequences,
        test_sequences,
        train_dataset,
        test_dataset,
        val_dataset,
        max_len,
    )


def prepare_data(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    val_path: Path = VAL_PATH,
):
    train_df, test_df, val_df  = load_datasets(
        train_path=train_path, test_path=test_path, val_path=val_path
    )
    feature_columns = [
        c
        for c in train_df.columns
        if c not in ["id", "time", "species", "disturbance_year"]
    ]
    label_column = "species"
    train_df, test_df, val_df, scaler = scale_features(
        train_df, test_df, val_df, feature_columns
    )
    train_df, test_df, val_df, le, n_classes = encode_labels(
        train_df, test_df, val_df, label_column
    )
    train_seq, test_seq, val_seq, train_ds, test_ds, val_ds, max_len = create_datasets(
        train_df, test_df, val_df, feature_columns, label_column
    )
    return {
        "train_sequences": train_seq,
        "test_sequences": test_seq,
        "val_sequences": val_seq,
        "train_dataset": train_ds,
        "test_dataset": test_ds,
        "val_dataset": val_ds,
        "feature_columns": feature_columns,
        "n_classes": n_classes,
        "label_encoder": le,
        "scaler": scaler,
        "max_len": max_len,
        "device": DEVICE,
    }


if __name__ == "__main__":
    data = prepare_data(train_path=TRAIN_PATH, test_path=TEST_PATH, val_path=VAL_PATH)
    print("Data prepared")
