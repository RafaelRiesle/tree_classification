import torch
import pandas as pd
import random
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lstm_utils.padded_species_dataset import PaddedSpeciesDataset
from lstm_utils.data_loader import DataLoader
from lstm_utils.support_function import df_to_sequences

pl.seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

def prepare_data(train_path, val_path, test_path):
    loader = DataLoader()
    train_df = loader.load_transform(train_path)
    val_df = loader.load_transform(val_path)
    test_df = loader.load_transform(test_path)

    # Feature-Spalten bestimmen
    feature_columns = [
        col
        for col in train_df.columns
        if col not in ["id", "time", "species", "disturbance_year"]
    ]
    label_column = "species"

    # --- Normalisierung der Features ---
    scaler = StandardScaler()
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    val_df[feature_columns] = scaler.transform(val_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])
    # --- Ende Normalisierung ---

    # Label-Encoding
    le = LabelEncoder()
    le.fit(train_df[label_column])
    train_df[label_column] = le.transform(train_df[label_column])
    val_df[label_column] = le.transform(val_df[label_column])
    test_df[label_column] = le.transform(test_df[label_column])
    n_classes = len(le.classes_)

    # Sequenzen erstellen
    train_sequences = df_to_sequences(train_df, feature_columns, label_column)
    val_sequences = df_to_sequences(val_df, feature_columns, label_column)
    test_sequences = df_to_sequences(test_df, feature_columns, label_column)

    max_len_train = max(len(X) for X, _ in train_sequences)

    train_dataset = PaddedSpeciesDataset(train_sequences, max_len=max_len_train)
    val_dataset = PaddedSpeciesDataset(val_sequences, max_len=max_len_train)
    test_dataset = PaddedSpeciesDataset(test_sequences, max_len=max_len_train)

    return {
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "test_sequences": test_sequences,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "feature_columns": feature_columns,
        "n_classes": n_classes,
        "label_encoder": le,
        "max_len_train": max_len_train,
        "device": device,
        "scaler": scaler  # falls du sie später für neue Daten brauchst
    }



if __name__ == "__main__":
    data = prepare_data(
        "../../../data/baseline_training/trainset.csv",
        "../../../data/baseline_training/valset.csv",
        "../../../data/baseline_training/testset.csv"
    )
    print("Data Prepared")

    random_seq_idx = random.randint(0, len(data["train_sequences"]) - 1)
    X_seq, y_seq = data["train_sequences"][random_seq_idx]
    df_seq = pd.DataFrame(X_seq, columns=data["feature_columns"])
    df_seq["label"] = y_seq  
    print(df_seq.head(1))  

