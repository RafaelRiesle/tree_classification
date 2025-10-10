import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl

from sklearn.preprocessing import LabelEncoder
from lstm_utils.padded_species_dataset import PaddedSpeciesDataset


from lstm_utils.data_loader import DataLoader
from lstm_utils.species_data_module import SpeciesDataModule
from lstm_utils.species_predictor import SpeciesPredictor
from lstm_utils.support_function import df_to_sequences, weighted_accuracy, get_predictions
pl.seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


if __name__ == "__main__":
    train_path = "../../../data/baseline_training/trainset.csv"
    val_path = "../../../data/baseline_training/valset.csv"
    test_path = "../../../data/baseline_training/testset.csv"

    loader = DataLoader()
    train_df = loader.load_transform(train_path)
    val_df = loader.load_transform(val_path)
    test_df = loader.load_transform(test_path)


    feature_columns = [
        col
        for col in train_df.columns
        if col not in ["id", "time", "species", "disturbance_year"]
    ]
    label_column = "species"


    le = LabelEncoder()
    le.fit(train_df[label_column])
    train_df[label_column] = le.transform(train_df[label_column])
    val_df[label_column] = le.transform(val_df[label_column])
    test_df[label_column] = le.transform(test_df[label_column])
    n_classes = len(le.classes_)


    train_sequences = df_to_sequences(train_df, feature_columns, label_column)
    val_sequences = df_to_sequences(val_df, feature_columns, label_column)
    test_sequences = df_to_sequences(test_df, feature_columns, label_column)


    max_len_train = max(len(X) for X, _ in train_sequences)
    train_dataset = PaddedSpeciesDataset(train_sequences, max_len=max_len_train)
    val_dataset = PaddedSpeciesDataset(val_sequences, max_len=max_len_train)
    test_dataset = PaddedSpeciesDataset(test_sequences, max_len=max_len_train)

    # -------------------------------------------------------------------------
    # Trainingsparameter
    # -------------------------------------------------------------------------
    n_features = len(feature_columns)
    batch_size = 50
    lr = 1e-3
    max_epochs = 2

    # -------------------------------------------------------------------------
    # DataModule & Model
    # -------------------------------------------------------------------------
    data_module = SpeciesDataModule(
        train_sequences, val_sequences, test_sequences, batch_size=batch_size
    )
    model = SpeciesPredictor(n_features=n_features, n_classes=n_classes, lr=lr)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, datamodule=data_module)

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    train_labels, train_preds = get_predictions(
        model, data_module.train_dataloader(), device
    )
    val_labels, val_preds = get_predictions(model, data_module.val_dataloader(), device)
    test_labels, test_preds = get_predictions(
        model, data_module.test_dataloader(), device
    )

    train_acc = weighted_accuracy(train_labels, train_preds)
    val_acc = weighted_accuracy(val_labels, val_preds)
    test_acc = weighted_accuracy(test_labels, test_preds)

    print(f"Weighted Accuracy (Train): {train_acc:.4f}")
    print(f"Weighted Accuracy (Validation): {val_acc:.4f}")
    print(f"Weighted Accuracy (Test): {test_acc:.4f}")

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    metrics = {"Train": train_acc, "Validation": val_acc, "Test": test_acc}
    plt.figure(figsize=(6, 4))
    plt.bar(metrics.keys(), metrics.values())
    plt.ylim(0, 1)
    plt.ylabel("Weighted Accuracy")
    plt.title("Weighted Accuracy across Datasets")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
