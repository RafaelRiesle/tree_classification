# evaluate.py
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from lstm_utils.support_function import weighted_accuracy, get_predictions
from lstm_utils.species_predictor import SpeciesPredictor
from lstm_utils.species_data_module import SpeciesDataModule
from pipeline.pipeline import prepare_data

if __name__ == "__main__":
    data = prepare_data(
        "../../../data/baseline_training/trainset.csv",
        "../../../data/baseline_training/valset.csv",
        "../../../data/baseline_training/testset.csv",
    )

    batch_size = 50
    data_module = SpeciesDataModule(
        data["train_sequences"],
        data["val_sequences"],
        data["test_sequences"],
        batch_size=batch_size,
    )
    train_loader = data_module.train_dataloader()
    # Modell laden
    model = SpeciesPredictor.load_from_checkpoint("trained_model/species_model.ckpt")
    device = data["device"]
    model.to(device)
    model.eval()

    # Vorhersagen
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

    # Plot
    metrics = {"Train": train_acc, "Validation": val_acc, "Test": test_acc}
    plt.figure(figsize=(6, 4))
    plt.bar(metrics.keys(), metrics.values())
    plt.ylim(0, 1)
    plt.ylabel("Weighted Accuracy")
    plt.title("Weighted Accuracy across Datasets")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
