import torch
from pathlib import Path
import matplotlib.pyplot as plt
from models.lstm.pipeline.pipeline import prepare_data
from models.lstm.lstm_utils.support_function import weighted_accuracy, get_predictions
from models.lstm.lstm_utils.species_predictor import SpeciesPredictor
from models.lstm.lstm_utils.species_data_module import SpeciesDataModule


BASE_DIR = Path(__file__).parents[4]
TRAIN_PATH = BASE_DIR / "data/processed/trainset.csv"
TEST_PATH = BASE_DIR / "data/processed/testset.csv"
VAL_PATH = BASE_DIR / "data/processed/valset.csv"


def get_checkpoint_path(filename="species_model.ckpt") -> Path:
    return Path(__file__).parent.parent / "experiments" / "trained_model" / filename


def load_model(checkpoint_path: str, device: torch.device):
    model = SpeciesPredictor.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


def evaluate_dataset(model, dataloader, device) -> float:
    labels, preds = get_predictions(model, dataloader, device)
    return weighted_accuracy(labels, preds)


def plot_metrics(metrics: dict):
    plt.figure(figsize=(6, 4))
    plt.bar(metrics.keys(), metrics.values())
    plt.ylim(0, 1)
    plt.ylabel("Weighted Accuracy")
    plt.title("Weighted Accuracy across Datasets")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def evaluate_model(
    checkpoint_path: str,
    train_path: Path,
    test_path: Path,
    val_path: Path,
    batch_size: int = 50,
):

    data = prepare_data(train_path=train_path, test_path=test_path, val_path=val_path)

    data_module = SpeciesDataModule(
        train_seqs=data["train_sequences"],
        test_seqs=data["test_sequences"],
        val_seqs=data["val_sequences"],
        batch_size=batch_size,
    )

    device = data["device"]
    model = load_model(checkpoint_path, device)

    metrics = {
        "Train": evaluate_dataset(model, data_module.train_dataloader(), device),
        "Test": evaluate_dataset(model, data_module.test_dataloader(), device),
        "Validation": evaluate_dataset(model, data_module.val_dataloader(), device),
    }

    for k, v in metrics.items():
        print(f"Weighted Accuracy ({k}): {v:.4f}")

    plot_metrics(metrics)


def run_lstm_evaluation(
    train_path: Path,
    test_path: Path,
    val_path: Path,
    batch_size: int = 50,
):

    checkpoint = get_checkpoint_path()
    evaluate_model(
        checkpoint_path=str(checkpoint),
        train_path=train_path,
        test_path=test_path,
        val_path=val_path,
        batch_size=batch_size,
    )


if __name__ == "__main__":

    run_lstm_evaluation(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        val_path=VAL_PATH,
        batch_size=50,
    )
