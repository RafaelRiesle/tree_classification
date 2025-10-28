import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    balanced_accuracy_score,
)


BASE_DIR = Path(__file__).parents[4]
RESULTS_DIR = BASE_DIR / "data/lstm_training/results"


class ModelEvaluator:
    def __init__(self, model, data_module, feature_columns, save_dir=RESULTS_DIR):
        self.model = model
        self.data_module = data_module
        self.feature_columns = feature_columns
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------- Plots -------------------------
    def plot_training_history(self, filename="training_history.png"):
        min_len = min(
            len(self.model.train_acc_history),
            len(self.model.train_loss_history),
            len(self.model.val_acc_history),
        )
        epochs = range(1, min_len + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.model.train_acc_history[:min_len], label="Train Accuracy")
        plt.plot(epochs, self.model.val_acc_history[:min_len], label="Validation Accuracy")
        plt.plot(
            epochs,
            self.model.train_loss_history[:min_len],
            label="Train Loss",
            linestyle="--",
            color="red",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Train/Validation Accuracy & Train Loss per Epoch")
        plt.legend()
        plt.grid(True)
        path = self.save_dir / filename
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        return path

    def evaluate_confusion_matrix(self, split="test", filename=None):
        self.model.eval()
        all_labels, all_preds = [], []

        if split == "train":
            data_loader = self.data_module.train_dataloader()
        elif split == "val":
            data_loader = self.data_module.val_dataloader()
        else:
            data_loader = self.data_module.test_dataloader()

        for batch in data_loader:
            x = batch["sequence"].to(self.device)
            y = batch["label"].to(self.device)
            with torch.no_grad():
                outputs = self.model(x)
                preds = torch.argmax(outputs, dim=1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {split.capitalize()} Set")
        if filename is None:
            filename = f"confusion_matrix_{split}.png"
        path = self.save_dir / filename
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        return path

    # ------------------------- Permutation Importance -------------------------
    def permutation_importance(self, split="test", filename=None):
        if split == "train":
            data_loader = self.data_module.train_dataloader()
        elif split == "val":
            data_loader = self.data_module.val_dataloader()
        else:
            data_loader = self.data_module.test_dataloader()

        baseline_acc = self._evaluate_accuracy(data_loader)
        importances = np.zeros(len(self.feature_columns))

        for i in range(len(self.feature_columns)):
            acc_drop = []
            for batch in data_loader:
                x = batch["sequence"].clone()
                y = batch["label"]
                idx = torch.randperm(x.size(0))
                x[:, :, i] = x[idx, :, i]
                with torch.no_grad():
                    outputs = self.model(x.to(self.device))
                    preds = torch.argmax(outputs, dim=1)
                acc_drop.append((preds.cpu() == y).float().mean().item())
            importances[i] = baseline_acc - np.mean(acc_drop)

        # Plot top 15 features
        sorted_idx = np.argsort(importances)[::-1]
        top_n = 15
        top_features = [self.feature_columns[i] for i in sorted_idx[:top_n]]
        top_importances = importances[sorted_idx[:top_n]]

        plt.figure(figsize=(12, 6))
        plt.barh(top_features[::-1], top_importances[::-1], color="skyblue")
        plt.xlabel("Permutation Importance")
        plt.title(f"Top 15 Feature Importances - {split.capitalize()} Set")
        if filename is None:
            filename = f"permutation_importance_{split}.png"
        path = self.save_dir / filename
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        return path

    # ------------------------- Accuracy Hilfsfunktion -------------------------
    def _evaluate_accuracy(self, data_loader):
        correct, total = 0, 0
        for batch in data_loader:
            x = batch["sequence"].to(self.device)
            y = batch["label"].to(self.device)
            with torch.no_grad():
                preds = torch.argmax(self.model(x), dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        return correct / total

    # ------------------------- Metrics Evaluation -------------------------
    def evaluate_metrics(self, split="test"):
        """Berechnet verschiedene Metriken für train/val/test"""
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []

        if split == "train":
            data_loader = self.data_module.train_dataloader()
        elif split == "val":
            data_loader = self.data_module.val_dataloader()
        else:
            data_loader = self.data_module.test_dataloader()

        for batch in data_loader:
            x = batch["sequence"].to(self.device)
            y = batch["label"].to(self.device)
            with torch.no_grad():
                outputs = self.model(x)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            if probs.shape[1] > 1:
                all_probs.extend(probs[:, 1].cpu().numpy())
            else:
                all_probs.extend([0]*len(y))

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        metrics = {
            "accuracy": np.mean(all_preds == all_labels),
            "precision": precision_score(all_labels, all_preds, average="weighted"),
            "recall": recall_score(all_labels, all_preds, average="weighted"),
            "f1_score": f1_score(all_labels, all_preds, average="weighted"),
            "weighted_accuracy": balanced_accuracy_score(all_labels, all_preds),
        }

        if len(np.unique(all_labels)) == 2:
            metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)

        print(f"\n📊 {split.capitalize()} Metrics:")
        for k, v in metrics.items():
            print(f"{k:>20}: {v:.4f}")

        print("\n🧾 Classification Report:")
        print(classification_report(all_labels, all_preds))

        return metrics
