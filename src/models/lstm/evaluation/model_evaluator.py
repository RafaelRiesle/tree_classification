import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
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

    def plot_training_history(self, filename="training_history.png"):
        min_len = min(
            len(self.model.train_acc_history),
            len(self.model.train_loss_history),
            len(self.model.val_acc_history),
        )
        epochs = range(1, min_len + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.model.train_acc_history[:min_len], label="Train Accuracy")
        plt.plot(
            epochs, self.model.val_acc_history[:min_len], label="Validation Accuracy"
        )
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

    def evaluate_confusion_matrix(self, filename="confusion_matrix.png"):
        self.model.eval()
        all_labels, all_preds = [], []

        for batch in self.data_module.test_dataloader():
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
        plt.title("Confusion Matrix - Test Set")
        path = self.save_dir / filename
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        return path

    def permutation_importance(self, filename="permutation_importance.png"):
        baseline_acc = self._evaluate_accuracy(self.data_module.test_dataloader())
        importances = np.zeros(len(self.feature_columns))

        for i in range(len(self.feature_columns)):
            acc_drop = []
            for batch in self.data_module.test_dataloader():
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
        plt.title("Top 15 Feature Importances")
        plt.tight_layout()
        path = self.save_dir / filename
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        return path

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
    
    def evaluate_metrics(self):
        """Berechnet verschiedene Testmetriken."""
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []

        for batch in self.data_module.test_dataloader():
            x = batch["sequence"].to(self.device)
            y = batch["label"].to(self.device)

            with torch.no_grad():
                outputs = self.model(x)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # falls binäre Klassifikation

        # In NumPy konvertieren
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # Verschiedene Metriken berechnen
        metrics = {
            "accuracy": np.mean(all_preds == all_labels),
            "precision": precision_score(all_labels, all_preds, average="weighted"),
            "recall": recall_score(all_labels, all_preds, average="weighted"),
            "f1_score": f1_score(all_labels, all_preds, average="weighted"),
        }

        # ROC-AUC nur für binäre Klassifikation
        if len(np.unique(all_labels)) == 2:
            metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)

        print("\n📊 Test Metrics:")
        for k, v in metrics.items():
            print(f"{k:>10}: {v:.4f}")

        print("\n🧾 Classification Report:")
        print(classification_report(all_labels, all_preds))

        return metrics

