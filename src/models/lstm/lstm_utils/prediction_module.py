import torch
import pandas as pd
from lstm_utils.species_dataset import SpeciesDataset
from lstm_utils.data_loader import DataLoader


class PredictionModule:
    def __init__(self, model, feature_columns, label_encoder=None, device="cpu"):
        self.model = model.to(device)
        self.feature_columns = feature_columns
        self.label_encoder = label_encoder
        self.device = device

    def predict(self, sequences):
        """
        sequences: Liste von DataFrames oder (DataFrame_features, label) tuples
        Returns: predictions list
        """
        dataset = SpeciesDataset(
            sequences=[
                (seq, 0) if isinstance(seq, pd.DataFrame) else seq for seq in sequences
            ],
            feature_columns=self.feature_columns,
            label_encoder=self.label_encoder,
            max_length=None,
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        self.model.eval()
        predictions = []
        labels = []

        with torch.no_grad():
            for batch in loader:
                seq_batch = batch["sequence"].to(self.device)
                label_batch = batch["label"].to(self.device)

                _, outputs = self.model(seq_batch)
                batch_preds = torch.argmax(outputs, dim=1)

                predictions.extend(batch_preds.cpu().tolist())
                labels.extend(label_batch.cpu().tolist())

        return predictions, labels

    def evaluate_accuracy(self, sequences_with_labels):
        predictions, labels = self.predict(sequences_with_labels)
        from sklearn.metrics import accuracy_score

        acc = accuracy_score(labels, predictions)
        return acc, predictions, labels
