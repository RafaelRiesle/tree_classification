import torch
from sklearn.metrics import accuracy_score
from collections import Counter


def df_to_sequences(df, feature_columns, label_column):
    """Teilt DataFrame in Sequenzen pro ID auf"""
    sequences = []
    for _, group in df.groupby("id"):
        X = group[feature_columns]
        y = group[label_column].iloc[0]
        sequences.append((X, y))
    return sequences


def weighted_accuracy(y_true, y_pred):
    """Berechnet gewichtete Accuracy basierend auf Klassenhäufigkeit"""
    class_counts = Counter(y_true)
    sample_weights = [1.0 / class_counts[y] for y in y_true]
    return accuracy_score(y_true, y_pred, sample_weight=sample_weights)


def get_predictions(model, dataloader, device="cpu"):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            seq_batch = batch["sequence"].to(device)
            label_batch = batch["label"].to(device)
            outputs = model(seq_batch)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            labels.extend(label_batch.cpu().tolist())
    return labels, predictions
