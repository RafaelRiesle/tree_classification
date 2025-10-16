import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class PaddedSpeciesDataset(Dataset):
    def __init__(self, sequences, pad_value=0.0, max_len=None):
        self.sequences = sequences
        self.pad_value = pad_value
        self.max_len = (
            max_len if max_len is not None else max(len(X) for X, _ in sequences)
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, y = self.sequences[idx]
        if isinstance(X, pd.DataFrame):
            X = X.values
        seq_len, n_features = X.shape
        padded_X = np.full((self.max_len, n_features), self.pad_value, dtype=np.float32)
        padded_X[:seq_len, :] = X
        return {
            "sequence": torch.tensor(padded_X, dtype=torch.float32),
            "label": torch.tensor(y, dtype=torch.long),
            "length": seq_len,
        }
