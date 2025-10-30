import torch.nn as nn


class ImprovedSequenceModel(nn.Module):
    def __init__(
        self, n_features, n_classes, n_hidden=256, n_layers=2, bidirectional=True
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=bidirectional,
        )
        self.fc = nn.Sequential(
            nn.Linear(n_hidden * (2 if bidirectional else 1), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        return self.fc(out)
