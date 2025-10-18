import numpy as np


def df_to_sequences(df, feature_columns, label_column):
    sequences = []
    for _, group in df.groupby("id"):
        X = group[feature_columns].astype(np.float32)
        y = group[label_column].iloc[0]
        sequences.append((X, y))
    return sequences
