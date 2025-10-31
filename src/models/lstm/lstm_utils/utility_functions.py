import numpy as np

def df_to_sequences(df, feature_columns, label_column=None):
    sequences = []
    for _, group in df.groupby("id"):
        X = group[feature_columns].astype(np.float32).values
        if label_column is not None and label_column in group.columns:
            y = group[label_column].iloc[0]
            sequences.append((X, y))
        else:
            sequences.append(X)
    return sequences
