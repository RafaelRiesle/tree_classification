import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)


def drop_unwanted_columns(df, cols_to_drop=None):
    if cols_to_drop is None:
        cols_to_drop = ["disturbance_year", "is_disturbed", "date_diff"]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


def split_into_X_y(df_train, df_test, target_col="species_encoded"):
    """
    Split train and test dataframes into X and y, align columns, fill missing values.
    """
    X_train = df_train.drop(columns=["id", "species", target_col])
    y_train = df_train[target_col]

    X_test = df_test.drop(columns=["id", "species"], errors="ignore")
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    return X_train, y_train, X_test


def evaluate_model(model, X_test, df_test, label_encoder):
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained classifier
        X_test (pd.DataFrame): Test features
        df_test (pd.DataFrame): Original test dataframe with true labels (optional)
        label_encoder: Fitted LabelEncoder for decoding classes
    """
    print("Predicting on test data...")
    y_pred = model.predict(X_test)

    if "species" in df_test.columns:
        y_test_true = label_encoder.transform(df_test["species"])

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test_true, y_pred))

        print("\nClassification Report:")
        print(
            classification_report(
                y_test_true, y_pred, target_names=label_encoder.classes_
            )
        )

        bal_acc = balanced_accuracy_score(y_test_true, y_pred)
        print(f"\nBalanced Accuracy: {bal_acc:.4f}")
    else:
        print("No species labels found in test data — only predictions generated.")
