from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import pickle
import os
from models.lstm.lstm_utils.data_loader import CSVDataLoader
from models.lstm.lstm_utils.utility_functions import df_to_sequences


class DataProcessor:
    def __init__(self, train_path, val_path, test_path, label_column="species"):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.label_column = label_column
        self.feature_columns = None
        self.loader = CSVDataLoader()

        self.exclude_columns = [
            "time",
            "id",
            "disturbance_year",
            "is_disturbed",
            "date_diff",
            "year",
            "doy",
        ]

    def load_data(self):
        self.train_df = self.loader.load_transform(self.train_path)
        self.val_df = self.loader.load_transform(self.val_path)
        self.test_df = self.loader.load_transform(self.test_path)
        self.feature_columns = [
            c
            for c in self.train_df.columns
            if c not in self.exclude_columns + [self.label_column]
        ]

        print("\n=== TRAIN COLUMNS ===")
        print(self.train_df.columns.tolist())
        print("\n=== VALIDATION COLUMNS ===")
        print(self.val_df.columns.tolist())
        print("\n=== TEST COLUMNS ===")
        print(self.test_df.columns.tolist())

    def preprocess(self):
        self.le = LabelEncoder()
        self.train_df[self.label_column] = self.le.fit_transform(
            self.train_df[self.label_column]
        )
        self.val_df[self.label_column] = self.le.transform(
            self.val_df[self.label_column]
        )
        self.test_df[self.label_column] = self.le.transform(
            self.test_df[self.label_column]
        )

        categorical_cols = [
            "season",
            "is_growing_season",
            "month_num",
            "biweek_of_year",
        ]
        self.train_df = pd.get_dummies(self.train_df, columns=categorical_cols)
        self.val_df = pd.get_dummies(self.val_df, columns=categorical_cols)
        self.test_df = pd.get_dummies(self.test_df, columns=categorical_cols)
        # 
        self.val_df = self.val_df.reindex(columns=self.train_df.columns, fill_value=0)
        self.test_df = self.test_df.reindex(columns=self.train_df.columns, fill_value=0)

        self.feature_columns = [
            c
            for c in self.train_df.columns
            if c not in self.exclude_columns + [self.label_column]
        ]

        self.scaler = StandardScaler()
        self.train_df[self.feature_columns] = self.scaler.fit_transform(
            self.train_df[self.feature_columns]
        )
        self.val_df[self.feature_columns] = self.scaler.transform(
            self.val_df[self.feature_columns]
        )
        self.test_df[self.feature_columns] = self.scaler.transform(
            self.test_df[self.feature_columns]
        )

    def create_sequences_and_weights(self):
        self.train_sequences = df_to_sequences(
            self.train_df, self.feature_columns, self.label_column
        )
        self.val_sequences = df_to_sequences(
            self.val_df, self.feature_columns, self.label_column
        )
        self.test_sequences = df_to_sequences(
            self.test_df, self.feature_columns, self.label_column
        )

        labels = [y for _, y in self.train_sequences]
        counts = Counter(labels)
        self.class_weights = [
            1.0 / counts[i] if i in counts else 0.0
            for i in range(len(self.le.classes_))
        ]

    def save_encoders(self, save_dir="encoders"):
        """Speichert LabelEncoder, StandardScaler und Feature-Namen als Pickle-Dateien."""
        os.makedirs(save_dir, exist_ok=True)

        # LabelEncoder speichern
        le_path = os.path.join(save_dir, "label_encoder.pkl")
        with open(le_path, "wb") as f:
            pickle.dump(self.le, f)
        print(f"LabelEncoder gespeichert unter: {le_path}")

        # StandardScaler speichern
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"StandardScaler gespeichert unter: {scaler_path}")

        # Feature-Namen speichern
        features_path = os.path.join(save_dir, "feature_columns.pkl")
        with open(features_path, "wb") as f:
            pickle.dump(self.feature_columns, f)
        print(f"Feature-Namen gespeichert unter: {features_path}")

    def run(self):
        self.load_data()
        self.preprocess()
        self.create_sequences_and_weights()
        self.save_encoders()

        print("\nFeature Columns after Encoding & Scaling:")
        print(len(self.feature_columns))
        print(self.feature_columns)

        return {
            "train_sequences": self.train_sequences,
            "val_sequences": self.val_sequences,
            "test_sequences": self.test_sequences,
            "feature_columns": self.feature_columns,
            "class_weights": self.class_weights,
            "n_classes": len(self.le.classes_),
            "le": self.le,
            "scaler": self.scaler,
        }
