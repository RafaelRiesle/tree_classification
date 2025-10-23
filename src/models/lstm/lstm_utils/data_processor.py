from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

    def load_data(self):
        self.train_df = self.loader.load_transform(self.train_path)
        self.val_df = self.loader.load_transform(self.val_path)
        self.test_df = self.loader.load_transform(self.test_path)
        self.feature_columns = [
            c
            for c in self.train_df.columns
            if c not in ["time", "id", self.label_column, "disturbance_year"]
        ]

    def preprocess(self):
        # Encode labels
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

        # Scale features
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

    def run(self):
        self.load_data()
        self.preprocess()
        self.create_sequences_and_weights()
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
