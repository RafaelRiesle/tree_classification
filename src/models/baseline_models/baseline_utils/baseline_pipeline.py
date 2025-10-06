import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class BasePipeline:
    def __init__(self, target_col):
        self.target_col = target_col
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.fitted = False
        self.categorical_cols = []

    def date_feature_extraction(self, df):
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        df["month_num"] = df["time"].dt.month
        df["year"] = df["time"].dt.year

        seasons = ["Winter", "Spring", "Summer", "Autumn"]
        df["season"] = df["month_num"].apply(lambda m: seasons[((m % 12) // 3)])

        # Calculate difference in days per 'id'
        df["date_diff"] = df.groupby("id")["time"].diff().dt.days.fillna(0)
        return df

    def drop_columns(self, df):
        # Drop columns that are not needed
        return df.drop(
            columns=["time", "id", "disturbed", "disturbance_year"], errors="ignore"
        )

    def fit(self, train_df):
        df = train_df.copy()
        df = self.date_feature_extraction(df)
        df = self.drop_columns(df)

        # Encode target variable
        df[self.target_col] = self.label_encoder.fit_transform(df[self.target_col])

        # Identify categorical features
        self.categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self.categorical_cols = [
            col for col in self.categorical_cols if col != self.target_col
        ]

        # One-hot encode categorical features
        df = pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        self.fitted = True
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), y

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Pipeline must first be trained with fit().")

        df = df.copy()
        df = self.date_feature_extraction(df)
        df = self.drop_columns(df)

        if self.target_col in df.columns:
            df[self.target_col] = self.label_encoder.transform(df[self.target_col])

        # One-hot encode categorical features (same columns as in training)
        df = pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)

        # Add missing columns with zeros
        for col in set(self.categorical_cols) - set(df.columns):
            df[col] = 0

        # Align column order with training features
        X = df.drop(columns=[self.target_col]) if self.target_col in df.columns else df
        X = X.reindex(columns=self.scaler.feature_names_in_, fill_value=0)

        # Scale features
        X_scaled = self.scaler.transform(X)

        y = df[self.target_col] if self.target_col in df.columns else None
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        return (X_scaled_df, y) if y is not None else X_scaled_df
