import pandas as pd
from general_utils.constants import spectral_bands, indices
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb

from preprocessing.features.statistical_features import StatisticalFeatures

bands_and_indices = spectral_bands + indices

class AdjustLabels:
    def __init__(self, on=True):
        self.on = on
        self.bands_and_indices = bands_and_indices
        self.stats = StatisticalFeatures()

    def get_soil_disturbed_col(self, df):
        """
        Where 'disturbance_year' != 0.0 and 'species' == 'soil',
        rename 'species' to 'soil_disturbed'.
        """
        df = df.copy()
        mask = (df["disturbance_year"] != 0.0) & (df["species"] == "soil")
        df.loc[mask, "species"] = "soil_disturbed"
        return df

    def specify_label_disturbed(self, df):
        """
        Train a classifier to distinguish between Norway_spruce and Scots_pine,
        then apply it to 'disturbed' samples to specify their likely species.
        """
        df_train = self.stats.calculate_keyfigures_per_id(df, self.bands_and_indices)

        df_disturbed = df[df["species"] == "disturbed"]
        df_disturbed_before = df_disturbed[df_disturbed["time"].dt.year < df_disturbed["disturbance_year"]]

        df_spruce_pine = df_train[df_train["species"].isin(["Norway_spruce", "Scots_pine"])]
        df_spruce_pine = df_spruce_pine.drop(columns="id")

        df_spruce_pine.loc[:, "species"] = df_spruce_pine["species"].map({
            "Norway_spruce": 0,
            "Scots_pine": 1
        })

        X = df_spruce_pine.drop("species", axis=1)
        y = df_spruce_pine["species"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        y_test = y_test.astype(int)

        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
        )
        xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)
        # print("Confusion Matrix:")
        # print(confusion_matrix(y_test, y_pred))
        # print("\nClassification Report:")
        # print(classification_report(y_test, y_pred))

        df_disturbed_prepared = self.stats.calculate_keyfigures_per_id(
            df_disturbed_before, self.bands_and_indices
        )

        ids = df_disturbed_prepared["id"]
        X_disturbed = df_disturbed_prepared.drop(columns=["species", "id"], errors="ignore")
        y_pred_disturbed_class = xgb_model.predict(X_disturbed)

        df_disturbed_labels = pd.DataFrame({
            "id": ids,
            "species": y_pred_disturbed_class
        })

        label_map = {
            0: "Norway_spruce_disturbed",
            1: "Scots_pine_disturbed"
        }
        df_disturbed_labels["species"] = df_disturbed_labels["species"].map(label_map)

        df_updated = df.merge(
            df_disturbed_labels[["id", "species"]],
            on="id",
            how="left",
            suffixes=("", "_pred")
        )

        df_updated.loc[df_updated["species_pred"].notnull(), "species"] = df_updated["species_pred"]
        df_updated = df_updated.drop(columns=["species_pred"], errors="ignore")

        return df_updated

    def run(self, df):
        """
        Runs both label adjustments on a dataframe.
        """
        if not self.on:
            return df
        df = self.get_soil_disturbed_col(df)
        df = self.specify_label_disturbed(df)
        return df
