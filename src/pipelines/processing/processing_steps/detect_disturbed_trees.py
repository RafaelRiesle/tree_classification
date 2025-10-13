import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb

from preprocessing.preprocessing_pipeline.constants import spectral_bands, indices


class DetectDisturbedTrees:
    
    def __init__(self, scale_pos_weight=4, random_state=42):
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.bands_and_indices = spectral_bands + indices
        self.model = None

    def scale_data(self, df):
        df_scaled = df.copy()
        df_scaled[self.bands_and_indices] = df.groupby("species")[self.bands_and_indices].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        return df_scaled

    def get_yearly_data(self, df_scaled):
        return df_scaled.groupby(["id", "year"])[self.bands_and_indices].mean().reset_index()

    def get_std(self, df_yearly):
        return df_yearly.groupby("id")[self.bands_and_indices].std().add_suffix("_std").reset_index()

    def get_slope(self, df_yearly):
        slope_data = []
        for id_, group in df_yearly.groupby("id"):
            if group["year"].nunique() < 2:
                continue
            slopes = {f"{col}_slope": linregress(group["year"], group[col]).slope for col in self.bands_and_indices}
            slopes["id"] = id_
            slope_data.append(slopes)
        return pd.DataFrame(slope_data)
   
    def get_label(self, df, df_std_slope):
        labels = df.groupby("id")["is_disturbed"].first().reset_index()
        return df_std_slope.merge(labels, on="id", how="left")

    def prepare_data(self, df):
        df_scaled = self.scale_data(df)
        df_yearly = self.get_yearly_data(df_scaled)
        df_std = self.get_std(df_yearly)
        df_slope = self.get_slope(df_yearly)
        df_std_slope = df_std.merge(df_slope, on="id", how="left")
        df_std_slope = self.get_label(df, df_std_slope)
        return df_std_slope
    

    def get_balanced_train_data(self, df_std_slope):
        disturbed = df_std_slope[df_std_slope["is_disturbed"]]
        healthy = df_std_slope[~df_std_slope["is_disturbed"]]
        top_features = ["b11_slope", "b5_slope", "b11_std", "gndvi_std"]

        healthy = healthy.copy()
        healthy["combi_top_features"] = healthy[top_features].mean(axis=1)
        healthy_sub = (
            healthy.sort_values(by="combi_top_features")
            .head(20000)
            .sample(n=10000, random_state=self.random_state)
        )

        return pd.concat([disturbed, healthy_sub]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
       

    def train_model(self, train_df):
        feature_cols = [c for c in train_df.columns if c.endswith(("_std", "_slope"))]
        X, y = train_df[feature_cols], train_df["is_disturbed"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.3, random_state=self.random_state
        )

        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=self.random_state,
            scale_pos_weight=self.scale_pos_weight,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        # print("Confusion Matrix:\n", confusion_matrix(y_test, model.predict(X_test)))
        # print("\nClassification Report:\n", classification_report(y_test, model.predict(X_test)))
        return model

    def apply_model(self, model, df_std_slope):
        feature_cols = [c for c in df_std_slope.columns if c.endswith(("_std", "_slope"))]
        df_std_slope["is_disturbed_pred"] = model.predict(df_std_slope[feature_cols])
        return df_std_slope[["id", "is_disturbed_pred"]]
    
    def change_disturbed_labels(self, df_final):
        """
        Updates species names and is_disturbed flags based on predicted disturbances
        """
        disturbed_mask = df_final["is_disturbed_pred"] == True
        df_final.loc[
            disturbed_mask & ~df_final["species"].str.endswith("_disturbed"),
            "species"
        ] = df_final.loc[
            disturbed_mask & ~df_final["species"].str.endswith("_disturbed"),
            "species"
        ].astype(str) + "_disturbed"

        df_final.loc[
            disturbed_mask & (df_final["is_disturbed"] == False),
            "is_disturbed"
        ] = True

        df_final = df_final.drop(columns=["is_disturbed_pred"])
        return df_final
    
    def run(self, df):
        full_df = self.prepare_data(df)
        train_df = self.get_balanced_train_data(full_df)
        model = self.train_model(train_df)
        self.model = model

        # Apply model to healthy trees only
        df_healthy = full_df[~full_df["is_disturbed"]].copy()
        df_pred = self.apply_model(model, df_healthy)

        df_disturbed = full_df[full_df["is_disturbed"]].copy()
        df_disturbed["is_disturbed_pred"] = True

        df_all_pred = pd.concat([df_pred, df_disturbed], ignore_index=True)
        df_final = df.merge(df_all_pred[["id", "is_disturbed_pred"]], on="id", how="left")

        df_final = self.change_disturbed_labels(df_final)

        return df_final
