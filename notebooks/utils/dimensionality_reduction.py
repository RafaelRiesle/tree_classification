import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import plotly.express as px


class SITS_DimensionalityReduction:
    def __init__(
        self, method="PCA", n_components=2, scale=True, random_state=42, **kwargs
    ):
        """
        Parameters
        ----------
        method : str
            Dimensionality reduction method: 'PCA', 'tSNE', or 'UMAP'
        n_components : int
            Number of dimensions for projection
        scale : bool
            Whether to standardize the features
        random_state : int
            Random seed for reproducibility
        kwargs : dict
            Additional parameters for t-SNE or UMAP
        """
        self.method = method
        self.n_components = n_components
        self.scale = scale
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None
        self.transformed_ = None
        self.labels_ = None
        self.feature_names_ = None

    def fit(self, df, label_col=None, band_prefix="b", id_col="id"):
        band_cols = [c for c in df.columns if c.startswith(band_prefix)]
        if not band_cols:
            raise ValueError(f"No bands found with prefix '{band_prefix}'.")

        if id_col not in df.columns:
            raise ValueError(f"'{id_col}' column not found in DataFrame.")

        df_agg = df.groupby(id_col)[band_cols].mean().reset_index()

        X = df_agg[band_cols].values
        self.feature_names_ = band_cols
        self.ids_ = df_agg[id_col].values 

        if self.scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        if label_col and label_col in df.columns:
            # Take the first label per ID (or mode)
            df_labels = df.groupby(id_col)[label_col].first().reset_index()
            self.labels_ = df_labels[label_col].values
        else:
            self.labels_ = None

        if self.method.upper() == "PCA":
            self.model = PCA(
                n_components=self.n_components, random_state=self.random_state
            )
            self.transformed_ = self.model.fit_transform(X)
        elif self.method.upper() == "TSNE":
            self.model = TSNE(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs,
            )
            self.transformed_ = self.model.fit_transform(X)
        elif self.method.upper() == "UMAP":
            self.model = umap.UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs,
            )
            self.transformed_ = self.model.fit_transform(X)
        else:
            raise ValueError("Method must be 'PCA', 'tSNE', or 'UMAP'.")

        return self

    def plot_2d(self):
        if self.transformed_ is None:
            raise RuntimeError("Please run fit() first.")
        if self.transformed_.shape[1] < 2:
            raise ValueError("At least 2 components are required for a 2D plot.")

        df_plot = pd.DataFrame(self.transformed_[:, :2], columns=["Dim1", "Dim2"])
        df_plot["Label"] = self.labels_ if self.labels_ is not None else "Sample"
        if hasattr(self, "ids_"):
            df_plot["ID"] = self.ids_

        fig = px.scatter(
            df_plot,
            x="Dim1",
            y="Dim2",
            color="Label",
            hover_data=["ID"],
            title=f"{self.method} 2D Projection",
            opacity=0.8,
        )
        fig.show()
