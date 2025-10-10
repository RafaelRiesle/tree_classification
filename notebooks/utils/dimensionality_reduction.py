import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.constants import spectral_bands


class SITS_DimensionalityReduction:
    def __init__(
        self, method="PCA", n_components=2, scale=True, random_state=42, **kwargs
    ):
        self.method = method.upper()
        self.n_components = n_components
        self.scale = scale
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None
        self.transformed_ = None
        self.labels_ = None
        self.ids_ = None

    def fit(self, df, label_col=None, id_col="id"):
        if id_col not in df.columns:
            raise ValueError(f"'{id_col}' not in DataFrame.")
        df_agg = df.groupby(id_col)[spectral_bands].mean().reset_index()
        X = df_agg[spectral_bands].values
        self.ids_ = df_agg[id_col].values
        if self.scale:
            X = StandardScaler().fit_transform(X)
        if label_col in df.columns:
            self.labels_ = df.groupby(id_col)[label_col].first().values

        if self.method == "PCA":
            self.model = PCA(
                n_components=self.n_components, random_state=self.random_state
            )
        elif self.method == "TSNE":
            self.model = TSNE(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs,
            )
        else:
            raise ValueError("Method must be 'PCA' or 'TSNE'.")
        self.transformed_ = self.model.fit_transform(X)
        return self

    def plot_2d(self, figsize=(10, 6), marker_size=5, alpha=0.6):
        if self.transformed_ is None:
            raise RuntimeError("Call fit() first.")
        x, y = self.transformed_[:, 0], self.transformed_[:, 1]
        plt.figure(figsize=figsize)
        if self.labels_ is not None:
            for l in np.unique(self.labels_):
                idx = self.labels_ == l
                plt.scatter(x[idx], y[idx], s=marker_size, alpha=alpha, label=str(l))
            plt.legend(title="Label")
        else:
            plt.scatter(x, y, s=marker_size, alpha=alpha)
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.title(f"{self.method} 2D Projection")
        plt.grid(True)
        plt.show()
