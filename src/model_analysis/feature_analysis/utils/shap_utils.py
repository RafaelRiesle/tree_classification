import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap


class ShapVisualizer:
    """
    A class for generating, saving, and visualizing SHAP plots.
    """

    def __init__(
        self, shap_values, X_test, le, output_folder="feature_analysis/shap_plots"
    ):
        self.shap_values = shap_values.values
        self.X_test = X_test
        self.le = le
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    # ---------- Helpers ---------- #
    def _save_plot(self, filename, dpi=200):
        path = os.path.join(self.output_folder, filename)
        plt.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close()

    def _parse_features(self):
        def parse(col):
            parts = col.split("_")
            return parts[:3] if len(parts) >= 3 else ["unknown"] * 3

        feat_info = pd.DataFrame(
            [parse(c) for c in self.X_test.columns],
            columns=["band", "stat", "month"],
            index=self.X_test.columns,
        )
        return feat_info

    def _mean_abs_shap(self, axis=(0, 2)):
        return pd.Series(
            np.mean(np.abs(self.shap_values), axis=axis), index=self.X_test.columns
        )

    # ---------- Global Importance ---------- #
    def plot_global_mean(self, top_n=30, figsize=(10, 6), title=None):
        """
        Plot global feature importance based on mean(|SHAP|) averaged across samples and classes.
        """
        feat_imp = self._mean_abs_shap().sort_values(ascending=False)
        plt.figure(figsize=figsize)
        sns.barplot(x=feat_imp.values[:top_n], y=feat_imp.index[:top_n])
        plt.title(title or f"Top {top_n} Features by Mean |SHAP|")
        plt.xlabel("Mean |SHAP|")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()

    # ---------- Class Plots ---------- #
    def save_class_summary_plots(self, max_display=10, dpi=200):
        """
        Generate and save SHAP summary (beeswarm) plots for each class.
        """
        for i, cls in enumerate(self.le.classes_):
            shap.summary_plot(
                self.shap_values[:, :, i],
                self.X_test,
                show=False,
                max_display=max_display,
            )
            plt.title(f"SHAP Summary - {cls}")
            plt.tight_layout()
            self._save_plot(f"shap_{cls}.png", dpi)
        print(f"Saved summary plots in: {self.output_folder}")

    def plot_top_feature_dependence_per_class(self, dpi=200):
        """
        For each class, identify its top SHAP feature and generate a dependence plot.
        """
        top_feats = {}
        for i, cls in enumerate(self.le.classes_):
            feat_imp = np.mean(np.abs(self.shap_values[:, :, i]), axis=0)
            top_feat = self.X_test.columns[np.argmax(feat_imp)]
            top_feats[cls] = top_feat

            shap.dependence_plot(
                top_feat, self.shap_values[:, :, i], self.X_test, show=False
            )
            plt.title(f"Dependence: {top_feat} ({cls})")
            plt.tight_layout()
            self._save_plot(f"dependence_{cls}.png", dpi)
        print(f"Saved dependence plots in: {self.output_folder}")
        return top_feats

    def plot_summary_grid(self, plot_type="shap", ncols=3, figsize_scale=(6, 5)):
        """
        Display all saved SHAP class plots or dependence plots in a single grid.
        """
        files = [f"{plot_type}_{cls}.png" for cls in self.le.classes_]
        n = len(files)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(figsize_scale[0] * ncols, figsize_scale[1] * nrows)
        )
        axes = axes.flatten()
        for i, (cls, file) in enumerate(zip(self.le.classes_, files)):
            path = os.path.join(self.output_folder, file)
            if os.path.exists(path):
                axes[i].imshow(plt.imread(path))
                axes[i].set_title(cls)
                axes[i].axis("off")
        # Remove extra unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

    # ---------- Grouped Importance ---------- #
    def plot_grouped_importance(self):
        """
        Compute and visualize mean absolute SHAP values grouped by feature components
        (band, statistic, month). Expects feature names in the format 'band_stat_month'.
        """
        feat_info = self._parse_features()
        shap_mean = self._mean_abs_shap()

        def group_plot(by, title, ax):
            vals = shap_mean.groupby(feat_info[by]).mean().sort_values(ascending=False)
            sns.barplot(x=vals.values[:10], y=vals.index[:10], ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Mean |SHAP|")
            ax.set_ylabel("")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        group_plot("month", "Mean |SHAP| by Month", axes[0])
        group_plot("band", "Mean |SHAP| by Band (Top 10)", axes[1])
        group_plot("stat", "Mean |SHAP| by Statistic", axes[2])
        plt.tight_layout()
        plt.show()

    # ---------- Heatmaps ---------- #
    def plot_heatmaps_grouped(self):
        """
        Generate heatmaps showing SHAP importance grouped by combinations of
        band, statistic, and month.
        """
        feat_info = self._parse_features()
        shap_mean = self._mean_abs_shap()

        # Compute top 10 bands
        top10_bands = shap_mean.groupby(feat_info["band"]).mean().nlargest(10).index
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        def make_heatmap(idx, col, top_bands=None):
            data = feat_info.copy()
            if top_bands is not None and idx == "band":
                data = data[data["band"].isin(top_bands)]
            mat = (
                data.groupby([idx, col])
                .apply(lambda x: shap_mean[x.index].mean())
                .unstack()
            )
            if "month" in (idx, col):
                if col == "month":
                    mat = mat[[m for m in months if m in mat.columns]]
                else:
                    mat = mat.loc[[m for m in months if m in mat.index]]
            return mat

        heatmaps = [
            ("Band vs Month", make_heatmap("band", "month", top10_bands)),
            ("Band vs Statistic", make_heatmap("band", "stat", top10_bands)),
            ("Month vs Statistic", make_heatmap("month", "stat")),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, (title, data) in zip(axes, heatmaps):
            sns.heatmap(data, cmap="YlGnBu", ax=ax)
            ax.set_title(title)
            ax.set_xlabel(data.columns.name)
            ax.set_ylabel(data.index.name)
        plt.tight_layout()
        plt.show()
