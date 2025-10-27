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

    def __init__(self, shap_values, X_test, le, output_folder="feature_analysis/shap_plots"):
        self.shap_values = shap_values
        self.X_test = X_test
        self.le = le
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def plot_global_mean(self, top_n=30, figsize=(10, 6), title=None):
        """
        Plot global feature importance based on mean(|SHAP|) averaged across samples and classes.
        """
        mean_abs = np.mean(np.abs(self.shap_values.values), axis=(0, 2))
        feat_imp = pd.Series(mean_abs, index=self.X_test.columns).sort_values(ascending=False)

        plt.figure(figsize=figsize)
        sns.barplot(x=feat_imp.values[:top_n], y=feat_imp.index[:top_n])
        plt.title(title or f"Mean |SHAP| (averaged across classes) - top {top_n} features")
        plt.xlabel("Mean |SHAP|")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()


    def save_class_summary_plots(self, max_display=10, dpi=200):
        """
        Generate and save SHAP summary (beeswarm) plots for each class.
        """
        for i, cls in enumerate(self.le.classes_):
            shap.summary_plot(
                self.shap_values.values[:, :, i],
                self.X_test,
                show=False,
                max_display=max_display
            )
            plt.title(f"SHAP summary (beeswarm) - class: {cls}")
            plt.tight_layout()

            save_path = os.path.join(self.output_folder, f"shap_{cls}.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
            plt.close()

        print(f"SHAP summary plots saved in: {self.output_folder}")


    def plot_top_feature_dependence_per_class(self, max_display=10, dpi=200):
        """
        For each class, identify its top SHAP feature and generate a dependence plot.
        """
        os.makedirs(self.output_folder, exist_ok=True)

        n_classes = len(self.le.classes_)
        top_features = []

        for i, cls in enumerate(self.le.classes_):
            # Compute mean(|SHAP|) across samples for this class
            mean_abs = np.mean(np.abs(self.shap_values.values[:, :, i]), axis=0)
            feat_imp = pd.Series(mean_abs, index=self.X_test.columns).sort_values(ascending=False)

            # Top feature for this class
            top_feat = feat_imp.index[0]
            top_features.append(top_feat)

            # Create dependence plot
            shap.dependence_plot(
                top_feat,
                self.shap_values.values[:, :, i],
                self.X_test,
                interaction_index="auto",
                show=False
            )
            plt.title(f"SHAP Dependence: {top_feat} ({cls})")
            plt.tight_layout()

            save_path = os.path.join(self.output_folder, f"dependence_{cls}.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
            plt.close()

        print(f"Dependence plots saved in: {self.output_folder}")

        return dict(zip(self.le.classes_, top_features))


    def plot_summary_grid(self, ncols=3, figsize_scale=(6, 5), plot_type="shap"):
        """
        Display all saved SHAP class plots or dependence plots in a single grid.
        """
        valid_types = {"shap", "dependence"}
        if plot_type not in valid_types:
            raise ValueError(f"plot_type must be one of {valid_types}")

        n = len(self.le.classes_)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(figsize_scale[0] * ncols, figsize_scale[1] * nrows)
        )
        axes = axes.flatten()

        for i, cls in enumerate(self.le.classes_):
            img_path = os.path.join(self.output_folder, f"{plot_type}_{cls}.png")
            if not os.path.exists(img_path):
                print(f"⚠️ Missing file: {img_path}")
                continue
            img = plt.imread(img_path)
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"{cls}", fontsize=14)

        # Remove extra unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=3.0)
        plt.show()


    def plot_grouped_importance(self):
        """
        Compute and visualize mean absolute SHAP values grouped by feature components
        (band, statistic, month). Expects feature names in the format 'band_stat_month'.
        """

        mean_abs = np.mean(np.abs(self.shap_values.values), axis=(0, 2))
        shap_mean_abs = pd.Series(mean_abs, index=self.X_test.columns)

        # Parse feature names into components
        def parse_feature_name(col):
            parts = col.split("_")
            if len(parts) >= 3:
                return parts[0], parts[1], parts[2]
            else:
                return "unknown", "unknown", "unknown"

        feat_info = pd.DataFrame(
            [parse_feature_name(c) for c in self.X_test.columns],
            columns=["band", "stat", "month"],
            index=self.X_test.columns
        )

        # Aggregate SHAP values by parsed feature attributes
        by_month = shap_mean_abs.groupby(feat_info["month"]).mean().sort_values(ascending=False)
        by_band = shap_mean_abs.groupby(feat_info["band"]).mean().sort_values(ascending=False).head(10)
        by_stat = shap_mean_abs.groupby(feat_info["stat"]).mean().sort_values(ascending=False)

        def plot_shap_bar(values, labels, ax, title, color):
            sns.barplot(x=values, y=labels, ax=ax, color=color)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Mean |SHAP|")
            ax.set_ylabel("")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        plot_shap_bar(by_month.values, by_month.index, axes[0], "Mean |SHAP| by Month", "#1f77b4")
        plot_shap_bar(by_band.values, by_band.index, axes[1], "Mean |SHAP| by Top 10 Bands and Indices", "#76bb74")
        plot_shap_bar(by_stat.values, by_stat.index, axes[2], "Mean |SHAP| by Statistic", "#8977c5")

        plt.tight_layout()
        plt.show()