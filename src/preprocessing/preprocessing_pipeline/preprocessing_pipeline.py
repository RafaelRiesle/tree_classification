from preprocessing.preprocessing_pipeline.data_loader import DataLoader
from preprocessing.preprocessing_steps.outlier_cleaner import SITSOutlierCleaner
from preprocessing.preprocessing_steps.interpolation import Interpolation
from preprocessing.features.spectral_indices import CalculateIndices
from preprocessing.features.basic_features import BasicFeatures
from preprocessing.features.temporal_features import TemporalFeatures
from preprocessing.preprocessing_pipeline.constants import spectral_bands


class PreprocessingPipeline:
    def __init__(
        self,
        path,
        basic_features=True,
        calculate_indices=True,
        temporal_features=True,
        interpolate_b4 = True,
        outlier_cleaner=False,
    ):
        self.path = path
        self.basic_features_flag = basic_features
        self.calculate_indices_flag = calculate_indices
        self.temporal_features_flag = temporal_features
        self.interpolate_b4_flag = interpolate_b4
        self.outlier_cleaner_flag = outlier_cleaner

        self.data_loader = DataLoader()
        self.basic_feat = BasicFeatures()
        self.indices_calc = CalculateIndices()
        self.temporal_feat = TemporalFeatures()
        self.interpolator = Interpolation()       
        self.outlier_cleaner = SITSOutlierCleaner()

        self.df = None

    def run(self):
        
        df = self.data_loader.load_transform(self.path)

        if self.basic_features_flag:
            df = self.basic_feat.add_disturbance_flag(df)

        if self.calculate_indices_flag:
            df = self.indices_calc.add_all_indices(df)

        if self.temporal_features_flag:
            df = self.temporal_feat.run(df)

        if self.interpolate_b4_flag:
            df = self.interpolator.interpolate_b4(df)

        if self.outlier_cleaner_flag:
            self.outlier_cleaner.fit_transform(df, spectral_bands)
            self.outlier_cleaner.add_any_outlier_flag()
            df = self.outlier_cleaner.get_interpolated_only()

        self.df = df
        return df