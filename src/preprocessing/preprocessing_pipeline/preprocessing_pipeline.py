from preprocessing.preprocessing_pipeline.data_loader import DataLoader
from preprocessing.preprocessing_steps.outlier_cleaner import SITSOutlierCleaner
from preprocessing.preprocessing_pipeline.constants import spectral_bands, indices

class PreprocessingPipeline:
    def __init__(
        self,
        path,
        outlier_cleaning=False,
    ):
        self.path = path
        self.outlier_cleaning = outlier_cleaning

        self.data_loader = DataLoader()
        self.outlier_cleaner = SITSOutlierCleaner()

        self.df = None

    def run(self):
        df = self.data_loader.load_transform(self.path)

        if self.outlier_cleaning:
            df = self.outlier_cleaner.fit_transform(df, spectral_bands)
            df = self.outlier_cleaner.add_any_outlier_flag()

        return df