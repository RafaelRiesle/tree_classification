from preprocessing.preprocessing_steps.interpolation import Interpolation
from preprocessing.preprocessing_steps.detect_disturbed_trees import DetectDisturbedTrees
from preprocessing.features.spectral_indices import CalculateIndices
from preprocessing.features.basic_features import BasicFeatures
from preprocessing.features.temporal_features import TemporalFeatures
from preprocessing.preprocessing_steps.data_augmentation import DataAugmentation
from general_utils.constants import spectral_bands
from preprocessing.preprocessing_steps.adjust_labels import AdjustLabels


class ProcessingPipeline:
    def __init__(
        self,
        df,
        basic_features=True,
        calculate_indices=True,
        temporal_features=True,
        interpolate_b4=True,
        detect_disturbed_trees=True,
        data_augmentation=True,
        specify_disturbed_labels =True
    ):

        self.basic_features_flag = basic_features
        self.calculate_indices_flag = calculate_indices
        self.temporal_features_flag = temporal_features
        self.interpolate_b4_flag = interpolate_b4
        self.detect_disturbed_trees_flag = detect_disturbed_trees
        self.data_augmentation_flag = data_augmentation
        self.specify_disturbed_labels_flag = specify_disturbed_labels

        self.basic_feat = BasicFeatures()
        self.indices_calc = CalculateIndices()
        self.temporal_feat = TemporalFeatures()
        self.interpolator = Interpolation()
        self.disturbed_detector = DetectDisturbedTrees()
        self.data_augmenter = DataAugmentation()
        self.label_adjuster = AdjustLabels(bands_and_indices=spectral_bands)

        self.df = None

    def run(self, df):

        if self.basic_features_flag:
            df = self.basic_feat.add_disturbance_flag(df)

        if self.calculate_indices_flag:
            df = self.indices_calc.add_all_indices(df)

        if self.temporal_features_flag:
            df = self.temporal_feat.run(df)

        if self.interpolate_b4_flag:
            df = self.interpolator.interpolate_b4(df)

        if self.detect_disturbed_trees_flag:
            df = self.disturbed_detector.run(df)

        if self.data_augmentation_flag:
            df = self.data_augmenter.run(df)
        
        if self.specify_disturbed_labels_flag:
            df = self.label_adjuster.run(df)

        self.df = df
        return df