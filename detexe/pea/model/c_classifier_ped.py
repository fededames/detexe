import lightgbm
import numpy as np
from secml.array import CArray
from secml.ml.classifiers import CClassifier

from ...ped.extractor import PEFeatureExtractor


class CClassifierPED(CClassifier):
    """
    The wrapper for PED
    """

    def __init__(self, tree_path: str, config: str):
        """
        Create the PED classifier.

        Parameters
        ----------
        tree_path : str
                path to the tree parameters
        """
        super(CClassifierPED, self).__init__()
        self.model_config = config
        self._lgbm = self._load_tree(tree_path)
        self.features = self._n_features
        self.extractor = PEFeatureExtractor(self.model_config, truncate=True)

    def extract_features(self, x: CArray) -> CArray:
        """
        Extract features

        Parameters
        ----------
        x : CArray
                program sample
        Returns
        -------
        CArray
                selected features
        """

        x = x.atleast_2d()
        size = x.shape[0]
        features = []
        for i in range(size):
            x_i = x[i, :]
            x_bytes = bytes(x_i.astype(np.int).tolist()[0])
            features.append(
                np.array(self.extractor.feature_vector(x_bytes), dtype=np.float32)
            )
        features = CArray(features)
        return features

    def _backward(self, w):
        pass

    def _fit(self, x, y):
        raise NotImplementedError("Fit is not implemented.")

    def _load_tree(self, tree_path):
        booster = lightgbm.Booster(model_file=tree_path)
        self._classes = 2
        self._n_features = booster.num_feature()
        return booster

    def _forward(self, x):
        x = x.atleast_2d()
        scores = self._lgbm.predict(x.tondarray())
        confidence = [[1 - c, c] for c in scores]
        confidence = CArray(confidence)
        return confidence
