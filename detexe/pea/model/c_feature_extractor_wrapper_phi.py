from secml.array import CArray

from detexe.pea.model.c_classifier_ped import CClassifierPED

from .c_wrapper_phi import CWrapperPhi


class CFeatureExtractorWrapperPhi(CWrapperPhi):
    """
    Class that wraps a GBDT classifier with EMBER feature set.
    """

    def __init__(self, model: CClassifierPED):
        """
        Creates the wrapper of a CClassifierEmber.

        Parameters
        ----------
        model : CClassifierEmber
        The GBDT models to wrap
        """
        if not isinstance(model, CClassifierPED):
            raise ValueError(f"Input models is {type(model)} and not CClassifierEmber")
        super().__init__(model)

    def extract_features(self, x):
        """
        It extracts the EMBER hand-crafted features

        Parameters
        ----------
        x : CArray
                The sample in the input space.
        Returns
        -------
        CArray
                The feature space representation of the input sample.
        """
        x = x.atleast_2d()
        clf: CClassifierPED = self.classifier
        feature_vectors = CArray.zeros((x.shape[0], self.classifier.features))
        for i in range(x.shape[0]):
            x_i = x[i, :]
            padding_positions = x_i.find(x_i == 256)
            if padding_positions:
                feature_vectors[i, :] = clf.extract_features(
                    x_i[0, : padding_positions[0]]
                )
            else:
                feature_vectors[i, :] = clf.extract_features(x_i)
        return feature_vectors
