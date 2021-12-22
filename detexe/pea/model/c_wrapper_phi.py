from abc import abstractmethod

from secml.array import CArray
from secml.ml.classifiers import CClassifier


class CWrapperPhi:
    """
    Abstract class that encapsulates a models for being used in a black-box way.
    """

    def __init__(self, model: CClassifier):
        """
        Creates the wrapper.

        Parameters
        ----------
        model : CClassifier
        The models to wrap
        """
        self.classifier = model

    @abstractmethod
    def extract_features(self, x: CArray):
        """
        It maps the input sample inside the feature space of the wrapped models.

        Parameters
        ----------
        x : CArray
                The sample in the input space.
        Returns
        -------
        CArray
                The feature space representation of the input sample.
        """
        raise NotImplementedError(
            "This method is abstract, you should implement it somewhere else!"
        )

    def predict(self, x: CArray, return_decision_function: bool = True):
        """
        Returns the prediction of the sample (in input space).

        Parameters
        ----------
        x : CArray
                The input sample in input space.
        return_decision_function : bool, default True
                If True, it also returns the decision function value, rather than only the label.
                Default is True.
        Returns
        -------
        CArray, (CArray)
                Returns the label of the sample.
                If return_decision_function is True, it also returns the output of the decision function.
        """
        x = x.atleast_2d()
        # feature_vectors = []
        # for i in range(x.shape[0]):
        # 	x_i = x[i, :]
        # 	padding_position = x_i.find(x_i == 256)
        # 	if padding_position:
        # 		x_i = x_i[0, :padding_position[0]]
        # 	feature_vectors.append(self.extract_features(x_i))
        # feature_vectors = CArray(feature_vectors)
        feature_vectors = self.extract_features(x)
        return self.classifier.predict(
            feature_vectors, return_decision_function=return_decision_function
        )
