import lief

LIEF_MAJOR, LIEF_MINOR, _ = lief.__version__.split(".")
LIEF_EXPORT_OBJECT = int(LIEF_MAJOR) > 0 or (
    int(LIEF_MAJOR) == 0 and int(LIEF_MINOR) >= 10
)


class FeatureType(object):
    """Base class from which each feature type may inherit"""

    name = ""
    dim = 0

    def __repr__(self):
        return "{}({})".format(self.name, self.dim)

    def raw_features(self, bytez, lief_binary):
        """Generate a JSON-able representation of the file"""
        raise (NotImplementedError)

    def process_raw_features(self, raw_obj):
        """Generate a feature vector from the raw features"""
        raise (NotImplementedError)

    def feature_vector(self, bytez, lief_binary):
        """Directly calculate the feature vector from the sample itself. This should only be implemented differently
        if there are significant speedups to be gained from combining the two functions."""
        return self.process_raw_features(self.raw_features(bytez, lief_binary))

    def feature_names(self):
        """Generate a list that specifies the name of the columns"""
        raise (NotImplementedError)
