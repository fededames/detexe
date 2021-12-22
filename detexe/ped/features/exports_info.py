import numpy as np
from sklearn.feature_extraction import FeatureHasher

from .base_feature import LIEF_EXPORT_OBJECT, FeatureType


class ExportsInfo(FeatureType):
    """Information about exported functions. Note that the total number of exported
    functions is contained in GeneralFileInfo.
    """

    name = "exports"
    dim = 128

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        if lief_binary is None:
            return []

        # Clipping assumes there are diminishing returns on the discriminatory power of exports beyond
        #  the first 10000 characters, and this will help limit the dataset size
        if LIEF_EXPORT_OBJECT:
            # export is an object with .name attribute (0.10.0 and later)
            clipped_exports = [
                export.name[:10000] for export in lief_binary.exported_functions
            ]
        else:
            # export is a string (LIEF 0.9.0 and earlier)
            clipped_exports = [
                export[:10000] for export in lief_binary.exported_functions
            ]
        return clipped_exports

    def process_raw_features(self, raw_obj):
        exports_hashed = (
            FeatureHasher(self.dim, input_type="string")
            .transform([raw_obj])
            .toarray()[0]
        )
        return exports_hashed.astype(np.float32)

    def feature_names(self):
        return [self.name + "_H" + str(i) for i in range(128)]
