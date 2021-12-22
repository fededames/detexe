import numpy as np

from .base_feature import FeatureType


class ByteHistogram(FeatureType):
    """Byte histogram (count + non-normalized) over the entire binary file"""

    name = "histogram"
    dim = 256

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        return counts.tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized

    def feature_names(self):
        return [self.name + "_" + str(0) for i in range(0, self.dim)]
