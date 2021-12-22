import numpy as np

from .base_feature import FeatureType


class GeneralFileInfo(FeatureType):
    """General information about the file"""

    name = "general"
    dim = 10

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        if lief_binary is None:
            return {
                "size": len(bytez),
                "vsize": 0,
                "has_debug": 0,
                "exports": 0,
                "imports": 0,
                "has_relocations": 0,
                "has_resources": 0,
                "has_signature": 0,
                "has_tls": 0,
                "symbols": 0,
            }

        return {
            "size": len(bytez),
            "vsize": lief_binary.virtual_size,
            "has_debug": int(lief_binary.has_debug),
            "exports": len(lief_binary.exported_functions),
            "imports": len(lief_binary.imported_functions),
            "has_relocations": int(lief_binary.has_relocations),
            "has_resources": int(lief_binary.has_resources),
            "has_signature": int(lief_binary.has_signature),
            "has_tls": int(lief_binary.has_tls),
            "symbols": len(lief_binary.symbols),
        }

    def process_raw_features(self, raw_obj):
        return np.asarray(
            [
                raw_obj["size"],
                raw_obj["vsize"],
                raw_obj["has_debug"],
                raw_obj["exports"],
                raw_obj["imports"],
                raw_obj["has_relocations"],
                raw_obj["has_resources"],
                raw_obj["has_signature"],
                raw_obj["has_tls"],
                raw_obj["symbols"],
            ],
            dtype=np.float32,
        )

    def feature_names(self):
        return [
            self.name + "_" + x
            for x in [
                "size",
                "vsize",
                "has_debug",
                "exports",
                "imports",
                "has_relocations",
                "has_resources",
                "has_signature",
                "has_tls",
                "symbols",
            ]
        ]
