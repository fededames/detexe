import numpy as np

from .base_feature import FeatureType


class DataDirectories(FeatureType):
    """Extracts size and virtual address of the first 15 data directories"""

    name = "datadirectories"
    dim = 15 * 2

    def __init__(self):
        super(FeatureType, self).__init__()
        self._name_order = [
            "EXPORT_TABLE",
            "IMPORT_TABLE",
            "RESOURCE_TABLE",
            "EXCEPTION_TABLE",
            "CERTIFICATE_TABLE",
            "BASE_RELOCATION_TABLE",
            "DEBUG",
            "ARCHITECTURE",
            "GLOBAL_PTR",
            "TLS_TABLE",
            "LOAD_CONFIG_TABLE",
            "BOUND_IMPORT",
            "IAT",
            "DELAY_IMPORT_DESCRIPTOR",
            "CLR_RUNTIME_HEADER",
        ]

    def raw_features(self, bytez, lief_binary):
        output = []
        if lief_binary is None:
            return output

        for data_directory in lief_binary.data_directories:
            output.append(
                {
                    "name": str(data_directory.type).replace("DATA_DIRECTORY.", ""),
                    "size": data_directory.size,
                    "virtual_address": data_directory.rva,
                }
            )
        return output

    def process_raw_features(self, raw_obj):
        features = np.zeros(2 * len(self._name_order), dtype=np.float32)
        for i in range(len(self._name_order)):
            if i < len(raw_obj):
                features[2 * i] = raw_obj[i]["size"]
                features[2 * i + 1] = raw_obj[i]["virtual_address"]
        return features

    def feature_names(self):
        out = []
        for x in self._name_order:
            out.append(self.name + "_" + x + "_Size")
            out.append(self.name + "_" + x + "_VirtAdr")
        assert len(out) == self.dim
        return out
