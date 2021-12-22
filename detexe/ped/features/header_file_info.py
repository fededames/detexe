import numpy as np
from sklearn.feature_extraction import FeatureHasher

from .base_feature import FeatureType


class HeaderFileInfo(FeatureType):
    """Machine, architecure, OS, linker and other information extracted from header"""

    name = "header"
    dim = 62

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        raw_obj = {}
        raw_obj["coff"] = {"timestamp": 0, "machine": "", "characteristics": []}
        raw_obj["optional"] = {
            "subsystem": "",
            "dll_characteristics": [],
            "magic": "",
            "major_image_version": 0,
            "minor_image_version": 0,
            "major_linker_version": 0,
            "minor_linker_version": 0,
            "major_operating_system_version": 0,
            "minor_operating_system_version": 0,
            "major_subsystem_version": 0,
            "minor_subsystem_version": 0,
            "sizeof_code": 0,
            "sizeof_headers": 0,
            "sizeof_heap_commit": 0,
        }
        if lief_binary is None:
            return raw_obj

        raw_obj["coff"]["timestamp"] = lief_binary.header.time_date_stamps
        raw_obj["coff"]["machine"] = str(lief_binary.header.machine).split(".")[-1]
        raw_obj["coff"]["characteristics"] = [
            str(c).split(".")[-1] for c in lief_binary.header.characteristics_list
        ]
        raw_obj["optional"]["subsystem"] = str(
            lief_binary.optional_header.subsystem
        ).split(".")[-1]
        raw_obj["optional"]["dll_characteristics"] = [
            str(c).split(".")[-1]
            for c in lief_binary.optional_header.dll_characteristics_lists
        ]
        raw_obj["optional"]["magic"] = str(lief_binary.optional_header.magic).split(
            "."
        )[-1]
        raw_obj["optional"][
            "major_image_version"
        ] = lief_binary.optional_header.major_image_version
        raw_obj["optional"][
            "minor_image_version"
        ] = lief_binary.optional_header.minor_image_version
        raw_obj["optional"][
            "major_linker_version"
        ] = lief_binary.optional_header.major_linker_version
        raw_obj["optional"][
            "minor_linker_version"
        ] = lief_binary.optional_header.minor_linker_version
        raw_obj["optional"][
            "major_operating_system_version"
        ] = lief_binary.optional_header.major_operating_system_version
        raw_obj["optional"][
            "minor_operating_system_version"
        ] = lief_binary.optional_header.minor_operating_system_version
        raw_obj["optional"][
            "major_subsystem_version"
        ] = lief_binary.optional_header.major_subsystem_version
        raw_obj["optional"][
            "minor_subsystem_version"
        ] = lief_binary.optional_header.minor_subsystem_version
        raw_obj["optional"]["sizeof_code"] = lief_binary.optional_header.sizeof_code
        raw_obj["optional"][
            "sizeof_headers"
        ] = lief_binary.optional_header.sizeof_headers
        raw_obj["optional"][
            "sizeof_heap_commit"
        ] = lief_binary.optional_header.sizeof_heap_commit
        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack(
            [
                raw_obj["coff"]["timestamp"],
                FeatureHasher(10, input_type="string")
                .transform([[raw_obj["coff"]["machine"]]])
                .toarray()[0],
                FeatureHasher(10, input_type="string")
                .transform([raw_obj["coff"]["characteristics"]])
                .toarray()[0],
                FeatureHasher(10, input_type="string")
                .transform([[raw_obj["optional"]["subsystem"]]])
                .toarray()[0],
                FeatureHasher(10, input_type="string")
                .transform([raw_obj["optional"]["dll_characteristics"]])
                .toarray()[0],
                FeatureHasher(10, input_type="string")
                .transform([[raw_obj["optional"]["magic"]]])
                .toarray()[0],
                raw_obj["optional"]["major_image_version"],
                raw_obj["optional"]["minor_image_version"],
                raw_obj["optional"]["major_linker_version"],
                raw_obj["optional"]["minor_linker_version"],
                raw_obj["optional"]["major_operating_system_version"],
                raw_obj["optional"]["minor_operating_system_version"],
                raw_obj["optional"]["major_subsystem_version"],
                raw_obj["optional"]["minor_subsystem_version"],
                raw_obj["optional"]["sizeof_code"],
                raw_obj["optional"]["sizeof_headers"],
                raw_obj["optional"]["sizeof_heap_commit"],
            ]
        ).astype(np.float32)

    def feature_names(self):
        out_vec = (
            [self.name + "_" + "timestamp"]
            + [self.name + "_" + "coff_machine_H" + str(i) for i in range(10)]
            + [self.name + "_" + "coff_characteristics_H" + str(i) for i in range(10)]
            + [self.name + "_" + "optional_subsystem_H" + str(i) for i in range(10)]
            + [
                self.name + "_" + "optional_dll_characteristics_H" + str(i)
                for i in range(10)
            ]
            + [self.name + "_" + "optional_magic" + str(i) for i in range(10)]
            + [
                self.name + "_" + "optional_" + x
                for x in [
                    "major_image_version",
                    "minor_image_version",
                    "major_linker_version",
                    "minor_linker_version",
                    "major_operating_system_version",
                    "minor_operating_system_version",
                    "major_subsystem_version",
                    "minor_subsystem_version",
                    "sizeof_code",
                    "sizeof_headers",
                    "sizeof_heap_commit",
                ]
            ]
        )
        assert len(out_vec) == self.dim
        return out_vec
