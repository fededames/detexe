import lief
import numpy as np
from sklearn.feature_extraction import FeatureHasher

from .base_feature import FeatureType


class SectionInfo(FeatureType):
    """Information about section names, sizes and entropy.  Uses hashing trick
    to summarize all this section info into a feature vector.
    """

    name = "section"
    dim = 5 + 50 + 50 + 50 + 50 + 50

    def __init__(self):
        super(FeatureType, self).__init__()

    @staticmethod
    def _properties(s):
        return [str(c).split(".")[-1] for c in s.characteristics_lists]

    def raw_features(self, bytez, lief_binary):
        if lief_binary is None:
            return {"entry": "", "sections": []}

        # properties of entry point, or if invalid, the first executable section
        try:
            entry_section = lief_binary.section_from_offset(lief_binary.entrypoint).name
        except lief.not_found:
            # bad entry point, let's find the first executable section
            entry_section = ""
            for s in lief_binary.sections:
                if (
                    lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE
                    in s.characteristics_lists
                ):
                    entry_section = s.name
                    break

        raw_obj = {"entry": entry_section}
        raw_obj["sections"] = [
            {
                "name": s.name,
                "size": s.size,
                "entropy": s.entropy,
                "vsize": s.virtual_size,
                "props": self._properties(s),
            }
            for s in lief_binary.sections
        ]
        return raw_obj

    def process_raw_features(self, raw_obj):
        sections = raw_obj["sections"]
        general = [
            len(sections),  # total number of sections
            # number of sections with nonzero size
            sum(1 for s in sections if s["size"] == 0),
            # number of sections with an empty name
            sum(1 for s in sections if s["name"] == ""),
            # number of RX
            sum(
                1
                for s in sections
                if "MEM_READ" in s["props"] and "MEM_EXECUTE" in s["props"]
            ),
            # number of W
            sum(1 for s in sections if "MEM_WRITE" in s["props"]),
        ]
        # gross characteristics of each section
        section_sizes = [(s["name"], s["size"]) for s in sections]
        section_sizes_hashed = (
            FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
        )
        section_entropy = [(s["name"], s["entropy"]) for s in sections]
        section_entropy_hashed = (
            FeatureHasher(50, input_type="pair")
            .transform([section_entropy])
            .toarray()[0]
        )
        section_vsize = [(s["name"], s["vsize"]) for s in sections]
        section_vsize_hashed = (
            FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
        )
        entry_name_hashed = (
            FeatureHasher(50, input_type="string")
            .transform([raw_obj["entry"]])
            .toarray()[0]
        )
        characteristics = [
            p for s in sections for p in s["props"] if s["name"] == raw_obj["entry"]
        ]
        characteristics_hashed = (
            FeatureHasher(50, input_type="string")
            .transform([characteristics])
            .toarray()[0]
        )

        return np.hstack(
            [
                general,
                section_sizes_hashed,
                section_entropy_hashed,
                section_vsize_hashed,
                entry_name_hashed,
                characteristics_hashed,
            ]
        ).astype(np.float32)

    def feature_names(self):
        return (
            [
                self.name + "_" + x
                for x in [
                    "totalNoSections",
                    "NoSectionsNonZeroSize",
                    "NoSectionsEmptyName",
                    "NoRX",
                    "NoW",
                ]
            ]
            + [self.name + "_" + "SectionSizes_H" + str(i) for i in range(50)]
            + [self.name + "_" + "SectionEntropies_H" + str(i) for i in range(50)]
            + [self.name + "_" + "SectionVSizes_H" + str(i) for i in range(50)]
            + [self.name + "_" + "SectionEntryName_H" + str(i) for i in range(50)]
            + [self.name + "_" + "CharacteristicsEntry_H" + str(i) for i in range(50)]
        )
