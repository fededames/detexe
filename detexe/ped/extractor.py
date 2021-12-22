#!/usr/bin/python
""" Extracts some basic features from PE pe_files. Based on
https://github.com/elastic/ember/blob/master/ember/features.py

NOTE: In this file, feature extraction has been modified.
It is recommended making a diff to default ember features.py file.
"""

import hashlib
import logging
import typing

import lief
import numpy as np

from .features import *  # noqa

log = logging.getLogger(__name__)
LIEF_MAJOR, LIEF_MINOR, _ = lief.__version__.split(".")
LIEF_EXPORT_OBJECT = int(LIEF_MAJOR) > 0 or (
    int(LIEF_MAJOR) == 0 and int(LIEF_MINOR) >= 10
)


def parse_pe_file(bytez: bytes) -> lief.PE.Binary:
    """Return a lief binary object from the bytes of a PE file"""
    lief_errors = (
        lief.bad_format,
        lief.bad_file,
        lief.pe_error,
        lief.parser_error,
        lief.read_out_of_bound,
        RuntimeError,
    )
    try:
        lief_binary = lief.PE.parse(list(bytez))
    except lief_errors as e:
        log.warning(f"LIEF error: {e}")
        lief_binary = None  # TODO Should we raise Exception to malware prediction?
    except Exception:  # everything else (KeyboardInterrupt, SystemExit, ValueError):
        raise
    return lief_binary


class WrongFeatureSelection(Exception):
    def __init__(self):
        pass


class PEFeatureExtractor:
    """
    Extract the selected features with a config file from a PE file, and return a vector.
    """

    def __init__(self, config, truncate=True):
        features_list = self._get_selected_features(config)
        self.truncate = truncate
        self.features = [eval(feature + "()") for feature in features_list]
        self.dim = sum([fe.dim for fe in self.features])

    def _get_selected_features(self, config):
        features = []
        for line in open(config):
            str_line = line.strip()
            if str_line and not str_line.startswith("#"):
                features.append(str_line)
        if not features:
            log.error(
                "No feature selected from features selection file. Check features_selection.txt"
            )
            raise WrongFeatureSelection
        return features

    def raw_features(
        self, bytez: bytes, pe_binary: typing.Optional[lief.PE.Binary] = None
    ):
        """
        If truncate is true,the file will be truncated at the end of the virtual address.
        This leaves some overlay code if present. The idea is just to prevent very simple attacks that
        append a lot of bytes at the end of the PE file.
        """

        if pe_binary is not None:
            lief_binary = pe_binary
        else:
            lief_binary = parse_pe_file(bytez)

        if self.truncate and lief_binary:
            bytez = bytez[: lief_binary.virtual_size]

        features = {"sha256": hashlib.sha256(bytez).hexdigest()}
        features.update(
            {
                fe.name: fe.raw_features(bytez=bytez, lief_binary=lief_binary)
                for fe in self.features
            }
        )
        return features

    def process_raw_features(self, raw_obj):
        feature_vectors = [
            fe.process_raw_features(raw_obj[fe.name]) for fe in self.features
        ]
        return np.hstack(feature_vectors).astype(np.float32)

    def feature_vector(
        self,
        bytez: bytes,
        pe_binary: typing.Optional[lief.PE.Binary] = None,
    ):
        return self.process_raw_features(
            self.raw_features(bytez=bytez, pe_binary=pe_binary)
        )

    def feature_names(self):
        feature_vectors = [fe.feature_names() for fe in self.features]
        return np.hstack(feature_vectors)  # .astype(np.float32)
