"""
file_vectorizer.py: Set of functions, to convert pe files and directories to vectors
"""

import concurrent.futures
import logging
import os
import pathlib
from pathlib import PurePath
from typing import Iterator, List, Tuple, Union

import lief
import numpy as np
from sklearn.model_selection import train_test_split

from .extractor import PEFeatureExtractor

log = logging.getLogger(__name__)


def is_pe_file(filename) -> bool:
    """Check if file is PE format"""
    output = os.popen("file {}".format(filename)).read()
    return "PE" in output and "executable" in output


def files_from_dirs(input_dirs: List[Union[str, pathlib.Path]]) -> Iterator[str]:
    """Yield file from a list of directories"""
    for directory in input_dirs:
        for root, dirs, files in os.walk(directory):
            for f in files:
                filename = os.path.join(root, f)
                yield filename


def remove_broken_pe_from_dirs(input_dirs: Union[str, pathlib.Path]) -> None:
    """Remove the files that can not be parsed thoprugh a PE parser"""
    for file in files_from_dirs(input_dirs):
        file_data = open(file, "rb").read()
        try:
            lief.PE.parse(list(file_data))
        except Exception:  # everything else (KeyboardInterrupt, SystemExit, ValueError):
            log.info(f"Removing no PE file: {file}")
            os.remove(file)


def pe_files_from_dirs(input_dirs: List[str]) -> str:
    """Yield files with only PE format from a list of directories"""
    for directory in input_dirs:
        for root, dirs, files in os.walk(directory):
            for f in files:
                filename = os.path.join(root, f)
                if is_pe_file(filename):
                    yield filename


def vec_from_pe_file(
    extractor: PEFeatureExtractor, pe_path: Union[str, pathlib.Path]
) -> np.ndarray:
    """Return a representative vector of a PE file, regarding the specified feature extractor"""
    log.info(f"Parsing {pe_path}")
    file_data = open(pe_path, "rb").read()
    vec_features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
    return np.expand_dims(vec_features, axis=0)


def vec_files_from_pe_dir(
    directory: Union[str, pathlib.Path],
    config: Union[str, pathlib.Path],
    verbose: bool,
) -> np.ndarray:
    """Return representative vectors for the PE files contained in a specified directory.
    The extracted features will be determined by the specified config file"""
    if not verbose:
        lief.logging.disable()
    futures = []
    extractor = PEFeatureExtractor(config=config, truncate=True)
    files_vec = np.empty((0, extractor.dim), int)
    """
    # debug mode:
    for pe_path in pe_files_from_dirs([directory]):
    files_vec = np.append(files_vec, vec_from_pe_file(extractor, pe_path), axis=0)
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        for pe_path in pe_files_from_dirs([directory]):
            futures.append(pool.submit(vec_from_pe_file, extractor, pe_path))
        for future in concurrent.futures.as_completed(futures):
            files_vec = np.append(files_vec, future.result(), axis=0)

    return files_vec


def get_features_from_malware_benign_dirs(
    malware_dir: PurePath, benign_dir: PurePath, config: PurePath, verbose: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Return representative vectors for the the PE files contained in malware and bening directories.
    The extracted features will be determined by the specified config file"""
    return vec_files_from_pe_dir(malware_dir, config, verbose), vec_files_from_pe_dir(
        benign_dir, config, verbose
    )


def label_and_split_vectorized_dataset(
    malware_vec: np.ndarray, benign_vec: np.ndarray
) -> Tuple[np.ndarray]:
    """Split into train and test dataset, returning both sets with its corresponding label."""
    x = np.concatenate((malware_vec, benign_vec))
    if len(x) < 10:
        log.error(
            "Not enough PE Files contained in data directory to train a detector."
        )
        raise InsufficientTrainingData
    y = np.array([1] * len(malware_vec) + [0] * len(benign_vec))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )
    return x_train, y_train, x_test, y_test


class InsufficientTrainingData(Exception):
    def __init__(self):
        pass
