import concurrent.futures
import logging
import os
from typing import List, Optional

from gensim.models import Word2Vec

from detexe.ped.extractor import parse_pe_file
from detexe.ped.features.opcode_vectors import OpCodeVectors
from detexe.ped.file_vectorizer import pe_files_from_dirs
from detexe.settings import (check_layout_exist, check_root_path,
                             read_directories_from_root)

log = logging.getLogger(__name__)


class EmptyData(Exception):
    def __init__(self):
        pass


def _get_ngrams_from_pe(pe_path: str, gram_size: int) -> List:
    log.info(f"Parsing {pe_path}")
    lief_binary = parse_pe_file(open(pe_path, "rb").read())
    opcode_info = OpCodeVectors(load=False)
    instructions = opcode_info.raw_features(lief_binary=lief_binary)
    return opcode_info.get_ngrams_from_instructions(instructions, gram_size=gram_size)


def train_opcode_vectors(
    malware_dir: Optional[str] = None,
    benign_dir: Optional[str] = None,
    output_w2v_model: Optional[str] = os.path.dirname(os.path.abspath(__file__))
    + "/w2v200_opcode",
    vector_size: Optional[int] = 200,
    window_length: Optional[int] = 10,
    gram_size: Optional[int] = 3,
) -> None:
    """Train W2V model considering the opcodes of a dataset of files."""
    if not malware_dir or not benign_dir:
        root_dir = check_root_path()
        check_layout_exist(root_dir)
        project_dirs = read_directories_from_root(root_dir)
        if not malware_dir:
            malware_dir = project_dirs.malware_dir
        if not benign_dir:
            benign_dir = project_dirs.benign_dir

    all_ngrams_features = []
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        for pe_path in pe_files_from_dirs([malware_dir, benign_dir]):
            futures.append(pool.submit(_get_ngrams_from_pe, pe_path, gram_size))
        for future in concurrent.futures.as_completed(futures):
            if not future.result():
                continue
            all_ngrams_features.append(future.result())
    min_word_count = 3
    downsampling = 1e-3
    num_workers = 8
    if not all_ngrams_features:
        log.error(
            "No data found to build W2V vocabulary. "
            "Please, introduce files in your data directories"
        )
        raise EmptyData
    model = Word2Vec(
        all_ngrams_features,
        vector_size=vector_size,
        window=window_length,
        min_count=min_word_count,
        sample=downsampling,
        workers=num_workers,
    )
    model.init_sims(replace=True)
    model.save(output_w2v_model)
