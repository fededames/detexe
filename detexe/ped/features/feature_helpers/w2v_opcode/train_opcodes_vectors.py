import concurrent.futures
import logging
from typing import List

from gensim.models import Word2Vec

from detexe.ped.extractor import parse_pe_file
from detexe.ped.features.opcode_vectors import OpCodeVectors
from detexe.ped.file_vectorizer import pe_files_from_dirs

log = logging.getLogger(__name__)


def get_ngrams_from_pe(pe_path, gram_size):
    log.info(f"Parsing {pe_path}")
    lief_binary = parse_pe_file(open(pe_path, "rb").read())
    opcode_info = OpCodeVectors(load=False)
    instructions = opcode_info.raw_features(lief_binary=lief_binary)
    return opcode_info.get_ngrams_from_instructions(instructions, gram_size=gram_size)


def w2v_model_from_dirs(
    directories: List[str],
    model_name: str,
    vector_size: int,
    window_length: int,
    gram_size: int,
) -> None:
    """Train W2v model considering the opcodes of a dataset of files."""
    all_ngrams_features = []
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        for pe_path in pe_files_from_dirs(directories):
            futures.append(pool.submit(get_ngrams_from_pe, pe_path, gram_size))
        for future in concurrent.futures.as_completed(futures):
            if not future.result():
                continue
            all_ngrams_features.append(future.result())
    min_word_count = 3
    downsampling = 1e-3
    num_workers = 8
    model = Word2Vec(
        all_ngrams_features,
        vector_size=vector_size,
        window=window_length,
        min_count=min_word_count,
        sample=downsampling,
        workers=num_workers,
    )
    model.init_sims(replace=True)
    model.save(model_name)


if __name__ == "__main__":
    # Training parameters setup
    vector_size = 200
    window_length = 10
    gram_size = 3

    output_w2v_model = "w2v_model_opcodes"
    malware_dir = (
        "/home/ubuntu/Desktop/projects/detexe/data/malware"  # /path/to/malware/dir
    )
    benign_dir = (
        "/home/ubuntu/Desktop/projects/detexe/data/benign"  # /path/to/benign/dir
    )

    w2v_model_from_dirs(
        [malware_dir, benign_dir],
        model_name=output_w2v_model,
        vector_size=vector_size,
        window_length=window_length,
        gram_size=gram_size,
    )
