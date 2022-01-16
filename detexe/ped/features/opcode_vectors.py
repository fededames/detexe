import logging
import pathlib
import re

import lief
import numpy as np
from capstone import CS_ARCH_X86, CS_MODE_32, Cs
from gensim.models import Word2Vec
from nltk import ngrams

from .base_feature import FeatureType

W2V_DIR = pathlib.Path(__file__).parent.absolute() / "feature_helpers"
W2V_PATH = W2V_DIR / "w2v200_opcode"
log = logging.getLogger(__name__)

gensim_logger = logging.getLogger("gensim")
gensim_logger.setLevel(logging.WARNING)


class OpCodeVectors(FeatureType):
    """
    Extract Opcode vectors utilizing a pretrained w2v model
    """

    name = "Opcode"
    dim = 200
    w2v_path = W2V_PATH

    def __init__(self, load=True):
        super(FeatureType, self).__init__()
        if load:
            logging.info("Loading w2v model..")
            self.model = Word2Vec.load(str(self.w2v_path))

    def raw_features(self, lief_binary, bytez=None):
        hex_mem_address = "-?0[xX][0-9a-fA-F]+"
        if lief_binary is None:
            return []
        # properties of entry point, or if invalid, the first executable section
        try:
            entry_section = lief_binary.section_from_offset(lief_binary.entrypoint)
        except lief.not_found:
            entry_section = None
            # bad entry point, let's find the first executable section
            for s in lief_binary.sections:
                if (
                    lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE
                    in s.characteristics_lists
                ):
                    entry_section = s
                    break
        if not entry_section:
            return []
        disassembler = Cs(CS_ARCH_X86, CS_MODE_32)
        instructions = []
        # disassemble the code
        for instruction in disassembler.disasm(
            bytes(entry_section.content), entry_section.size
        ):
            clean_operand = re.sub(
                hex_mem_address, "memadd", instruction.op_str
            ).replace(" ", "")
            instructions.append("%s_%s" % (instruction.mnemonic, clean_operand))
        return instructions

    def avg_vectorize_opcodes(self, file_opcodes):
        """
        Average the word vectors for a set of words
        """

        feature_vec = np.zeros(
            (self.model.vector_size,), dtype="float32"
        )  # pre-initialize (for speed)
        nwords = 0.0
        index2word_set = set(self.model.wv.index_to_key)  # words known to the models
        for opcode in file_opcodes:
            if opcode in index2word_set:
                nwords = nwords + 1
                feature_vec = np.add(feature_vec, self.model.wv[opcode])
        if nwords > 0:
            feature_vec = np.divide(feature_vec, nwords)
        else:
            log.debug("0 features")
        return feature_vec

    def get_ngrams_from_instructions(self, instructions, gram_size=3):
        grams = []
        if instructions:
            grams = [
                "_".join(splitted_ngram)
                for splitted_ngram in ngrams(instructions, gram_size)
            ]
        return grams

    def process_raw_features(self, raw_obj):
        grams = self.get_ngrams_from_instructions(raw_obj)
        w2v_vector = self.avg_vectorize_opcodes(grams)
        return w2v_vector
