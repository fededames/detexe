import logging
import os
from typing import Optional, Tuple

import lightgbm.basic
import numpy as np
from secml.array import CArray

from ..settings import (check_layout_exist, check_root_path,
                        read_directories_from_root)
from .blackbox.c_black_box_format_exploit_evasion import \
    CBlackBoxContentShiftingEvasionProblem
from .blackbox.c_blackbox_header_problem import CBlackBoxHeaderEvasionProblem
from .blackbox.c_blackbox_problem import CBlackBoxProblem
from .blackbox.c_gamma_evasion import CGammaEvasionProblem
from .blackbox.c_gamma_sections_evasion import CGammaSectionsEvasionProblem
from .blackbox.ga.c_base_genetic_engine import CGeneticAlgorithm
from .model.c_classifier_ped import CClassifierPED
from .model.c_feature_extractor_wrapper_phi import CFeatureExtractorWrapperPhi

log = logging.getLogger(__name__)


def get_carray_from_file(file):
    with open(file, "rb") as f:
        malware_bytes = f.read()
    return CArray(np.frombuffer(malware_bytes, dtype=np.uint8))


class WrongMalware(Exception):
    def __init__(self):
        pass


class Attacker:
    """An attacker of a detector model, given its configuration and directory o benign files to create
    evasions"""

    def __init__(self, model: str, benign_dir: Optional[str] = None):
        root_dir = check_root_path()
        check_layout_exist(root_dir)
        projects_dir = read_directories_from_root(root_dir)

        if not benign_dir:
            benign_dir = root_dir + "/data/benign"
        self._malware = None
        self.malware_filename = None
        self.model_dir = projects_dir.models_dir + "/" + model
        self.model = self.model_dir + "/" + model + ".model"
        self.benign_dir = benign_dir

        self.config_features = self.model_dir + "/features_selection.txt"
        try:
            c_ember = CClassifierPED(self.model, self.config_features)
            self.blackbox_model = CFeatureExtractorWrapperPhi(c_ember)
        except lightgbm.basic.LightGBMError:
            log.info("The specified model does not exist.")
            exit()
        self.adv_path = None
        self.c_malware = None
        self.preattack_prediction = None

    @property
    def malware(self):
        """Malware attribute. The selected attack will be done using it."""
        return self._malware

    @malware.setter
    def malware(self, malware_path: str):
        """Malware attribute. The selected attack will be done using it."""
        if not os.path.isfile(malware_path):
            log.error("The specified malware does not exist")
            raise WrongMalware
        self._malware = malware_path
        _, self.malware_filename = os.path.split(self._malware)
        self.preattack_prediction, self.c_malware = self._pre_attack()

    def _check_specified_malware(self):
        if not self._malware:
            log.error("Please define a malware to apply the attack.")
            raise WrongMalware

    def _pre_attack(self):
        """Preprocessing steps before launching attack"""
        c_malware = get_carray_from_file(self._malware)
        _, pre_prediction = self.blackbox_model.predict(c_malware, True)
        preattack_prediction = pre_prediction[0, 1].item()
        log.info(
            f"Model prediction before adversarial attack: {preattack_prediction:.2f}\n---"
        )
        return preattack_prediction, c_malware

    def _launch_attack(self, attack: CBlackBoxProblem) -> float:
        """Launch the selected attack and store the modified sample"""
        self._check_specified_malware()
        engine = CGeneticAlgorithm(attack)
        y_pred, adv_score, adv_malware, f_obj = engine.run(
            self.c_malware, self.preattack_prediction
        )
        engine.write_adv_to_file(adv_malware.X[0, :], self.adv_path)
        c_adversarial = get_carray_from_file(self.adv_path)
        _, postattack_prediction = self.blackbox_model.predict(c_adversarial, True)
        _, postattack_prediction = self.blackbox_model.predict(c_adversarial, True)
        metric_postattack_prediction = postattack_prediction[0, 1].item()
        log.info(
            f"Model prediction after adversarial attack: {metric_postattack_prediction:.2f}"
        )
        return metric_postattack_prediction

    def dos(self) -> float:
        """Perturbs the DOS header"""
        log.info("DOS Header attack")
        attack = CBlackBoxHeaderEvasionProblem(
            self.blackbox_model,
            population_size=10,
            optimize_all_dos=True,
            penalty_regularizer=1e-6,
            iterations=10,
        )

        self.adv_path = self.model_dir + f"/adv_dos_{self.malware_filename}"
        return self._launch_attack(attack)

    def shift(self) -> float:
        """Shifts the first section including a sequence of bytes."""
        log.info("Section shift attack")
        attack = CBlackBoxContentShiftingEvasionProblem(
            self.blackbox_model,
            population_size=10,
            penalty_regularizer=1e-6,
            iterations=10,
        )

        self.adv_path = self.model_dir + f"/adv_shift_{self.malware_filename}"
        return self._launch_attack(attack)

    def section_injection(self, sections: Optional[Tuple[str]] = (".data",)) -> float:
        """Injects sections from the files contained in the benign directory.
        The sections to inject from these files are given by the parameter sections.
        """
        log.info("Section injection attack")
        (
            section_population,
            what_from_who,
        ) = CGammaSectionsEvasionProblem.create_section_population_from_folder(
            self.benign_dir, how_many=10, sections_to_extract=sections
        )
        attack = CGammaSectionsEvasionProblem(
            section_population,
            self.blackbox_model,
            population_size=10,
            penalty_regularizer=1e-6,
            iterations=10,
            threshold=0,
        )

        self.adv_path = self.model_dir + f"/adv_section_{self.malware_filename}"
        return self._launch_attack(attack)

    def padding(self, sections: Optional[Tuple[str]] = (".rdata",)) -> float:
        """Appends bytes from the files contained in the benign directory to the end of the file.
        The bytes to inject from those files are coming from the indicated sections with the parameter sections.
        """
        log.info("Padding attack")
        (
            section_population,
            what_from_who,
        ) = CGammaSectionsEvasionProblem.create_section_population_from_folder(
            self.benign_dir, how_many=10, sections_to_extract=sections
        )

        attack = CGammaEvasionProblem(
            section_population,
            model_wrapper=self.blackbox_model,
            population_size=10,
            penalty_regularizer=1e-6,
            iterations=10,
        )

        self.adv_path = self.model_dir + f"/adv_padding_{self.malware_filename}"
        return self._launch_attack(attack)

    def all_attacks(self) -> Tuple[float, float, float, float]:
        """Launchs different attacks: Perturb DOS header, shift first section in file,
        append bytes to the end of the file, inject new sections"""
        return self.dos(), self.shift(), self.padding(), self.section_injection()
