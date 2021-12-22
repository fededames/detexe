import logging
import os
import traceback
from typing import Optional

from .ped.features import features

log = logging.getLogger(__name__)


class WrongEnvironmentVariable(Exception):
    def __init__(self):
        pass


class WrongLayout(Exception):
    def __init__(self):
        pass


def read_directories_from_root(root: str) -> dict:
    """Define data directories and model directory having root path as reference"""
    data_dir = root + "/data"
    models_dir = root + "/models"
    malware_dir = data_dir + "/malware"
    benign_dir = data_dir + "/benign"
    return {
        "data_dir": data_dir,
        "models_dir": models_dir,
        "malware_dir": malware_dir,
        "benign_dir": benign_dir,
    }


def check_layout_exist(root) -> None:
    project_dirs = read_directories_from_root(root)
    for key, value in project_dirs.items():
        if not os.path.isdir(value):
            log.error(
                "Wrong model/data structure. Please use the option to configure the project layout."
            )
            raise WrongLayout


def check_root_path() -> str:
    """Check if DETEXE_ROOT environment variable was set up correctly"""
    root_dir = os.environ.get("DETEXE_ROOT")
    if not root_dir:
        log.error("Please, set up environment variable DETEXE_ROOT.")
        raise WrongEnvironmentVariable

    if not os.path.isdir(root_dir):
        log.error(
            "environment variable DETEXE_ROOT is pointing to a non existing path."
        )
        raise WrongEnvironmentVariable

    return root_dir


def _create_default_selection_file(location: str) -> None:
    features_selection_file = location + "/features_selection.txt"
    if not os.path.isfile(features_selection_file):
        with open(features_selection_file, "a") as file:
            for feature in features:
                file.write("# " + feature + "\n")


def configure_layout(root_dir: Optional[str] = None) -> None:
    """Create project layout."""
    if not root_dir:
        root_dir = os.path.dirname(traceback.extract_stack()[-2].filename)
        os.environ["DETEXE_ROOT"] = root_dir
    project_dirs = read_directories_from_root(root_dir)
    _create_default_selection_file(root_dir)

    def _create_models_dir():
        os.mkdir(project_dirs["models_dir"])
        log.info("Models directory created.")

    def _create_benign_dir():
        os.makedirs(project_dirs["benign_dir"])
        log.info("Benign directory created.")

    def _create_malware_dir():
        os.makedirs(project_dirs["malware_dir"])
        log.info("Malware directory created.")

    for create_dir in [
        _create_models_dir,
        _create_benign_dir,
        _create_malware_dir,
    ]:
        try:
            create_dir()
        except Exception as e:
            log.error(e)
