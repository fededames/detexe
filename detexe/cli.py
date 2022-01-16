import argparse
import logging

from .pea.attacker import Attacker
from .ped.detector import Detector, ExistingModel, NonExistingModel, compare
from .ped.extractor import WrongFeatureSelection
from .ped.features.feature_helpers.train_opcodes_vectors import \
    train_opcode_vectors
from .settings import (WrongLayout, check_layout_exist, check_root_path,
                       configure_layout, read_directories_from_root)

log = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="EXE-Scan CLI")
    action_parser = parser.add_subparsers(dest="command", title="actions")
    action_parser.required = True
    action_parser.add_parser(
        "train",
        help="Train a model with the data contained in data directory, using the features_selection.txt file.",
    )
    action_parser.add_parser("tune", help="Train model and tune hyperparamenters.")
    action_parser.add_parser("scan", help="Scan EXE file.")

    for action, subparser in action_parser.choices.items():
        subparser.add_argument(
            "--model",
            required=True,
            type=str,
            help="set the model name for the selected action",
        )
        subparser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="show additional debugging info",
        )
        if action == "scan":
            subparser.add_argument(
                "--exe",
                required=True,
                type=str,
                help="select malware you want to scan",
            )
        elif action == "tune":
            subparser.add_argument(
                "-t",
                "--timeout",
                default=30,
                type=int,
                help="Select the limit of minutes tu run the hyperparameters serach",
            )

    action_parser.add_parser(
        "layout", help="Build directories layout with models and data directories."
    )
    action_parser.add_parser("compare", help="Compare the saved models.")
    attack = action_parser.add_parser(
        "attack", help="Adversarial Attack. Edit PE file to decrease detection rate."
    )
    action_parser.add_parser("opcodesw2v", help="Build vectors from opcodes.")
    attack_parser = attack.add_subparsers(dest="attack_mode", help="attack type")
    attack_parser.required = True
    shift = attack_parser.add_parser("shift")  # noqa
    inject = attack_parser.add_parser("inject")  # noqa
    all_attacks = attack_parser.add_parser("all")  # noqa
    for _, subparser in attack_parser.choices.items():
        subparser.add_argument(
            "--model",
            required=True,
            type=str,
            help="set the model name for the selected action",
        )

        subparser.add_argument(
            "--malware", required=True, type=str, help="malware path"
        )

    input = parser.parse_args()
    if not input.command:
        log.info("Use any of the options displayed with -h, --help.")
        exit()
    parser_selection(vars(input))


def parser_selection(cmd_args):
    root_dir = check_root_path()
    try:
        check_layout_exist
    except WrongLayout:
        exit()
    project_dirs = read_directories_from_root(root_dir)
    command = cmd_args.pop("command")

    if command == "setup":
        configure_layout(root_dir)
        exit()
    if command == "compare":
        compare("compare.png")
        exit()
    if command == "opcodesw2v":
        try:
            train_opcode_vectors()
        except WrongFeatureSelection:
            pass
        exit()
    model_name = cmd_args.pop("model")
    if command == "scan":
        config_dir = project_dirs.models_dir + "/" + model_name
    else:
        config_dir = root_dir
    if command == "attack":
        command = cmd_args.pop("attack_mode")
        attacker = Attacker(
            model=model_name,
        )
        attacker.malware = cmd_args.pop("malware")
        log.info(f"Malware selected: {attacker.malware}")
        actions = {"shift": attacker.shift, "all": attacker.all_attacks}

    else:
        detector = Detector(
            model=model_name,
            config_features=config_dir + "/features_selection.txt",
            verbose=cmd_args.pop("verbose"),
        )

        actions = {
            "train": detector.train,
            "tune": detector.tune,
            "scan": detector.scan,
        }
    try:
        actions[command](**cmd_args)
    except (ExistingModel, NonExistingModel):
        pass
