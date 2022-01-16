import json
import logging
import os
from shutil import copyfile
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..settings import (check_layout_exist, check_root_path,
                        read_directories_from_root)
from .classifier import (bayessian_lgb_search, optimize_hyperparam,
                         predict_from_npz, predict_sample,
                         train_from_feature_vectors)
from .file_vectorizer import (get_features_from_malware_benign_dirs,
                              label_and_split_vectorized_dataset)
from .metrics import (get_best_threshold, get_metrics_model, include_in_plot,
                      save_plot)

log = logging.getLogger(__name__)


class ExistingModel(Exception):
    def __init__(self):
        pass


class NonExistingModel(Exception):
    def __init__(self):
        pass


class Detector:
    """A custom detector for a specified dataset given by malware and benign directories.
    A config file should be provided with the selected features to extract from the dataset.
    """

    def __init__(
        self,
        model: str,
        config_features: str,
        verbose: bool = False,
    ):
        root_dir = check_root_path()
        check_layout_exist(root_dir)
        project_dirs = read_directories_from_root(root_dir)
        self.model_name = model
        self.malware_dir = project_dirs.malware_dir
        self.benign_dir = project_dirs.benign_dir
        self.model_dir = project_dirs.models_dir + "/" + self.model_name
        self.model_path = self.model_dir + "/" + self.model_name + ".model"
        self.config_features = config_features
        self.params_file = self.model_dir + "/optimal_params.json"
        self.verbose = verbose

    def tune(self, timeout: int = 30) -> Tuple[lgb.LGBMModel, float]:
        """Tune and train detector with the found hyper parameters."""
        train_npz_path = self.model_dir + "/train.npz"
        test_npz_path = self.model_dir + "/test.npz"
        if not os.path.isfile(train_npz_path) or not os.path.isfile(test_npz_path):
            self.train()
        if not os.path.isfile(self.params_file):
            self._tune(timeout)

        npz_train = np.load(train_npz_path)
        npz_test = np.load(test_npz_path)

        with open(self.params_file, "r") as params_file:
            params = json.load(params_file)

        return train_from_feature_vectors(
            npz_train["x"],
            npz_train["y"],
            npz_test["x"],
            npz_test["y"],
            model_path=self.model_path,
            params=params,
        )

    def train(self) -> Tuple[lgb.LGBMModel, float]:
        """Train detector."""
        if os.path.isdir(self.model_dir):
            log.error(
                "There is already an existing model directory with the specified model name."
            )
            raise ExistingModel
        malware_vectors, benign_vectors = get_features_from_malware_benign_dirs(
            malware_dir=self.malware_dir,
            benign_dir=self.benign_dir,
            config=self.config_features,
            verbose=self.verbose,
        )

        x_train, y_train, x_test, y_test = label_and_split_vectorized_dataset(
            malware_vec=malware_vectors, benign_vec=benign_vectors
        )
        os.mkdir(self.model_dir)
        copyfile(
            self.config_features,
            self.model_dir + "/features_selection.txt",
        )
        np.savez(self.model_dir + "/train", x=x_train, y=y_train)
        np.savez(self.model_dir + "/test", x=x_test, y=y_test)
        return train_from_feature_vectors(
            x_train, y_train, x_test, y_test, model_path=self.model_path
        )

    def scan(self, exe: str) -> float:
        """Scan exe file."""
        threshold_file = self.model_dir + "/threshold.txt"
        if os.path.isfile(threshold_file):
            with open(threshold_file) as f:
                threshold = float(f.readline().replace(" ", ""))
        else:
            threshold = 0.5
        with open(exe, "rb") as f:
            data_file = f.read()
        prediction, class_prediction = predict_sample(
            model_path=self.model_path,
            config_path=self.config_features,
            data_file=data_file,
            threshold=threshold,
        )
        if class_prediction >= threshold:
            log.info(f"Malware -> {prediction:.3f}, threshold: {threshold:.3f}")
        else:
            log.info(f"Benign -> {prediction:0.3f}, threshold: {threshold:.3f}")
        return prediction

    def _tune(self, minutes_limit: int) -> dict:
        """Tune hyper parameters."""
        log.info("Searching hyper parameters..")
        if not os.path.isdir(self.model_dir):
            log.error(
                "The indicated model does not exist. Please use first the command train with the indicated model name."
            )
            exit()
        npz_train = np.load(self.model_dir + "/train.npz")
        params = optimize_hyperparam(
            self.model_dir,
            x_train=npz_train["x"],
            y_train=npz_train["y"],
            minutes_limit=minutes_limit,
        )
        params["verbose"] = -1

        with open(self.params_file, "w") as f:
            json.dump(params, f)
        return params


def compare(plot_name: str) -> Tuple[str, float, float]:
    """Compare the detectors saved under the models_dir path. This method provides as output the most precise
    detector and also creates a precision_recall comparison graph under models_dir path"""
    root_dir = check_root_path()
    project_dirs = read_directories_from_root(root_dir)
    models_subdirectories = [x[0] for x in os.walk(project_dirs.models_dir)][1:]
    auc_models = {}
    treshold_models = {}
    model_stats = pd.DataFrame()
    for subdir in models_subdirectories:
        _, model_name = os.path.split(subdir)
        try:
            y_preds, y_test = predict_from_npz(
                subdir + "/test.npz", subdir + "/" + model_name + ".model"
            )
        except (FileNotFoundError, lgb.basic.LightGBMError):
            log.info(subdir + ": Error reading data model. Remove this directory.")
            continue
        roc_auc, recall, precision, thresholds = get_metrics_model(y_preds, y_test)
        model_stats = pd.DataFrame(
            {
                "model": model_name,
                "auc": roc_auc,
                "recall": [recall],
                "precision": [precision],
                "thresholds": [thresholds],
            }
        )
        auc_models[model_name] = roc_auc
        treshold_models[model_name] = thresholds
        label = f"AUC {model_name} = {roc_auc:.2f}"
        include_in_plot(recall, precision, label)
    if model_stats.empty:
        log.error("There are no models to get metrics.")
        raise NonExistingModel
    ix = model_stats["auc"].idxmax()
    row = model_stats.iloc[ix]

    th, fscore = get_best_threshold(row["recall"], row["precision"], row["thresholds"])

    log.info(f"Model with best results: {row['model']}")
    log.info(f"Best Threshold: {th:.2f}, with F1 score: {fscore:.2f}")
    with open(
        project_dirs.models_dir + "/" + row["model"] + "/threshold.txt", "w"
    ) as f:
        f.write(str(th))
    save_plot(project_dirs.models_dir + "/" + plot_name)

    return row["model"], th, fscore
