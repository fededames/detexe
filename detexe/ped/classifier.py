"""
Classsifier: Set of functions, for training and predicting feature vectorsof a PE file.
"""

import logging
import pathlib
from functools import partial
from pprint import pformat
from time import time
from typing import Tuple, Union

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.integration import LightGBMPruningCallback
from optuna.visualization.matplotlib import (plot_optimization_history,
                                             plot_param_importances)
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, log_loss, make_scorer)
from sklearn.model_selection import KFold, StratifiedKFold
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Integer, Real

from .extractor import PEFeatureExtractor

log = logging.getLogger(__name__)


def train_from_feature_vectors(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_path: str,
    params: dict = None,
    threshold: float = 0.5,
) -> Tuple[lgb.LGBMModel, float]:
    """Train a model with the specified vectors."""
    if params is None:
        params = {"verbose": -1}

    lgb_dataset = lgb.Dataset(x_train, y_train)
    lgb_model = lgb.train({"n_estimators": 100, "verbose": -1}, lgb_dataset)
    y_preds = lgb_model.predict(x_test)
    lgb_model.save_model(filename=model_path)
    log.info(f"Specified threshold: {threshold}")
    y_preds_int = np.where(y_preds > threshold, 1, 0)
    log.info("Confusion matrix:")
    print(confusion_matrix(y_test, y_preds_int))
    f1_metric = f1_score(y_test, y_preds_int)
    log.info(f"F1 Score: {f1_metric:.2f}")
    return lgb_model, f1_metric


def predict_from_npz(
    numpy_file: Union[str, pathlib.Path], model_path: Union[str, pathlib.Path]
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict the result of a trained model for an array contained in a npz file."""
    npz_test = np.load(numpy_file)
    model = lgb.Booster(model_file=model_path)
    y_preds = model.predict(npz_test["x"])
    return y_preds, npz_test["y"]


def predict_from_feature_vector(
    model_path: Union[str, pathlib.Path],
    feature_vector: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray]:
    """
    Predict a vector of a PE file with LightGBM.
    """
    lgbm_model = lgb.Booster(model_file=model_path)
    y_preds = lgbm_model.predict([feature_vector])[0]
    y_preds_int = np.where(y_preds > threshold, 1, 0)
    return y_preds, y_preds_int


def predict_sample(
    model_path: Union[str, pathlib.Path],
    config_path: Union[str, pathlib.Path],
    data_file: Union[str, pathlib.Path],
    threshold: float = 0.5,
) -> Tuple[np.ndarray]:
    """
    Predict a PE file using LightGBM.
    """
    extractor = PEFeatureExtractor(config=config_path, truncate=True)
    lgbm_model = lgb.Booster(model_file=model_path)
    features = np.array(extractor.feature_vector(data_file), dtype=np.float32)

    y_preds = lgbm_model.predict([features])[0]
    y_preds_int = np.where(y_preds > threshold, 1, 0)
    return y_preds, y_preds_int


def optimize(trial, x: np.ndarray, y: np.ndarray):

    params_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 30, 10000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, int(len(x) / 2)),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 2, 200),
        "max_bin": trial.suggest_int("max_bin", 100, 300),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100.0, log=True),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 0.01, 15.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.2, 0.95, step=0.1),
        "subsample_freq": trial.suggest_categorical("bagging_freq", [1]),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.2, 0.95, step=0.1
        ),
    }

    model = lgb.LGBMClassifier(
        objective="binary", n_jobs=1, random_state=0, **params_grid, verbose=-1
    )
    cv = StratifiedKFold(n_splits=5)
    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        X_train, X_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(
            X_train,
            y_train,
            verbose=-1,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, "binary_logloss")
            ],  # Add a pruning callback
        )
        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

    return np.mean(cv_scores)


def optimize_hyperparam(
    model_dir: str, x_train: np.ndarray, y_train: np.ndarray, minutes_limit: int
):
    study = optuna.create_study(direction="minimize")
    optimization_function = partial(optimize, x=x_train, y=y_train)
    study.optimize(
        optimization_function, n_trials=100000000, timeout=60 * minutes_limit
    )
    plot_param_importances(study)
    plt.savefig(model_dir + "/parameter_importance.jpg", bbox_inches="tight")
    plot_optimization_history(study)
    plt.savefig(model_dir + "/optimization_history.jpg", bbox_inches="tight")
    log.info(f"Best value (log loss): {study.best_value:.5f}")
    log.info("Best hyperparameters:")
    logging.info(pformat(dict(study.best_params.items())))
    return study.best_params


def bayessian_lgb_search(
    x_train: np.ndarray, y_train: np.ndarray, minutes_limit: int
) -> dict:
    """
    Search optimal parameters for a given dataset.
    """
    # Setting the scoring function
    scoring = make_scorer(average_precision_score)
    # Setting the validation strategy
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    model = lgb.LGBMClassifier(
        boosting_type="gbdt", n_jobs=1, verbose=-1, random_state=0
    )
    # Setting the search space
    search_spaces = {
        "learning_rate": Real(0.01, 0.3, "log-uniform"),  # Boosting learning rate
        "n_estimators": Integer(30, 10000),  # Number of boosted trees to fit
        "num_leaves": Integer(20, 3000),  # Maximum tree leaves for base learners
        "max_depth": Integer(
            -1, 15
        ),  # Maximum tree depth for base learners, <=0 means no limit
        "min_child_samples": Integer(2, 200),  # Minimal number of data in one leaf
        "max_bin": Integer(
            100, 300
        ),  # Max number of bins that feature values will be bucketed
        "subsample": Real(
            0.1, 0.95, "uniform"
        ),  # Subsample ratio of the training instance
        "subsample_freq": Integer(0, 1),  # Frequency of subsample, <=0 means no enable
        "colsample_bytree": Real(
            0.1, 1.0, "uniform"
        ),  # Subsample ratio of columns when constructing each tree
        "min_child_weight": Real(1e-4, 10.0, "log-uniform"),
        # Minimum sum of instance weight (hessian) needed in a child (leaf)
        "reg_lambda": Real(1e-9, 100.0, "log-uniform"),  # L2 regularization
        "reg_alpha": Real(1e-9, 100.0, "log-uniform"),  # L1 regularization
    }

    # Wrapping everything up into the Bayesian optimizer
    opt = BayesSearchCV(
        estimator=model,
        search_spaces=search_spaces,
        scoring=scoring,
        cv=kf,
        n_iter=10000,  # max number of trials
        n_points=3,  # number of hyperparameter sets evaluated at the same time
        n_jobs=-1,  # number of jobs
        iid=False,  # if not iid it optimizes on the cv score
        return_train_score=False,
        refit=False,
        optimizer_kwargs={"base_estimator": "GP"},
        # optmizer parameters: we use Gaussian Process (GP)
        random_state=0,
        verbose=0,
    )  # random state for replicability

    # Running the optimizer
    overdone_control = DeltaYStopper(
        delta=0.0001
    )  # We stop if the gain of the optimization becomes too small
    time_limit_control = DeadlineStopper(
        total_time=60 * minutes_limit
    )  # limit of minutes

    best_params = _report_performance(
        opt,
        x_train,
        y_train,
        callbacks=[overdone_control, time_limit_control],
    )
    best_params["verbose"] = -1
    return best_params


def _report_performance(optimizer, x, y, callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers

    optimizer = a sklearn or a skopt optimizer
    X = the training set
    y = our target
    """
    start = time()

    if callbacks is not None:
        optimizer.fit(x, y, callback=callbacks)
    else:
        optimizer.fit(x, y)

    d = pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_

    logging.info(
        f"Bayesian optimization ran during {time() - start:.2f} seconds,  candidates checked: {len(optimizer.cv_results_['params'])}, best CV score: {best_score:.3f} \u00B1 {best_score_std:.3f}"
    )
    logging.info("Best parameters:")
    logging.info(pformat(dict(best_params.items())))
    return best_params
