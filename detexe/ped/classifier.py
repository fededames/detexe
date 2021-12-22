"""
Classsifier: Set of functions, for training and predicting feature vectorsof a PE file.
"""

import logging
import pathlib
from pprint import pformat
from time import time
from typing import Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, make_scorer)
from sklearn.model_selection import KFold
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
    lgb_model = lgb.train(params, lgb_dataset)
    lgb_model.save_model(filename=model_path)
    y_preds = lgb_model.predict(x_test)
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


def bayessian_lgb_search(
    x_train: np.ndarray, y_train: np.ndarray, minutes_limit: int, verbose: int = -1
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
