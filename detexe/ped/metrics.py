"""
metrics.py: Set of functions to obtain metrics
"""
import pathlib
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
from sklearn.metrics import auc, precision_recall_curve


def include_in_plot(x: np.ndarray, y: np.ndarray, label: str) -> None:
    "Include representation in the graph to be printed out later."
    plt.plot(x, y, marker=".", label=label)


def save_plot(path: Union[str, pathlib.Path]) -> None:
    "Include representation in the graph to be printed out later."
    plt.legend(loc="lower left")
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.savefig(path, bbox_inches="tight")


def get_metrics_model(
    y_preds: np.ndarray, y_test: np.ndarray
) -> Tuple[float, float, float, float]:
    """Returns the metris for the predicted results"""
    precision, recall, thresholds = precision_recall_curve(y_test, y_preds)
    precision_recall_auc = auc(recall, precision)
    return precision_recall_auc, recall, precision, thresholds


def get_best_threshold(
    recall: float, precision: float, thresholds
) -> Tuple[float, float]:
    "From a range of threshold return the optimal threshold with its fscore metric"
    fscore = (2 * precision * recall) / (precision + recall)
    ix = argmax(fscore)
    plt.scatter(recall[ix], precision[ix], marker="o", color="black", label="Best")
    return thresholds[ix], fscore[ix]
