from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def test_compute_accuracy():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 1, 3])
    accuracy = accuracy_score(y_true, y_pred)

    assert 0.0 <= accuracy <= 1.0


def test_f1_score():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 2])
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    assert macro_f1 == pytest.approx(2 / 3, rel=1e-3)


def test_confusion_matrix_shape():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 1, 3, 2, 1, 2, 0])
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    assert matrix.shape == (4, 4)


def test_auc_roc():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_prob = np.array(
        [
            [0.92, 0.03, 0.03, 0.02],
            [0.05, 0.88, 0.04, 0.03],
            [0.04, 0.08, 0.82, 0.06],
            [0.03, 0.05, 0.06, 0.86],
            [0.81, 0.08, 0.06, 0.05],
            [0.06, 0.83, 0.05, 0.06],
            [0.05, 0.07, 0.84, 0.04],
            [0.04, 0.06, 0.08, 0.82],
        ]
    )
    auc_value = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

    assert 0.0 <= auc_value <= 1.0
