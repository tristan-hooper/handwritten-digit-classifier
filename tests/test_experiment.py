"""Check split integrity and verify exported results against the source data."""

import csv
import json
from collections import Counter

import numpy as np
import pytest
from PIL import Image
from sklearn.datasets import load_digits

from train_evaluate import run_experiment, score_predictions, split_indices


def test_split_is_complete_disjoint_and_stratified():
    y = load_digits().target
    train, test = split_indices(y)
    assert len(train) == 1437
    assert len(test) == 360
    assert set(train).isdisjoint(test)
    assert sorted([*train, *test]) == list(range(len(y)))
    assert set(y[train]) == set(y[test]) == set(range(10))
    for label, count in Counter(y).items():
        assert abs(np.count_nonzero(y[test] == label) - count * 0.2) <= 1


def test_split_is_reproducible():
    y = load_digits().target
    first = split_indices(y)
    second = split_indices(y)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_metrics_match_a_hand_calculated_example():
    scores = score_predictions(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert scores["correct"] == 3
    assert scores["accuracy"] == 0.75
    assert scores["confusion_matrix"][0][:2] == [1, 1]
    assert scores["confusion_matrix"][1][:2] == [0, 2]
    # F1(0)=2/3, F1(1)=4/5, other eight labels=0; average across ten labels.
    assert scores["macro_f1"] == pytest.approx(((2 / 3) + (4 / 5)) / 10)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    directory = tmp_path_factory.mktemp("reports")
    return directory, run_experiment(directory)


def test_exports_agree_with_source_labels_and_manual_counts(experiment):
    directory, result = experiment
    digits = load_digits()
    indices = json.loads((directory / "split_indices.json").read_text())
    with (directory / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result["test_samples"] == 360
    assert [int(row["source_index"]) for row in rows] == indices["test"]
    counts = Counter(digits.target[indices["train"]])
    majority = min(counts, key=lambda label: (-counts[label], label))
    matrix = [[0] * 10 for _ in range(10)]
    for row in rows:
        source, true, predicted, baseline = (int(value) for value in row.values())
        assert true == int(digits.target[source])
        assert baseline == majority
        matrix[true][predicted] += 1
    scores = result["models"]["logistic_regression"]
    assert scores["confusion_matrix"] == matrix
    assert scores["correct"] == sum(matrix[i][i] for i in range(10))
    assert scores["accuracy"] == scores["correct"] / len(rows)
    assert json.loads((directory / "metrics.json").read_text()) == result


def test_figures_are_readable_and_summary_uses_actual_results(experiment):
    directory, result = experiment
    for name in ("training_examples.png", "confusion_matrix.png", "misclassified_examples.png"):
        with Image.open(directory / name) as image:
            image.verify()
    summary = (directory / "summary.md").read_text(encoding="utf-8")
    scores = result["models"]["logistic_regression"]
    assert f"{scores['correct']} / {scores['total']}" in summary


def test_repeated_experiment_preserves_predictions_and_metrics(experiment, tmp_path):
    directory, first = experiment
    second = run_experiment(tmp_path)
    assert first == second
    for name in ("predictions.csv", "split_indices.json"):
        assert (directory / name).read_bytes() == (tmp_path / name).read_bytes()
