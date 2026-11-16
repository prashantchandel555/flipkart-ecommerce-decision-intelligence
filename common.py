"""Shared data and model routines; importing this module never runs a Part."""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "records.csv"
OUTPUT_DIR = ROOT / "outputs"


def load_data():
    with DATA_PATH.open(newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[0], np.asarray([[float(value) for value in row[1:]] for row in rows[1:]], dtype=float)


def write_results(path, values, integer_keys=()):
    integer_keys = set(integer_keys)
    with path.open("w", newline="\n") as handle:
        for key, value in values.items():
            if isinstance(value, str):
                rendered = value
            elif key in integer_keys:
                rendered = str(int(value))
            else:
                rendered = f"{float(value):.4f}"
            handle.write(f"{key} = {rendered}\n")


def write_csv(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([
                f"{item:.4f}" if isinstance(item, (float, np.floating)) else item
                for item in row
            ])


def stratified_split(y):
    train_parts, test_parts = [], []
    for label in (0, 1):
        positions = np.flatnonzero(y == label)
        boundary = 7 * len(positions) // 10
        train_parts.extend(positions[:boundary])
        test_parts.extend(positions[boundary:])
    train = np.sort(np.asarray(train_parts, dtype=int))
    test = np.sort(np.asarray(test_parts, dtype=int))
    return train, test


def standardise_train_test(x, train, test):
    mean = np.mean(x[train], axis=0)
    sd = np.std(x[train], axis=0, ddof=1)
    return (x[train] - mean) / sd, (x[test] - mean) / sd, mean, sd


def sigmoid(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss(y, logits):
    return float(np.mean(np.logaddexp(0.0, logits) - y * logits))


def fit_logistic(x, y, learning_rate=0.1, updates=1000, checkpoints=()):
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    losses = {0: logistic_loss(y, np.zeros(len(y)))}
    for epoch in range(1, updates + 1):
        probabilities = sigmoid(x @ weights + bias)
        error = probabilities - y
        weights -= learning_rate * (x.T @ error) / len(y)
        bias -= learning_rate * np.mean(error)
        if epoch in checkpoints:
            losses[epoch] = logistic_loss(y, x @ weights + bias)
    return weights, bias, losses


def confusion(y, pred):
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / len(y),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
    }


def knn_probabilities(reference_x, reference_y, query_x, k):
    probabilities = []
    for query in query_x:
        distances = np.sum((reference_x - query) ** 2, axis=1)
        nearest = np.argsort(distances, kind="stable")[:k]
        probabilities.append(np.mean(reference_y[nearest]))
    return np.asarray(probabilities)


def knn_cross_validation(x, y):
    folds = np.arange(len(y)) % 5
    scores = []
    for k in range(1, 22, 2):
        fold_f1 = []
        for fold in range(5):
            held_out = folds == fold
            probability = knn_probabilities(x[~held_out], y[~held_out], x[held_out], k)
            fold_f1.append(confusion(y[held_out], (probability >= 0.5).astype(int))["f1"])
        scores.append((k, float(np.mean(fold_f1))))
    best_k = max(scores, key=lambda item: (item[1], -item[0]))[0]
    return scores, best_k


def prepare_classification():
    _, raw = load_data()
    x = raw[:, :7]
    y = raw[:, 7].astype(int)
    train, test = stratified_split(y)
    x_train, x_test, mean, sd = standardise_train_test(x, train, test)
    return raw, x, y, train, test, x_train, x_test, mean, sd
