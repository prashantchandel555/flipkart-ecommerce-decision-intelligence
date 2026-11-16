"""Part 4: model-choice calculations and architecture parameter counts."""
import numpy as np

from common import (
    OUTPUT_DIR, confusion, fit_logistic, knn_cross_validation, knn_probabilities,
    prepare_classification, sigmoid, write_csv, write_results,
)


def entropy(positive, total):
    if total == 0:
        return 0.0
    probabilities = (positive / total, (total - positive) / total)
    return -sum(p * np.log2(p) for p in probabilities if p > 0)


def gini(positive, total):
    if total == 0:
        return 0.0
    p = positive / total
    return 1.0 - p * p - (1.0 - p) ** 2


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw, _, y, train, test, x_train, x_test, _, _ = prepare_classification()
    train_raw, test_raw = raw[train], raw[test]
    y_train, y_test = y[train], y[test]
    train_x_original = train_raw[:, :7]
    test_x_original = test_raw[:, :7]
    root_p = int(np.sum(y_train))
    root_entropy = entropy(root_p, len(y_train))
    root_gini = gini(root_p, len(y_train))
    result = {"root_entropy": root_entropy, "root_gini": root_gini}

    conditions = {
        "S1": train_x_original[:, 5] == 1,
        "S2": train_x_original[:, 6] == 1,
        "S3": train_x_original[:, 2] > 0.5,
        "S4": train_x_original[:, 0] <= 24,
    }
    split_rows = []
    split_rates = {}
    for name, yes_mask in conditions.items():
        no_mask = ~yes_mask
        n_yes, n_no = int(np.sum(yes_mask)), int(np.sum(no_mask))
        p_yes, p_no = int(np.sum(y_train[yes_mask])), int(np.sum(y_train[no_mask]))
        weighted_entropy = (
            n_yes / len(y_train) * entropy(p_yes, n_yes)
            + n_no / len(y_train) * entropy(p_no, n_no)
        )
        weighted_gini = (
            n_yes / len(y_train) * gini(p_yes, n_yes)
            + n_no / len(y_train) * gini(p_no, n_no)
        )
        info_gain = root_entropy - weighted_entropy
        gini_gain = root_gini - weighted_gini
        split_rows.append((name, n_yes, p_yes, n_no, p_no, info_gain, gini_gain))
        split_rates[name] = (
            p_yes / n_yes if n_yes else 0.0,
            p_no / n_no if n_no else 0.0,
        )
    best_info = max(split_rows, key=lambda row: row[5])[0]
    best_gini = max(split_rows, key=lambda row: row[6])[0]
    result["best_split_by_info_gain"] = best_info
    result["best_split_by_gini_gain"] = best_gini
    result["stump_rate_yes"], result["stump_rate_no"] = split_rates[best_info]
    write_csv(
        OUTPUT_DIR / "part4_splits.csv",
        ("split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"),
        split_rows,
    )

    weights, bias, _ = fit_logistic(x_train, y_train, 0.1, 1000)
    logistic_probability = sigmoid(x_test @ weights + bias)
    cv_scores, best_k = knn_cross_validation(x_train, y_train)
    del cv_scores
    knn_probability = knn_probabilities(x_train, y_train, x_test, best_k)
    stump_condition = {
        "S1": test_x_original[:, 5] == 1,
        "S2": test_x_original[:, 6] == 1,
        "S3": test_x_original[:, 2] > 0.5,
        "S4": test_x_original[:, 0] <= 24,
    }[best_info]
    stump_probability = np.where(stump_condition, result["stump_rate_yes"], result["stump_rate_no"])
    predictions = {
        "logistic": logistic_probability >= 0.3,
        "knn": knn_probability >= 0.5,
        "stump": stump_probability >= 0.3,
    }
    predictions["hard_vote"] = (
        predictions["logistic"].astype(int)
        + predictions["knn"].astype(int)
        + predictions["stump"].astype(int)
    ) >= 2
    predictions["soft_vote"] = (
        (logistic_probability + knn_probability + stump_probability) / 3.0 >= 0.3
    )
    voting_rows = []
    for name in ("logistic", "knn", "stump", "hard_vote", "soft_vote"):
        cm = confusion(y_test, predictions[name].astype(int))
        voting_rows.append((name, cm["tp"], cm["fp"], cm["tn"], cm["fn"],
                            cm["precision"], cm["recall"], cm["f1"]))
    write_csv(
        OUTPUT_DIR / "part4_voting.csv",
        ("model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"),
        voting_rows,
    )

    true = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    weak_rules = (
        np.array([1, 1, -1, -1, -1, -1, -1, -1]),
        np.array([-1, 1, 1, -1, 1, -1, 1, -1]),
    )
    point_weights = np.full(8, 1.0 / 8.0)
    adaboost_rows = []
    alphas, errors = [], []
    misclassified_weight_after_round1 = None
    for round_no, hypothesis in enumerate(weak_rules, start=1):
        before = point_weights.copy()
        error = float(np.sum(before[hypothesis != true]))
        alpha = 0.5 * np.log((1.0 - error) / error)
        unnormalised = before * np.exp(-alpha * true * hypothesis)
        point_weights = unnormalised / np.sum(unnormalised)
        errors.append(error)
        alphas.append(alpha)
        for point in range(8):
            adaboost_rows.append((round_no, point + 1, before[point], unnormalised[point],
                                  point_weights[point]))
        if round_no == 1:
            misclassified_weight_after_round1 = float(np.sum(point_weights[hypothesis != true]))
    ensemble_score = alphas[0] * weak_rules[0] + alphas[1] * weak_rules[1]
    result["round1_error"] = errors[0]
    result["round1_alpha"] = alphas[0]
    result["round2_error"] = errors[1]
    result["round2_alpha"] = alphas[1]
    result["ensemble_accuracy"] = float(np.mean(np.sign(ensemble_score) == true))
    result["round1_misclassified_weight_after_normalisation"] = misclassified_weight_after_round1
    write_csv(
        OUTPUT_DIR / "part4_adaboost.csv",
        ("round", "point", "weight_before", "weight_unnormalised", "weight_after"),
        adaboost_rows,
    )

    result["oob_fraction"] = (1.0 - 1.0 / 560) ** 560
    architecture_rows = [
        ("mlp_hidden1", 784 * 16 + 16),
        ("mlp_hidden2", 16 * 16 + 16),
        ("mlp_output", 16 * 10 + 10),
        ("cnn_conv1", 5 * 5 * 1 * 6 + 6),
        ("cnn_conv2", 5 * 5 * 6 * 16 + 16),
        ("cnn_dense1", 5 * 5 * 16 * 32 + 32),
        ("cnn_dense2", 32 * 10 + 10),
    ]
    write_csv(OUTPUT_DIR / "part4_architecture.csv", ("layer", "parameters"), architecture_rows)
    result["mlp_total"] = sum(count for name, count in architecture_rows if name.startswith("mlp_"))
    result["cnn_total"] = sum(count for name, count in architecture_rows if name.startswith("cnn_"))
    result["fc_layer_params"] = (128 * 128 * 3) * 1024 + 1024
    result["conv_layer_params"] = 3 * 3 * 3 * 64 + 64
    result["rnn_total"] = 64 * 128 + 128 * 128 + 128 * 16 + 128 + 16
    write_results(
        OUTPUT_DIR / "part4_results.txt", result,
        integer_keys=("mlp_total", "cnn_total", "fc_layer_params", "conv_layer_params", "rnn_total"),
    )


if __name__ == "__main__":
    main()
