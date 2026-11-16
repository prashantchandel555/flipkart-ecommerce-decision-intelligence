"""Part 2: scratch logistic regression, threshold evaluation, and KNN."""
import numpy as np

from common import (
    OUTPUT_DIR, confusion, fit_logistic, knn_cross_validation, knn_probabilities,
    logistic_loss, prepare_classification, sigmoid, write_csv, write_results,
)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw, _, y, train, test, x_train, x_test, _, _ = prepare_classification()
    y_train, y_test = y[train], y[test]
    weights, bias, losses = fit_logistic(
        x_train, y_train, learning_rate=0.1, updates=1000, checkpoints=(1, 10, 100, 1000)
    )
    train_logits = x_train @ weights + bias
    test_logits = x_test @ weights + bias
    test_probability = sigmoid(test_logits)

    result = {
        "train_rows": len(train), "train_positives": int(np.sum(y_train)),
        "test_rows": len(test), "test_positives": int(np.sum(y_test)),
        "loss_epoch_0": losses[0], "loss_epoch_1": losses[1],
        "loss_epoch_10": losses[10], "loss_epoch_100": losses[100],
        "loss_epoch_1000": losses[1000],
        "test_bce": logistic_loss(y_test, test_logits),
    }
    for index, weight in enumerate(weights, start=1):
        result[f"weight_X{index}"] = weight
        result[f"odds_ratio_X{index}"] = np.exp(weight)
    result["bias"] = bias

    for rate in (0.01, 0.1, 1.0):
        _, _, rate_losses = fit_logistic(
            x_train, y_train, learning_rate=rate, updates=200, checkpoints=(200,)
        )
        result[f"loss_epoch_200_eta_{rate}"] = rate_losses[200]

    for threshold in (0.5, 0.3):
        cm = confusion(y_test, (test_probability >= threshold).astype(int))
        prefix = f"tau_{threshold}"
        for key in ("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1", "specificity"):
            result[f"{prefix}_{key}"] = cm[key]
    result["blind_accuracy"] = float(np.mean(y_test == 0))

    threshold_rows = []
    for threshold in np.arange(0.1, 1.0, 0.1):
        cm = confusion(y_test, (test_probability >= threshold).astype(int))
        threshold_rows.append((
            round(float(threshold), 1), cm["tp"], cm["fp"], cm["tn"], cm["fn"],
            cm["recall"], cm["fp"] / (cm["fp"] + cm["tn"]) if cm["fp"] + cm["tn"] else 0.0,
        ))
    write_csv(OUTPUT_DIR / "part2_thresholds.csv",
              ("tau", "tp", "fp", "tn", "fn", "tpr", "fpr"), threshold_rows)

    cv_scores, best_k = knn_cross_validation(x_train, y_train)
    write_csv(OUTPUT_DIR / "part2_knn_cv.csv", ("K", "mean_cv_f1"), cv_scores)
    knn_probability = knn_probabilities(x_train, y_train, x_test, best_k)
    knn_cm = confusion(y_test, (knn_probability >= 0.5).astype(int))
    result["knn_best_k"] = best_k
    for key in ("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1"):
        result[f"knn_{key}"] = knn_cm[key]

    train_probability = sigmoid(train_logits)
    cost_candidates = []
    for step in range(1, 20):
        threshold = step / 20
        cm = confusion(y_train, (train_probability >= threshold).astype(int))
        cost_candidates.append((50 * cm["fp"] + 400 * cm["fn"], threshold))
    train_cost, cost_threshold = min(cost_candidates, key=lambda item: (item[0], item[1]))
    test_at_tau = confusion(y_test, (test_probability >= cost_threshold).astype(int))
    test_at_half = confusion(y_test, (test_probability >= 0.5).astype(int))
    result["cost_tau"] = cost_threshold
    result["cost_train_at_tau"] = train_cost
    result["cost_test_at_tau"] = 50 * test_at_tau["fp"] + 400 * test_at_tau["fn"]
    result["cost_test_at_0.5"] = 50 * test_at_half["fp"] + 400 * test_at_half["fn"]
    result["cost_test_flag_nobody"] = 400 * int(np.sum(y_test))
    write_results(
        OUTPUT_DIR / "part2_results.txt", result,
        integer_keys=(
            "train_rows", "train_positives", "test_rows", "test_positives", "knn_best_k",
            "tau_0.5_tp", "tau_0.5_fp", "tau_0.5_tn", "tau_0.5_fn",
            "tau_0.3_tp", "tau_0.3_fp", "tau_0.3_tn", "tau_0.3_fn",
            "knn_tp", "knn_fp", "knn_tn", "knn_fn", "cost_train_at_tau",
            "cost_test_at_tau", "cost_test_at_0.5", "cost_test_flag_nobody",
        ),
    )


if __name__ == "__main__":
    main()
