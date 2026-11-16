"""Part 1: estimation, inference, and regression baselines."""
import numpy as np

from common import OUTPUT_DIR, load_data, write_results


def line_fit(x, y):
    slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)
    intercept = np.mean(y) - slope * np.mean(x)
    fitted = intercept + slope * x
    residual = y - fitted
    tss = np.sum((y - np.mean(y)) ** 2)
    ess = np.sum((fitted - np.mean(y)) ** 2)
    rss = np.sum(residual ** 2)
    return slope, intercept, fitted, residual, tss, ess, rss


def main():
    _, raw = load_data()
    predictors = raw[:, :7]
    x4 = predictors[:, 3]
    y = raw[:, 8]
    n = len(y)
    mean_value = float(np.mean(y))
    sd_value = float(np.std(y, ddof=1))
    se_value = sd_value / np.sqrt(n)
    ci_low = mean_value - 1.96 * se_value
    ci_high = mean_value + 1.96 * se_value

    slope, intercept, fitted, residual, tss, ess, rss = line_fit(x4, y)
    rmse = np.sqrt(rss / n)
    mae = np.mean(np.abs(residual))

    x4_sd = np.std(x4, ddof=1)
    z = (x4 - np.mean(x4)) / x4_sd
    m = c = 0.0
    gd_cost = {0: float(np.mean(y ** 2))}
    for iteration in range(1, 501):
        error = y - (m * z + c)
        grad_m = -2.0 * np.sum(z * error) / n
        grad_c = -2.0 * np.sum(error) / n
        m -= 0.1 * grad_m
        c -= 0.1 * grad_c
        if iteration in (1, 10, 100, 500):
            gd_cost[iteration] = float(np.mean((y - (m * z + c)) ** 2))
    gd_slope = m / x4_sd
    gd_intercept = c - m * np.mean(x4) / x4_sd

    order = np.argsort(fitted, kind="stable")
    raw_lower = float(np.std(residual[order[:400]], ddof=1))
    raw_upper = float(np.std(residual[order[400:]], ddof=1))
    log_x = np.log(x4)
    log_y = np.log(y)
    log_slope, log_intercept, log_fitted, log_residual, log_tss, _, log_rss = line_fit(log_x, log_y)
    log_order = np.argsort(log_fitted, kind="stable")
    log_lower = float(np.std(log_residual[log_order[:400]], ddof=1))
    log_upper = float(np.std(log_residual[log_order[400:]], ddof=1))

    design = np.column_stack((
        np.ones(n), log_x, predictors[:, 0], predictors[:, 1], predictors[:, 2],
        predictors[:, 4], predictors[:, 5], predictors[:, 6],
    ))
    beta = np.linalg.solve(design.T @ design, design.T @ log_y)
    mr_residual = log_y - design @ beta
    mr_r2 = 1.0 - np.sum(mr_residual ** 2) / np.sum((log_y - np.mean(log_y)) ** 2)
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - 7 - 1)
    log_r2 = 1.0 - log_rss / log_tss
    log_adj_r2 = 1.0 - (1.0 - log_r2) * (n - 1) / (n - 1 - 1)

    result = {
        "n": n,
        "mean_value": mean_value, "sd_value": sd_value, "se_value": se_value,
        "ci95_low": ci_low, "ci95_high": ci_high,
        "z_stat": (mean_value - 8.0) / se_value,
        "reject_h0": "yes" if abs((mean_value - 8.0) / se_value) > 1.96 else "no",
        "n_required": int(np.ceil((1.96 * sd_value / 0.5) ** 2)),
        "reg_slope": slope, "reg_intercept": intercept, "tss": tss, "ess": ess, "rss": rss,
        "tss_equals_ess_plus_rss": "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no",
        "r2": 1.0 - rss / tss, "rmse": rmse, "mae": mae,
        "gd_J0": gd_cost[0], "gd_J1": gd_cost[1], "gd_J10": gd_cost[10],
        "gd_J100": gd_cost[100], "gd_J500": gd_cost[500],
        "gd_slope": gd_slope, "gd_intercept": gd_intercept,
        "gd_slope_gap": abs(gd_slope - slope),
        "resid_sd_lower_raw": raw_lower, "resid_sd_upper_raw": raw_upper,
        "resid_sd_ratio_raw": raw_upper / raw_lower,
        "loglog_intercept": log_intercept, "loglog_slope": log_slope,
        "loglog_r2": log_r2, "loglog_rmse": np.sqrt(log_rss / n),
        "resid_sd_lower_log": log_lower, "resid_sd_upper_log": log_upper,
        "resid_sd_ratio_log": log_upper / log_lower,
    }
    for index, coefficient in enumerate(beta):
        result[f"mr_beta_{index}"] = coefficient
    result["mr_r2"] = mr_r2
    result["mr_adj_r2"] = mr_adj_r2
    result["loglog_adj_r2"] = log_adj_r2
    OUTPUT_DIR.mkdir(exist_ok=True)
    write_results(OUTPUT_DIR / "part1_results.txt", result, integer_keys=("n", "n_required"))


if __name__ == "__main__":
    main()
