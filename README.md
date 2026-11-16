Domain: E-commerce

## How to run

From the repository root on macOS or Linux, use Python 3.13 (the pinned NumPy version also supports Python 3.10–3.12). If using one of those versions instead, replace `python3.13` in the first command with its executable name. Python 3.14 is not supported by the pinned NumPy version.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

The generator should print `800 10 170 c6f249ba693f1ebeed318d6d5e18de23`. All analysis is offline; the scripts write their results under `outputs/`.

## Part 1 findings

- **R1.1:** We reject H0 because `z_stat = 2.8930` exceeds the two-tailed critical value 1.96. A Type I error here would be concluding that the mean order value differs from INR 800 when the true population mean is INR 800. The observed gap is 0.9382 hundreds of INR, or INR 93.82; that is about 11.7% of the planning assumption and could matter when setting category budgets, although statistical significance alone does not establish business impact.
- **R1.2:** The required sample is `n_required = 1293`, versus 800 available, a shortfall of 493. The current sample does not meet the stated ±INR 50, 95%-confidence precision target, so the planning team would need more observations to target that margin.
- **R1.3:** `gd_slope_gap = 0.0000`. Gradient descent reaches the closed-form line because the squared-error objective is convex and the full-batch updates converge to its minimum with the chosen learning rate and sufficient iterations.
- **R1.4:** `resid_sd_ratio_raw = 3.7525`, indicating a strong funnel in the raw model; after logging, `resid_sd_ratio_log = 1.0396`, so the residual spreads are nearly equal and the log model largely removes it.
- **R1.5:** The coefficients with absolute value above 0.05 are ln(X4), X2, X3 and X6. The multiple-regression `mr_adj_r2 = 0.7403` exceeds the log-log model's `loglog_adj_r2 = 0.6561`, so the added predictors improve adjusted in-sample fit; the 0.05 cutoff is mechanical and is not an importance ranking.

## Part 2 findings

- **R2.1:** The 200-update losses from lowest to highest are η=1.0 (`0.3703`), η=0.1 (`0.3776`), and η=0.01 (`0.5091`). The small rate learns more slowly, while the larger rate converges faster here; a rate that is too large can overshoot or diverge, though that did not occur in these results.
- **R2.2:** The three largest absolute weights are X3 (`0.9013`), X1 (`-0.5554`) and X2 (`0.5505`). X3 has the largest odds ratio (`2.4629`): holding the other standardised predictors fixed, a one-training-standard-deviation increase in discount-order share multiplies estimated churn odds by about 2.46.
- **R2.3:** The all-zero `blind_accuracy` is `0.7875`, close to the logistic accuracy of `0.8042` at τ=0.5. Accuracy is weak here because only 51 of 240 test records are positive, so predicting the majority class already scores highly. Lowering τ from 0.5 to 0.3 raises recall from `0.4706` to `0.7451` while precision falls from `0.5455` to `0.4634`.
- **R2.4:** At τ=0.3, logistic regression has `f1 = 0.5714` and `recall = 0.7451`; KNN has `f1 = 0.3696` and `recall = 0.3333`.
- **R2.5:** I would deploy the logistic model with the training-selected `cost_tau = 0.1500`: its test cost is INR 6,050, compared with INR 11,800 at τ=0.5 and INR 20,400 when nobody is flagged. The threshold is chosen on training rows so the test set remains an evaluation set rather than being used to tune the decision.

## Part 3 findings

- **R3.1:** The successive WCSS drops are 659.8282 (K=2→3), 108.0739 (3→4), 171.0575 (4→5), and 45.5466 (5→6). The elbow rule supports K=3 because the next drop is less than half the preceding drop at that K; the later drops do not shrink steadily because these pinned starting rows give a reproducible solution, not necessarily each K's global optimum.
- **R3.2:** Segment A (322 customers) is **established, steady shoppers**: tenure averages 47.50 months, discount share 0.2250, loyalty 62.16 and outcome rate 0.0186; reinforce repeat purchasing with a loyalty benefit rather than a blanket discount. Segment B (255) is **high-friction, higher-basket shoppers**: complaint count averages 6.07, mean value is INR 1,712.48, and outcome rate is 0.2353; prioritize complaint resolution and service recovery. Segment C (223) is **discount-led, at-risk shoppers**: discount share averages 0.7118 and outcome rate is the highest at `0.4664`; test a targeted, time-limited retention offer. These descriptions follow the observed profiles rather than asserting unmeasured seasonal behavior.
- **R3.3:** At the single-linkage cut, the clusters are `{E1}`, `{E2,E3,E6}`, `{E4}`, and `{E5}`. At the complete-linkage cut, they are `{E1,E4,E5}` and `{E2,E3,E6}`. Single linkage joins groups as soon as any pair of members is close, so its largest height gap occurs after two merges; complete linkage waits for the farthest cross-cluster members to be close, and its largest gap is after four merges.
- **R3.4:** The first component contrasts positive X1 (`0.5317`) and X5 (`0.4418`) loadings with negative X3 (`-0.5187`), X2 (`-0.4100`) and X4 (`-0.2916`), so tenure and loyalty oppose discount share, complaints and average order value on this axis. The standardised matrix has a zero mean vector; three components are needed to retain at least 80% of variance (`0.8616`). Mean PC1 scores are A=`1.7773`, B=`-0.7831` and C=`-1.6709`, separating established shoppers most strongly from the discount-led segment.

## Part 4 findings

- **R4.1:** S4 (X1 ≤ 24) is best under both information gain and Gini gain, so it produces the purest weighted child nodes among the four tested choices. A tree grown until every leaf is pure can memorize noise and small-sample quirks instead of generalising. A maximum-depth constraint or minimum-leaf-size rule can stop excessive growth.
- **R4.2:** I would ship logistic regression: at the comparison threshold τ=0.3 it has `f1 = 0.5714` and `recall = 0.7451`. Combining predictions did not help on this test split: hard vote has F1 `0.5344` and recall `0.6863`, while soft vote has F1 `0.5507` and recall `0.7451`.
- **R4.3:** The two points misclassified by h1 carry total normalised weight `0.5000` after round 1. Round 2 gives h2 more influence because its weighted error (`0.1667`) is lower than h1's (`0.2500`), producing the larger coefficient (`0.8047` versus `0.5493`).
- **R4.4:** For Model A option 2, the input is 28×28×1. The table gives the output shape and trainable parameter count for every operation:

  | Layer | Output shape | Parameters |
  |---|---:|---:|
  | Conv1 (5×5, 6 filters, padding 2) | 28×28×6 | 156 |
  | ReLU | 28×28×6 | 0 |
  | Max-pool1 (2×2) | 14×14×6 | 0 |
  | Conv2 (5×5, 16 filters, no padding) | 10×10×16 | 2,416 |
  | ReLU | 10×10×16 | 0 |
  | Max-pool2 (2×2) | 5×5×16 | 0 |
  | Flatten | 1×1×400 | 0 |
  | Dense1 (32 units) | 1×1×32 | 12,832 |
  | Dense2 (10 classes) | 1×1×10 | 330 |

  For Model B, `W_xh` is 128×64, `W_hh` is 128×128 and `W_hy` is 16×128; the biases have 128 and 16 parameters, respectively, and the total is `rnn_total = 26768`. The RNN reuses the same weights at each time step, so the parameter count does not grow with sequence length. For the image classifier I would use ReLU in hidden layers and softmax with categorical cross-entropy for the ten mutually exclusive classes. A very deep plain network can suffer vanishing gradients, leaving early layers with almost no learning signal; skip connections provide a direct path for information and gradients across layers.

## Model-choice memo

**To: Head of the Retention Team**

Use the logistic-regression classifier as the initial structured-data production candidate, with alerts at τ=0.15. The test-set cost at that threshold is INR 6,050, against INR 11,800 at τ=0.5 and INR 20,400 for flagging nobody; at the τ=0.3 model comparison it also leads the alternatives with F1 0.5714 and recall 0.7451. The cost-minimising threshold was selected on training data, but the evidence is limited to synthetic records and one fixed split with 51 positive test cases, so validate on real, time-separated data before operationalising coupons. Neither the image model (A) nor the review-text model (B) should be built before the structured retention model is production-ready.

## Limitations

- The records are synthetic; their distributions and outcomes do not describe actual Flipkart customers or business results.
- Evaluation uses one fixed train/test split, so performance and the cost-selected threshold may vary on other samples or future periods.
- The test set contains only 51 positive outcomes, making recall, precision and INR cost estimates sensitive to a small number of records.
- The K-means results depend on the mandated initial centroids, and the clustering and regression findings are descriptive rather than causal.
