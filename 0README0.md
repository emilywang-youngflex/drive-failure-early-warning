# Drive Failure Early Warning

A leakage-aware machine-learning project for detecting storage-drive failures using public Backblaze SMART telemetry.

## Overview

This project investigates whether a compact set of SMART health indicators can support an early-warning system for storage-drive failure.

Rather than treating the task as a generic row-level classification problem, the project focuses on operational questions:

- How many failed drives can be detected?
- How many healthy drives would be alerted?
- How early before failure does the first valid alert occur?
- At the same alert workload, does a more complex feature set outperform a simpler baseline?

The final pipeline uses grouped cross-validation, out-of-fold predictions, drive-level alert aggregation, lead-time analysis, error analysis, and matched-workload model comparison.

## Project Context

This was a self-directed educational project completed during an internship period at Silicon Motion and inspired by the company’s storage domain.

The project:

- uses public Backblaze drive telemetry;
- does not use proprietary Silicon Motion data;
- was not a deployed Silicon Motion production system;
- should not be interpreted as validation on Silicon Motion SSD hardware.

## Why This Problem Is Difficult

1. **Extreme class imbalance** - Failure observations represent less than 1% of the data, so accuracy alone would be misleading.
2. **Repeated observations per drive** - Each physical drive appears across many dates. A random row split could place the same drive in both training and validation data, causing leakage.
3. **Operational evaluation** - A single drive may generate many high-risk rows, but maintenance teams investigate physical drives, not individual rows.
4. **Warning timing** - A useful alert must occur on or before the recorded failure date. Post-failure alerts do not count as successful early warnings.
5. **Score comparability** - Different models and folds can produce scores on different scales, making fixed numerical thresholds unreliable across model families.

## Dataset

The project uses a cleaned subset of public Backblaze drive telemetry.

Expected input file:

```text
df_cleaned.parquet
```

Required columns:

```text
serial_number
date
failure
smart_5_raw
smart_197_raw
smart_198_raw
```

The full dataset is not stored in this repository because of its size.

## Baseline Features

```python
BASE_FEATURES = [
    "smart_5_raw",
    "smart_197_raw",
    "smart_198_raw",
]
```

## Leakage-Safe Validation

Because each drive appears on multiple dates, validation was performed with `StratifiedGroupKFold`, using `serial_number` as the grouping key.

This keeps all observations from one physical drive in a single fold, reducing drive-identity leakage while helping distribute rare failures across folds.

## Models Evaluated

- Logistic Regression
- Random Forest
- XGBoost

Logistic Regression was retained as **Baseline v1** because it showed the most stable validation behavior under the selected features and grouped validation design.

This does not mean Logistic Regression is universally better. It was the most defensible baseline for this dataset, feature set, and evaluation framework.

## Out-of-Fold Predictions

The baseline generated out-of-fold predictions with this schema:

```text
fold
serial_number
date
y_true
y_prob
```

Each prediction came from a model that did not train on that drive’s fold. Preserving drive identifiers and dates enabled later drive-level thresholding and lead-time analysis.

## Drive-Level Evaluation

The raw prediction unit was a drive-day, but the operational unit was a physical drive. Predictions were therefore aggregated by `serial_number`.

For failed drives, an alert counted as successful only when it occurred on or before the recorded failure date.

### Evaluation Population

```text
Total unique drives: 281,192
Failed drives: 324
Healthy drives: 280,868
```

## Key Result

At the selected exploratory threshold of **0.70**, Baseline v1:

- alerted **2,138 of 281,192 drives**;
- detected **152 of 324 failed drives**;
- missed **172 failed drives**;
- alerted **1,986 healthy drives**;
- achieved **46.9% drive-level recall**;
- achieved **7.11% drive-level precision**;
- alerted approximately **0.76% of the total fleet**;
- produced a **17-day median lead time** among detected failures.

This threshold was selected as a balanced exploratory operating point, not as a production threshold.

## Threshold Trade-Offs

| Threshold | Drives Alerted | Failed Drives Detected | Failed Drives Missed | Drive Recall | Drive Precision |
|---:|---:|---:|---:|---:|---:|
| 0.35 | 6,735 | 228 | 96 | 70.4% | 3.39% |
| 0.70 | 2,138 | 152 | 172 | 46.9% | 7.11% |
| 0.75 | 2,032 | 149 | 175 | 46.0% | 7.33% |
| 0.90 | 1,519 | 133 | 191 | 41.0% | 8.76% |

Lower thresholds detected more failures but created a larger investigation workload.

## Lead-Time Analysis

Lead time was calculated from the first valid pre-failure alert to the recorded failure date.

| Threshold | Median Lead Time | Mean Lead Time |
|---:|---:|---:|
| 0.35 | 22.5 days | 28.6 days |
| 0.70 | 17 days | 24.6 days |
| 0.90 | 15 days | 24.1 days |

The 17-day median at threshold 0.70 applies only to failed drives that were successfully detected.

## Error Analysis

At threshold 0.70:

```text
False-negative failed drives: 172
False-positive healthy drives: 1,986
```

### Missed Failures

| Miss Pattern | Drives | Approximate Share |
|---|---:|---:|
| Weak signal below 0.35 | 69 | 40.1% |
| Moderate signal from 0.35 to 0.60 | 46 | 26.7% |
| Short history under 7 observed days | 45 | 26.2% |
| Near miss from 0.60 to 0.70 | 12 | 7.0% |

Only 12 missed failures were close to the selected threshold. This suggested that threshold tuning was no longer the main limitation. Many missed failures showed weak or absent warning signal in the selected SMART attributes.

### Healthy-Drive Alerts

Most false-positive drives produced repeated alerts rather than isolated one-day spikes.

Possible explanations include degraded drives that did not fail during the observation period, right-censored outcomes, removed drives, drive-model-specific SMART behavior, and genuine false alarms.

Their eventual outcomes were unknown, so these cases were not treated as proof that the model was correct.

## Temporal Feature Experiment

A reduced temporal feature set was tested to capture persistence and recent change. It added:

- current non-zero indicators;
- abnormal SMART counts;
- one-observation deltas;
- seven-observation rolling maxima;
- seven-observation non-zero counts.

Together with the three raw SMART values, the temporal model used 16 features.

The screening experiment used a 10% sample, three grouped folds, and `SGDClassifier` with log-loss. The same estimator was used for both the baseline and temporal feature sets so the comparison isolated the effect of the features.

## Methodological Correction: Probability Saturation

The first matched-workload comparison used `predict_proba`. Many high-risk predictions became exactly `1.0`, creating large ties and destroying ranking resolution.

The evaluation was corrected by saving `decision_score` from `decision_function`.

Decision scores were ranked within each validation fold, and equal alert budgets were assigned inside each fold before combining results. This avoided treating raw score magnitudes as directly comparable across models.

## Matched-Workload Results

| Alert Budget | Baseline Detected | Temporal Detected | Result |
|---:|---:|---:|---|
| 0.5% | 37 | 33 | Baseline wins |
| 1% | 47 | 44 | Baseline wins |
| 2% | 62 | 69 | Temporal wins |
| 5% | 81 | 81 | Tie, but severe boundary tie |

Drive-level ranking metrics:

| Model | Drive PR-AUC | Drive ROC-AUC |
|---|---:|---:|
| Baseline v1 | 0.032105 | 0.8817 |
| Temporal v2a | 0.055255 | 0.8730 |

The temporal feature set improved drive-level PR-AUC and performed better at the 2% alert budget, but it did not consistently outperform the simpler baseline at the most restrictive workloads.

The 5% result was weakened by a severe score-boundary tie and was not treated as strong evidence.

## Final Model Decision

**Baseline v1 was retained.**

Reasons:

- it detected more failed drives at the 0.5% and 1% alert budgets;
- temporal v2a won only at 2%;
- the 5% result was weakened by a severe boundary tie;
- the temporal model was more complex and slower;
- the operational improvement was not consistent enough to justify promotion.

The final decision favored simplicity, interpretability, and performance under constrained alert capacity.

## Repository Structure

```text
drive-failure-early-warning/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_data_and_problem_overview.ipynb
│   ├── 02_grouped_model_validation.ipynb
│   ├── 03_baseline_oof_predictions.ipynb
│   ├── 04_drive_level_evaluation.ipynb
│   ├── 05_error_analysis.ipynb
│   └── 06_temporal_feature_screening.ipynb
├── reports/
│   └── figures/
└── sample_outputs/
    ├── drive_level_threshold_summary.csv
    ├── lead_time_summary.csv
    └── matched_workload_summary.csv
```

## Running the Project

The repository is being curated from the original experimental notebooks. The public version is intended to reproduce the modeling and evaluation workflow from a cleaned Parquet dataset.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place the cleaned dataset at:

```text
data/df_cleaned.parquet
```

Then run the notebooks in numerical order.

## Limitations

- Failure prevalence was extremely low.
- The final baseline used only three raw SMART attributes.
- SMART values were sparse and heavy-tailed.
- Some failed drives showed little visible pre-failure signal.
- Some drives had short observed histories.
- False positives may include right-censored or degraded drives.
- Performance varied across drive models.
- The temporal experiment used a 10% sample and three folds.
- SGD probabilities were not calibrated.
- The corrected comparison relied on within-fold ranking.
- The 5% matched-workload result had a severe boundary tie.
- Grouped validation prevented drive-identity leakage but was not a strict future-time holdout.
- Public Backblaze HDD telemetry is not equivalent to proprietary SSD-controller telemetry.
- The project was exploratory and was not deployed.

## Future Work

Potential extensions include:

- temporal holdout or rolling-origin validation;
- model-specific or hierarchical features;
- additional carefully selected SMART attributes;
- persistence-based alert rules;
- calibrated fold-level probabilities;
- survival or time-to-event modeling;
- explicit investigation and failure-cost modeling;
- evaluation on SSD-specific telemetry.

## Main Engineering Lessons

1. Group-aware validation is essential when entities have repeated observations.
2. Row-level metrics do not automatically translate into operational usefulness.
3. Thresholds cannot be transferred blindly between models.
4. Probabilities can lose ranking resolution through numerical saturation.
5. Matched alert workloads can be more meaningful than equal numerical thresholds.
6. Error analysis can reveal limits in the available information.
7. A simpler model should be retained when added complexity does not produce a consistent operational advantage.

## Status

```text
Technical modeling: complete
Final selected model: Baseline v1
Current phase: public repository curation and recruiting preparation
```
