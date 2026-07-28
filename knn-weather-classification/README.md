# K-Nearest Neighbors — Weather Condition Classification

Multi-class classification predicting a weather category (4 classes) from 7 numeric weather measurements
using KNN.

**Dataset:** ~141,000 hourly weather readings — pressure, solar radiation, temperature (mean/min/max),
wind speed, wind bearing. Sourced/labeled as `normalized_label` (classes 0-3). A 10,000-row stratified
sample is used for tractable KNN runtime.

**Approach:** stratified sampling and train/test split, `StandardScaler` fit on the training set only
(no leakage), *k* chosen via 5-fold cross-validation elbow method rather than guessed, evaluated with
accuracy, macro F1, weighted F1, and a full classification report.

**Results (k=13, chosen by cross-validation):**

| Metric | Value |
|---|---|
| Accuracy | 0.728 |
| Macro F1 | 0.612 |
| Weighted F1 | 0.702 |

Per-class recall is uneven — classes `0` and `2` (the larger classes) are recalled well (0.93 / 0.79),
while classes `1` and `3` (the smaller classes, ~18% and ~9% of the data) are noticeably harder (0.34 /
0.29). That gap is reported directly rather than hidden behind the headline accuracy number.

**What I learned:** KNN is a distance-based algorithm, so feature scaling isn't optional — and the scaler
must be fit on the training set only, or test-set information leaks into the transform. I also learned
that a large, unjustified *k* silently biases predictions toward the majority class, and that accuracy
alone can hide poor performance on minority classes in an imbalanced dataset — macro F1 and a full
classification report make that visible.

**Note:** `weather.csv` is ~5MB, well within GitHub's file size limits, so it's included directly for
reproducibility.
