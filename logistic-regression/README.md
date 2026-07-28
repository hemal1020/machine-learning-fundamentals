# Logistic Regression — Exam Pass Prediction

Binary classification exercise predicting whether a student passes an exam based on hours studied.

**Dataset:** 1,000 students, single feature (`study_hours`), binary target (`passed_exam`). No missing values.

**Approach:** stratified train/test split, `StandardScaler` + `LogisticRegression` pipeline, evaluated with
accuracy, precision, recall, F1, and ROC-AUC, validated with 5-fold stratified cross-validation.

**Results:**

| Metric | Test Set | 5-Fold CV |
|---|---|---|
| Accuracy | 0.86 | 0.846 (+/- 0.036) |
| Precision | 0.87 | — |
| Recall | 0.90 | — |
| F1 | 0.89 | 0.875 (+/- 0.030) |
| ROC-AUC | 0.95 | — |

**What I learned:** how the sigmoid function maps a continuous feature to a pass probability, why
stratified splitting matters with imbalanced classes, and why a target column should never be
mean/median-imputed (only features should be — a missing label means the row should be dropped, not
guessed). I also tested feature scaling and polynomial features to see if they'd help; they didn't move
accuracy, which makes sense since the study-hours-to-outcome relationship is already close to linear —
included here to show the process rather than only the metrics that improved.
