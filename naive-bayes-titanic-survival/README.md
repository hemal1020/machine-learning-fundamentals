# Gaussian Naive Bayes — Titanic Survival Prediction

Binary classification predicting Titanic passenger survival using `GaussianNB`, with feature engineering
and a walk-through of how Naive Bayes makes its predictions.

**Dataset:** 891 passengers, genuine historical outcomes (real `train.csv`, not Kaggle's demo
`gender_submission.csv` baseline — see note below).

**Approach:** median imputation for skewed numeric columns, engineered `family_size` and `Title` (from
passenger name) features, one-hot encoding, stratified train/test split, evaluated with accuracy,
precision, recall, F1, ROC curve, and 5-fold cross-validation.

**Results:**

| Metric | Test Set | 5-Fold CV |
|---|---|---|
| Accuracy | 0.776 | 0.783 (+/- 0.022) |
| Precision | 0.709 | — |
| Recall | 0.709 | — |
| F1 | 0.709 | — |

These numbers are consistent with published Titanic benchmarks (~0.76-0.83 for simple models) — a
realistic, credible result.

**Important note on the data:** the file originally used for this exercise turned out to be Kaggle's
`test.csv` merged with `gender_submission.csv`, which is Kaggle's *example starter submission* — a trivial
rule ("every female survived, every male didn't"), not real outcomes. Every model trained on that file hit
100% accuracy, because the label column was a literal copy of the `Sex` column. This is a well-known
beginner trap with this specific dataset. Caught it before modeling and swapped in the real 891-passenger
training data (genuine historical outcomes: 81 women died, 109 men survived — a real, imperfect pattern).

**What I learned:** how Bayes' theorem and the "naive" independence assumption combine to make Gaussian
Naive Bayes work — it fits a mean/std per feature per class and multiplies per-feature likelihoods rather
than learning one joint distribution. Also learned to sanity-check a dataset's labels before trusting model
results — 100% accuracy is a red flag for data leakage, not a win.
