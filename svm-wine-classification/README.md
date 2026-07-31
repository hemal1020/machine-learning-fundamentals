# Wine Quality Classification (SVM)

Predicts a wine's quality category — **low / medium / high** — from four physicochemical
measurements: `fixed_acidity`, `residual_sugar`, `alcohol`, and `density`.

## Files

| File | Description |
|---|---|
| `svm_wine_quality.ipynb` | Full notebook: EDA, cleaning, preprocessing, model tuning, evaluation |
| `wine_quality_classification.csv` | Source data (1,000 rows, 5 columns) |

## What changed from the original notebook

The original notebook loaded the data, scaled it, fit a single `SVC(C=10)` with no tuning,
and reported accuracy/classification report with no other diagnostics. This version adds:

- **EDA & visualization** — class balance, feature histograms, boxplots by class,
  a correlation heatmap, and a pairplot.
- **Cleaning checks** — missing values, duplicate rows, IQR-based outlier counts (all clean
  in this dataset).
- **Feature engineering** — two derived features (`sugar_acid_ratio`,
  `alcohol_density_interaction`).
- **Leak-free preprocessing** — the scaler is fit only on the training split, then applied
  to the test split.
- **Stratified train/test split** so class proportions are preserved.
- **Hyperparameter tuning** — `GridSearchCV` over `C`, `gamma`, and `kernel` with 5-fold
  stratified cross-validation, optimizing macro-F1 rather than accuracy.
- **Full evaluation suite** — accuracy, precision, recall, F1 (macro *and* weighted),
  a full classification report, and a confusion matrix heatmap.
- **A baseline comparison** — a "predict the most common class" `DummyClassifier` and a
  tuned `RandomForestClassifier`, so the SVM's score can be read in context rather than
  in isolation.
- **A reusable, correctly-scoped prediction function** and saved model artifacts
  (`.joblib`) for reuse outside the notebook.

## Results

| Model | Accuracy | F1 (macro) |
|---|---|---|
| Baseline (always predict majority class) | 0.355 | 0.175 |
| Tuned SVM (`C=10`, `gamma=0.1`, `rbf`) | 0.325 | 0.319 |
| Random Forest (500 trees, depth 6) | 0.335 | 0.326 |

**Read this table carefully — accuracy alone is misleading here.** The naive baseline gets a
higher raw accuracy (0.355) than either real model simply because it always guesses the
largest class and is never penalized for the two classes it never predicts. Its macro-F1
(0.175) exposes that: it scores 0 precision/recall on `low` and `high`. The tuned SVM and
Random Forest both trade a bit of raw accuracy for much better macro-F1, meaning they actually
attempt to distinguish all three classes rather than defaulting to one. **Macro-F1, not
accuracy, is the fairer metric to optimize and report for this dataset.**

## Why accuracy is capped around ~33–36%

This is the more important finding from the analysis, and it's a data issue, not a modeling
one:

- The boxplots (Section 4.3) show the four features have almost completely overlapping
  distributions across the `low`, `medium`, and `high` classes.
- The correlation heatmap and per-class means show no feature is meaningfully associated
  with the class label.
- A tuned SVM, a tuned Random Forest, and a majority-class baseline all land within ~3
  points of each other — extensive hyperparameter search and feature engineering could not
  move the needle beyond that band.

In other words: for a 3-class problem, "random guessing" already scores about 33%, and every
model tested here — however well tuned — sits close to that floor. This strongly suggests the
four available features do not carry enough information to reliably predict this particular
`quality_label` column, rather than indicating the SVM implementation is deficient.

### Suggested next steps if higher accuracy is required

- Check the data source / generation process — if this is a synthetic or randomly-labeled
  dataset for a course exercise, no model can do meaningfully better than chance.
- If working with real wine chemistry data (e.g. the UCI Wine Quality dataset), add the
  additional chemically relevant features it provides — volatile acidity, citric acid,
  chlorides, free/total sulfur dioxide, pH, and sulphates — which carry much more signal
  than acidity/sugar/alcohol/density alone.
- If more data can be collected, a larger sample size would make it easier to tell whether
  a weak-but-real signal exists versus none at all.

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
jupyter notebook svm_wine_quality_improved.ipynb
```

Run all cells top to bottom. The notebook will regenerate all plots, retrain the model, and
save `svm_wine_quality_model.joblib`, `scaler.joblib`, and `label_encoder.joblib` in the same
directory.

## Using the saved model elsewhere

```python
import joblib
import pandas as pd

model = joblib.load('svm_wine_quality_model.joblib')
scaler = joblib.load('scaler.joblib')
le = joblib.load('label_encoder.joblib')

sample = pd.DataFrame([{
    'fixed_acidity': 9.3,
    'residual_sugar': 6.4,
    'alcohol': 13.6,
    'density': 1.0005,
    'sugar_acid_ratio': 6.4 / 9.3,
    'alcohol_density_interaction': 13.6 * 1.0005
}])

pred = model.predict(scaler.transform(sample))
print(le.inverse_transform(pred)[0])
```
