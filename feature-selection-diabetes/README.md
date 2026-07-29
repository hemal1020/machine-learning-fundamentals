# Feature Selection with SelectKBest — Pima Diabetes Dataset

Ranking predictive features using `SelectKBest` with the ANOVA F-test, plus a real model comparison
proving the selection matters, not just a static ranked list.

**Dataset:** 768 patients, 8 numeric medical features, binary `Outcome` (diabetic / not). Public,
de-identified, widely used teaching dataset (Pima Indians Diabetes).

**Key finding:** `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` all encode missing
values as `0`, which `pandas.isnull()` doesn't catch (up to 49% of `Insulin` values are these fake zeros).
Left unfixed, this significantly distorts the feature ranking:

| Feature | F-score (zeros as real) | F-score (zeros treated as missing) |
|---|---|---|
| SkinThickness | 4.3 (ranked last) | 35.5 (ranked 5th of 8) |
| Insulin | 13.3 | 39.4 |

**Approach:** train/test split before any imputation or scoring (avoids leakage), zeros marked as missing
and imputed with training-set medians, `SelectKBest`/`f_classif` scored on training data only, followed by
a direct model comparison: logistic regression trained on all 8 features vs. only the top 5 by F-score.

**Result:** the top-5-feature model matches (or slightly beats) the full 8-feature model on both accuracy
and F1 — confirming `Pregnancies`, `BloodPressure`, and `DiabetesPedigreeFunction` add little beyond the
other 5 features for this dataset.

**What I learned:** a dataset can report zero missing values via `.isnull()` while still having a large
hidden data-quality problem (missing values encoded as `0` instead of `NaN`) — worth checking for
domain-implausible values, not just null counts. Also learned that feature scoring/imputation, like scaling,
needs to be fit on the training set only to avoid leaking test-set information into the selection decision.
