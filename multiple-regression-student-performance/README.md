# Multiple Linear Regression — Student Performance Index

Predicting a continuous Performance Index (0-100) from 5 study-related features, with full evaluation,
VIF multicollinearity check, and residual diagnostics.

**Dataset:** 10,000 students — Hours Studied, Previous Scores, Extracurricular Activities (Yes/No), Sleep
Hours, Sample Question Papers Practiced, target Performance Index. No missing values.

**Approach:** binary label encoding (fine here — only 2 categories, no false-ordering risk), VIF check for
multicollinearity, train/test split, evaluated with R², MAE, MSE, RMSE, standardized coefficients for fair
feature-importance comparison, and residual diagnostics.

**Results:**

| Metric | Value |
|---|---|
| R² | 0.989 |
| MAE | 1.61 |
| MSE | 4.07 |
| RMSE | 2.02 |

VIF for all 5 features ≈ 1.0 — no meaningful multicollinearity. Standardized coefficients show
`Previous Scores` and `Hours Studied` dominate the prediction; `Extracurricular Activities`, `Sleep Hours`,
and `Sample Question Papers Practiced` contribute comparatively little.

**What I learned:** the biggest gap in the original notebook was that it imported evaluation metrics,
computed predictions, and then never actually called the metrics — a model isn't "done" until it's
evaluated. Also learned that `variance_inflation_factor` needs an intercept/constant column added first
or the VIF numbers come out misleadingly inflated (a specific gotcha with that function), and that raw
regression coefficients aren't directly comparable across differently-scaled features — standardizing
first gives a fair read on relative feature importance. Residual plots also matter specifically for linear
regression, as a way to visually confirm a linear model is actually appropriate rather than trusting R²
alone.

**Note on the dataset:** the original notebook referenced `Student_Performance.csv` without including it.
This is the standard public Kaggle "Student Performance (Multiple Linear Regression)" dataset (documented
as synthetic, created for illustrative purposes), sourced here from a public GitHub mirror.
