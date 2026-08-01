# Simple Linear Regression — Salary vs. Years of Experience

Predicting salary from a single feature (years of experience) using ordinary least squares — the most
basic supervised learning exercise, one feature and one target.

**Dataset:** 30 employees, `YearsExperience` and `Salary`. No missing values. A widely-used public
teaching dataset for simple linear regression.

**Approach:** train/test split, fitted line interpretation (slope/intercept), evaluated with R², RMSE,
MAE, cross-validated (since the dataset is small enough that a single split alone is noisy), residual
check, and an explicit note on the risk of extrapolating predictions beyond the observed data range.

**Results:**

| Metric | Value |
|---|---|
| Equation | Salary ≈ 25,322 + 9,424 × YearsExperience |
| R² (test) | 0.902 |
| RMSE | ≈ 7,059 |
| MAE | ≈ 6,286 |
| 5-fold CV R² | reported in notebook, given the small sample size |

**What I learned:** the mechanics behind ordinary least squares — minimizing total squared distance
between the line and every point — and how to read a slope/intercept directly (each year of experience
associated with ~$9,424 more predicted salary). Also learned to flag extrapolation risk: the model will
happily predict for 12+ years of experience even though the training data only goes up to 10.5 years, and
that prediction is less trustworthy than one inside the observed range. With only 30 rows, a single
train/test split is noisy, so cross-validation is used to get a steadier performance estimate rather than
trusting one split's R² alone.
