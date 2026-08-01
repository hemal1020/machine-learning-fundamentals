# Laptop Price Prediction — KNN Regression

Predicts laptop prices from hardware specs (brand, processor speed, RAM, storage,
screen size, weight) using a K-Nearest Neighbors regressor.

## Files

| File | Description |
|---|---|
| `laptop_knn_regression.ipynb` | Main notebook: EDA, preprocessing, k-tuning, training, evaluation, and a sample prediction |
| `Laptop_price.csv` | Dataset — 1,000 laptops with specs and price |

## Dataset

7 columns, 1,000 rows, no missing values:

- `Brand` — Asus, Acer, Lenovo, HP, Dell (categorical)
- `Processor_Speed` — GHz
- `RAM_Size` — GB
- `Storage_Capacity` — GB
- `Screen_Size` — inches
- `Weight` — kg
- `Price` — target variable

## Approach

1. **EDA** — distribution of price, price by brand, correlation heatmap, price vs. each spec
2. **Preprocessing**
   - One-hot encode `Brand` (nominal category, no natural order)
   - Train/test split (80/20) done **before** scaling
   - `StandardScaler` fit only on the training set, then applied to both — avoids data leakage
3. **Model selection — with an overfitting check, not just the lowest CV error**
   - Initial pass: picking `k` by lowest 5-fold cross-validated RMSE alone selects **k=2**
   - Checking the gap between training-fold and validation-fold RMSE (via
     `cross_validate(..., return_train_score=True)`, entirely within the training set) shows that
     choice has a **96.7% gap** — training error far lower than validation error, a clear
     overfitting signature. KNN with a very low k relies on just 1-2 neighbors per prediction,
     which memorizes local noise rather than learning a real pattern.
   - Fixed by selecting the smallest `k` whose train/validation gap drops under 10%: **k=14**
     (gap ≈ 9.7%)
4. **Training** — `KNeighborsRegressor` fit on the scaled training set with k=14
5. **Evaluation** — MSE, RMSE, MAE, R² on the held-out test set, plus a final train-vs-test gap
   check (4.7%) confirming the chosen k generalizes well — consistent with the cross-validation
   estimate, and never touching the test set until this one confirmation step
6. **Diagnostics** — actual-vs-predicted and residual plots
7. **Inference** — example prediction for a new laptop's specs

## Results

| Metric | Value |
|---|---|
| Selected k | 14 (chosen via train/validation gap, not lowest CV error alone) |
| Test RMSE | ≈ 3,109 |
| Test MAE | ≈ 2,616 |
| Test R² | ≈ 0.894 |
| Train/test RMSE gap | ≈ 4.7% |

**Note:** an earlier version of this notebook picked k=2 by lowest cross-validated RMSE alone,
which looked stronger on paper (R² ≈ 0.949) but had a 70% higher test RMSE than training RMSE —
a real overfitting problem masked by a good-looking headline number. k=14 is a small step down in
raw R² but is the result that can actually be trusted to generalize to new laptops.

## Requirements

See `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

```bash
pip install -r requirements.txt
jupyter notebook laptop_knn_regression.ipynb
```

Run all cells top to bottom — `Laptop_price.csv` must be in the same directory as the notebook.
