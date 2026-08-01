# Laptop Price Prediction — KNN Regression

Predicts laptop prices from hardware specs (brand, processor speed, RAM, storage,
screen size, weight) using a K-Nearest Neighbors regressor.

## Files

| File | Description |
|---|---|
| `knnReg_improved.ipynb` | Main notebook: EDA, preprocessing, k-tuning, training, evaluation, and a sample prediction |
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
3. **Model selection** — 5-fold cross-validation over `k = 1..40`, picking the `k` with the lowest RMSE
4. **Training** — `KNeighborsRegressor` fit on the scaled training set with the best `k`
5. **Evaluation** — MSE, RMSE, MAE, R² on the held-out test set, plus actual-vs-predicted and residual plots
6. **Inference** — example prediction for a new laptop's specs

## Results

| Metric | Value |
|---|---|
| Best k | 2 |
| RMSE | ≈ 2,161 |
| MAE | ≈ 1,129 |
| R² | ≈ 0.949 |

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Usage

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook knnReg_improved.ipynb
```

Run all cells top to bottom — `Laptop_price.csv` must be in the same directory as the notebook.
