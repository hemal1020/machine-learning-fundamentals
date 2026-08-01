# ML Fundamentals — Practice

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

Hands-on exercises implementing core machine learning algorithms on different datasets, built to
strengthen my understanding of how these models work individually before applying them together in
my thesis research (a phishing SMS detection system using a Hard Voting Ensemble).

This is a learning/practice repository, not a research project — each folder is a focused exercise
on one algorithm, run end-to-end on real data with full evaluation, not just a script that runs
without checking whether the result is actually good.

## What's Inside

| Algorithm | Dataset | Notes |
|---|---|---|
| Linear Regression | *(fill in dataset name)* | *(fill in: what you predicted, key takeaway)* |
| Multiple Linear Regression | Student Performance (10,000 students) | R² = 0.989, RMSE ≈ 2.02. VIF confirmed no multicollinearity; standardized coefficients showed `Previous Scores` and `Hours Studied` dominate. |
| Logistic Regression | Student exam pass/fail (1,000 students) | Accuracy 0.86, F1 0.89, ROC-AUC 0.95, 5-fold CV accuracy 0.846. |
| KNN Classification | Weather condition, 4 classes (141K readings) | k=13 chosen by cross-validation (not an arbitrary k=90). Accuracy 0.728, Macro F1 0.612 — reported alongside accuracy since the classes are imbalanced. |
| KNN Regression | Laptop price prediction (1,000 laptops) | k=14, R² 0.894, RMSE ≈ 3,109. Caught and fixed a real overfitting problem: the naive "lowest CV error" choice (k=2) had a 70% train/test gap; fixed by selecting k via train/validation gap analysis instead. |
| Naive Bayes | Titanic survival (891 passengers) | Accuracy 0.776, F1 0.709, 5-fold CV 0.783. Caught a mislabeled dataset before modeling — an initial file had `Survived` copied directly from `Sex` (Kaggle's demo baseline, not real outcomes) and was replaced with the genuine historical data. |
| K-Means Clustering | Student performance (150 students) | k=3 across all feature pairings, silhouette scores 0.672–0.681, confirmed with a 3-feature + PCA clustering. |
| Feature Selection (SelectKBest) | Pima Diabetes (768 patients) | Found that 5 columns encode missing values as `0` (undetected by `.isnull()`), which was distorting the feature ranking. Top-5-feature model matched the full 8-feature model (Accuracy 0.72, F1 0.57). |
| Feature Correlation | California Housing (20,640 districts) | Demonstrated a real encoding bug: ordinal-encoding a nominal category collapsed a strong relationship (-0.485 for `INLAND`) into a near-meaningless 0.08. Top predictor: `median_income` (+0.688). |
| SVM | Wine quality classification, 3 classes (1,000 samples) | Tuned SVM: Accuracy 0.325, Macro F1 0.319 — barely above a majority-class baseline (Acc 0.355, F1 0.175). Diagnosed *why*: the 4 available features carry almost no separable signal for this label, confirmed by testing a Random Forest and baseline for comparison rather than assuming the SVM was misconfigured. |

## Structure

```
machine-learning-fundamentals/
├── linear-regression/
│   └── linear_regression.ipynb
├── multiple-regression-student-performance/
│   ├── multiple_regression.ipynb
│   ├── Student_Performance.csv
│   ├── README.md
│   └── requirements.txt
├── logistic-regression-exam-prediction/
│   ├── logistic_regression.ipynb
│   ├── student_exam_data.csv
│   ├── README.md
│   └── requirements.txt
├── knn-weather-classification/
│   ├── knn_classification.ipynb
│   ├── weather.csv
│   ├── README.md
│   └── requirements.txt
├── knn-regression-laptop-price/
│   ├── laptop_knn_regression.ipynb
│   ├── Laptop_price.csv
│   ├── README.md
│   └── requirements.txt
├── naive-bayes-titanic-survival/
│   ├── naive_bayes_titanic.ipynb
│   ├── titanic.csv
│   ├── README.md
│   └── requirements.txt
├── kmeans-student-clustering/
│   ├── kmeans_clustering.ipynb
│   ├── student_clustering_data.csv
│   ├── README.md
│   └── requirements.txt
├── feature-selection-diabetes/
│   ├── feature_selection.ipynb
│   ├── diabetes.csv
│   ├── README.md
│   └── requirements.txt
├── feature-correlation-housing/
│   ├── feature_correlation.ipynb
│   ├── housing.csv
│   ├── README.md
│   └── requirements.txt
├── svm-wine-classification/
│   ├── svm_wine_quality.ipynb
│   ├── wine_quality_classification.csv
│   ├── README.md
│   └── requirements.txt
└── README.md
```

Each subfolder has its own README with the full write-up: methodology, why any given design choice
was made, and results in more detail than the summary table above.

## Tech Stack

- **Language:** Python
- **Core libraries:** scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Also used:** statsmodels (VIF / multicollinearity check), joblib (model persistence)

## Recurring Themes Across These Exercises

A few issues showed up repeatedly while working through these — worth calling out here since they're
the actual skill being demonstrated, not just "ran a model and got a number":

- **Data leakage** — fitting a scaler or feature selector before the train/test split, rather than
  after, appeared in multiple notebooks and was fixed each time by re-ordering the pipeline.
- **Overfitting hiding behind a good-looking metric** — the KNN regression exercise is the clearest
  example: the naive "lowest cross-validation error" choice looked best on paper but generalized
  badly; catching this required checking a train/validation gap, not just the headline score.
- **Encoding choices changing results, not just style** — using ordinal encoding on unordered
  (nominal) categories manufactured fake relationships in two separate exercises, measurably
  distorting correlation and prediction results until fixed with one-hot encoding.
- **A model with a low score isn't automatically a bad model** — the SVM wine classification exercise
  is a deliberately-included example of this: a well-tuned model, correctly evaluated, showing that
  the data itself doesn't support high accuracy for that label, confirmed against a baseline rather
  than assumed.

## Why This Repo Exists

Before combining multiple models into the Hard Voting Ensemble used in my thesis (Phishing SMS
Detection in Bangladeshi Mobile Banking), I practiced each algorithm individually to understand its
assumptions, strengths, and limitations — for example, how regularization affects Logistic Regression,
or how the choice of kernel changes an SVM's decision boundary.

## Related

- [Phishing SMS Detection in Bangladeshi Mobile Banking System (Thesis Project)](https://github.com/hemal1020/Phishing-SMS-Detection-in-Bangladeshi-Mobile-Banking-System.git) — where these fundamentals were applied in a real ensemble system
