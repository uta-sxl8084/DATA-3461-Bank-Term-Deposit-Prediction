# Bank Term Deposit Subscription Prediction

**DATA 3461 — Machine Learning Project**
**Members: Ahnaf Mahiat Samiha Harappa, Sunaihra Zeb Lamia**

## Project Overview

This project uses machine learning to predict whether a bank customer will subscribe to a term deposit following a marketing campaign. The goal is to identify potential subscribers using customer demographic, financial, and campaign-related information.

The project compares Logistic Regression, Random Forest, and Gradient Boosting against a majority-class baseline.

## Dataset

The project uses the UCI Bank Marketing dataset (`bank-additional-full.csv`).

- Original records: 41,188
- Records after duplicate removal: 41,176
- Original features: 20
- Features used for prediction: 19
- Target variable: `y` (subscription: yes/no)
- Positive class proportion: approximately 11.27%

The dataset is imbalanced, so evaluation emphasizes precision, recall, F1-score, and average precision (PR-AUC summary) rather than accuracy alone.

The dataset is not included in this repository, to reproduce the analysis:

- Download the dataset from the UCI Machine Learning Repository.
- Extract the downloaded files.
- Locate bank-additional-full.csv.
- Place the CSV file inside the project's data/ directory.
- Run notebooks/bank_marketing_analysis.ipynb.

The notebook uses the relative path ../data/bank-additional-full.csv.

## Methodology

1. Perform exploratory data analysis and identify missing, unknown, and duplicate values.
2. Remove duplicate records and encode the binary target variable.
3. Exclude call duration to prevent information leakage in pre-call predictions.
4. Split data into stratified training (70%), validation (15%), and testing (15%) sets.
5. Apply numerical scaling and one-hot encoding using a preprocessing pipeline.
6. Train a majority-class baseline, Logistic Regression, Random Forest, and Gradient Boosting.
7. Tune Gradient Boosting hyperparameters using stratified cross-validation.
8. Optimize classification thresholds using validation data.
9. Evaluate the selected model on the held-out test set.

## Model Comparison

Initial validation results:

| Model | Accuracy | Precision | Recall | F1-score | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.8873 | 0.0000 | 0.0000 | 0.0000 | 0.1127 |
| Logistic Regression | 0.8251 | 0.3491 | 0.6379 | 0.4512 | 0.4430 |
| Random Forest | 0.8722 | 0.4337 | 0.4368 | 0.4352 | 0.3881 |
| Gradient Boosting | 0.9011 | 0.6840 | 0.2270 | 0.3409 | 0.4709 |

Gradient Boosting was selected because of its strong precision-recall ranking performance and improved F1-score after threshold optimization.

## Final Model Results

**Selected model:** Gradient Boosting Classifier

**Decision threshold:** 0.1857

Held-out test results:

| Metric | Score |
|---|---:|
| Accuracy | 0.8813 |
| Precision | 0.4789 |
| Recall | 0.6034 |
| F1-score | 0.5340 |
| PR-AUC | 0.4896 |

The model correctly identified 420 of 696 actual subscribers in the test set.

## Information Leakage Experiment

Call duration is only available after a marketing call and was therefore excluded from the final model.

| Feature Configuration | Validation PR-AUC |
|---|---:|
| Without duration | 0.4709 |
| With duration | 0.6551 |

Including duration increased apparent performance but would make the model unsuitable for predicting subscriptions before calls occur.

## Single-Customer Prediction Demo

The trained model is saved with its preprocessing pipeline and classification threshold.

Example prediction:

- Estimated subscription probability: 0.0388
- Decision threshold: 0.1857
- Predicted outcome: No

## Technologies

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook, and Joblib.

## Repository Structure

```text
DATA-3461-Bank-Term-Deposit-Prediction/
├── data/
│   └── .gitkeep
├── models/
│   └── bank_subscription_model.joblib
├── notebooks/
│   └── bank_marketing_analysis.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset Source

UCI Machine Learning Repository — Bank Marketing Dataset

https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset is excluded from GitHub and must be downloaded separately.

## Limitations

The dataset contains historical marketing campaign information, which may not fully represent current customer behavior. The positive class is relatively rare, and the selected threshold produces false positives. Predicted probabilities are not independently calibrated. The model is intended as an academic demonstration rather than a production banking decision system.
