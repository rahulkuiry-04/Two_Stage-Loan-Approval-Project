# Two-Stage Loan Approval & Amount Prediction System

A machine learning pipeline that predicts **loan approval status** and, for approved applicants, the **likely sanctioned loan amount** — using a two-stage Random Forest architecture built with scikit-learn.


**🔗 Live Demo:** [twostage-loan-approval-project.streamlit.app](https://twostage-loan-approval-project-e2btftf2owaaxctehreqkm.streamlit.app/)

---

## Project Summary & Objective

Loan underwriting traditionally requires manual review of an applicant's financial profile — income, credit history, assets, and existing obligations — to decide both **whether** to approve a loan and **how much** to sanction. This project automates that decision process using a **two-stage supervised learning pipeline**:

1. **Stage 1 (Classification):** Predicts whether a loan application will be `Approved` or `Rejected`.
2. **Stage 2 (Regression):** For applications predicted as `Approved`, estimates the likely loan amount to be sanctioned.

The objective is to demonstrate an end-to-end, production-style ML workflow — from raw tabular data to a deployable two-stage inference function — with reproducible preprocessing, hyperparameter tuning, and clear performance validation at each stage.

---

## Dataset Description

- **Source:** [Loan Approval Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- **Size:** 4,269 rows × 13 columns
- **Target variables:**
  - `loan_status` (binary: `Approved` / `Rejected`) — used for Stage 1 classification
  - `loan_amount` (continuous) — used for Stage 2 regression (approved applicants only)
- **Feature columns:**

| Feature | Type | Description |
|---|---|---|
| `no_of_dependents` | Numeric | Number of dependents |
| `education` | Categorical | Graduate / Not Graduate |
| `self_employed` | Categorical | Yes / No |
| `income_annum` | Numeric | Annual income |
| `loan_amount` | Numeric | Requested loan amount |
| `loan_term` | Numeric | Loan term (years) |
| `cibil_score` | Numeric | Credit score |
| `residential_assets_value` | Numeric | Value of residential assets |
| `commercial_assets_value` | Numeric | Value of commercial assets |
| `luxury_assets_value` | Numeric | Value of luxury assets |
| `bank_asset_value` | Numeric | Value of bank assets |
| `loan_status` | Categorical | Target (Stage 1) |


---

##  Model Architectures & Methodology

### Preprocessing Pipeline
Built using `sklearn.compose.ColumnTransformer` inside a `Pipeline`, so preprocessing and modeling are bundled into a single serializable object:

- **Numerical features:** `SimpleImputer(strategy='median')` → `StandardScaler()`
- **Categorical features:** `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`

### Stage 1 — Classification (Approval Prediction)
- **Model:** `RandomForestClassifier`
- **Hyperparameter tuning:** `GridSearchCV` (5-fold CV, scoring = `f1`) over `n_estimators`, `max_depth`, `max_features`
- **Best parameters found:** `n_estimators=300`, `max_depth=10`, `max_features=None`
- **Final baseline model:** `n_estimators=400`, `max_depth=None`, `oob_score=True`, `random_state=10`

### Stage 2 — Regression (Loan Amount Estimation)
- **Model:** `RandomForestRegressor`
- **Hyperparameter tuning:** `GridSearchCV` (5-fold CV, scoring = `r2`) over `n_estimators`, `max_depth`, `max_features`, `min_samples_split`
- **Best parameters found:** `n_estimators=200`, `max_depth=8`, `max_features=None`, `min_samples_split=5`

### Two-Stage Inference Function
A single `two_stage_predict()` function chains both models:
1. Runs the classifier on the applicant's data.
2. If predicted `Approved`, feeds the applicant (with `loan_status` set to `'Approve'`) into the regressor to estimate the loan amount.
3. If predicted `Rejected`, skips regression and returns the rejection decision only.

Both fitted pipelines are serialized with `joblib` for reuse in downstream applications (e.g. a Streamlit app).

---

## Model Comparison & Performance Evaluation

### Stage 1 — Classification Metrics

| Metric | Tuned Model (GridSearchCV) | Final Baseline Model |
|---|---|---|
| Accuracy | 0.99 | 0.99 |
| F1-score (weighted avg) | 0.99 | 0.99 |
| Precision (class 0 / 1) | 0.98 / 0.99 | 0.98 / 0.99 |
| Recall (class 0 / 1) | 0.98 / 0.99 | 0.98 / 0.99 |


**Top predictive features (Stage 1):** `cibil_score` (82.8% importance) dominates the decision, followed by `loan_term` (8.0%) and `loan_amount` (3.6%) — asset values and dependents contribute marginally.

### Stage 2 — Regression Metrics

| Metric | Value |
|---|---|
| R² Score | **0.864** |
| MAE | ₹25,12,447 |
| MSE | 1.12 × 10¹³ |

**Top predictive features (Stage 2):** `income_annum` dominates loan amount prediction (93.4% importance), with `cibil_score` and asset values contributing minor adjustments.

---

## Installation & Setup

### Prerequisites
- Python 3.12
- pip


### 1. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
pandas==2.3.0
numpy==2.2.4
scikit-learn==1.7.1
joblib==1.5.1
matplotlib==3.10.7
seaborn==0.13.2
```

### 3. Download the dataset
Download `loan_approval_dataset.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) and place it in the project root.

### 4. Run the notebook
```bash
jupyter notebook Two_Stage_Loan_Using_random_forest.ipynb
```
Run all cells top to bottom. The final cells serialize the trained pipelines:
```python
joblib.dump(rf_clf_pipeline, 'stage_1_rf_classifier_pipeline.pkl')
joblib.dump(rf_reg_pipeline, 'stage_2_rf_regression_pipeline.pkl')
```

---

## Key Findings & Conclusion

- The **Random Forest classifier** achieves **99% accuracy and 0.99 F1-score** on held-out test data, with an out-of-bag score of **0.984**, indicating a highly reliable approval decision model with minimal overfitting.
- **`cibil_score` is overwhelmingly the strongest predictor of approval**, consistent with real-world lending practice.
- The **Random Forest regressor** explains **86.4% of the variance (R² = 0.864)** in approved loan amounts, with **`income_annum`** as the dominant driver — an intuitive result, since sanctioned amounts typically scale with applicant income.
- The **two-stage architecture** mirrors real-world underwriting logic: a hard approval gate followed by a conditional amount estimation, rather than a single end-to-end regression that ignores rejection cases.
- **Future improvements** could include testing gradient-boosted models (XGBoost/LightGBM) for comparison, adding SHAP-based explainability (already scoped via the `shap` dependency), and calibrating classifier probabilities for risk-tiered decisions rather than a hard binary cutoff.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
