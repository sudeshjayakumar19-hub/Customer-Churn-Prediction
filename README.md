# Customer Churn Prediction Using Machine Learning

## Project Title
Customer Churn Prediction Using Machine Learning — IBM Telco Dataset

## Problem Statement
Customer churn is a critical business challenge in the telecommunications industry.
Losing existing customers is significantly more costly than acquiring new ones.
This project builds a machine learning system to identify customers who are likely
to cancel their service (churn), enabling the business to take proactive retention actions.

## Objectives
1. Understand the customer attributes associated with churn behaviour.
2. Build and compare multiple machine learning classification models.
3. Identify high-risk customers ranked by predicted churn probability.
4. Provide an interactive prediction interface via a Streamlit web application.

## Dataset Description
- **Source:** IBM Telco Customer Churn dataset (publicly available)
- **File:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Rows:** 7,043 customer records
- **Columns:** 21 (demographics, services, contract, billing, and churn target)
- **Target variable:** `Churn` (Yes = churned, No = retained)
- **Class distribution:** ~73.5% No Churn (5,174), ~26.5% Churn (1,869)

### Key Columns
| Category | Columns |
|----------|---------|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account | tenure, Contract, PaperlessBilling, PaymentMethod |
| Billing | MonthlyCharges, TotalCharges |
| Phone | PhoneService, MultipleLines |
| Internet | InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |

## Technologies Used
| Technology | Purpose |
|------------|---------|
| Python 3.13 | Core programming language |
| pandas, numpy | Data manipulation |
| matplotlib, seaborn | Visualisation |
| scikit-learn | Preprocessing, modelling, evaluation |
| XGBoost 3.4.1 | Gradient boosting model |
| joblib | Model persistence |
| Streamlit | Web application |
| Jupyter Notebook | Interactive analysis |

## Project Structure
```
Customer-Churn-Prediction/
|
|-- data/
|   `-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|
|-- notebooks/
|   `-- customer_churn_analysis.ipynb
|
|-- src/
|   |-- preprocessing.py
|   |-- train.py
|   `-- predict.py
|
|-- models/
|   `-- churn_model.joblib
|
|-- outputs/
|   |-- figures/
|   `-- high_risk_customers.csv
|
|-- app.py
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Installation Instructions

### 1. Clone or unzip the project
```bash
cd Customer-Churn-Prediction
```

### 2. (Recommended) Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## How to Run the Notebook

### Option A — Jupyter Notebook
```bash
jupyter notebook notebooks/customer_churn_analysis.ipynb
```

### Option B — JupyterLab
```bash
jupyter lab
```
Then open `notebooks/customer_churn_analysis.ipynb`.

### Option C — Run the training script directly
```bash
python src/train.py
```
This generates all figures, the model file, and the high-risk CSV without needing Jupyter.

## How to Run the Streamlit Application

**Important:** Run `python src/train.py` first to generate the model file before launching the app.

```bash
streamlit run app.py
```
The app will open at `http://localhost:8501` in your browser.

## Model Methodology

### Class Imbalance Handling
The dataset has approximately 26.5% churn rate — a moderate class imbalance.
Rather than using SMOTE (synthetic oversampling), which introduces artificial data,
we use `class_weight='balanced'` for scikit-learn models and `scale_pos_weight` for XGBoost.
This adjusts the loss function to penalise misclassification of the minority class more heavily,
without adding synthetic data. This approach is more interpretable for a business use case.

### Preprocessing Pipeline
All preprocessing is performed inside a scikit-learn Pipeline to prevent data leakage:
- Numerical columns: median imputation → StandardScaler
- Categorical columns: mode imputation → OneHotEncoder (drop first category)
- The pipeline is fit ONLY on training data and applied to test data

### Feature Engineering
Six derived features are created from the raw columns:
1. `tenure_group` — categorical tenure bucketing (0-12, 13-24, 25-48, 49-60, 61-72 months)
2. `total_services` — count of active optional services (0–6)
3. `avg_monthly_per_tenure` — MonthlyCharges / (tenure + 1)
4. `has_tech_support` — binary flag (1 = Yes)
5. `has_online_security` — binary flag (1 = Yes)
6. `is_month_to_month` — binary flag for contract type

### Models Trained
| Model | Imbalance Strategy |
|-------|--------------------|
| Logistic Regression | class_weight=balanced |
| Random Forest | class_weight=balanced |
| XGBoost | scale_pos_weight=2.77 |

## Evaluation Metrics (Actual Results from Training)

### Model Comparison Table
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7367 | 0.5026 | 0.7727 | 0.6091 | **0.8458** |
| Random Forest | 0.7679 | 0.5481 | 0.7166 | 0.6211 | 0.8349 |
| XGBoost | 0.7630 | 0.5392 | 0.7353 | 0.6222 | 0.8308 |

> All values computed from a stratified 20% hold-out test set (random_state=42).

### Selected Model: Logistic Regression
**Selected by highest ROC-AUC (0.8458).**

Model selection reasoning:
- Churn is the positive class; recall matters most for capturing actual churners.
- Logistic Regression achieves the highest ROC-AUC (0.8458), which reflects
  its overall discriminative ability across all thresholds.
- It also achieves the highest recall (0.7727), correctly identifying 77.3% of
  actual churners, at the cost of somewhat lower precision.
- The lower accuracy (0.7367) is expected when `class_weight=balanced` is used —
  the model is intentionally biased toward catching churners.
- Logistic Regression provides interpretable coefficients, which is valuable
  for business stakeholders.

### High-Risk Customer Summary (Full Dataset)
| Risk Level | Count | Percentage |
|------------|-------|-----------|
| High Risk (>70%) | 1,690 | 24.0% |
| Medium Risk (30–70%) | 2,321 | 33.0% |
| Low Risk (<30%) | 3,032 | 43.0% |

> Risk thresholds are project-defined, not statistically validated cutoffs.

## Limitations
1. **Dataset size:** 7,043 records is moderate. Larger datasets may improve model generalisation.
2. **Risk thresholds:** The Low/Medium/High risk boundaries (30%/70%) are project-defined
   and should be calibrated against actual business outcomes before operational use.
3. **No causal inference:** The model identifies correlates of churn, not causal factors.
   Review reasons are associated characteristics only.
4. **Static model:** The model is trained on a historical snapshot. Retraining is needed
   as customer behaviour patterns evolve.
5. **Missing features:** External factors (competitor pricing, service outages, etc.)
   that may influence churn are not in this dataset.
6. **Calibration:** Logistic Regression probabilities are generally well-calibrated,
   but a calibration plot is included in the notebook to verify.

## Future Improvements
1. Threshold optimisation: find the decision threshold that maximises F1 or satisfies
   a business-defined precision/recall trade-off.
2. Hyperparameter tuning: use GridSearchCV or RandomizedSearchCV.
3. SHAP values for more granular, per-prediction interpretability.
4. Automated retraining pipeline with data drift detection.
5. Feature store integration to enrich with real-time usage data.
6. A/B testing framework to measure the impact of retention interventions.

---
*This project was created as a Machine Learning internship capstone submission.*
*All metrics are computed from actual model execution on the IBM Telco dataset.*
