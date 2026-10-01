"""
train.py
--------
End-to-end training script for the Customer Churn Prediction project.

This script:
1. Loads and cleans the dataset
2. Engineers features
3. Splits data into train/test sets
4. Builds preprocessing + model pipelines
5. Trains Logistic Regression, Random Forest, and XGBoost (if available)
6. Evaluates all models and prints a comparison table
7. Saves all EDA and evaluation figures
8. Saves the best pipeline (by ROC-AUC) to models/churn_model.joblib
9. Generates the high-risk customer output CSV

Run from the project root directory:
    python src/train.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay, classification_report
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

warnings.filterwarnings("ignore")

# ── Resolve paths relative to project root ──────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "outputs", "figures")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Add src/ to path so we can import preprocessing ─────────────────────────
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from preprocessing import (
    load_data, clean_data, engineer_features,
    encode_target, get_column_groups,
    build_preprocessing_pipeline, RANDOM_STATE
)

CHURN_PALETTE = {"No": "#2196F3", "Yes": "#F44336"}
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

# ═══════════════════════════════════════════════════════════════════════════
# SECTION A  — Data loading, cleaning, feature engineering
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  CUSTOMER CHURN PREDICTION — TRAINING PIPELINE")
print("="*65)

df_raw = load_data(DATA_PATH)
df_clean = clean_data(df_raw)
df = engineer_features(df_clean)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION B  — EDA figures
# ═══════════════════════════════════════════════════════════════════════════

print("\n[EDA] Generating exploratory visualisations...")

# ── B1: Churn distribution ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
churn_counts = df["Churn"].value_counts()
ax.bar(churn_counts.index, churn_counts.values,
       color=[CHURN_PALETTE["No"], CHURN_PALETTE["Yes"]], edgecolor="white")
for i, (label, val) in enumerate(churn_counts.items()):
    pct = val / len(df) * 100
    ax.text(i, val + 40, f"{val}\n({pct:.1f}%)", ha="center", fontsize=10)
ax.set_title("Churn Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Churn"); ax.set_ylabel("Number of Customers")
ax.set_ylim(0, churn_counts.max() * 1.2)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "01_churn_distribution.png"))
plt.close()

# ── B2: Churn by Contract Type ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ct = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
ct.plot(kind="bar", ax=ax, color=[CHURN_PALETTE["No"], CHURN_PALETTE["Yes"]],
        edgecolor="white", rot=15)
ax.set_title("Churn by Contract Type", fontsize=13, fontweight="bold")
ax.set_xlabel("Contract"); ax.set_ylabel("Number of Customers")
ax.legend(title="Churn")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "02_churn_by_contract.png"))
plt.close()

# ── B3: Churn by Tenure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30,
             palette=CHURN_PALETTE, multiple="stack", ax=ax)
ax.set_title("Churn by Tenure", fontsize=13, fontweight="bold")
ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "03_churn_by_tenure.png"))
plt.close()

# ── B4: Churn by Monthly Charges ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(data=df, x="MonthlyCharges", hue="Churn", bins=30,
             palette=CHURN_PALETTE, multiple="stack", ax=ax)
ax.set_title("Churn by Monthly Charges", fontsize=13, fontweight="bold")
ax.set_xlabel("Monthly Charges ($)"); ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "04_churn_by_monthly_charges.png"))
plt.close()

# ── B5: Churn by Internet Service ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
is_ct = df.groupby(["InternetService", "Churn"]).size().unstack(fill_value=0)
is_ct.plot(kind="bar", ax=ax, color=[CHURN_PALETTE["No"], CHURN_PALETTE["Yes"]],
           edgecolor="white", rot=0)
ax.set_title("Churn by Internet Service", fontsize=13, fontweight="bold")
ax.set_xlabel("Internet Service"); ax.set_ylabel("Number of Customers")
ax.legend(title="Churn")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "05_churn_by_internet_service.png"))
plt.close()

# ── B6: Churn by Payment Method ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
pm_ct = df.groupby(["PaymentMethod", "Churn"]).size().unstack(fill_value=0)
pm_ct.plot(kind="bar", ax=ax, color=[CHURN_PALETTE["No"], CHURN_PALETTE["Yes"]],
           edgecolor="white", rot=20)
ax.set_title("Churn by Payment Method", fontsize=13, fontweight="bold")
ax.set_xlabel("Payment Method"); ax.set_ylabel("Number of Customers")
ax.legend(title="Churn")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "06_churn_by_payment_method.png"))
plt.close()

# ── B7: Churn by Tech Support ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ts_ct = df.groupby(["TechSupport", "Churn"]).size().unstack(fill_value=0)
ts_ct.plot(kind="bar", ax=ax, color=[CHURN_PALETTE["No"], CHURN_PALETTE["Yes"]],
           edgecolor="white", rot=0)
ax.set_title("Churn by Tech Support", fontsize=13, fontweight="bold")
ax.set_xlabel("Tech Support"); ax.set_ylabel("Number of Customers")
ax.legend(title="Churn")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "07_churn_by_tech_support.png"))
plt.close()

# ── B8: Churn by Senior Citizen ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
sc_ct = df.groupby(["SeniorCitizen", "Churn"]).size().unstack(fill_value=0)
sc_ct.plot(kind="bar", ax=ax, color=[CHURN_PALETTE["No"], CHURN_PALETTE["Yes"]],
           edgecolor="white", rot=0)
ax.set_title("Churn by Senior Citizen Status", fontsize=13, fontweight="bold")
ax.set_xlabel("Senior Citizen"); ax.set_ylabel("Number of Customers")
ax.legend(title="Churn")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "08_churn_by_senior_citizen.png"))
plt.close()

# ── B9: Correlation heatmap (numerical columns) ────────────────────────────
num_cols_for_corr = ["tenure", "MonthlyCharges", "TotalCharges",
                     "total_services", "avg_monthly_per_tenure"]
df_corr = df[num_cols_for_corr].copy()
df_corr["Churn"] = (df["Churn"] == "Yes").astype(int)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(df_corr.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Correlation Matrix (Numerical Features + Churn)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "09_correlation_heatmap.png"))
plt.close()

# ── B10: Churn by Tenure Group ─────────────────────────────────────────────
tenure_order = ["0-12 months", "13-24 months", "25-48 months",
                "49-60 months", "61-72 months"]
tg_ct = (df.groupby(["tenure_group", "Churn"])
           .size().unstack(fill_value=0)
           .reindex(tenure_order))
fig, ax = plt.subplots(figsize=(9, 4))
tg_ct.plot(kind="bar", ax=ax, color=[CHURN_PALETTE["No"], CHURN_PALETTE["Yes"]],
           edgecolor="white", rot=15)
ax.set_title("Churn by Tenure Group", fontsize=13, fontweight="bold")
ax.set_xlabel("Tenure Group"); ax.set_ylabel("Number of Customers")
ax.legend(title="Churn")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "10_churn_by_tenure_group.png"))
plt.close()

print(f"[EDA] Saved 10 figures to {FIGURES_DIR}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION C  — Prepare ML data
# ═══════════════════════════════════════════════════════════════════════════

print("\n[ML Prep] Preparing features and target...")
X, y = encode_target(df)
numerical_cols, categorical_cols = get_column_groups(X)

# Stratified train/test split — preserves class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"[ML Prep] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"[ML Prep] Train churn rate: {y_train.mean()*100:.1f}%  "
      f"| Test churn rate: {y_test.mean()*100:.1f}%")

# Build the preprocessor (fit only on training data!)
preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION D  — Define models
# ═══════════════════════════════════════════════════════════════════════════

print("\n[Models] Building model pipelines...")

models = {}

# ── Logistic Regression ────────────────────────────────────────────────────
models["Logistic Regression"] = Pipeline([
    ("preprocessor", build_preprocessing_pipeline(numerical_cols, categorical_cols)),
    ("classifier", LogisticRegression(
        class_weight="balanced",    # handles class imbalance via weighting
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )),
])

# ── Random Forest ─────────────────────────────────────────────────────────
models["Random Forest"] = Pipeline([
    ("preprocessor", build_preprocessing_pipeline(numerical_cols, categorical_cols)),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])

# ── XGBoost (optional — install gracefully handles failure) ────────────────
xgb_available = False
try:
    import xgboost as xgb
    # Calculate scale_pos_weight to handle imbalance
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    spw = neg_count / pos_count
    models["XGBoost"] = Pipeline([
        ("preprocessor", build_preprocessing_pipeline(numerical_cols, categorical_cols)),
        ("classifier", xgb.XGBClassifier(
            n_estimators=300,
            scale_pos_weight=spw,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    xgb_available = True
    print(f"[Models] XGBoost loaded successfully. scale_pos_weight={spw:.2f}")
except ImportError:
    print("[Models] WARNING: XGBoost not available. "
          "Using Logistic Regression and Random Forest only.")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION E  — Train and evaluate
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(pipeline, X_tr, y_tr, X_te, y_te, model_name):
    """Fit a pipeline and return evaluation metrics dict."""
    print(f"\n[Training] {model_name}...")
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    y_prob = pipeline.predict_proba(X_te)[:, 1]

    metrics = {
        "model_name": model_name,
        "pipeline": pipeline,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy":  accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall":    recall_score(y_te, y_pred, zero_division=0),
        "f1":        f1_score(y_te, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_te, y_prob),
    }
    return metrics

results = []
for name, pipeline in models.items():
    res = evaluate_model(pipeline, X_train, y_train, X_test, y_test, name)
    results.append(res)
    print(f"  Accuracy : {res['accuracy']:.4f}")
    print(f"  Precision: {res['precision']:.4f}")
    print(f"  Recall   : {res['recall']:.4f}")
    print(f"  F1-score : {res['f1']:.4f}")
    print(f"  ROC-AUC  : {res['roc_auc']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION F  — Confusion matrices
# ═══════════════════════════════════════════════════════════════════════════

print("\n[Eval] Generating confusion matrices...")
fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
if len(results) == 1:
    axes = [axes]
for ax, res in zip(axes, results):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{res['model_name']}\nConfusion Matrix", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "11_confusion_matrices.png"))
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION G  — ROC curves
# ═══════════════════════════════════════════════════════════════════════════

print("[Eval] Generating ROC curves...")
fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#2196F3", "#4CAF50", "#FF5722"]
for res, color in zip(results, colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{res['model_name']} (AUC={res['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "12_roc_curves.png"))
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION H  — Precision-Recall curves
# ═══════════════════════════════════════════════════════════════════════════

print("[Eval] Generating Precision-Recall curves...")
fig, ax = plt.subplots(figsize=(7, 5))
for res, color in zip(results, colors):
    prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
    ax.plot(rec, prec, color=color, lw=2, label=res["model_name"])
baseline_pr = y_test.mean()
ax.axhline(y=baseline_pr, color="k", linestyle="--", lw=1,
           label=f"Baseline ({baseline_pr:.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — All Models", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "13_precision_recall_curves.png"))
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION I  — Model comparison table
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  MODEL COMPARISON TABLE")
print("="*65)
comparison_data = []
for res in results:
    comparison_data.append({
        "Model":     res["model_name"],
        "Accuracy":  f"{res['accuracy']:.4f}",
        "Precision": f"{res['precision']:.4f}",
        "Recall":    f"{res['recall']:.4f}",
        "F1-score":  f"{res['f1']:.4f}",
        "ROC-AUC":   f"{res['roc_auc']:.4f}",
    })
df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION J  — Select best model (by ROC-AUC)
# ═══════════════════════════════════════════════════════════════════════════

best_result = max(results, key=lambda r: r["roc_auc"])
best_model_name = best_result["model_name"]
best_pipeline = best_result["pipeline"]
print(f"\n[Selection] Best model by ROC-AUC: {best_model_name} "
      f"(AUC={best_result['roc_auc']:.4f})")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION K  — Calibration plot (best model)
# ═══════════════════════════════════════════════════════════════════════════

print(f"[Calibration] Generating calibration plot for {best_model_name}...")
prob_true, prob_pred = calibration_curve(y_test, best_result["y_prob"], n_bins=10)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(prob_pred, prob_true, marker="o", color="#2196F3",
        label=f"{best_model_name}")
ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives (True Probability)")
ax.set_title(f"Calibration Curve — {best_model_name}", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "14_calibration_curve.png"))
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION L  — Feature importance
# ═══════════════════════════════════════════════════════════════════════════

print("[Importance] Generating feature importance plots...")

def get_feature_names(pipeline):
    """Extract feature names after one-hot encoding from a fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    num_names = preprocessor.transformers_[0][2]   # numerical col names
    ohe = preprocessor.transformers_[1][1].named_steps["encoder"]
    cat_names = ohe.get_feature_names_out(
        preprocessor.transformers_[1][2]
    ).tolist()
    return list(num_names) + cat_names

# --- Logistic Regression feature importance (coefficients) ---
lr_result = next(r for r in results if r["model_name"] == "Logistic Regression")
lr_feature_names = get_feature_names(lr_result["pipeline"])
lr_coefs = lr_result["pipeline"].named_steps["classifier"].coef_[0]
lr_importance_df = pd.DataFrame({
    "Feature": lr_feature_names,
    "Coefficient": lr_coefs
}).sort_values("Coefficient", key=abs, ascending=False).head(20)

fig, ax = plt.subplots(figsize=(9, 6))
colors_lr = ["#F44336" if c > 0 else "#2196F3" for c in lr_importance_df["Coefficient"]]
ax.barh(lr_importance_df["Feature"][::-1], lr_importance_df["Coefficient"][::-1],
        color=colors_lr[::-1])
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_title("Logistic Regression — Top 20 Feature Coefficients\n"
             "(Positive = higher churn likelihood, Negative = lower)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "15_lr_feature_importance.png"))
plt.close()

# --- Random Forest feature importance ---
rf_result = next(r for r in results if r["model_name"] == "Random Forest")
rf_feature_names = get_feature_names(rf_result["pipeline"])
rf_importances = rf_result["pipeline"].named_steps["classifier"].feature_importances_
rf_importance_df = pd.DataFrame({
    "Feature": rf_feature_names,
    "Importance": rf_importances
}).sort_values("Importance", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(rf_importance_df["Feature"][::-1], rf_importance_df["Importance"][::-1],
        color="#4CAF50")
ax.set_title("Random Forest — Top 20 Feature Importances (Gini Impurity)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "16_rf_feature_importance.png"))
plt.close()

# --- XGBoost feature importance (if available) ---
if xgb_available:
    try:
        xgb_result = next(r for r in results if r["model_name"] == "XGBoost")
        xgb_feature_names = get_feature_names(xgb_result["pipeline"])
        xgb_importances = xgb_result["pipeline"].named_steps["classifier"].feature_importances_
        xgb_importance_df = pd.DataFrame({
            "Feature": xgb_feature_names,
            "Importance": xgb_importances
        }).sort_values("Importance", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(xgb_importance_df["Feature"][::-1], xgb_importance_df["Importance"][::-1],
                color="#FF5722")
        ax.set_title("XGBoost — Top 20 Feature Importances", fontsize=11, fontweight="bold")
        ax.set_xlabel("Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "17_xgb_feature_importance.png"))
        plt.close()
    except Exception as e:
        print(f"[Importance] XGBoost importance plot skipped: {e}")

print(f"[Importance] Feature importance figures saved.")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION M  — High-risk customer identification
# ═══════════════════════════════════════════════════════════════════════════

print("\n[High-Risk] Generating high-risk customer table...")

# Use the FULL dataset (not just test set) for business output
# Re-run best pipeline predict_proba on full dataset
# Note: the pipeline was already trained on X_train; we predict on all X

# Reconstruct full X (before train/test split) for customer ranking
df_full = df.copy()
X_full, _ = encode_target(df_full)
full_probs = best_pipeline.predict_proba(X_full)[:, 1]

# Build the high-risk table
high_risk_df = pd.DataFrame({
    "customerID":       df_full["customerID"].values,
    "churn_probability": np.round(full_probs, 4),
    "tenure":           df_full["tenure"].values,
    "Contract":         df_full["Contract"].values,
    "MonthlyCharges":   df_full["MonthlyCharges"].values,
    "InternetService":  df_full["InternetService"].values,
    "TechSupport":      df_full["TechSupport"].values,
    "PaymentMethod":    df_full["PaymentMethod"].values,
    "SeniorCitizen":    df_full["SeniorCitizen"].values,
    "actual_churn":     df_full["Churn"].values,
})

# Risk level labels — project-defined thresholds, not statistically validated
def assign_risk_level(prob):
    if prob < 0.30:
        return "Low Risk"
    elif prob <= 0.70:
        return "Medium Risk"
    else:
        return "High Risk"

high_risk_df["risk_level"] = high_risk_df["churn_probability"].apply(assign_risk_level)

# Generate suggested review reasons based on actual customer attributes
def suggest_review_reason(row):
    reasons = []
    if row["Contract"] == "Month-to-month":
        reasons.append("month-to-month contract")
    if row["tenure"] <= 12:
        reasons.append("short tenure (<=12 months)")
    if row["MonthlyCharges"] > 70:
        reasons.append("high monthly charges (>$70)")
    if row["TechSupport"] in ("No", "No internet service"):
        reasons.append("no tech support")
    if row["InternetService"] == "Fiber optic":
        reasons.append("fiber optic service")
    if row["PaymentMethod"] == "Electronic check":
        reasons.append("electronic check payment")
    if row["SeniorCitizen"] == "Yes":
        reasons.append("senior citizen")
    if not reasons:
        reasons.append("no single dominant indicator")
    return "; ".join(reasons)

high_risk_df["suggested_review_reason"] = high_risk_df.apply(suggest_review_reason, axis=1)

# Sort by churn probability descending
high_risk_df = high_risk_df.sort_values("churn_probability", ascending=False).reset_index(drop=True)

# Save full table
output_path = os.path.join(OUTPUTS_DIR, "high_risk_customers.csv")
high_risk_df.to_csv(output_path, index=False)
print(f"[High-Risk] Saved {len(high_risk_df)} customers to {output_path}")

# Summary
for level in ["High Risk", "Medium Risk", "Low Risk"]:
    count = (high_risk_df["risk_level"] == level).sum()
    pct = count / len(high_risk_df) * 100
    print(f"  {level}: {count} customers ({pct:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION N  — Save model pipeline
# ═══════════════════════════════════════════════════════════════════════════

model_save_path = os.path.join(MODELS_DIR, "churn_model.joblib")
# Save pipeline + metadata together
save_bundle = {
    "pipeline": best_pipeline,
    "model_name": best_model_name,
    "numerical_cols": numerical_cols,
    "categorical_cols": categorical_cols,
    "metrics": {
        "accuracy":  best_result["accuracy"],
        "precision": best_result["precision"],
        "recall":    best_result["recall"],
        "f1":        best_result["f1"],
        "roc_auc":   best_result["roc_auc"],
    },
    "comparison_table": df_comparison.to_dict("records"),
}
joblib.dump(save_bundle, model_save_path)
print(f"\n[Save] Model bundle saved to: {model_save_path}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION O  — Final summary
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  TRAINING COMPLETE — FINAL SUMMARY")
print("="*65)
print(f"\nBest model  : {best_model_name}")
print(f"ROC-AUC     : {best_result['roc_auc']:.4f}")
print(f"Recall      : {best_result['recall']:.4f}")
print(f"F1-score    : {best_result['f1']:.4f}")
print(f"Precision   : {best_result['precision']:.4f}")
print(f"Accuracy    : {best_result['accuracy']:.4f}")
print(f"\nAll figures saved to: {FIGURES_DIR}")
print(f"High-risk CSV saved : {output_path}")
print(f"Model saved to      : {model_save_path}")
print("="*65)
