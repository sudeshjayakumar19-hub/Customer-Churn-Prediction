"""
predict.py
----------
Prediction utilities for the Customer Churn Prediction project.

Used by both the Streamlit app (app.py) and any downstream batch
prediction script.

Provides:
- load_model_bundle()     : Load the saved joblib bundle
- prepare_single_customer(): Build a one-row DataFrame from user inputs
- predict_churn()         : Return probability, risk level, and reasons
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# ── Ensure src/ is on the path so preprocessing constants are accessible ──
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from preprocessing import (
    engineer_features,
    TENURE_GROUP_COL,
    TOTAL_SERVICES_COL,
    AVG_MONTHLY_PER_TENURE_COL,
)

# ---------------------------------------------------------------------------
# Constants — these mirror the thresholds documented in train.py
# ---------------------------------------------------------------------------

# NOTE: These are project-defined business thresholds, not statistically
#       validated cutoffs. They exist to segment output into actionable groups.
RISK_LOW_THRESHOLD    = 0.30   # probability < 0.30 → Low Risk
RISK_HIGH_THRESHOLD   = 0.70   # probability > 0.70 → High Risk


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_bundle(model_path: str = None) -> dict:
    """
    Load the saved joblib model bundle.

    Parameters
    ----------
    model_path : str, optional
        Path to the .joblib file. Defaults to models/churn_model.joblib
        relative to the project root.

    Returns
    -------
    dict
        Dictionary containing 'pipeline', 'model_name', and 'metrics'.
    """
    if model_path is None:
        model_path = os.path.join(_PROJECT_ROOT, "models", "churn_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            "Please run 'python src/train.py' first."
        )

    bundle = joblib.load(model_path)
    print(f"[predict] Loaded model: {bundle['model_name']}")
    return bundle


# ---------------------------------------------------------------------------
# Feature preparation for a single customer
# ---------------------------------------------------------------------------

def prepare_single_customer(customer_dict: dict) -> pd.DataFrame:
    """
    Convert a dictionary of raw customer attributes into a preprocessed
    DataFrame ready for model.predict_proba().

    The function applies the same feature engineering transformations
    that were applied during training.

    Parameters
    ----------
    customer_dict : dict
        Keys should match raw dataset column names:
        customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
        PhoneService, MultipleLines, InternetService, OnlineSecurity,
        OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
        StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
        MonthlyCharges, TotalCharges

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all engineered features, ready for
        the sklearn pipeline (which handles encoding and scaling).
    """
    # Build a one-row DataFrame
    row = pd.DataFrame([customer_dict])

    # Apply feature engineering (same as training)
    row = engineer_features(row)

    # Drop identifier and target columns (target doesn't exist in inference)
    cols_to_drop = ["customerID", "Churn"]
    row = row.drop(columns=[c for c in cols_to_drop if c in row.columns])

    return row


# ---------------------------------------------------------------------------
# Risk level assignment
# ---------------------------------------------------------------------------

def get_risk_level(probability: float) -> str:
    """
    Convert a churn probability to a labelled risk tier.

    Thresholds (project-defined, not validated cutoffs):
    - Low Risk    : probability < 0.30
    - Medium Risk : 0.30 <= probability <= 0.70
    - High Risk   : probability > 0.70
    """
    if probability < RISK_LOW_THRESHOLD:
        return "Low Risk"
    elif probability <= RISK_HIGH_THRESHOLD:
        return "Medium Risk"
    else:
        return "High Risk"


# ---------------------------------------------------------------------------
# Review reason generation
# ---------------------------------------------------------------------------

def get_review_reasons(customer_dict: dict) -> list:
    """
    Generate a list of potential review reasons based on the customer's
    attributes. These are associated characteristics, NOT causal factors.

    Parameters
    ----------
    customer_dict : dict
        Raw customer attributes.

    Returns
    -------
    list of str
        Human-readable potential review reasons.
    """
    reasons = []

    if customer_dict.get("Contract") == "Month-to-month":
        reasons.append("Month-to-month contract (associated with higher churn rates)")

    tenure = customer_dict.get("tenure", 999)
    if isinstance(tenure, (int, float)) and tenure <= 12:
        reasons.append(f"Short tenure ({tenure} months — new customers may be more at risk)")

    monthly = customer_dict.get("MonthlyCharges", 0)
    if isinstance(monthly, (int, float)) and monthly > 70:
        reasons.append(f"High monthly charges (${monthly:.2f}/month)")

    if customer_dict.get("TechSupport") in ("No", "No internet service"):
        reasons.append("No tech support subscription")

    if customer_dict.get("InternetService") == "Fiber optic":
        reasons.append("Fiber optic internet service (associated characteristic)")

    if customer_dict.get("PaymentMethod") == "Electronic check":
        reasons.append("Electronic check payment method")

    if customer_dict.get("SeniorCitizen") in ("Yes", 1):
        reasons.append("Senior citizen status")

    if customer_dict.get("Partner") == "No" and customer_dict.get("Dependents") == "No":
        reasons.append("No partner or dependents (single-household customer)")

    if not reasons:
        reasons.append("No single dominant review indicator identified")

    return reasons


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict_churn(customer_dict: dict, bundle: dict) -> dict:
    """
    Predict churn probability for a single customer.

    Parameters
    ----------
    customer_dict : dict
        Raw customer attributes (before feature engineering).
    bundle : dict
        Loaded model bundle from load_model_bundle().

    Returns
    -------
    dict
        {
          "churn_probability": float,
          "risk_level": str,
          "review_reasons": list[str],
          "model_name": str,
        }
    """
    pipeline = bundle["pipeline"]
    X_single = prepare_single_customer(customer_dict)

    prob = float(pipeline.predict_proba(X_single)[0, 1])
    risk_level = get_risk_level(prob)
    reasons = get_review_reasons(customer_dict)

    return {
        "churn_probability": round(prob, 4),
        "risk_level": risk_level,
        "review_reasons": reasons,
        "model_name": bundle["model_name"],
    }
