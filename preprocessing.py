"""
preprocessing.py
----------------
Data loading, cleaning, feature engineering, and sklearn pipeline
construction for the Customer Churn Prediction project.

All preprocessing steps are encapsulated here to keep the training
script and notebook clean.
"""

import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

COLUMNS_TO_DROP = ["customerID"]

TENURE_GROUP_COL = "tenure_group"
TOTAL_SERVICES_COL = "total_services"
AVG_MONTHLY_PER_TENURE_COL = "avg_monthly_per_tenure"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[load_data] Loaded {df.shape[0]} rows x {df.shape[1]} columns.")
    return df


# ---------------------------------------------------------------------------
# Data inspection helpers
# ---------------------------------------------------------------------------

def inspect_data(df):
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "missing_counts": df.isnull().sum(),
        "duplicate_count": df.duplicated().sum(),
    }
    return summary


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def clean_data(df):
    df = df.copy()

    # Fix TotalCharges dtype (spaces become NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing_tc = df["TotalCharges"].isnull().sum()
    if n_missing_tc > 0:
        median_tc = df["TotalCharges"].median()
        df["TotalCharges"].fillna(median_tc, inplace=True)
        print(f"[clean_data] Imputed {n_missing_tc} missing TotalCharges with median ({median_tc:.2f}).")

    # Convert SeniorCitizen from 0/1 to No/Yes string
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Drop duplicate rows
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        df.drop_duplicates(inplace=True)
        print(f"[clean_data] Dropped {n_dupes} duplicate rows.")
    else:
        print("[clean_data] No duplicate rows found.")

    # Strip whitespace from object columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    print(f"[clean_data] Cleaned data: {df.shape[0]} rows x {df.shape[1]} columns.")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df):
    df = df.copy()

    # tenure_group
    bins = [0, 12, 24, 48, 60, 72]
    labels = ["0-12 months", "13-24 months", "25-48 months", "49-60 months", "61-72 months"]
    df[TENURE_GROUP_COL] = pd.cut(
        df["tenure"], bins=bins, labels=labels, right=True, include_lowest=True
    ).astype(str)

    # total_services
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
    df[TOTAL_SERVICES_COL] = (df[service_cols] == "Yes").sum(axis=1)

    # avg_monthly_per_tenure
    df[AVG_MONTHLY_PER_TENURE_COL] = (df["MonthlyCharges"] / (df["tenure"] + 1)).round(4)

    # binary flags
    df["has_tech_support"] = (df["TechSupport"] == "Yes").astype(int)
    df["has_online_security"] = (df["OnlineSecurity"] == "Yes").astype(int)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    print(f"[engineer_features] Added 6 new features. DataFrame now has {df.shape[1]} columns.")
    return df


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------

def encode_target(df):
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"] + COLUMNS_TO_DROP)
    print(f"[encode_target] X: {X.shape}, y distribution:")
    print(y.value_counts())
    return X, y


# ---------------------------------------------------------------------------
# Column identification helpers
# ---------------------------------------------------------------------------

def get_column_groups(X):
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"[get_column_groups] Numerical ({len(numerical_cols)}): {numerical_cols}")
    print(f"[get_column_groups] Categorical ({len(categorical_cols)}): {categorical_cols}")
    return numerical_cols, categorical_cols


# ---------------------------------------------------------------------------
# Preprocessing pipeline builder
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(numerical_cols, categorical_cols):
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ], remainder="drop")

    return preprocessor
