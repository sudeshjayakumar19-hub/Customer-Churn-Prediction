"""
setup_model.py
--------------
Utility script that trains the model and saves it.
Used during deployment (Streamlit Cloud) to generate churn_model.joblib
on the deployment server, avoiding sklearn version mismatch issues.

Run with:
    python setup_model.py
"""

import os
import sys

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from preprocessing import (
    load_data, clean_data, engineer_features,
    encode_target, get_column_groups,
    build_preprocessing_pipeline, RANDOM_STATE
)

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def main():
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load and preprocess
    df = engineer_features(clean_data(load_data(DATA_PATH)))
    X, y = encode_target(df)
    numerical_cols, categorical_cols = get_column_groups(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Train models
    models = {}
    models["Logistic Regression"] = Pipeline([
        ("preprocessor", build_preprocessing_pipeline(numerical_cols, categorical_cols)),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1000,
            random_state=RANDOM_STATE, solver="lbfgs",
        )),
    ])
    models["Random Forest"] = Pipeline([
        ("preprocessor", build_preprocessing_pipeline(numerical_cols, categorical_cols)),
        ("classifier", RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            max_depth=None, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])

    try:
        import xgboost as xgb
        spw = int((y_train == 0).sum()) / int((y_train == 1).sum())
        models["XGBoost"] = Pipeline([
            ("preprocessor", build_preprocessing_pipeline(numerical_cols, categorical_cols)),
            ("classifier", xgb.XGBClassifier(
                n_estimators=300, scale_pos_weight=spw,
                max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss",
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ])
    except ImportError:
        pass

    # Evaluate and select best
    best_name, best_pipeline, best_auc = None, None, -1
    all_metrics = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_pred = pipe.predict(X_test)
        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "roc_auc":   roc_auc_score(y_test, y_prob),
        }
        all_metrics[name] = metrics
        auc = metrics["roc_auc"]
        print(f"{name}: ROC-AUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipeline = pipe

    save_path = os.path.join(MODELS_DIR, "churn_model.joblib")
    joblib.dump({
        "pipeline": best_pipeline,
        "model_name": best_name,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "metrics": all_metrics[best_name],
        "comparison_table": [
            {"Model": n, **{k: f"{v:.4f}" for k, v in m.items()}}
            for n, m in all_metrics.items()
        ],
    }, save_path)
    print(f"\nSaved best model ({best_name}) to {save_path}")


if __name__ == "__main__":
    main()
