"""
Train and evaluate an interpretable churn propensity model (Logistic
Regression) alongside a Random Forest benchmark, with disciplined
evaluation: train/test split, calibration, and feature importance.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
from sklearn.calibration import calibration_curve

PROCESSED_PATH = "outputs/processed_churn_data.csv"
SCORES_PATH = "outputs/churn_scores.csv"


def load_processed_data():
    return pd.read_csv(PROCESSED_PATH)


def split_data(df, target="Exited", test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def train_logistic_regression(X_train, y_train):
    """Interpretable model"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_random_forest(X_train, y_train):
    """Benchmark model to compare interpretability vs. lift trade-off."""
    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, scaler=None, name="Model"):
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    y_prob = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    print(f"\n{name} Evaluation")
    print(f"ROC AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return y_prob, auc


def plot_calibration(y_test, y_prob_dict, save_path="outputs/calibration_curve.png"):
    """Calibration matters more than raw accuracy when model output drives a real retention-targeting decision. A miscalibrated score misleads business sizing of the intervention."""
    plt.figure(figsize=(6, 6))
    for name, y_prob in y_prob_dict.items():
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve: Predicted vs. Actual Churn Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Calibration plot saved to {save_path}")


def get_logistic_odds_ratios(model, feature_names):
    """Translate coefficients into odds ratios for business-readable
    interpretation — e.g. 'inactive members are 2.3x more likely to churn'."""
    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)
    result = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "odds_ratio": odds_ratios
    }).sort_values("odds_ratio", ascending=False)
    return result


def get_rf_feature_importance(model, feature_names):
    return pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)


def run_churn_modeling():
    df = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Logistic Regression (primary, interpretable model)
    log_model, scaler = train_logistic_regression(X_train, y_train)
    y_prob_log, auc_log = evaluate_model(log_model, X_test, y_test, scaler, "Logistic Regression")

    odds_ratios = get_logistic_odds_ratios(log_model, X_train.columns)
    print("\nTop churn drivers (odds ratios):")
    print(odds_ratios.head(10))

    # Random Forest (benchmark)
    rf_model = train_random_forest(X_train, y_train)
    y_prob_rf, auc_rf = evaluate_model(rf_model, X_test, y_test, name="Random Forest")

    feat_importance = get_rf_feature_importance(rf_model, X_train.columns)
    print("\nTop churn drivers (RF feature importance):")
    print(feat_importance.head(10))

    # Calibration comparison
    plot_calibration(y_test, {"Logistic Regression": y_prob_log, "Random Forest": y_prob_rf})

    print(f"\nInterpretability vs. lift trade-off: Logistic AUC={auc_log:.3f}, "
          f"RF AUC={auc_rf:.3f}. Logistic regression is retained as the primary "
          f"model for deployment due to its transparent, auditable coefficients, "
          f"consistent with the small AUC gap not justifying the loss of interpretability.")

    # Score the FULL dataset with the chosen (logistic) model for downstream targeting
    X_full = df.drop(columns=["Exited"])
    X_full_scaled = scaler.transform(X_full)
    df["churn_probability"] = log_model.predict_proba(X_full_scaled)[:, 1]

    os.makedirs("outputs", exist_ok=True)
    df.to_csv(SCORES_PATH, index=False)
    print(f"\nFull dataset scored and saved to {SCORES_PATH}")

    return df, odds_ratios, feat_importance


if __name__ == "__main__":
    run_churn_modeling()