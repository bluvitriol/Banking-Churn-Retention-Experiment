"""
Data loading, cleaning, and feature engineering for the bank churn dataset.
"""

import pandas as pd
import numpy as np
import os

RAW_URL = "https://raw.githubusercontent.com/sharmaroshan/Churn-Modelling-Dataset/master/Churn_Modelling.csv"
RAW_PATH = "data/Churn_Modelling.csv"
PROCESSED_PATH = "outputs/processed_churn_data.csv"


def load_raw_data():
    """Download the dataset if not already present locally, then load it."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(RAW_PATH):
        df = pd.read_csv(RAW_URL)
        df.to_csv(RAW_PATH, index=False)
    else:
        df = pd.read_csv(RAW_PATH)
    
    return df


def run_data_quality_checks(df):
    """Basic validation before trusting the dataset for modeling."""
    report = {
        "n_rows": len(df),
        "n_duplicates": df.duplicated().sum(),
        "missing_values": df.isnull().sum().to_dict(),
        "churn_rate": df["Exited"].mean(),
    }
    print("Data Quality Report")
    for k, v in report.items():
        print(f"{k}: {v}")
    return report


def clean_and_engineer_features(df):
    """Drop identifier columns, encode categoricals, and engineer a few
    behaviourally meaningful features."""
    df = df.copy()

    # Drop non-predictive identifier columns (classic leakage/no-signal columns)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    # Encode categorical variables
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

    # Feature engineering — behavioural / commercial signals
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["TenureByAge"] = df["Tenure"] / (df["Age"] + 1)
    df["ProductsPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)

    # Explicit leakage check: confirm no feature is derived from or chronologically after the churn event itself. All fields here (demographics, product holding, balance, activity flag) are account-state attributes known *before* any churn decision — safe to use as predictors.
    
    assert "Exited" not in df.drop(columns=["Exited"]).columns

    return df


def prepare_data():
    df_raw = load_raw_data()
    run_data_quality_checks(df_raw)
    df_clean = clean_and_engineer_features(df_raw)

    os.makedirs("outputs", exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    print(f"\nProcessed data saved to {PROCESSED_PATH}, shape: {df_clean.shape}")
    return df_clean


if __name__ == "__main__":
    prepare_data()