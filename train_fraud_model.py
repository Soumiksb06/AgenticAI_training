# =====================================================================
# INTELLIGENT INSURANCE CLAIMS RISK & INVESTIGATION ASSISTANT
# =====================================================================
#
# ML Component:
# Predict whether an insurance claim is:
#
#   0 -> Legitimate / Normal
#   1 -> Fraudulent / Suspicious
#
# Models:
#   - XGBoost
#   - LightGBM
#
# Important:
# Historical features are calculated using only information available
# BEFORE the current claim.
#
# This prevents temporal and target leakage.
# =====================================================================


# =====================================================================
# 0. IMPORTS
# =====================================================================

import os
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix
)

import xgboost as xgb
import lightgbm as lgb
import shap


warnings.filterwarnings("ignore")


RANDOM_STATE = 42


# =====================================================================
# 1. HISTORICAL AGGREGATION HELPER
# =====================================================================

def add_historical_features(
    df,
    group_col,
    date_col,
    amount_col="ClaimAmount"
):
    """
    Calculate historical claim statistics using ONLY claims
    occurring strictly before the current claim date.

    Same-day claims are excluded.

    Returns:
        Original dataframe plus:
            historical_claim_count
            historical_claim_amount_sum
            historical_avg_claim_amount
    """

    result = df.copy()

    # ---------------------------------------------------------------
    # Aggregate claims at entity + date level.
    #
    # This is important because claims occurring on the same day
    # should not be considered historical relative to each other.
    # ---------------------------------------------------------------

    daily = (
        result.groupby(
            [group_col, date_col],
            as_index=False
        )
        .agg(
            daily_claim_count=(amount_col, "size"),
            daily_claim_amount_sum=(amount_col, "sum")
        )
    )

    daily = daily.sort_values(
        [group_col, date_col]
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Cumulative values shifted by one date.
    #
    # This means the current date is excluded.
    # ---------------------------------------------------------------

    daily["historical_claim_count"] = (
        daily
        .groupby(group_col)["daily_claim_count"]
        .cumsum()
        .groupby(daily[group_col])
        .shift(1)
    )

    daily["historical_claim_amount_sum"] = (
        daily
        .groupby(group_col)["daily_claim_amount_sum"]
        .cumsum()
        .groupby(daily[group_col])
        .shift(1)
    )

    # ---------------------------------------------------------------
    # First observation for each group has no history.
    # ---------------------------------------------------------------

    daily["historical_claim_count"] = (
        daily["historical_claim_count"]
        .fillna(0)
    )

    daily["historical_claim_amount_sum"] = (
        daily["historical_claim_amount_sum"]
        .fillna(0)
    )

    # ---------------------------------------------------------------
    # Historical average claim amount.
    # ---------------------------------------------------------------

    daily["historical_avg_claim_amount"] = np.where(
        daily["historical_claim_count"] > 0,

        daily["historical_claim_amount_sum"]
        / daily["historical_claim_count"],

        np.nan
    )

    # ---------------------------------------------------------------
    # Merge historical statistics back into original data.
    # ---------------------------------------------------------------

    result = result.merge(
        daily[
            [
                group_col,
                date_col,
                "historical_claim_count",
                "historical_claim_amount_sum",
                "historical_avg_claim_amount"
            ]
        ],
        on=[
            group_col,
            date_col
        ],
        how="left"
    )

    return result


# =====================================================================
# 2. BUILD FEATURE PIPELINE
# =====================================================================

def build_feature_pipeline(file_path: str) -> pd.DataFrame:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Dataset missing at location: {file_path}"
        )

    # =================================================================
    # INGESTION
    # =================================================================

    print("=" * 75)
    print("1. INGESTING RAW CLAIMS DATASET")
    print("=" * 75)

    df = pd.read_excel(file_path)

    print(
        f"Raw dataset shape: {df.shape}"
    )

    # =================================================================
    # CLEAN COLUMN NAMES
    # =================================================================

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # =================================================================
    # REQUIRED COLUMNS
    # =================================================================

    required_columns = [

        "ClaimID",

        "PatientID",

        "ProviderID",

        "ClaimAmount",

        "ClaimDate",

        "DiagnosisCode",

        "ProcedureCode",

        "PatientAge",

        "PatientGender",

        "ProviderSpecialty",

        "ClaimStatus",

        "PatientIncome",

        "PatientMaritalStatus",

        "PatientEmploymentStatus",

        "ProviderLocation",

        "ClaimType",

        "ClaimSubmissionMethod",

        "Cluster",

        "ClaimLegitimacy"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "The following required columns are missing:\n"
            + "\n".join(missing_columns)
        )

    # =================================================================
    # DATE CLEANING
    # =================================================================

    df["ClaimDate"] = pd.to_datetime(
        df["ClaimDate"],
        errors="coerce"
    )

    invalid_dates = (
        df["ClaimDate"]
        .isna()
        .sum()
    )

    if invalid_dates > 0:

        print(
            f"WARNING: Removing {invalid_dates} rows "
            "with invalid ClaimDate."
        )

        df = df.dropna(
            subset=["ClaimDate"]
        ).copy()

    # =================================================================
    # NUMERICAL CLEANING
    # =================================================================

    numerical_columns = [

        "ClaimAmount",

        "PatientAge",

        "PatientIncome"
    ]

    for col in numerical_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # ---------------------------------------------------------------
    # Claim amount
    #
    # Do NOT replace missing claim amount with zero.
    # Zero would create an artificial claim.
    # ---------------------------------------------------------------

    df["ClaimAmount"] = (
        df["ClaimAmount"]
        .fillna(
            df["ClaimAmount"].median()
        )
    )

    df["PatientAge"] = (
        df["PatientAge"]
        .fillna(
            df["PatientAge"].median()
        )
    )

    df["PatientIncome"] = (
        df["PatientIncome"]
        .fillna(
            df["PatientIncome"].median()
        )
    )

    # =================================================================
    # CATEGORICAL CLEANING
    # =================================================================

    categorical_columns = [

        "ClaimType",

        "ProviderID",

        "ProviderLocation",

        "PatientGender",

        "PatientMaritalStatus",

        "PatientEmploymentStatus",

        "ProviderSpecialty",

        "ClaimSubmissionMethod",

        "DiagnosisCode",

        "ProcedureCode",

        "ClaimStatus"
    ]

    for col in categorical_columns:

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    # =================================================================
    # ID CLEANING
    # =================================================================

    df["PatientID"] = (
        df["PatientID"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    df["ProviderID"] = (
        df["ProviderID"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    # =================================================================
    # TARGET CREATION
    # =================================================================

    target_clean = (
        df["ClaimLegitimacy"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    target_mapping = {

        "legitimate": 0,

        "normal": 0,

        "fraudulent": 1,

        "fraud": 1
    }

    df["target"] = (
        target_clean
        .map(target_mapping)
    )

    # ---------------------------------------------------------------
    # Validate target.
    # ---------------------------------------------------------------

    if df["target"].isna().any():

        unknown_targets = (
            df.loc[
                df["target"].isna(),
                "ClaimLegitimacy"
            ]
            .value_counts(
                dropna=False
            )
        )

        print(
            "\nUnrecognized target values:"
        )

        print(
            unknown_targets
        )

        raise ValueError(
            "ClaimLegitimacy contains values that "
            "were not mapped."
        )

    # =================================================================
    # TARGET DISTRIBUTION
    # =================================================================

    print(
        "\nTarget distribution:"
    )

    print(
        df["target"]
        .value_counts()
    )

    print(
        "\nTarget percentages:"
    )

    print(
        (
            df["target"]
            .value_counts(
                normalize=True
            )
            * 100
        ).round(2)
    )

    # =================================================================
    # CHRONOLOGICAL SORT
    # =================================================================

    df = (
        df
        .sort_values(
            [
                "ClaimDate",
                "PatientID",
                "ClaimID"
            ]
        )
        .reset_index(drop=True)
    )

    # Internal row identifier.
    #
    # Used only for alignment and never becomes a model feature.

    df["_row_id"] = np.arange(
        len(df)
    )

    # =================================================================
    # CREATE PROCESSED DATAFRAME
    # =================================================================

    processed = pd.DataFrame(
        index=np.arange(len(df))
    )

    # =================================================================
    # 3. DIRECT CLAIM FEATURES
    # =================================================================

    print("\n" + "=" * 75)
    print("2. CREATING DIRECT CLAIM FEATURES")
    print("=" * 75)

    processed["claim_amount"] = (
        df["ClaimAmount"]
        .values
    )

    processed["claim_type"] = (
        df["ClaimType"]
        .values
    )

    processed["provider_hospital"] = (
        df["ProviderID"]
        .values
    )

    processed["geography"] = (
        df["ProviderLocation"]
        .values
    )

    processed["diagnosis_code"] = (
        df["DiagnosisCode"]
        .values
    )

    processed["procedure_code"] = (
        df["ProcedureCode"]
        .values
    )

    processed["claim_status"] = (
        df["ClaimStatus"]
        .values
    )

    processed["submission_method"] = (
        df["ClaimSubmissionMethod"]
        .values
    )

    # =================================================================
    # 4. CUSTOMER / PATIENT TENURE
    # =================================================================

    print("\n" + "=" * 75)
    print("3. CREATING PATIENT HISTORICAL FEATURES")
    print("=" * 75)

    # ---------------------------------------------------------------
    # IMPORTANT:
    #
    # Dataset does not contain actual policy creation date.
    #
    # Therefore this represents:
    #
    # "Days since patient's first observed claim"
    #
    # It is NOT claimed to be policy tenure.
    # ---------------------------------------------------------------

    first_patient_date = (
        df
        .groupby("PatientID")["ClaimDate"]
        .transform("min")
    )

    processed["customer_tenure"] = (
        df["ClaimDate"]
        - first_patient_date
    ).dt.days.values

    # =================================================================
    # 5. PATIENT HISTORICAL CLAIM FEATURES
    # =================================================================

    patient_hist = add_historical_features(
        df,

        group_col="PatientID",

        date_col="ClaimDate",

        amount_col="ClaimAmount"
    )

    processed["patient_previous_claim_count"] = (
        patient_hist[
            "historical_claim_count"
        ]
        .fillna(0)
        .astype(int)
        .values
    )

    processed["avg_historical_claim_amount"] = (
        patient_hist[
            "historical_avg_claim_amount"
        ]
        .fillna(
            df["ClaimAmount"]
        )
        .round(2)
        .values
    )

    # =================================================================
    # 6. PREVIOUS 12-MONTH CLAIM COUNT
    # =================================================================

    print(
        "Calculating previous 12-month claim counts..."
    )

    claims_last_12m = np.zeros(
        len(df),
        dtype=np.int64
    )

    # ---------------------------------------------------------------
    # Process each patient independently.
    #
    # Only claims:
    #
    # current_date - 365 days <= claim_date < current_date
    #
    # are counted.
    #
    # Same-day claims are excluded.
    # ---------------------------------------------------------------

    for patient_id, indices in (
        df.groupby(
            "PatientID",
            sort=False
        ).groups.items()
    ):

        indices = np.asarray(
            indices
        )

        dates = (
            df.loc[
                indices,
                "ClaimDate"
            ]
            .values
            .astype("datetime64[ns]")
        )

        left_dates = (
            dates
            - np.timedelta64(
                365,
                "D"
            )
        )

        left_positions = (
            np.searchsorted(
                dates,
                left_dates,
                side="left"
            )
        )

        right_positions = (
            np.searchsorted(
                dates,
                dates,
                side="left"
            )
        )

        counts = (
            right_positions
            - left_positions
        )

        claims_last_12m[
            indices
        ] = counts

    processed["num_claims_last_12m"] = (
        claims_last_12m
    )

    # =================================================================
    # 7. PREVIOUSLY REJECTED CLAIMS
    # =================================================================

    print(
        "Calculating historical rejected/denied claims..."
    )

    rejected_flag = (
        df["ClaimStatus"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "rejected",
                "denied"
            ]
        )
        .astype(int)
    )

    rejected_daily = (
        pd.DataFrame({

            "PatientID":
                df["PatientID"].values,

            "ClaimDate":
                df["ClaimDate"].values,

            "rejected_flag":
                rejected_flag.values
        })
        .groupby(
            [
                "PatientID",
                "ClaimDate"
            ],
            as_index=False
        )
        ["rejected_flag"]
        .sum()
    )

    rejected_daily = (
        rejected_daily
        .sort_values(
            [
                "PatientID",
                "ClaimDate"
            ]
        )
    )

    rejected_daily[
        "previous_rejected_claims"
    ] = (
        rejected_daily
        .groupby(
            "PatientID"
        )["rejected_flag"]
        .cumsum()
        .groupby(
            rejected_daily["PatientID"]
        )
        .shift(1)
        .fillna(0)
    )

    # ---------------------------------------------------------------
    # Build lookup and align to original claim rows.
    #
    # This avoids duplicate ClaimDate_x / ClaimDate_y columns.
    # ---------------------------------------------------------------

    rejected_lookup = (
        rejected_daily[
            [
                "PatientID",
                "ClaimDate",
                "previous_rejected_claims"
            ]
        ]
        .rename(
            columns={
                "previous_rejected_claims":
                    "_previous_rejected_claims"
            }
        )
    )

    rejected_values = pd.merge(

        df[
            [
                "PatientID",
                "ClaimDate"
            ]
        ],

        rejected_lookup,

        on=[
            "PatientID",
            "ClaimDate"
        ],

        how="left"
    )["_previous_rejected_claims"]

    processed[
        "previously_rejected_claims"
    ] = (
        rejected_values
        .fillna(0)
        .astype(int)
        .values
    )

    # =================================================================
    # 8. PROVIDER HISTORICAL FEATURES
    # =================================================================

    print("\n" + "=" * 75)
    print("4. CREATING PROVIDER HISTORICAL FEATURES")
    print("=" * 75)

    provider_hist = add_historical_features(

        df,

        group_col="ProviderID",

        date_col="ClaimDate",

        amount_col="ClaimAmount"
    )

    processed[
        "provider_claim_frequency"
    ] = (
        provider_hist[
            "historical_claim_count"
        ]
        .fillna(0)
        .astype(int)
        .values
    )

    processed[
        "provider_historical_avg_claim"
    ] = (
        provider_hist[
            "historical_avg_claim_amount"
        ]
        .fillna(
            df["ClaimAmount"]
        )
        .round(2)
        .values
    )

    # =================================================================
    # 9. HISTORICAL PROVIDER FRAUD RATE
    # =================================================================

    print(
        "Calculating historical provider fraud rates..."
    )

    df["_fraud_flag"] = (
        df["target"]
        .astype(int)
    )

    provider_fraud_daily = (
        df.groupby(
            [
                "ProviderID",
                "ClaimDate"
            ],
            as_index=False
        )
        .agg(

            daily_claims=(
                "_fraud_flag",
                "size"
            ),

            daily_fraud_count=(
                "_fraud_flag",
                "sum"
            )
        )
    )

    provider_fraud_daily = (
        provider_fraud_daily
        .sort_values(
            [
                "ProviderID",
                "ClaimDate"
            ]
        )
    )

    # ---------------------------------------------------------------
    # Historical totals BEFORE current date.
    # ---------------------------------------------------------------

    provider_fraud_daily[
        "historical_claims"
    ] = (
        provider_fraud_daily
        .groupby(
            "ProviderID"
        )["daily_claims"]
        .cumsum()
        .groupby(
            provider_fraud_daily["ProviderID"]
        )
        .shift(1)
        .fillna(0)
    )

    provider_fraud_daily[
        "historical_fraud_count"
    ] = (
        provider_fraud_daily
        .groupby(
            "ProviderID"
        )["daily_fraud_count"]
        .cumsum()
        .groupby(
            provider_fraud_daily["ProviderID"]
        )
        .shift(1)
        .fillna(0)
    )

    provider_fraud_daily[
        "historical_fraud_rate"
    ] = np.where(

        provider_fraud_daily[
            "historical_claims"
        ] > 0,

        provider_fraud_daily[
            "historical_fraud_count"
        ]
        /
        provider_fraud_daily[
            "historical_claims"
        ],

        0.0
    )

    # ---------------------------------------------------------------
    # Map back to original rows.
    # ---------------------------------------------------------------

    provider_lookup = (
        provider_fraud_daily[
            [
                "ProviderID",
                "ClaimDate",
                "historical_fraud_rate"
            ]
        ]
        .rename(
            columns={
                "historical_fraud_rate":
                    "_provider_historical_fraud_rate"
            }
        )
    )

    provider_values = pd.merge(

        df[
            [
                "ProviderID",
                "ClaimDate"
            ]
        ],

        provider_lookup,

        on=[
            "ProviderID",
            "ClaimDate"
        ],

        how="left"
    )[
        "_provider_historical_fraud_rate"
    ]

    processed[
        "provider_historical_fraud_rate"
    ] = (
        provider_values
        .fillna(0)
        .clip(0, 1)
        .values
    )

    # =================================================================
    # 10. PEER CLAIM ANALYSIS
    # =================================================================

    print("\n" + "=" * 75)
    print("5. CREATING HISTORICAL PEER-BASED FEATURES")
    print("=" * 75)

    # ---------------------------------------------------------------
    # Define peer group:
    #
    # ProcedureCode + ClaimType
    #
    # This compares claims against similar procedures and claim types.
    # ---------------------------------------------------------------

    df["_peer_group"] = (

        df["ProcedureCode"]
        .astype(str)

        + "||"

        + df["ClaimType"]
        .astype(str)
    )

    peer_daily = (
        df.groupby(
            [
                "_peer_group",
                "ClaimDate"
            ],
            as_index=False
        )
        .agg(

            peer_daily_count=(
                "ClaimAmount",
                "size"
            ),

            peer_daily_amount_sum=(
                "ClaimAmount",
                "sum"
            )
        )
    )

    peer_daily = (
        peer_daily
        .sort_values(
            [
                "_peer_group",
                "ClaimDate"
            ]
        )
    )

    # ---------------------------------------------------------------
    # Historical peer claim count.
    # ---------------------------------------------------------------

    peer_daily[
        "historical_peer_count"
    ] = (
        peer_daily
        .groupby(
            "_peer_group"
        )["peer_daily_count"]
        .cumsum()
        .groupby(
            peer_daily["_peer_group"]
        )
        .shift(1)
        .fillna(0)
    )

    # ---------------------------------------------------------------
    # Historical peer amount.
    # ---------------------------------------------------------------

    peer_daily[
        "historical_peer_sum"
    ] = (
        peer_daily
        .groupby(
            "_peer_group"
        )["peer_daily_amount_sum"]
        .cumsum()
        .groupby(
            peer_daily["_peer_group"]
        )
        .shift(1)
        .fillna(0)
    )

    # ---------------------------------------------------------------
    # Historical peer average.
    # ---------------------------------------------------------------

    peer_daily[
        "historical_peer_mean"
    ] = np.where(

        peer_daily[
            "historical_peer_count"
        ] > 0,

        peer_daily[
            "historical_peer_sum"
        ]
        /
        peer_daily[
            "historical_peer_count"
        ],

        np.nan
    )

    # ---------------------------------------------------------------
    # Map peer statistics back to claims.
    #
    # IMPORTANT:
    # We do not merge directly into processed because processed
    # will later contain ClaimDate.
    # ---------------------------------------------------------------

    peer_lookup = (
        peer_daily[
            [
                "_peer_group",
                "ClaimDate",
                "historical_peer_mean"
            ]
        ]
        .rename(
            columns={
                "historical_peer_mean":
                    "_historical_peer_mean"
            }
        )
    )

    peer_values = pd.merge(

        df[
            [
                "_peer_group",
                "ClaimDate"
            ]
        ],

        peer_lookup,

        on=[
            "_peer_group",
            "ClaimDate"
        ],

        how="left"
    )[
        "_historical_peer_mean"
    ]

    processed[
        "historical_peer_mean"
    ] = peer_values.values

    # =================================================================
    # 11. DEVIATION FROM PEER CLAIMS
    # =================================================================

    peer_mean = (
        processed[
            "historical_peer_mean"
        ]
    )

    processed[
        "deviation_from_peer_claims"
    ] = np.where(

        peer_mean.notna()
        &
        (peer_mean != 0),

        (
            processed[
                "claim_amount"
            ]
            - peer_mean
        )
        /
        peer_mean,

        0.0
    )

    processed[
        "deviation_from_peer_claims"
    ] = (
        processed[
            "deviation_from_peer_claims"
        ]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            0
        )
        .fillna(0)
        .round(4)
    )

    processed.drop(
        columns=[
            "historical_peer_mean"
        ],
        inplace=True
    )

    # =================================================================
    # 12. ADDITIONAL BEHAVIORAL FEATURES
    # =================================================================

    print("\n" + "=" * 75)
    print("6. CREATING ADDITIONAL BEHAVIORAL FEATURES")
    print("=" * 75)

    # ---------------------------------------------------------------
    # Claim amount vs patient's historical average.
    # ---------------------------------------------------------------

    processed[
        "claim_amount_vs_patient_average"
    ] = np.where(

        processed[
            "avg_historical_claim_amount"
        ] > 0,

        processed[
            "claim_amount"
        ]
        /
        processed[
            "avg_historical_claim_amount"
        ],

        1.0
    )

    # ---------------------------------------------------------------
    # Claim amount vs provider historical average.
    # ---------------------------------------------------------------

    processed[
        "claim_amount_vs_provider_average"
    ] = np.where(

        processed[
            "provider_historical_avg_claim"
        ] > 0,

        processed[
            "claim_amount"
        ]
        /
        processed[
            "provider_historical_avg_claim"
        ],

        1.0
    )

    # ---------------------------------------------------------------
    # Log-transformed claim amount.
    # ---------------------------------------------------------------

    processed[
        "log_claim_amount"
    ] = np.log1p(
        processed[
            "claim_amount"
        ].clip(
            lower=0
        )
    )

    # =================================================================
    # 13. PATIENT / PROVIDER CONTEXT
    # =================================================================

    processed[
        "patient_age"
    ] = df[
        "PatientAge"
    ].values

    processed[
        "patient_income"
    ] = df[
        "PatientIncome"
    ].values

    processed[
        "patient_gender"
    ] = df[
        "PatientGender"
    ].values

    processed[
        "marital_status"
    ] = df[
        "PatientMaritalStatus"
    ].values

    processed[
        "employment_status"
    ] = df[
        "PatientEmploymentStatus"
    ].values

    processed[
        "provider_specialty"
    ] = df[
        "ProviderSpecialty"
    ].values

    # =================================================================
    # 14. CLUSTER
    # =================================================================
    #
    # Cluster is intentionally NOT included as a model feature.
    #
    # We don't yet know how this cluster was created.
    # If it was generated using target information, it would leak
    # the fraud label.
    #
    # We can investigate Cluster separately during EDA.
    # =================================================================

    # =================================================================
    # 15. ADD CLAIM DATE + TARGET
    # =================================================================

    processed[
        "ClaimDate"
    ] = df[
        "ClaimDate"
    ].values

    processed[
        "target"
    ] = df[
        "target"
    ].values

    # =================================================================
    # 16. CLEAN TEMPORARY DATA
    # =================================================================

    df.drop(
        columns=[
            "_fraud_flag",
            "_peer_group",
            "_row_id"
        ],
        inplace=True,
        errors="ignore"
    )

    # =================================================================
    # 17. HANDLE INF / NA
    # =================================================================

    processed = processed.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    # ---------------------------------------------------------------
    # Numerical features
    # ---------------------------------------------------------------

    numeric_features = (
        processed
        .select_dtypes(
            include=[
                "number"
            ]
        )
        .columns
        .tolist()
    )

    numeric_features = [
        col
        for col in numeric_features
        if col != "target"
    ]

    for col in numeric_features:

        processed[
            col
        ] = (
            processed[
                col
            ]
            .fillna(
                processed[
                    col
                ].median()
            )
        )

    # ---------------------------------------------------------------
    # Categorical features
    # ---------------------------------------------------------------

    categorical_features = (
        processed
        .select_dtypes(
            include=[
                "object",
                "category"
            ]
        )
        .columns
        .tolist()
    )

    for col in categorical_features:

        processed[
            col
        ] = (
            processed[
                col
            ]
            .fillna(
                "Unknown"
            )
            .astype(str)
            .str.strip()
        )

    # =================================================================
    # 18. FINAL CHRONOLOGICAL SORT
    # =================================================================

    processed = (
        processed
        .sort_values(
            "ClaimDate"
        )
        .reset_index(
            drop=True
        )
    )

    # =================================================================
    # 19. FINAL OUTPUT
    # =================================================================

    print("\n" + "=" * 75)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 75)

    print(
        f"Final processed shape: "
        f"{processed.shape}"
    )

    print(
        "\nFinal feature columns:"
    )

    for i, col in enumerate(
        processed.columns,
        start=1
    ):

        print(
            f"{i:2d}. {col}"
        )

    print(
        "\nTarget distribution:"
    )

    print(
        processed[
            "target"
        ].value_counts()
    )

    return processed


# =====================================================================
# 3. MODEL TRAINING
# =====================================================================

def train_production_pipeline(
    df: pd.DataFrame
):

    print("\n" + "=" * 75)
    print("STARTING MODEL TRAINING")
    print("=" * 75)

    # =================================================================
    # SORT CHRONOLOGICALLY
    # =================================================================

    df = (
        df
        .sort_values(
            "ClaimDate"
        )
        .reset_index(
            drop=True
        )
    )

    # =================================================================
    # FEATURES / TARGET
    # =================================================================

    X = df.drop(
        columns=[
            "target",
            "ClaimDate"
        ]
    )

    y = df[
        "target"
    ]

    # =================================================================
    # TEMPORAL TRAIN / VALIDATION / TEST SPLIT
    # =================================================================
    #
    # 70% -> Training
    # 10% -> Validation
    # 20% -> Final Test
    #
    # The final test set is untouched for:
    #   - model selection
    #   - threshold selection
    # =================================================================

    n = len(df)

    train_end = int(
        n * 0.70
    )

    validation_end = int(
        n * 0.80
    )

    X_train = (
        X.iloc[
            :train_end
        ]
        .copy()
    )

    X_valid = (
        X.iloc[
            train_end:
            validation_end
        ]
        .copy()
    )

    X_test = (
        X.iloc[
            validation_end:
        ]
        .copy()
    )

    y_train = (
        y.iloc[
            :train_end
        ]
        .copy()
    )

    y_valid = (
        y.iloc[
            train_end:
            validation_end
        ]
        .copy()
    )

    y_test = (
        y.iloc[
            validation_end:
        ]
        .copy()
    )

    print(
        f"\nTraining Set   : "
        f"{len(X_train)} rows"
    )

    print(
        f"Validation Set : "
        f"{len(X_valid)} rows"
    )

    print(
        f"Test Set       : "
        f"{len(X_test)} rows"
    )

    # =================================================================
    # CLASS DISTRIBUTION
    # =================================================================

    def show_distribution(
        name,
        y_data
    ):

        counts = (
            y_data
            .value_counts()
            .reindex(
                [0, 1],
                fill_value=0
            )
        )

        total = len(
            y_data
        )

        print(
            f"\n{name}"
        )

        print(
            f"Normal (0): "
            f"{counts[0]} "
            f"({counts[0] / total * 100:.2f}%)"
        )

        print(
            f"Fraud (1): "
            f"{counts[1]} "
            f"({counts[1] / total * 100:.2f}%)"
        )

    show_distribution(
        "TRAIN DISTRIBUTION",
        y_train
    )

    show_distribution(
        "VALIDATION DISTRIBUTION",
        y_valid
    )

    show_distribution(
        "TEST DISTRIBUTION",
        y_test
    )

    # =================================================================
    # CLASS IMBALANCE WEIGHT
    # =================================================================

    train_counts = (
        y_train
        .value_counts()
        .reindex(
            [0, 1],
            fill_value=0
        )
    )

    neg_count = int(
        train_counts[0]
    )

    pos_count = int(
        train_counts[1]
    )

    scale_pos_weight = (
        neg_count
        /
        max(
            1,
            pos_count
        )
    )

    print(
        f"\nscale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # =================================================================
    # CATEGORICAL / NUMERICAL FEATURES
    # =================================================================

    categorical_cols = (
        X_train
        .select_dtypes(
            include=[
                "object",
                "category"
            ]
        )
        .columns
        .tolist()
    )

    numerical_cols = (
        X_train
        .select_dtypes(
            include=[
                "int64",
                "float64",
                "int32",
                "float32"
            ]
        )
        .columns
        .tolist()
    )

    print(
        f"\nNumerical features: "
        f"{len(numerical_cols)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_cols)}"
    )

    # =================================================================
    # ONE-HOT ENCODING
    # =================================================================

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "num",

                "passthrough",

                numerical_cols
            ),

            (
                "cat",

                OneHotEncoder(

                    handle_unknown="ignore",

                    max_categories=20,

                    sparse_output=False
                ),

                categorical_cols
            )
        ],

        remainder="drop"
    )

    # ---------------------------------------------------------------
    # FIT ONLY ON TRAINING DATA.
    # ---------------------------------------------------------------

    X_train_trans = (
        preprocessor
        .fit_transform(
            X_train
        )
    )

    X_valid_trans = (
        preprocessor
        .transform(
            X_valid
        )
    )

    X_test_trans = (
        preprocessor
        .transform(
            X_test
        )
    )

    # =================================================================
    # FEATURE NAMES
    # =================================================================

    cat_feature_names = (
        preprocessor
        .named_transformers_[
            "cat"
        ]
        .get_feature_names_out(
            categorical_cols
        )
    )

    all_feature_names = (
        numerical_cols
        +
        list(
            cat_feature_names
        )
    )

    print(
        f"\nEncoded feature count: "
        f"{len(all_feature_names)}"
    )

    # =================================================================
    # MODELS
    # =================================================================

    models = {

        "XGBoost": xgb.XGBClassifier(

            n_estimators=250,

            learning_rate=0.03,

            max_depth=6,

            subsample=0.8,

            colsample_bytree=0.8,

            scale_pos_weight=
                scale_pos_weight,

            random_state=
                RANDOM_STATE,

            eval_metric=
                "logloss",

            n_jobs=-1
        ),

        "LightGBM": lgb.LGBMClassifier(

            n_estimators=250,

            learning_rate=0.03,

            max_depth=6,

            num_leaves=31,

            scale_pos_weight=
                scale_pos_weight,

            random_state=
                RANDOM_STATE,

            verbosity=-1,

            n_jobs=-1
        )
    }

    # =================================================================
    # MODEL COMPARISON
    # =================================================================

    best_model = None

    best_name = ""

    best_valid_pr_auc = -1

    best_valid_probs = None

    best_test_probs = None

    best_train_probs = None

    model_results = []

    print("\n" + "=" * 75)
    print(
        "MODEL COMPARISON"
    )
    print("=" * 75)

    for name, model in (
        models.items()
    ):

        print(
            f"\nTraining {name}..."
        )

        model.fit(
            X_train_trans,
            y_train
        )

        # =============================================================
        # TRAIN PROBABILITIES
        # =============================================================

        train_probs = (
            model
            .predict_proba(
                X_train_trans
            )[:, 1]
        )

        train_roc_auc = (
            roc_auc_score(
                y_train,
                train_probs
            )
        )

        train_pr_auc = (
            average_precision_score(
                y_train,
                train_probs
            )
        )

        # =============================================================
        # VALIDATION PROBABILITIES
        # =============================================================

        valid_probs = (
            model
            .predict_proba(
                X_valid_trans
            )[:, 1]
        )

        valid_roc_auc = (
            roc_auc_score(
                y_valid,
                valid_probs
            )
        )

        valid_pr_auc = (
            average_precision_score(
                y_valid,
                valid_probs
            )
        )

        # =============================================================
        # TEST PROBABILITIES
        # =============================================================
        #
        # Reported only.
        #
        # NOT used to select model or threshold.
        # =============================================================

        test_probs = (
            model
            .predict_proba(
                X_test_trans
            )[:, 1]
        )

        test_roc_auc = (
            roc_auc_score(
                y_test,
                test_probs
            )
        )

        test_pr_auc = (
            average_precision_score(
                y_test,
                test_probs
            )
        )

        print(
            f"\n--- {name} Performance ---"
        )

        print(
            f"TRAIN      -> "
            f"ROC-AUC: {train_roc_auc:.4f} | "
            f"PR-AUC: {train_pr_auc:.4f}"
        )

        print(
            f"VALIDATION -> "
            f"ROC-AUC: {valid_roc_auc:.4f} | "
            f"PR-AUC: {valid_pr_auc:.4f}"
        )

        print(
            f"TEST       -> "
            f"ROC-AUC: {test_roc_auc:.4f} | "
            f"PR-AUC: {test_pr_auc:.4f}"
        )

        model_results.append({

            "Model":
                name,

            "Train_ROC_AUC":
                train_roc_auc,

            "Train_PR_AUC":
                train_pr_auc,

            "Validation_ROC_AUC":
                valid_roc_auc,

            "Validation_PR_AUC":
                valid_pr_auc,

            "Test_ROC_AUC":
                test_roc_auc,

            "Test_PR_AUC":
                test_pr_auc
        })

        # -----------------------------------------------------------
        # Select model ONLY using validation PR-AUC.
        # -----------------------------------------------------------

        if (
            valid_pr_auc
            >
            best_valid_pr_auc
        ):

            best_valid_pr_auc = (
                valid_pr_auc
            )

            best_name = name

            best_model = model

            best_valid_probs = (
                valid_probs
            )

            best_test_probs = (
                test_probs
            )

            best_train_probs = (
                train_probs
            )

    # =================================================================
    # MODEL COMPARISON TABLE
    # =================================================================

    results_df = (
        pd.DataFrame(
            model_results
        )
        .sort_values(
            "Validation_PR_AUC",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print("\n" + "=" * 75)
    print(
        "MODEL COMPARISON SUMMARY"
    )
    print("=" * 75)

    print(
        results_df.to_string(
            index=False
        )
    )

    # =================================================================
    # THRESHOLD OPTIMIZATION
    # =================================================================
    #
    # Threshold is selected using VALIDATION data.
    #
    # Test remains untouched.
    # =================================================================

    print("\n" + "=" * 75)
    print(
        "THRESHOLD OPTIMIZATION"
    )
    print("=" * 75)

    precisions, recalls, thresholds = (
        precision_recall_curve(
            y_valid,
            best_valid_probs
        )
    )

    # precision_recall_curve returns
    # len(thresholds) + 1 precision values.

    f1_scores = (
        2
        *
        precisions
        *
        recalls
        /
        (
            precisions
            +
            recalls
            +
            1e-10
        )
    )

    valid_f1_scores = (
        f1_scores[
            :-1
        ]
    )

    if len(
        thresholds
    ) > 0:

        best_threshold_idx = (
            np.argmax(
                valid_f1_scores
            )
        )

        optimal_threshold = float(
            thresholds[
                best_threshold_idx
            ]
        )

        best_validation_f1 = float(
            valid_f1_scores[
                best_threshold_idx
            ]
        )

    else:

        optimal_threshold = 0.5

        best_validation_f1 = 0.0

    print(
        f"Selected model: "
        f"{best_name}"
    )

    print(
        f"Validation PR-AUC: "
        f"{best_valid_pr_auc:.4f}"
    )

    print(
        f"Optimal threshold: "
        f"{optimal_threshold:.4f}"
    )

    print(
        f"Validation F1: "
        f"{best_validation_f1:.4f}"
    )

    # =================================================================
    # FEATURE IMPORTANCE
    # =================================================================
    try:
        feature_importance_list = None

        if hasattr(best_model, "feature_importances_"):
            fi = best_model.feature_importances_

        elif hasattr(best_model, "coef_"):
            fi = np.abs(best_model.coef_).ravel()

        else:
            fi = None

        if fi is not None and len(fi) == len(all_feature_names):
            feature_importances = dict(zip(all_feature_names, fi))
            sorted_fi = sorted(
                feature_importances.items(),
                key=lambda x: x[1],
                reverse=True
            )

            feature_importance_list = sorted_fi

            print("\n" + "=" * 75)
            print("FEATURE IMPORTANCE (Top 20)")
            print("=" * 75)

            for name, val in sorted_fi[:20]:
                print(f"{name:<45s} {val: .6f}")

        else:
            print("[WARNING] Could not extract feature importances from the chosen model or feature length mismatch.")

    except Exception as e:
        print("[WARNING] Feature importance extraction failed:", str(e))

    # =================================================================
    # TRAIN CLASSIFICATION
    # =================================================================

    train_preds = (
        best_train_probs
        >= optimal_threshold
    ).astype(int)

    print("\n" + "=" * 75)
    print(
        "TRAINING SET CLASSIFICATION METRICS"
    )
    print("=" * 75)

    print(
        classification_report(

            y_train,

            train_preds,

            target_names=[
                "Normal (0)",
                "Fraud (1)"
            ],

            zero_division=0
        )
    )

    print(
        "Train Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_train,
            train_preds
        )
    )

    # =================================================================
    # VALIDATION CLASSIFICATION
    # =================================================================

    valid_preds = (
        best_valid_probs
        >= optimal_threshold
    ).astype(int)

    print("\n" + "=" * 75)
    print(
        "VALIDATION SET CLASSIFICATION METRICS"
    )
    print("=" * 75)

    print(
        classification_report(

            y_valid,

            valid_preds,

            target_names=[
                "Normal (0)",
                "Fraud (1)"
            ],

            zero_division=0
        )
    )

    print(
        "Validation Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_valid,
            valid_preds
        )
    )

    # =================================================================
    # FINAL TEST EVALUATION
    # =================================================================

    test_preds = (
        best_test_probs
        >= optimal_threshold
    ).astype(int)

    print("\n" + "=" * 75)
    print(
        "FINAL UNTOUCHED TEST SET EVALUATION"
    )
    print("=" * 75)

    print(
        classification_report(

            y_test,

            test_preds,

            target_names=[
                "Normal (0)",
                "Fraud (1)"
            ],

            zero_division=0
        )
    )

    print(
        "Test Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            test_preds
        )
    )

    # =================================================================
    # FINAL TEST AUC METRICS
    # =================================================================

    final_test_roc_auc = (
        roc_auc_score(
            y_test,
            best_test_probs
        )
    )

    final_test_pr_auc = (
        average_precision_score(
            y_test,
            best_test_probs
        )
    )

    print(
        "\nFinal Test ROC-AUC:"
    )

    print(
        f"{final_test_roc_auc:.4f}"
    )

    print(
        "\nFinal Test PR-AUC:"
    )

    print(
        f"{final_test_pr_auc:.4f}"
    )

    # =================================================================
    # SHAP
    # =================================================================

    print("\n" + "=" * 75)
    print(
        "INITIALIZING SHAP EXPLAINER"
    )
    print("=" * 75)

    try:

        explainer = shap.TreeExplainer(
            best_model
        )

        print(
            "[✓] SHAP TreeExplainer initialized successfully."
        )

    except Exception as e:

        print(
            "[WARNING] SHAP TreeExplainer "
            "could not be initialized:"
        )

        print(
            str(e)
        )

        explainer = None

    # =================================================================
    # SAVE ARTIFACTS
    # =================================================================

    os.makedirs(
        "output",
        exist_ok=True
    )

    artifact_path = (
        "output/fraud_detection_model.pkl"
    )

    artifacts = {

        "model_name":
            best_name,

        "model":
            best_model,

        "preprocessor":
            preprocessor,

        "optimal_threshold":
            optimal_threshold,

        "explainer":
            explainer,

        "feature_names":
            all_feature_names,

        "required_features":
            list(X.columns),

        "categorical_columns":
            categorical_cols,

        "numerical_columns":
            numerical_cols,

        "validation_pr_auc":
            best_valid_pr_auc,

        "test_roc_auc":
            final_test_roc_auc,

        "test_pr_auc":
            final_test_pr_auc,

        "model_comparison":
            results_df
        ,
        "feature_importances":
            feature_importance_list
    }

    joblib.dump(
        artifacts,
        artifact_path
    )

    print(
        f"\n[✓] Pipeline artifacts successfully saved to:"
    )

    print(
        artifact_path
    )

    # =================================================================
    # SAVE MODEL COMPARISON
    # =================================================================

    results_path = (
        "output/model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print(
        f"[✓] Model comparison saved to:"
    )

    print(
        results_path
    )

    # =================================================================
    # RETURN RESULTS
    # =================================================================

    return {

        "best_model":
            best_model,

        "best_model_name":
            best_name,

        "preprocessor":
            preprocessor,

        "explainer":
            explainer,

        "threshold":
            optimal_threshold,

        "X_train":
            X_train,

        "X_valid":
            X_valid,

        "X_test":
            X_test,

        "y_train":
            y_train,

        "y_valid":
            y_valid,

        "y_test":
            y_test,

        "X_train_trans":
            X_train_trans,

        "X_valid_trans":
            X_valid_trans,

        "X_test_trans":
            X_test_trans,

        "feature_names":
            all_feature_names,

        "train_probs":
            best_train_probs,

        "valid_probs":
            best_valid_probs,

        "test_probs":
            best_test_probs,

        "test_predictions":
            test_preds,

        "model_results":
            results_df
    }


# =====================================================================
# 4. MAIN
# =====================================================================

if __name__ == "__main__":

    DATA_PATH = (
        "Health Insurance Fraud Claims.xlsx"
    )

    # =================================================================
    # STEP 1:
    # FEATURE ENGINEERING
    # =================================================================

    processed_data = (
        build_feature_pipeline(
            DATA_PATH
        )
    )

    # =================================================================
    # SAVE PROCESSED DATA
    # =================================================================

    os.makedirs(
        "output",
        exist_ok=True
    )

    processed_path = (
        "output/processed_claim_features.csv"
    )

    processed_data.to_csv(
        processed_path,
        index=False
    )

    print(
        f"\n[✓] Processed feature dataset saved to:"
    )

    print(
        processed_path
    )

    # =================================================================
    # STEP 2:
    # MODEL TRAINING
    # =================================================================

    pipeline_results = (
        train_production_pipeline(
            processed_data
        )
    )

    # =================================================================
    # COMPLETE
    # =================================================================

    print("\n" + "=" * 75)
    print(
        "COMPLETE ML PIPELINE FINISHED SUCCESSFULLY"
    )
    print("=" * 75)