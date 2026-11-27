from pathlib import Path
import pandas as pd
import yaml

RAW_DATA_CANDIDATES = [
    Path("data/raw/Telco-Customer-Churn.csv"),
    Path("data/raw/Telco-Customer-Churn"),
]
INGESTED_DATA_PATH = Path("data/ingested/telco_churn.csv")
TRAIN_DATA_PATH = Path("data/processed/train.csv")
TEST_DATA_PATH = Path("data/processed/test.csv")
X_TRAIN_PATH = Path("data/processed/X_train.csv")
X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")
PARAMS_PATH = Path("params.yaml")

EXPECTED_BASE_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
TARGET = "Churn"


def _load_params():
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_raw_data_path() -> Path:
    for candidate in RAW_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise AssertionError(
        f"Raw dataset not found at any expected path: {[str(p) for p in RAW_DATA_CANDIDATES]}"
    )


def test_raw_and_ingested_datasets_exist_and_valid():
    """Verify raw and ingested Telco Churn datasets exist, are non-empty, and contain target."""
    raw_path = _resolve_raw_data_path()
    raw_df = pd.read_csv(raw_path)
    assert not raw_df.empty, "Raw dataset is empty"
    assert TARGET in raw_df.columns, f"Missing target column '{TARGET}' in raw data"

    for col in EXPECTED_BASE_FEATURES:
        assert col in raw_df.columns, f"Missing expected feature '{col}' in raw data"

    if INGESTED_DATA_PATH.exists():
        ingested_df = pd.read_csv(INGESTED_DATA_PATH)
        assert len(ingested_df) == len(raw_df)
        assert TARGET in ingested_df.columns


def test_preprocessed_splits_exist_and_consistent():
    """Verify train/test splits exist, are non-empty, and preserve binary target."""
    assert TRAIN_DATA_PATH.exists(), f"Train set not found at {TRAIN_DATA_PATH}"
    assert TEST_DATA_PATH.exists(), f"Test set not found at {TEST_DATA_PATH}"

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    assert len(train_df) > 0, "Train split is empty"
    assert len(test_df) > 0, "Test split is empty"
    assert TARGET in train_df.columns and TARGET in test_df.columns

    # Binary target encoded as 0/1 in both splits
    assert set(train_df[TARGET].dropna().unique()).issubset({0, 1})
    assert set(test_df[TARGET].dropna().unique()).issubset({0, 1})

    params = _load_params()
    if bool(params.get("drop_customer_id", True)):
        assert "customerID" not in train_df.columns
        assert "customerID" not in test_df.columns

    combined = pd.concat([train_df, test_df], ignore_index=True)
    assert combined.duplicated().sum() == 0, "Duplicate rows found across train/test splits"


def test_feature_target_split_files_match():
    """Verify X_train, X_test, y_train, and y_test align in row counts and columns."""
    for p in (X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH):
        assert p.exists(), f"Missing split artifact: {p}"

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert TARGET not in X_train.columns
    assert TARGET not in X_test.columns
    assert TARGET in y_train.columns
    assert TARGET in y_test.columns
    assert list(X_train.columns) == list(X_test.columns)