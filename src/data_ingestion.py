from pathlib import Path
import pandas as pd

RAW_CANDIDATES = [
    Path("data/raw/Telco-Customer-Churn.csv"),
    Path("data/raw/Telco-Customer-Churn"),
]
OUTPUT_PATH = Path("data/ingested/telco_churn.csv")

REQUIRED_COLUMNS = [
    "customerID",
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
    "Churn",
]


def resolve_raw_path() -> Path:
    for candidate in RAW_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Dataset not found in data/raw/. Place Telco-Customer-Churn.csv at data/raw/Telco-Customer-Churn.csv."
    )


def main():
    raw_path = resolve_raw_path()
    data = pd.read_csv(raw_path)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data[REQUIRED_COLUMNS].copy()

    if data.empty:
        raise ValueError("The input dataset is empty.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Data ingestion completed: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()