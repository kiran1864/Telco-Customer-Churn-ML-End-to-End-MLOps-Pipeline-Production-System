from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("data/ingested/telco_churn.csv")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
TARGET = "Churn"


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    data = pd.read_csv(INPUT_PATH)

    # Drop identifier first so exact duplicates can be detected properly.
    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(data)} duplicate rows.")

    if TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    # Blank TotalCharges typically occur for new customers (tenure == 0).
    data["TotalCharges"] = (
        pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0.0)
    )
    data["tenure"] = pd.to_numeric(data["tenure"], errors="coerce")
    data["MonthlyCharges"] = pd.to_numeric(data["MonthlyCharges"], errors="coerce")

    # SeniorCitizen is a binary indicator; map to Yes/No for consistent categorical handling.
    if data["SeniorCitizen"].dtype != object:
        data["SeniorCitizen"] = data["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna("No")

    # Normalize target to binary 1/0 (Yes -> 1, No -> 0).
    y = (data[TARGET].astype(str).str.strip() == "Yes").astype(int)
    X = data.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(params["test_size"]),
        random_state=int(params["random_state"]),
        stratify=y,
    )

    train = X_train.copy()
    train[TARGET] = y_train.values
    test = X_test.copy()
    test[TARGET] = y_test.values

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print(f"Training data: {train.shape} (Churn rate: {y_train.mean():.3f})")
    print(f"Testing data: {test.shape} (Churn rate: {y_test.mean():.3f})")
    print(f"Saved: {TRAIN_PATH} and {TEST_PATH}")


if __name__ == "__main__":
    main()