"""Final evaluation of the automatically selected champion on the untouched test set."""
from pathlib import Path
import json
import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")
SELECTION_PATH = Path("reports/model_selection.json")
COMPARISON_CSV_PATH = Path("reports/model_comparison.csv")
METRICS_PATH = Path("reports/metrics.json")
DETAILS_PATH = Path("reports/evaluation_details.json")
TARGET = "Churn"

MODEL_NAMES = [
    "KNN",
    "DecisionTree",
    "SVM",
    "GaussianNB",
    "RandomForest",
    "GradientBoosting",
]


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    metrics_dict = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
        metrics_dict["roc_auc"] = float(roc_auc_score(y_test, scores))
        metrics_dict["avg_precision"] = float(average_precision_score(y_test, scores))
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        metrics_dict["roc_auc"] = float(roc_auc_score(y_test, scores))
        metrics_dict["avg_precision"] = float(average_precision_score(y_test, scores))
    return metrics_dict


def detailed_report(model, X_test, y_test):
    predictions = model.predict(X_test)
    return classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)[TARGET].astype(int)

    with open(SELECTION_PATH, "r", encoding="utf-8") as file:
        selection = json.load(file)

    champion_info = selection["champion"]
    champion_name = champion_info["model_name"]
    champion_path = Path(selection["champion_model_path"])
    champion = joblib.load(champion_path)
    criteria = champion_info["criteria"]
    policy = selection["selection_policy"]

    all_results = {}
    for model_name in MODEL_NAMES:
        model = joblib.load(Path("models") / f"{model_name}.pkl")
        all_results[model_name] = evaluate_model(model, X_test, y_test)

    champion_metrics = evaluate_model(champion, X_test, y_test)
    test_acc_pct = champion_metrics["accuracy"] * 100.0

    metrics = {
        "accuracy": champion_metrics["accuracy"],
        "precision": champion_metrics["precision"],
        "recall": champion_metrics["recall"],
        "f1": champion_metrics["f1"],
        "roc_auc": champion_metrics.get("roc_auc", 0.0),
        "avg_precision": champion_metrics.get("avg_precision", 0.0),
        "champion_cv_accuracy": champion_info["best_cv_accuracy"],
        "champion_training_accuracy_pct": criteria["training_accuracy_pct"],
        "champion_validation_accuracy_pct": criteria["validation_accuracy_pct"],
        "champion_accuracy_gap_pct": criteria["accuracy_gap_pct"],
        "champion_model_size_mb": criteria["model_size_mb"],
        "champion_training_time_seconds": criteria["training_time_seconds"],
        "champion_prediction_time_ms": criteria["prediction_time_ms"],
        "test_vs_validation_gap_pct": criteria["validation_accuracy_pct"] - test_acc_pct,
    }

    details = {
        "champion_model": champion_name,
        "selection_basis": (
            "automated multi-criteria selection: accuracy gap strictly below "
            f"{policy['max_accuracy_gap_pct (strictly below)']}%, then highest validation "
            "accuracy (within tolerance), then smaller model size, lower training "
            "time and lower prediction time"
        ),
        "selection_explanation": champion_info["explanation"],
        "selection_policy": policy,
        "selection_comparison": {
            name: {
                k: selection[name][k]
                for k in (
                    "model_size_mb",
                    "training_time_seconds",
                    "prediction_time_ms",
                    "training_accuracy_pct",
                    "validation_accuracy_pct",
                    "accuracy_gap_pct",
                    "eligible",
                    "status",
                )
            }
            for name in MODEL_NAMES
        },
        "all_models_test_metrics": all_results,
        "champion_classification_report": detailed_report(champion, X_test, y_test),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)
    with open(DETAILS_PATH, "w", encoding="utf-8") as file:
        json.dump(details, file, indent=4)

    tracking_uri = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name="champion_evaluation"):
        mlflow.log_param("champion_model", champion_name)
        mlflow.log_param("selection_method", "multi_criteria_gap_acc_size_time")
        mlflow.log_metrics({
            "test_accuracy": champion_metrics["accuracy"],
            "test_precision": champion_metrics["precision"],
            "test_recall": champion_metrics["recall"],
            "test_f1": champion_metrics["f1"],
            "test_roc_auc": champion_metrics.get("roc_auc", 0.0),
            "test_avg_precision": champion_metrics.get("avg_precision", 0.0),
            "champion_training_accuracy_pct": criteria["training_accuracy_pct"],
            "champion_validation_accuracy_pct": criteria["validation_accuracy_pct"],
            "champion_accuracy_gap_pct": criteria["accuracy_gap_pct"],
            "champion_model_size_mb": criteria["model_size_mb"],
            "champion_training_time_seconds": criteria["training_time_seconds"],
            "champion_prediction_time_ms": criteria["prediction_time_ms"],
        })
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(DETAILS_PATH))
        mlflow.log_artifact(str(SELECTION_PATH))
        if COMPARISON_CSV_PATH.exists():
            mlflow.log_artifact(str(COMPARISON_CSV_PATH))

    print(f"Champion model: {champion_name}")
    print(f"Why: {champion_info['explanation']}")
    print(
        f"Validation accuracy: {criteria['validation_accuracy_pct']:.2f}%  "
        f"Gap: {criteria['accuracy_gap_pct']:.2f}pp"
    )
    print(f"Test accuracy: {champion_metrics['accuracy']:.4f}")
    print(f"Test F1: {champion_metrics['f1']:.4f}")
    if "roc_auc" in champion_metrics:
        print(f"Test ROC-AUC: {champion_metrics['roc_auc']:.4f}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()