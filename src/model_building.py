"""Model building with automated multi-criteria champion selection for Telco Churn."""
from pathlib import Path
import json
import os
import shutil
import time
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

X_TRAIN_PATH = Path("data/processed/X_train.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
SELECTION_PATH = REPORTS_DIR / "model_selection.json"
COMPARISON_CSV_PATH = REPORTS_DIR / "model_comparison.csv"
COMPARISON_MD_PATH = REPORTS_DIR / "model_comparison.md"
TARGET = "Churn"

NUMERIC_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
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
]

MODEL_NAMES = [
    "KNN",
    "DecisionTree",
    "SVM",
    "GaussianNB",
    "RandomForest",
    "GradientBoosting",
]
DISPLAY_NAMES = {
    "KNN": "KNN",
    "DecisionTree": "Decision Tree",
    "SVM": "SVM",
    "GaussianNB": "Gaussian NB",
    "RandomForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
}


def suggest_params(model_name, trial, max_k=30):
    scaler_type = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
    k = trial.suggest_int("k", 5, max_k)

    if model_name == "KNN":
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 21, step=2),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 3),
        }
    elif model_name == "DecisionTree":
        params = {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        }
    elif model_name == "SVM":
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
        params = {
            "kernel": kernel,
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        }
        if kernel in ["rbf", "poly", "sigmoid"]:
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 4)
    elif model_name == "GaussianNB":
        params = {
            "var_smoothing": trial.suggest_float("var_smoothing", 1e-11, 1e-7, log=True)
        }
    elif model_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": trial.suggest_int("max_depth", 5, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    elif model_name == "GradientBoosting":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return scaler_type, k, params


def build_pipeline(model_name, params, scaler_type, k=20, random_state=42):
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, NUMERIC_COLUMNS),
        ("cat", cat_transformer, CATEGORICAL_COLUMNS),
    ])

    if model_name == "KNN":
        estimator = KNeighborsClassifier(**params)
    elif model_name == "DecisionTree":
        estimator = DecisionTreeClassifier(**params, random_state=random_state)
    elif model_name == "SVM":
        estimator = SVC(**params, probability=True, random_state=random_state)
    elif model_name == "GaussianNB":
        estimator = GaussianNB(**params)
    elif model_name == "RandomForest":
        estimator = RandomForestClassifier(**params, random_state=random_state, n_jobs=1)
    elif model_name == "GradientBoosting":
        estimator = GradientBoostingClassifier(**params, random_state=random_state)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("Preprocessor", preprocessor),
        ("FeatureSelection", SelectKBest(score_func=f_classif, k=k)),
        ("Model", estimator),
    ])


def fit_with_timing(model_name, params, scaler_type, k, X, y, repeats, random_state=42):
    times = []
    pipeline = None
    for _ in range(max(1, repeats)):
        pipeline = build_pipeline(model_name, params, scaler_type, k=k, random_state=random_state)
        start = time.perf_counter()
        pipeline.fit(X, y)
        times.append(time.perf_counter() - start)
    return pipeline, float(np.median(times))


def measure_prediction_ms(model, X, n_samples, repeats, seed):
    sample = X.sample(n=min(n_samples, len(X)), random_state=seed)
    rows = [sample.iloc[[i]] for i in range(len(sample))]

    for row in rows[:3]:
        model.predict(row)

    per_repeat_ms = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        for row in rows:
            model.predict(row)
        per_repeat_ms.append((time.perf_counter() - start) / len(rows) * 1000.0)
    return float(np.median(per_repeat_ms))


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def _normalise(values):
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def select_champion(results, cfg):
    gap_limit = float(cfg["max_accuracy_gap_pct"])
    tolerance = float(cfg["accuracy_tolerance_pct"])
    weights = cfg["weights"]

    for r in results.values():
        r["eligible"] = bool(r["accuracy_gap_pct"] < gap_limit)

    eligible = {n: r for n, r in results.items() if r["eligible"]}
    fallback_used = False

    if not eligible:
        mode = cfg.get("fallback_if_none_eligible", "smallest_gap")
        if mode == "error":
            raise RuntimeError(f"No model has an accuracy gap strictly below {gap_limit}%.")
        warnings.warn("No eligible model; falling back to smallest accuracy gap.")
        fallback_used = True
        eligible = {n: r for n, r in results.items()}

    best_val = max(r["validation_accuracy_pct"] for r in eligible.values())
    if fallback_used:
        smallest_gap = min(r["accuracy_gap_pct"] for r in eligible.values())
        finalists = {
            n: r for n, r in eligible.items()
            if r["accuracy_gap_pct"] - smallest_gap < 1e-9
        }
    else:
        finalists = {
            n: r for n, r in eligible.items()
            if best_val - r["validation_accuracy_pct"] <= tolerance + 1e-9
        }

    norm_size = _normalise({n: r["model_size_mb"] for n, r in finalists.items()})
    norm_train = _normalise({n: r["training_time_seconds"] for n, r in finalists.items()})
    norm_pred = _normalise({n: r["prediction_time_ms"] for n, r in finalists.items()})

    for n, r in finalists.items():
        r["selection_cost"] = float(
            weights["model_size_mb"] * norm_size[n]
            + weights["training_time_seconds"] * norm_train[n]
            + weights["prediction_time_ms"] * norm_pred[n]
        )

    ranked = sorted(
        finalists,
        key=lambda n: (
            round(finalists[n]["selection_cost"], 12),
            -finalists[n]["validation_accuracy_pct"],
            n,
        ),
    )
    champion = ranked[0]

    for n, r in results.items():
        if n == champion:
            r["status"] = "champion"
        elif n in finalists:
            r["status"] = "finalist_runner_up"
        elif r["eligible"]:
            r["status"] = "eligible_but_below_accuracy_tolerance"
        else:
            r["status"] = f"rejected_accuracy_gap_>={gap_limit}%"

    for rank, n in enumerate(ranked, start=1):
        results[n]["finalist_rank"] = rank

    explanation = (
        f"{DISPLAY_NAMES[champion]} selected: "
        f"{'no model met the gap rule, so smallest gap was used' if fallback_used else f'passed gap rule (< {gap_limit}%)'}; "
        f"{len(finalists)} finalist(s) within {tolerance} pp of best validation "
        f"accuracy ({best_val:.2f}%); lowest weighted size/train-time/predict-time cost."
    )
    return champion, explanation, fallback_used, ranked


def write_comparison_reports(results, champion):
    rows = []
    for name in MODEL_NAMES:
        r = results[name]
        rows.append({
            "Model": DISPLAY_NAMES[name],
            "Size MB": round(r["model_size_mb"], 4),
            "Train time s": round(r["training_time_seconds"], 4),
            "Predict ms": round(r["prediction_time_ms"], 4),
            "Train acc %": round(r["training_accuracy_pct"], 2),
            "Val acc %": round(r["validation_accuracy_pct"], 2),
            "Gap %": round(r["accuracy_gap_pct"], 2),
            "Eligible": "Yes" if r["eligible"] else "No",
            "Status": r["status"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(COMPARISON_CSV_PATH, index=False)

    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    md = "\n".join(
        ["# Model Comparison", "", header, sep, *body, "",
         f"**Champion:** {DISPLAY_NAMES[champion]}", ""]
    )
    COMPARISON_MD_PATH.write_text(md, encoding="utf-8")
    return df


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    sel_cfg = config["selection"]
    meas_cfg = config["measurement"]
    gap_limit = float(sel_cfg["max_accuracy_gap_pct"])
    use_penalty = bool(sel_cfg.get("gap_penalty_in_search", True))
    penalty_weight = float(sel_cfg.get("gap_penalty_weight", 1.0))

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)[TARGET].astype(int)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    random_state = int(config["random_state"])
    cv_folds = int(config["cv_folds"])
    n_trials = int(config["n_trials"])
    max_k = int(config.get("max_k", 30))

    tracking_uri = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["experiment_name"])

    results = {}

    for model_name in MODEL_NAMES:
        print(f"\n--- Optimizing {model_name} ---")

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv_folds", cv_folds)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("max_accuracy_gap_pct", gap_limit)

            def objective(trial):
                scaler_type, k, model_params = suggest_params(model_name, trial, max_k=max_k)
                candidate = build_pipeline(
                    model_name, model_params, scaler_type, k=k, random_state=random_state
                )
                cv = StratifiedKFold(
                    n_splits=cv_folds, shuffle=True, random_state=random_state
                )
                cv_result = cross_validate(
                    candidate, X_train, y_train, scoring="accuracy", cv=cv,
                    n_jobs=1, return_train_score=True,
                )
                val_acc = float(cv_result["test_score"].mean())
                train_acc = float(cv_result["train_score"].mean())
                gap_pct = (train_acc - val_acc) * 100.0

                objective_value = val_acc
                if use_penalty and gap_pct >= gap_limit:
                    objective_value -= penalty_weight * (gap_pct - gap_limit + 1.0) / 100.0

                trial.set_user_attr("cv_validation_accuracy", val_acc)
                trial.set_user_attr("cv_training_accuracy", train_acc)
                trial.set_user_attr("cv_gap_pct", gap_pct)

                with mlflow.start_run(
                    run_name=f"{model_name}_trial_{trial.number}", nested=True
                ):
                    mlflow.log_param("scaler_type", scaler_type)
                    mlflow.log_param("k", k)
                    mlflow.log_params(model_params)
                    mlflow.log_metric("cv_accuracy", val_acc)
                    mlflow.log_metric("cv_train_accuracy", train_acc)
                    mlflow.log_metric("cv_gap_pct", gap_pct)
                    mlflow.log_metric("objective_value", float(objective_value))

                return float(objective_value)

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=random_state),
            )
            search_start = time.time()
            study.optimize(objective, n_trials=n_trials)
            search_time = time.time() - search_start

            best_scaler = study.best_params["scaler_type"]
            best_k = int(study.best_params["k"])
            best_model_params = {
                k_name: v
                for k_name, v in study.best_params.items()
                if k_name not in {"scaler_type", "k"}
            }
            best_cv_val_acc = float(study.best_trial.user_attrs["cv_validation_accuracy"])

            final_pipeline, train_time = fit_with_timing(
                model_name,
                best_model_params,
                best_scaler,
                best_k,
                X_train,
                y_train,
                int(meas_cfg["training_time_repeats"]),
                random_state=random_state,
            )

            model_path = MODELS_DIR / f"{model_name}.pkl"
            joblib.dump(final_pipeline, model_path)
            size_mb = file_size_mb(model_path)

            train_acc_pct = 100.0 * float(
                accuracy_score(y_train, final_pipeline.predict(X_train))
            )
            val_acc_pct = 100.0 * best_cv_val_acc
            gap_pct = train_acc_pct - val_acc_pct

            pred_ms = measure_prediction_ms(
                final_pipeline, X_train,
                int(meas_cfg["prediction_n_samples"]),
                int(meas_cfg["prediction_repeats"]),
                random_state,
            )

            mlflow.log_params({f"best_{k_name}": v for k_name, v in study.best_params.items()})
            mlflow.log_metrics({
                "best_cv_accuracy": best_cv_val_acc,
                "training_accuracy_pct": train_acc_pct,
                "validation_accuracy_pct": val_acc_pct,
                "accuracy_gap_pct": gap_pct,
                "model_size_mb": size_mb,
                "training_time_seconds": train_time,
                "prediction_time_ms": pred_ms,
                "search_time_seconds": float(search_time),
            })
            mlflow.log_artifact(str(model_path), artifact_path="models")
            mlflow.sklearn.log_model(
                final_pipeline,
                name=f"{model_name}_telco_model",
                serialization_format="cloudpickle",
            )

            results[model_name] = {
                "best_params": study.best_params,
                "model_size_mb": float(size_mb),
                "training_time_seconds": float(train_time),
                "prediction_time_ms": float(pred_ms),
                "training_accuracy_pct": float(train_acc_pct),
                "validation_accuracy_pct": float(val_acc_pct),
                "accuracy_gap_pct": float(gap_pct),
                "search_time_seconds": float(search_time),
                "model_path": str(model_path).replace("\\", "/"),
            }

            print(
                f"{model_name}: val={val_acc_pct:.2f}% train={train_acc_pct:.2f}% "
                f"gap={gap_pct:.2f}pp size={size_mb:.3f}MB "
                f"fit={train_time:.3f}s predict={pred_ms:.3f}ms"
            )

    champion, explanation, fallback_used, ranked = select_champion(results, sel_cfg)

    champion_path = MODELS_DIR / "model.pkl"
    shutil.copy2(MODELS_DIR / f"{champion}.pkl", champion_path)

    champ = results[champion]
    selection = dict(results)
    selection["champion"] = {
        "model_name": champion,
        "best_cv_accuracy": champ["validation_accuracy_pct"] / 100.0,
        "display_name": DISPLAY_NAMES[champion],
        "criteria": {
            "model_size_mb": champ["model_size_mb"],
            "training_time_seconds": champ["training_time_seconds"],
            "prediction_time_ms": champ["prediction_time_ms"],
            "training_accuracy_pct": champ["training_accuracy_pct"],
            "validation_accuracy_pct": champ["validation_accuracy_pct"],
            "accuracy_gap_pct": champ["accuracy_gap_pct"],
        },
        "explanation": explanation,
    }
    selection["champion_model_path"] = str(champion_path).replace("\\", "/")
    selection["selection_policy"] = {
        "max_accuracy_gap_pct (strictly below)": gap_limit,
        "accuracy_tolerance_pct": float(sel_cfg["accuracy_tolerance_pct"]),
        "weights": sel_cfg["weights"],
        "fallback_used": fallback_used,
        "finalist_ranking": ranked,
        "selection_data": "training data only (CV validation); test set untouched",
    }

    with open(SELECTION_PATH, "w", encoding="utf-8") as file:
        json.dump(selection, file, indent=4)

    comparison = write_comparison_reports(results, champion)

    with mlflow.start_run(run_name="champion_selection"):
        mlflow.log_param("champion_model", champion)
        mlflow.log_param("fallback_used", fallback_used)
        mlflow.log_metrics({
            "champion_validation_accuracy_pct": champ["validation_accuracy_pct"],
            "champion_accuracy_gap_pct": champ["accuracy_gap_pct"],
            "champion_model_size_mb": champ["model_size_mb"],
            "champion_training_time_seconds": champ["training_time_seconds"],
            "champion_prediction_time_ms": champ["prediction_time_ms"],
        })
        mlflow.log_artifact(str(SELECTION_PATH))
        mlflow.log_artifact(str(COMPARISON_CSV_PATH))
        mlflow.log_artifact(str(COMPARISON_MD_PATH))

    print("\n=== Model comparison ===")
    print(comparison.drop(columns=["Status"]).to_string(index=False))
    print(f"\nChampion: {DISPLAY_NAMES[champion]}")
    print(explanation)
    print(f"Champion model saved to: {champion_path}")
    print(f"Selection details: {SELECTION_PATH}")
    print(f"Comparison report: {COMPARISON_CSV_PATH} / {COMPARISON_MD_PATH}")


if __name__ == "__main__":
    main()