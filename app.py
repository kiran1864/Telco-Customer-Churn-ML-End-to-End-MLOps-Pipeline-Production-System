from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.monitor_drift import (
    CATEGORICAL_FEATURE_COLUMNS,
    DRIFT_HTML_PATH,
    DRIFT_JSON_PATH,
    FEATURE_COLUMNS,
    INFERENCE_LOG_PATH,
    NUMERIC_FEATURE_COLUMNS,
    analyze_drift,
    generate_alert_markdown,
    generate_sample_inferences,
    load_inference_logs,
    log_inference,
    render_html_report,
)

MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("reports/metrics.json")
SELECTION_PATH = Path("reports/model_selection.json")
COMPARISON_CSV_PATH = Path("reports/model_comparison.csv")
REFERENCE_DATA_PATH = Path("data/processed/train.csv")

st.set_page_config(
    page_title="Telco Customer Churn Predictor & MLOps Monitor",
    page_icon="📡",
    layout="wide",
)


@st.cache_resource
def load_champion_model(model_path: Path):
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def load_json_if_exists(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_customer(model, input_df: pd.DataFrame):
    raw_pred = model.predict(input_df)[0]
    if isinstance(raw_pred, (int, np.integer)):
        pred_label = "Yes" if int(raw_pred) == 1 else "No"
    else:
        pred_label = str(raw_pred)

    probs = {"No": np.nan, "Yes": np.nan}
    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba(input_df)[0]
        classes = list(getattr(model, "classes_", []))
        for idx, cls in enumerate(classes):
            if cls in (0, "0", "No", "no"):
                probs["No"] = float(proba_arr[idx])
            elif cls in (1, "1", "Yes", "yes"):
                probs["Yes"] = float(proba_arr[idx])
    return pred_label, probs


def main():
    st.title("📡 Telco Customer Churn Prediction & MLOps Dashboard")
    st.caption(
        "Predict customer churn using the selected champion pipeline, inspect model "
        "evaluation metrics, and monitor production data drift."
    )

    model = load_champion_model(MODEL_PATH)
    if model is None:
        st.error(
            f"Champion model not found at `{MODEL_PATH}`. Run `dvc repro` first to train and select a model."
        )
        st.stop()

    tab_predict, tab_model, tab_drift = st.tabs(
        ["🔮 Churn Prediction", "📊 Model & Selection", "📉 Data Drift"]
    )

    # ------------------------------------------------------------------
    # Tab 1: Single-Customer Churn Prediction + Inference Logging
    # ------------------------------------------------------------------
    with tab_predict:
        st.subheader("Customer Profile Inputs")
        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.number_input("Tenure (months)", min_value=0.0, max_value=72.0, value=12.0, step=1.0)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.5, step=0.5)
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=15000.0,
                value=float(round(tenure * monthly_charges, 2)),
                step=1.0,
            )
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])

        with col2:
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

        with col3:
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        log_to_csv = st.checkbox("Log prediction to inference monitoring CSV", value=True)

        if st.button("Predict Churn", type="primary"):
            input_df = pd.DataFrame(
                [
                    {
                        "tenure": tenure,
                        "MonthlyCharges": monthly_charges,
                        "TotalCharges": total_charges,
                        "gender": gender,
                        "SeniorCitizen": senior_citizen,
                        "Partner": partner,
                        "Dependents": dependents,
                        "PhoneService": phone_service,
                        "MultipleLines": multiple_lines,
                        "InternetService": internet_service,
                        "OnlineSecurity": online_security,
                        "OnlineBackup": online_backup,
                        "DeviceProtection": device_protection,
                        "TechSupport": tech_support,
                        "StreamingTV": streaming_tv,
                        "StreamingMovies": streaming_movies,
                        "Contract": contract,
                        "PaperlessBilling": paperless_billing,
                        "PaymentMethod": payment_method,
                    }
                ],
                columns=FEATURE_COLUMNS,
            )

            pred_label, probs = predict_customer(model, input_df)
            if pred_label == "Yes":
                st.warning("Prediction: **Churn = Yes** (Customer is at higher risk of churning)")
            else:
                st.success("Prediction: **Churn = No** (Customer is likely to stay)")

            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    "Probability of Churn (Yes)",
                    f"{probs['Yes'] * 100:.2f}%" if pd.notna(probs["Yes"]) else "N/A",
                )
            with m2:
                st.metric(
                    "Probability of No Churn (No)",
                    f"{probs['No'] * 100:.2f}%" if pd.notna(probs["No"]) else "N/A",
                )

            if log_to_csv:
                log_inference(input_df, pred_label, probs, log_path=INFERENCE_LOG_PATH)
                st.caption(f"Logged inference row to `{INFERENCE_LOG_PATH}`.")

    # ------------------------------------------------------------------
    # Tab 2: Champion Selection & Test Evaluation Metrics
    # ------------------------------------------------------------------
    with tab_model:
        metrics = load_json_if_exists(METRICS_PATH)
        selection = load_json_if_exists(SELECTION_PATH)

        if selection and "champion" in selection:
            champ = selection["champion"]
            st.subheader(f"Champion Model: {champ.get('display_name', champ['model_name'])}")
            st.write(champ.get("explanation", ""))

        if metrics:
            st.markdown("### Held-Out Test Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test Accuracy", f"{metrics.get('accuracy', 0.0):.4f}")
            c2.metric("Test F1", f"{metrics.get('f1', 0.0):.4f}")
            c3.metric("Test ROC-AUC", f"{metrics.get('roc_auc', 0.0):.4f}")
            c4.metric("Train-Val Gap", f"{metrics.get('champion_accuracy_gap_pct', 0.0):.2f} pp")
        else:
            st.info("`reports/metrics.json` not found yet. Run `dvc repro` to generate evaluation metrics.")

        if COMPARISON_CSV_PATH.exists():
            st.markdown("### Candidate Model Comparison")
            st.dataframe(pd.read_csv(COMPARISON_CSV_PATH), use_container_width=True)

    # ------------------------------------------------------------------
    # Tab 3: Data Drift Monitoring
    # ------------------------------------------------------------------
    with tab_drift:
        st.subheader("Production vs. Reference Drift Check")
        action_col1, action_col2, action_col3 = st.columns(3)

        with action_col1:
            if st.button("Generate Healthy Sample Logs"):
                generate_sample_inferences(n_samples=50, drift=False, log_path=INFERENCE_LOG_PATH)
                st.rerun()

        with action_col2:
            if st.button("Simulate Drifted Traffic"):
                generate_sample_inferences(n_samples=50, drift=True, log_path=INFERENCE_LOG_PATH)
                st.rerun()

        with action_col3:
            run_check = st.button("Run Drift Analysis Now", type="primary")

        if run_check:
            if not REFERENCE_DATA_PATH.exists():
                st.error(f"Reference dataset missing at `{REFERENCE_DATA_PATH}`.")
            else:
                ref_df = pd.read_csv(REFERENCE_DATA_PATH)
                logs_df = load_inference_logs(INFERENCE_LOG_PATH)
                if logs_df.empty:
                    st.warning("No inference logs available yet. Log predictions or generate sample logs first.")
                else:
                    report = analyze_drift(ref_df, logs_df)
                    DRIFT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(DRIFT_JSON_PATH, "w", encoding="utf-8") as f:
                        json.dump(report, f, indent=4)
                    render_html_report(report, DRIFT_HTML_PATH)
                    generate_alert_markdown(report, DRIFT_JSON_PATH.parent / "drift_alert.md")
                    st.rerun()

        drift_report = load_json_if_exists(DRIFT_JSON_PATH)
        if drift_report:
            status_msg = (
                "🚨 Dataset Drift Detected"
                if drift_report.get("dataset_drift_detected")
                else "✅ Distribution Healthy"
            )
            st.markdown(f"**Status:** {status_msg}")
            st.write(
                f"Drifted features: **{drift_report.get('drifted_features', 0)} / "
                f"{drift_report.get('total_features', 0)}** "
                f"({drift_report.get('drift_share_pct', 0.0)}%)"
            )

            feat_rows = []
            for feat_name, info in drift_report.get("features", {}).items():
                feat_rows.append(
                    {
                        "Feature": feat_name,
                        "Test": info.get("test", ""),
                        "Drift": "Yes" if info.get("drift_detected") else "No",
                        "p-value": round(float(info.get("p_value", 0.0)), 4),
                        "Statistic": round(float(info.get("ks_statistic", 0.0)), 4),
                        "Reference": info.get("reference_summary", ""),
                        "Current": info.get("current_summary", ""),
                    }
                )
            if feat_rows:
                st.dataframe(pd.DataFrame(feat_rows), use_container_width=True)
        else:
            st.info("No drift report generated yet. Click **Run Drift Analysis Now** above.")


if __name__ == "__main__":
    main()