from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="EV Charging Demand Analytics",
    page_icon="⚡",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ChargingRecords_Cleaned.csv"
REPORTS_PATH = PROJECT_ROOT / "reports"
MODELS_PATH = PROJECT_ROOT / "models"


# ----------------------------- DATA HELPERS -----------------------------
@st.cache_data
def load_data(path_string: str) -> pd.DataFrame:
    data = pd.read_csv(path_string)
    for column in ("StartDatetime", "EndDatetime"):
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce")
    for column in ("Demand", "Duration", "StartHour", "StartMonth", "StartWeekday"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


@st.cache_data
def load_report(path_string: str) -> pd.DataFrame:
    return pd.read_csv(path_string)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def show_report(report_path: Path, title: str) -> None:
    st.markdown(f"**{title}**")
    try:
        report = load_report(str(report_path))
        st.dataframe(report, use_container_width=True, hide_index=True)
        csv_bytes = report.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"Download {title} CSV",
            data=csv_bytes,
            file_name=report_path.name,
            mime="text/csv",
            key=f"download_{report_path.name}",
        )
    except Exception as exc:
        st.warning(f"Could not load {report_path.name}: {exc}")


def load_joblib_model(path: Path):
    if not path.exists():
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception as exc:
        st.warning(f"Could not load model {path.name}: {exc}")
        return None



def make_prediction_input(
    metadata: dict,
    location,
    company,
    charger_type,
    start_hour: int,
    start_month: int,
    start_weekday: int,
    start_day_of_month: int,
):
    features = metadata.get("feature_columns", [])

    if not features:
        return None, "feature_columns missing from metadata"

    row = {}

    for feature in features:
        if feature == "ChargerCompany":
            row[feature] = company
        elif feature == "Location":
            row[feature] = location
        elif feature == "ChargerType":
            row[feature] = charger_type
        elif feature == "StartHour":
            row[feature] = start_hour
        elif feature == "StartMonth":
            row[feature] = start_month
        elif feature in ("StartWeekday", "StartDayOfWeek"):
            row[feature] = start_weekday
        elif feature == "StartDayOfMonth":
            row[feature] = start_day_of_month
        elif feature == "IsWeekend":
            row[feature] = int(start_weekday in (5, 6))
        elif feature.startswith("Location_"):
            # Match the one-hot encoding expected by XGBoost.
            category = feature[len("Location_"):]
            row[feature] = int(str(location) == category)
        else:
            return None, feature

    return pd.DataFrame([row], columns=features), None


# ----------------------------- LOAD DATA -----------------------------
st.title("⚡ EV Charging Demand Analytics")
st.markdown(
    "An interactive dashboard for EV charging usage, demand patterns, "
    "forecasting, model performance, and model-based predictions."
)

if not DATA_PATH.exists():
    st.error(
        f"Could not find the cleaned dataset at `{DATA_PATH}`. "
        "Place ChargingRecords_Cleaned.csv in data/processed/."
    )
    st.stop()

df = load_data(str(DATA_PATH))
required_columns = [
    "Location", "ChargerCompany", "ChargerType", "StartDatetime", "Demand"
]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    st.error("Missing required columns: " + ", ".join(missing_columns))
    st.stop()

df = df.dropna(subset=["Demand"]).copy()
df["CompanyLabel"] = df["ChargerCompany"].map(
    lambda value: f"Company {value}" if pd.notna(value) else "Unknown company"
)
df["ChargerTypeLabel"] = df["ChargerType"].map(
    lambda value: f"Charger Type {value}" if pd.notna(value) else "Unknown type"
)

# ----------------------------- FILTERS -----------------------------
st.sidebar.title("Dashboard Filters")
st.sidebar.caption("Filters apply to usage charts, KPIs, and the data preview.")

location_values = sorted(df["Location"].dropna().unique().tolist(), key=str)
company_values = sorted(df["ChargerCompany"].dropna().unique().tolist(), key=str)
type_values = sorted(df["ChargerType"].dropna().unique().tolist(), key=str)

selected_locations = st.sidebar.multiselect(
    "Location", options=location_values, default=location_values
)
company_label_map = {value: f"Company {value}" for value in company_values}
selected_company_labels = st.sidebar.multiselect(
    "Charger Company",
    options=[company_label_map[value] for value in company_values],
    default=[company_label_map[value] for value in company_values],
)
selected_companies = [
    value for value in company_values
    if company_label_map[value] in selected_company_labels
]
type_label_map = {value: f"Charger Type {value}" for value in type_values}
selected_type_labels = st.sidebar.multiselect(
    "Charger Type",
    options=[type_label_map[value] for value in type_values],
    default=[type_label_map[value] for value in type_values],
)
selected_types = [
    value for value in type_values if type_label_map[value] in selected_type_labels
]

valid_dates = df["StartDatetime"].dropna()
start_date = end_date = None
if not valid_dates.empty:
    min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
    chosen_dates = st.sidebar.date_input(
        "Start date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(chosen_dates, (tuple, list)):
        if len(chosen_dates) == 2:
            start_date, end_date = chosen_dates
        elif len(chosen_dates) == 1:
            start_date = end_date = chosen_dates[0]
    else:
        start_date = end_date = chosen_dates

mask = (
    df["Location"].isin(selected_locations)
    & df["ChargerCompany"].isin(selected_companies)
    & df["ChargerType"].isin(selected_types)
)
if start_date is not None and end_date is not None:
    mask &= (
        df["StartDatetime"].notna()
        & (df["StartDatetime"].dt.date >= start_date)
        & (df["StartDatetime"].dt.date <= end_date)
    )
filtered = df.loc[mask].copy()
st.sidebar.divider()
st.sidebar.caption(f"{len(filtered):,} records match the selected filters.")


if filtered.empty:
    st.warning(
        "No records match the selected filters. "
        "The dashboard tabs will still be available. "
        "Adjust the sidebar selections to display charts and records."
    )

# ----------------------------- TABS -----------------------------
overview_tab, prediction_tab, forecast_tab, explain_tab, reports_tab, data_tab = st.tabs(
    [
        "📊 Overview",
        "🤖 Predictions",
        "📈 Time Series",
        "🔎 Explainability",
        "🏆 Model Reports",
        "🗂️ Data Explorer",
    ]
)

with overview_tab:
    st.subheader("Overall Charging Usage")
    total_records = len(filtered)
    average_demand = filtered["Demand"].mean()
    median_demand = filtered["Demand"].median()
    total_duration = (
        filtered["Duration"].sum()
        if "Duration" in filtered.columns
        else np.nan
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Charging Records", f"{total_records:,}")
    k2.metric("Average Demand", f"{average_demand:.2f}")
    k3.metric("Median Demand", f"{median_demand:.2f}")
    k4.metric(
        "Total Duration",
        f"{total_duration:,.1f}" if pd.notna(total_duration) else "Not available",
    )

    st.caption(
        "Company and charger-type values are shown as coded categories because "
        "the source data does not provide a verified code-to-name mapping."
    )

    left, right = st.columns(2)
    with left:
        location_summary = (
            filtered.groupby("Location", as_index=False)["Demand"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "Average Demand", "count": "Sessions"})
            .sort_values("Average Demand", ascending=False)
        )
        fig = px.bar(
            location_summary,
            x="Location",
            y="Average Demand",
            hover_data=["Sessions"],
            title="Average Demand by Location",
            color="Average Demand",
            color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        type_summary = (
            filtered.groupby("ChargerTypeLabel", as_index=False)["Demand"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "Average Demand", "count": "Sessions"})
        )
        fig = px.bar(
            type_summary,
            x="ChargerTypeLabel",
            y="Average Demand",
            color="ChargerTypeLabel",
            hover_data=["Sessions"],
            title="Average Demand by Charger Type",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    company_summary = (
        filtered.groupby("CompanyLabel", as_index=False)["Demand"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Average Demand", "count": "Sessions"})
    )
    fig = px.bar(
        company_summary,
        x="CompanyLabel",
        y="Average Demand",
        hover_data=["Sessions"],
        color="CompanyLabel",
        title="Average Demand by Charger Company Category",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    if "StartDatetime" in filtered.columns:
        dated = filtered.dropna(subset=["StartDatetime"]).copy()
        if not dated.empty:
            dated["Date"] = dated["StartDatetime"].dt.floor("D")
            daily = (
                dated.groupby("Date", as_index=False)["Demand"]
                .mean()
                .rename(columns={"Demand": "Average Demand"})
                .sort_values("Date")
            )
            fig = px.line(
                daily, x="Date", y="Average Demand", markers=True,
                title="Daily Average Charging Demand"
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            time_left, time_right = st.columns(2)
            if "StartHour" in dated.columns:
                with time_left:
                    hourly = (
                        dated.groupby("StartHour", as_index=False)["Demand"]
                        .mean()
                        .rename(columns={"Demand": "Average Demand"})
                    )
                    fig = px.line(
                        hourly, x="StartHour", y="Average Demand", markers=True,
                        title="Average Demand by Start Hour"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            if "StartWeekday" in dated.columns:
                with time_right:
                    weekday = (
                        dated.groupby("StartWeekday", as_index=False)["Demand"]
                        .mean()
                        .rename(columns={"Demand": "Average Demand"})
                    )
                    weekday_names = {
                        0: "Monday", 1: "Tuesday", 2: "Wednesday",
                        3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
                    }
                    weekday["Weekday"] = weekday["StartWeekday"].map(weekday_names)
                    fig = px.bar(
                        weekday, x="Weekday", y="Average Demand",
                        title="Average Demand by Weekday", color="Weekday"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

with prediction_tab:
    st.subheader("Regression: Predict Charging Demand")
    st.write(
        "Enter the session's known, prediction-time features. The app uses the "
        "saved regression model and its metadata when compatible."
    )
    regression_model_path = MODELS_PATH / "best_regression_model.joblib"
    regression_metadata = read_json(MODELS_PATH / "best_regression_metadata.json")
    reg_model = load_joblib_model(regression_model_path)

    unique_locations = sorted(df["Location"].dropna().unique().tolist(), key=str)
    unique_companies = sorted(df["ChargerCompany"].dropna().unique().tolist(), key=str)
    unique_types = sorted(df["ChargerType"].dropna().unique().tolist(), key=str)

    with st.form("regression_prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            pred_location = st.selectbox("Location", unique_locations, key="reg_location")
            pred_company = st.selectbox(
                "Charger company category", unique_companies, key="reg_company"
            )
        with col2:
            pred_type = st.selectbox(
                "Charger type category", unique_types, key="reg_type"
            )
            pred_hour = st.slider("Start hour", 0, 23, 12, key="reg_hour")
        with col3:
            pred_month = st.selectbox("Start month", list(range(1, 13)), index=0)
            pred_weekday = st.selectbox(
                "Start weekday",
                options=list(range(7)),
                format_func=lambda x: [
                    "Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday"
                ][x],
                index=0,
            )
            pred_day = st.slider("Day of month", 1, 31, 15)
        submitted_reg = st.form_submit_button("Predict demand", type="primary")

    
    if submitted_reg:
        if reg_model is None:
            st.error("The saved regression model could not be loaded.")
        elif not isinstance(reg_model, dict):
            st.error(
                "Unexpected regression artifact format. "
                "Expected a dictionary with 'preprocessor' and 'model'."
            )
        elif "preprocessor" not in reg_model or "model" not in reg_model:
            st.error(
                "The regression artifact is missing its preprocessor or model."
            )
        else:
            x_input, unsupported = make_prediction_input(
                regression_metadata,
                pred_location,
                pred_company,
                pred_type,
                pred_hour,
                pred_month,
                pred_weekday,
                pred_day,
            )

            if x_input is None:
                st.error(f"Unsupported regression feature: {unsupported}")
            else:
                try:
                    preprocessor = reg_model["preprocessor"]
                    model = reg_model["model"]

                    transformed_input = preprocessor.transform(x_input)
                    prediction = float(
                        np.asarray(
                            model.predict(transformed_input)
                        ).reshape(-1)[0]
                    )

                    st.success(
                        f"Predicted charging demand: **{prediction:.3f}**"
                    )
                    st.caption(
                        "Prediction generated using the saved preprocessing "
                        "transformer and LightGBM regression model."
                    )

                except Exception as exc:
                    st.error(f"Regression prediction failed: {exc}")
                    st.info(
                        "The input features or preprocessing may not match "
                        "the original training configuration."
                    )



    st.divider()
    
    st.subheader("Classification: Predict Demand Class")

    st.write(
        "The saved tuned XGBoost classifier predicts whether a session is "
        "in the High or Normal demand class. The project used the median "
        "demand threshold of 14.1 for class labels."
    )

    classification_path = MODELS_PATH / "tuned_xgboost_model.joblib"
    classification_model = load_joblib_model(classification_path)
    classification_metadata = read_json(
        MODELS_PATH / "tuned_xgboost_metadata.json"
    )

    with st.form("classification_prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            cls_location = st.selectbox(
                "Location ", unique_locations, key="cls_location"
            )
            cls_company = st.selectbox(
                "Company category ", unique_companies, key="cls_company"
            )

        with c2:
            cls_type = st.selectbox(
                "Charger type ", unique_types, key="cls_type"
            )
            cls_hour = st.slider(
                "Start hour ", 0, 23, 12, key="cls_hour"
            )

        with c3:
            cls_month = st.selectbox(
                "Start month ",
                list(range(1, 13)),
                index=0,
                key="cls_month",
            )
            cls_weekday = st.selectbox(
                "Start weekday ",
                options=list(range(7)),
                format_func=lambda x: [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ][x],
                index=0,
                key="cls_weekday",
            )
            cls_day = st.slider(
                "Day of month ", 1, 31, 15, key="cls_day"
            )

        submitted_cls = st.form_submit_button(
            "Classify demand", type="primary"
        )

    if submitted_cls:
        if classification_model is None:
            st.error(
                "The saved classifier could not be loaded. "
                "Check models/tuned_xgboost_model.joblib."
            )
        else:
            cls_features = classification_metadata.get(
                "feature_columns", []
            )

            if not cls_features:
                st.error(
                    "Classification metadata does not include "
                    "feature_columns. The app cannot safely construct "
                    "an input row."
                )
            else:
                cls_meta = {"feature_columns": cls_features}

                x_cls, unsupported = make_prediction_input(
                    cls_meta,
                    cls_location,
                    cls_company,
                    cls_type,
                    cls_hour,
                    cls_month,
                    cls_weekday,
                    cls_day,
                )

                if x_cls is None:
                    st.error(
                        f"The classifier requires feature "
                        f"`{unsupported}` that this form does not collect. "
                        "Add the exact training-time feature before "
                        "using it."
                    )
                else:
                    try:
                        cls_prediction = np.asarray(
                            classification_model.predict(x_cls)
                        ).reshape(-1)[0]

                        model_classes = list(
                            getattr(classification_model, "classes_", [])
                        )

                        # Display readable labels for binary classes.
                        # Verify this mapping against training labels.
                        def demand_label(value):
                            value_text = str(value).strip().lower()

                            if value_text == "high":
                                return "High Demand"
                            if value_text in ("normal", "low"):
                                return "Normal Demand"
                            if value_text == "1":
                                return "High Demand"
                            if value_text == "0":
                                return "Normal Demand"

                            return f"Class {value}"

                        predicted_label = demand_label(cls_prediction)

                        st.success(
                            f"Predicted demand: **{predicted_label}**"
                        )
                        st.caption(
                            f"Raw model output: {cls_prediction}"
                        )

                        if hasattr(
                            classification_model, "predict_proba"
                        ):
                            probabilities = (
                                classification_model.predict_proba(x_cls)[0]
                            )

                            if not model_classes:
                                model_classes = list(
                                    range(len(probabilities))
                                )

                            probability_df = pd.DataFrame({
                                "Demand Class": [
                                    demand_label(value)
                                    for value in model_classes
                                ],
                                "Predicted Probability": [
                                    f"{probability:.2%}"
                                    for probability in probabilities
                                ],
                            })

                            st.dataframe(
                                probability_df,
                                use_container_width=True,
                                hide_index=True,
                            )

                    except Exception as exc:
                        st.error(
                            "The classifier input does not match the "
                            "saved model's expected features or "
                            f"preprocessing: {exc}"
                        )
                        st.info(
                            "The classification model may require the "
                            "exact training-time one-hot encoded feature "
                            "matrix. Save its preprocessing pipeline with "
                            "the classifier to enable reliable live "
                            "predictions."
                        )

with forecast_tab:
    st.subheader("Time-Series Demand Trends and Forecast")
    st.write(
        "The chart aggregates the filtered records into daily mean demand. "
        "The optional forecast uses a simple exponential-smoothing model and is "
        "computed from the selected records."
    )
    ts_data = filtered.dropna(subset=["StartDatetime"]).copy()
    if ts_data.empty:
        st.info("No valid timestamps are available for time-series analysis.")
    else:
        daily_series = (
            ts_data.set_index("StartDatetime")["Demand"]
            .resample("D").mean().dropna().sort_index()
        )
        if len(daily_series) < 10:
            st.info("At least 10 daily observations are recommended for a meaningful forecast.")
        else:
            hist = pd.DataFrame({
                "Date": daily_series.index,
                "Average Demand": daily_series.values,
                "Series": "Observed",
            })
            st.plotly_chart(
                px.line(hist, x="Date", y="Average Demand", title="Daily Average Demand"),
                use_container_width=True,
            )
            horizon = st.slider("Forecast horizon (days)", 7, 60, 14)
            if st.button("Generate forecast", type="primary"):
                try:
                    from statsmodels.tsa.holtwinters import ExponentialSmoothing
                    fit = ExponentialSmoothing(
                        daily_series, trend="add", seasonal=None,
                        initialization_method="estimated"
                    ).fit()
                    forecast_values = fit.forecast(horizon)
                    forecast_df = pd.DataFrame({
                        "Date": forecast_values.index,
                        "Average Demand": forecast_values.values,
                        "Series": "Forecast",
                    })
                    combined = pd.concat([hist, forecast_df], ignore_index=True)
                    fig = px.line(
                        combined, x="Date", y="Average Demand", color="Series",
                        title=f"Observed Demand and {horizon}-Day Forecast",
                        color_discrete_map={"Observed": "#2878B5", "Forecast": "#E07A28"},
                    )
                    fig.update_layout(hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download forecast CSV",
                        data=forecast_df.to_csv(index=False).encode("utf-8"),
                        file_name="ev_demand_forecast.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.error(f"Could not generate forecast: {exc}")
                    st.info("Install statsmodels in the project environment if it is missing.")

with explain_tab:
    st.subheader("Feature Importance and Model Explanations")
    importance_candidates = [
        REPORTS_PATH / "tuned_xgboost_feature_importance.csv",
        REPORTS_PATH / "lime_explanation.csv",
    ]
    importance_found = False
    for path in importance_candidates:
        if path.exists():
            importance_found = True
            show_report(path, path.stem.replace("_", " ").title())
            try:
                explanation_df = load_report(str(path))
                numeric_columns = explanation_df.select_dtypes(include=np.number).columns.tolist()
                name_column = find_column(
                    explanation_df,
                    ["feature", "feature_name", "variable", "term", "explanation", "name"]
                )
                score_column = find_column(
                    explanation_df,
                    ["importance", "importance_score", "mean_abs_shap", "shap_value",
                     "weight", "coefficient", "contribution", "score"]
                )
                if name_column is None:
                    non_numeric = explanation_df.select_dtypes(exclude=np.number).columns.tolist()
                    name_column = non_numeric[0] if non_numeric else None
                if score_column is None:
                    possible = [c for c in numeric_columns if c != name_column]
                    score_column = possible[0] if possible else None
                if name_column and score_column and len(explanation_df):
                    plot_df = explanation_df[[name_column, score_column]].copy()
                    plot_df[score_column] = pd.to_numeric(plot_df[score_column], errors="coerce")
                    plot_df = plot_df.dropna().sort_values(score_column, ascending=False).head(20)
                    if not plot_df.empty:
                        fig = px.bar(
                            plot_df.sort_values(score_column),
                            x=score_column, y=name_column, orientation="h",
                            title=f"Top explanation values — {path.stem.replace('_', ' ').title()}",
                        )
                        st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
    if not importance_found:
        st.info(
            "No feature-importance or LIME explanation CSV files were found in reports/. "
            "Generate and save these artifacts from the explainability notebook to display them here."
        )
    st.caption(
        "LIME explanations are local to the examples explained in the notebook. "
        "Do not interpret a single local explanation as a global causal relationship."
    )

with reports_tab:
    st.subheader("Model Performance Comparison")
    report_files = [
        ("Regression model comparison", REPORTS_PATH / "regression_model_comparison.csv"),
        ("Classification model comparison", REPORTS_PATH / "final_model_comparison.csv"),
        ("All classification models", REPORTS_PATH / "all_classification_models.csv"),
        ("Boosting model comparison", REPORTS_PATH / "boosting_model_comparison.csv"),
        ("Resampling comparison", REPORTS_PATH / "resampling_comparison.csv"),
        ("Tuned XGBoost results", REPORTS_PATH / "tuned_xgboost_results.csv"),
        ("XGBoost best parameters", REPORTS_PATH / "xgboost_best_parameters.csv"),
    ]
    found_reports = [(title, path) for title, path in report_files if path.exists()]
    if found_reports:
        for title, path in found_reports:
            with st.expander(title, expanded=(title in ("Regression model comparison", "Classification model comparison"))):
                show_report(path, title)
    else:
        st.info("No recognized model reports were found in the reports folder.")

    st.markdown("#### Saved model metadata")
    for metadata_file in (
        MODELS_PATH / "best_regression_metadata.json",
        MODELS_PATH / "tuned_xgboost_metadata.json",
    ):
        if metadata_file.exists():
            with st.expander(metadata_file.name):
                st.json(read_json(metadata_file))

with data_tab:
    st.subheader("Filtered Charging Records")
    display_df = filtered.copy()
    display_df["ChargerCompany"] = display_df["ChargerCompany"].map(company_label_map)
    display_df["ChargerType"] = display_df["ChargerType"].map(type_label_map)
    st.caption(f"Showing the first 500 of {len(display_df):,} filtered records.")
    st.dataframe(display_df.head(500), use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered records as CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_ev_charging_records.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Project note: company and charger-type names are not verified in the supplied data. "
    "Predictions are enabled only when saved model artifacts and metadata are compatible "
    "with the feature inputs provided by this dashboard."
)
