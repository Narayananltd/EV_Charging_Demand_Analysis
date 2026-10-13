
# EV Charging Demand Analysis

## Project Overview

The EV Charging Demand Analysis project explores electric vehicle charging-session data to understand charging demand patterns and build machine learning models for demand prediction.

The project includes data cleaning, exploratory data analysis, feature engineering, model evaluation, and an interactive Streamlit dashboard.

## Project Objectives

- Understand EV charging-session data and its quality.
- Clean and prepare charging records for analysis.
- Explore demand patterns by hour, month, weekday, and other features.
- Engineer features for machine learning.
- Predict charging demand using regression.
- Classify sessions into demand categories using classification.
- Present insights and model results through an interactive dashboard.

## Key Features

- Data understanding and cleaning
- Exploratory data analysis (EDA)
- Feature engineering
- Regression-based demand prediction
- Classification-based demand prediction
- Time-series analysis and forecasting
- Model explainability
- Model performance comparison
- Saved model metadata
- Interactive data exploration

## Dataset

The project uses EV charging-session records containing fields such as:

- UserID
- ChargerID
- ChargerCompany
- Location
- ChargerType
- StartDay and StartTime
- EndDay and EndTime
- StartDatetime and EndDatetime
- Duration
- Demand

The cleaned dataset contains **72,826 records** after removing 13 exact duplicate rows and 17 records with negative durations. Zero-duration records were retained during cleaning.

Additional engineered features are used for analysis and modeling.

## Machine Learning Models

### Regression

The saved best regression model is a LightGBM regressor. It predicts charging demand using the project's saved preprocessing transformer and trained model.

### Classification

The saved classification model is a tuned XGBoost classifier. It predicts the demand class and can display class probabilities.

The project uses a median demand threshold of approximately **14.10** to define demand classes.

## Dashboard

The Streamlit dashboard includes:

1. Overview
2. Predictions
3. Time Series
4. Explainability
5. Model Reports
6. Data Explorer

The dashboard supports interactive visualizations, model predictions, performance reports, and filtered charging-record exploration.

## Project Structure

The project contains notebooks, datasets, model artifacts, reports, and a Streamlit dashboard.

Key paths include:

- `data/processed/` — cleaned charging data
- `models/` — saved models and metadata
- `reports/` — generated analysis and model reports
- `dashboard/app.py` — Streamlit dashboard
- `README.md` — project documentation
- `requirements.txt` — Python dependencies

The exact folder structure may include additional files and notebooks.

## Environment Setup

This project was developed using Python and VS Code on Windows.

### 1. Open the project folder

Open `EV_Charging_Demand_Analysis` in VS Code.

### 2. Activate the Python environment

If the existing `.venv314` environment is available, activate it in PowerShell:

```powershell
.\.venv314\Scripts\Activate.ps1
```

Alternatively, run Python directly through that environment without activating it.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the dashboard

From the project root directory:

```powershell
python -m streamlit run dashboard/app.py --server.port 8503
```

Then open the local URL displayed in the terminal, usually `http://localhost:8503`.

If you are using the existing `.venv314` environment, you can run:

```powershell
.\.venv314\Scripts\python.exe -m streamlit run dashboard/app.py --server.port 8503
```

## Reproducibility

The project uses saved model artifacts and metadata for predictions and model reporting. Keep the required model files and data files available in their expected locations.

For a fresh installation, use compatible package versions and verify that all required model artifacts are included or otherwise made available.

## Future Improvements

- Improve demand forecasting and model validation.
- Expand model monitoring and explainability.
- Add more interactive filtering and visualizations.
- Improve deployment and reproducibility.

## Author

EV Charging Demand Analysis — Machine Learning Mini Project