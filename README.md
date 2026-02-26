# Energy Consumption Forecasting & Anomaly Detection

## Overview

This project presents an end-to-end machine learning system for forecasting household energy consumption and detecting anomalous usage patterns using residual-based analysis.

The system combines time-series feature engineering, supervised regression modeling, and a multi-page Streamlit dashboard to provide:

* Short-horizon energy forecasts (24 hours and 7 days)
* Residual-based anomaly detection
* Interactive data exploration
* CSV upload and scoring functionality

The goal is to simulate a real-world energy monitoring system that could be deployed in smart home or industrial monitoring environments.

---

## Problem Statement

Energy consumption patterns often follow cyclical and seasonal trends. However, sudden spikes or drops may indicate:

* Equipment malfunction
* Abnormal operational behavior
* Data quality issues
* Unexpected usage events

This project addresses two key objectives:

1. Forecast future energy usage based on historical patterns.
2. Detect anomalies by identifying unusually large deviations between predicted and actual values.

---

## Dataset

Dataset: Individual Household Electric Power Consumption
Source: UCI Machine Learning Repository

Target variable:

* `Global_active_power`

Processing steps:

* Original 1-minute resolution resampled to hourly frequency
* Cleaned missing values
* Converted datetime column
* Saved processed dataset for modeling and inference

The final working dataset contains over 34,000 hourly observations.

---

## Modeling Approach

### Baselines

* Naive forecast (lag-1)
* Seasonal naive (lag-24)

### Machine Learning Models

* Ridge Regression
* Random Forest Regressor
* Histogram Gradient Boosting Regressor (final selected model)

The final deployed model is a tree-based gradient boosting model trained on engineered lag and rolling features.

Time-based splitting was used:

* 70% training
* 15% validation
* 15% testing

This ensures no future data leakage.

---

## Feature Engineering Logic

The model relies on time-series derived features:

### Time-based features

* Hour of day (cyclical encoding: sine and cosine)
* Day of week
* Month

### Lag features

* lag_1
* lag_24
* lag_168

### Rolling statistics

* rolling_mean_24
* rolling_std_24
* rolling_mean_168

These features allow the model to capture:

* Daily cycles
* Weekly seasonality
* Short-term momentum
* Medium-term trend structure

---

## Evaluation Results

Model performance (validation set):

* MAE ≈ 0.25
* RMSE ≈ 0.33
* MAPE ≈ 30–35%

The tree-based boosting model outperformed linear regression and baseline methods.

---

## Anomaly Detection Logic

Anomalies are detected using residual-based thresholding.

Residual:
| Actual − Forecast |

Threshold strategy:

* Percentile-based (P95 of validation residuals)

If:
Residual > Threshold
→ Flag as anomaly

Severity score:
Residual / Threshold

This allows ranking anomalies by magnitude rather than binary detection alone.

---

## Streamlit Application

The application includes four functional pages:

### Overview

* Trend visualization
* Peak hour detection
* Weekly average usage
* Top 5% load alert

### Forecast

* 24-hour and 7-day backtest forecasts
* Performance metrics (MAE, RMSE, MAPE)
* Confidence band visualization

### Anomalies

* Highlighted anomaly points on time series
* Severity ranking table
* Downloadable anomaly report

### Upload & Score

* CSV upload
* Column validation
* Automatic feature generation
* Anomaly scoring
* Downloadable scored report

---

## Project Structure

```
Energy-Forecasting-Anomaly-Detection/
│
├── app/
│   ├── main.py
│   ├── utils.py
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Forecast.py
│       ├── 3_Anomalies.py
│       └── 4_Upload.py
│
├── models/
│   ├── forecast_pipeline_v1.joblib
│   └── threshold_v1.json
│
├── data/
│   └── processed/
│
├── notebooks/
└── README.md
```

---

## How to Run

### 1. Clone repository

```
git clone https://github.com/Saiful-Codes/ML-Projects
cd Energy-Forecasting-Anomaly-Detection
```

### 2. Create virtual environment

```
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```
.venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run Streamlit app

```
streamlit run app/main.py
```

---

## Future Improvements

* Probabilistic forecasting
* Advanced anomaly scoring (Isolation Forest / LSTM)
* Real-time streaming input
* Cloud deployment
* Model monitoring dashboard

---

## Author

Saiful Islam Shihab
Bachelor of Computer Science (Artificial Intelligence Major)
La Trobe University

This project was developed as part of a structured machine learning portfolio roadmap focused on building industry-ready systems.

---
