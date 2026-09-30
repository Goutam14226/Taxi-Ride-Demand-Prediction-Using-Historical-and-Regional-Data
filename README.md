<div align="center">

# 🚕 NYC Taxi Ride Demand Prediction

### Historical demand analysis • Geographic regions • Time-series features • Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)

**Forecasting taxi pickup demand across NYC regions using historical Yellow Taxi trip records.**

</div>

---

## 📌 Project Overview

This project explores NYC taxi pickup patterns and builds a baseline machine-learning workflow to forecast demand in geographic regions. The notebooks cover exploratory data analysis, outlier handling, geographic clustering, demand aggregation, feature engineering, baseline modelling, and model-selection experiments.

> **Dataset scope:** NYC Yellow Taxi trip data for January, February, and March 2016. This is a taxi-demand project—not an Uber trip dataset.

## 🎯 Problem Statement

Given historical pickup counts and time-related information, estimate the number of taxi pickups expected in a region during a future 15-minute interval. Such forecasts can support fleet positioning and operational planning.

## 📊 Exploratory Data Analysis

The EDA notebook investigates trip fields, distributions, pickup timing, and demand patterns. The visualizations below are extracted from the project notebooks.

### Pickup demand by hour and day of week

The hourly profiles show demand changing substantially throughout the day, with lower activity in the early morning and higher activity later in the day. The profiles also differ by weekday.

![Pickup demand by hour and weekday](images/demand-by-hour-and-weekday.png)

### Pickup-hour distribution

![Pickup-hour distribution](images/pickup-hour-distribution.png)

### Geographic distribution of pickups

Pickup coordinates are concentrated in Manhattan and nearby areas, with additional pickup locations across the wider NYC region.

![NYC pickup geography](images/nyc-pickup-geography.png)

## 🧹 Data Preparation & Feature Engineering

- Reviewed and removed/handled outliers in trip-related fields, including trip distance and fare amount.
- Scaled geographic coordinates before clustering.
- Used **MiniBatchKMeans** to divide pickup locations into **30 geographic regions**. The notebook includes a diagnostic exploration of several candidate cluster counts; the final value is set to 30 in the workflow.
- Aggregated pickups by region into **15-minute intervals**.
- Created historical demand features, including lagged pickup counts and an exponentially weighted moving average (EWMA; `alpha = 0.4`).
- Added calendar features such as day of week and month, and encoded categorical variables for modelling.

### Outlier inspection

The following notebook plots illustrate the distributions inspected during outlier handling.

| Trip distance | Fare amount |
|:--:|:--:|
| ![Trip distance boxplot](images/trip-distance-outliers.png) | ![Fare amount boxplot](images/fare-amount-outliers.png) |

## 🤖 Modelling Workflow

The baseline notebook uses lagged demand and calendar/geographic features. The data is split chronologically: January–February for training and March for testing. The baseline model is **Linear Regression**.

The model-selection notebook explores multiple regression approaches, including Linear Regression, Random Forest, Gradient Boosting, and XGBoost, with hyperparameter-search experiments and experiment tracking. The ZIP does not include the `models/` artifacts or all required data files, so saved-model inference and every experiment result cannot be reproduced from this archive alone.

### Baseline result reported in the notebook

| Metric | Reported value |
|---|---:|
| Training MAPE | 8.78% |
| Test MAPE | 7.93% |

These are the values reported by the baseline notebook; they should be interpreted in the context of the notebook’s preprocessing and evaluation setup.

## 🗂️ Repository Structure

```text
Taxi-Ride-Demand-Prediction-Using-Historical-and-Regional-Data-main/
├── EDA-Demand-Prediction.ipynb
├── Removing Outliers.ipynb
├── Creating-Historical-Data.ipynb
├── Breaking_NYC_to_Regions.ipynb
├── Training-Baseline-Model.ipynb
├── Model-Selection.ipynb
├── Plot-Map.ipynb
├── Taxi_Ride_Demand_Prediction_Report.pdf
├── README.md
└── images/
    ├── demand-by-hour-and-weekday.png
    ├── pickup-hour-distribution.png
    ├── nyc-pickup-geography.png
    ├── trip-distance-outliers.png
    └── fare-amount-outliers.png
```

## 🧰 Tech Stack

- **Python**
- **Pandas, NumPy** — data processing
- **Matplotlib, Seaborn, Plotly** — visualization
- **Scikit-learn** — preprocessing, clustering, and regression
- **XGBoost / Optuna / MLflow-DagsHub** — model-selection experiments and tracking, as used in the notebooks
- **Jupyter Notebook** — experimentation

## 🚀 Running the Notebooks

1. Clone or download this repository.
2. Install the libraries imported by the notebooks in your Python environment.
3. Place the required source datasets in the expected locations and update file paths if needed.
4. Run the notebooks in workflow order, starting with EDA and preprocessing, then region creation and modelling.

**Note:** The supplied ZIP does not contain the original `data/` or `models/` directories. Consequently, the notebooks are not guaranteed to run end-to-end without restoring those files and adjusting paths.

## ⚠️ Notes & Limitations

- The geographic clusters represent **spatial regions**, not learned demand-behaviour categories.
- In the historical-data workflow, zero pickup counts are replaced with `10`; this changes true zero-demand observations and should be reconsidered before production use.
- The final region count is set to 30 rather than being selected automatically by the diagnostic plot.
- Weather, traffic, and event information are possible future extensions; they are not included as implemented model features in the supplied notebooks.
- The ZIP does not include the trained model artifacts required by the map/inference notebook.

## 🔮 Future Improvements

- Compare the baseline against tuned tree-based models using a consistent held-out evaluation.
- Revisit the treatment of zero-demand intervals and outliers.
- Add weather, traffic, and event data if suitable sources are available.
- Package preprocessing and prediction into reusable scripts or an API.
- Add reproducible environment/dependency files and include the required data/model artifact instructions.

---

<div align="center">

**Built as a machine-learning project for NYC taxi demand analysis and forecasting.**

</div>
