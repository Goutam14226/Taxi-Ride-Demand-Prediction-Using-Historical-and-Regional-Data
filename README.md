<div align="center">

# 🚕 Taxi Ride Demand Prediction
### Forecasting Urban Taxi Demand Using Historical and Geographic Data

**A Machine Learning Project | New York City Yellow Taxi Data | Time-Series Forecasting**

<img src="https://commons.wikimedia.org/wiki/Special:FilePath/Yellow_cab.JPG" alt="Yellow taxi in New York City" width="850"/>

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange?style=for-the-badge)
![Time Series](https://img.shields.io/badge/Time%20Series-Forecasting-8A2BE2?style=for-the-badge)
![Geospatial](https://img.shields.io/badge/Geospatial-K--Means-2E8B57?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Research%20%26%20Development-blue?style=for-the-badge)

</div>

---

## 📌 Table of Contents

- [1. Project Overview](#-1-project-overview)
- [2. Real-World Problem](#-2-real-world-problem)
- [3. Project Objectives](#-3-project-objectives)
- [4. Project at a Glance](#-4-project-at-a-glance)
- [5. Dataset](#-5-dataset)
- [6. Complete Project Workflow](#-6-complete-project-workflow)
- [7. Exploratory Data Analysis](#-7-exploratory-data-analysis)
- [8. Data Cleaning and Outlier Removal](#-8-data-cleaning-and-outlier-removal)
- [9. Geographic Segmentation Using K-Means](#-9-geographic-segmentation-using-k-means)
- [10. Creating Historical Demand Data](#-10-creating-historical-demand-data)
- [11. EWMA Smoothing](#-11-ewma-smoothing)
- [12. Feature Engineering](#-12-feature-engineering)
- [13. Baseline Model: Linear Regression](#-13-baseline-model-linear-regression)
- [14. Model Selection and Hyperparameter Tuning](#-14-model-selection-and-hyperparameter-tuning)
- [15. Model Evaluation](#-15-model-evaluation)
- [16. Geographic Visualization](#-16-geographic-visualization)
- [17. Technology Stack](#-17-technology-stack)
- [18. Repository Structure](#-18-repository-structure)
- [19. Installation and Setup](#-19-installation-and-setup)
- [20. How to Run the Project](#-20-how-to-run-the-project)
- [21. Results](#-21-results)
- [22. Limitations](#-22-limitations)
- [23. Future Improvements](#-23-future-improvements)
- [24. Key Learnings](#-24-key-learnings)
- [25. References](#-25-references)

---

## 🚀 1. Project Overview

Taxi Ride Demand Prediction is a machine learning project designed to analyze and forecast taxi pickup demand across different geographic regions of New York City.

Taxi demand varies across both **time and location**. Some areas experience more pickups during working hours, while others may have higher demand during evenings or other periods.

To capture these variations, this project combines geographic clustering with time-series analysis and machine learning.

The project uses historical NYC Yellow Taxi trip records from January to March 2016. Pickup locations are grouped into **30 geographic regions** using MiniBatch K-Means. The pickup activity in each region is then aggregated into 15-minute intervals to create a regional demand time series.

Historical demand features, lag values, and calendar information are used to train regression models and estimate pickup demand.

### ✨ Key Highlights

- Exploratory Data Analysis of NYC taxi trip records
- Data cleaning and geographic outlier removal
- Geographic segmentation into 30 regions
- MiniBatch K-Means with incremental data processing
- 15-minute regional pickup demand aggregation
- Exponentially Weighted Moving Average (EWMA) smoothing
- Lag-based and calendar-based feature engineering
- Linear Regression baseline model
- Comparison of multiple machine learning models
- Hyperparameter optimization using Optuna
- Experiment tracking using MLflow and DagsHub
- Geographic visualization of taxi demand and predictions

> **Dataset clarification:** This project uses NYC Yellow Taxi trip data, not Uber trip data.

---

## 🌍 2. Real-World Problem

<div align="center">

<img src="https://commons.wikimedia.org/wiki/Special:FilePath/Taxi-cabs-New-York-0986.jpg" alt="Yellow taxis operating in New York City" width="850"/>

<em>Taxi demand is influenced by both where and when people need transportation.</em>

</div>

Taxi demand is not uniformly distributed across a city.

Different geographic areas can have different pickup patterns. Demand also changes throughout the day, across weekdays and weekends, and over time.

If taxi operators can estimate demand for different areas in advance, they may be able to make better decisions about fleet positioning and driver allocation.

### The challenges

<table>
<tr>
<th>Challenge</th>
<th>Description</th>
</tr>
<tr>
<td bgcolor="#E8F1FF"><b>Geographic variation</b></td>
<td>Pickup demand differs across locations within the city.</td>
</tr>
<tr>
<td bgcolor="#FFF1D6"><b>Temporal variation</b></td>
<td>Demand changes across time intervals and calendar periods.</td>
</tr>
<tr>
<td bgcolor="#FCE7F3"><b>Noisy data</b></td>
<td>Invalid coordinates and extreme observations can affect analysis.</td>
</tr>
<tr>
<td bgcolor="#E8F7E9"><b>Demand fluctuations</b></td>
<td>Pickup counts can vary considerably between consecutive intervals.</td>
</tr>
<tr>
<td bgcolor="#F0E8FF"><b>Forecasting complexity</b></td>
<td>Historical demand patterns must be converted into useful predictive features.</td>
</tr>
</table>

### 🎯 Business applications

- Taxi fleet positioning
- Driver allocation and dispatch planning
- Identifying high-demand areas
- Demand-aware resource allocation
- Urban mobility analysis
- Transportation planning

**The objective is to estimate how many taxi pickups may occur in a geographic region during a future time interval.**

---

## 🎯 3. Project Objectives

The project aims to:

1. Understand the geographic and temporal characteristics of taxi pickup data.
2. Clean the raw data and handle unsuitable observations.
3. Divide NYC into geographic regions using clustering.
4. Convert individual trips into regional demand time series.
5. Smooth demand fluctuations using EWMA.
6. Create lag-based and calendar-based predictive features.
7. Train a baseline regression model.
8. Compare alternative machine learning models.
9. Tune model hyperparameters and track experiments.
10. Evaluate forecasting performance on a later time period.
11. Visualize geographic locations and demand predictions.

---

## 📊 4. Project at a Glance

<div align="center">

<table>
<tr>
<td align="center" bgcolor="#E8F1FF">
<h3>🚕 Data</h3>
NYC Yellow Taxi<br/>Trip Records
</td>
<td align="center" bgcolor="#FFF1D6">
<h3>🗓️ Period</h3>
January–March<br/>2016
</td>
</tr>
<tr>
<td align="center" bgcolor="#E8F7E9">
<h3>🗺️ Regions</h3>
30 Geographic<br/>Clusters
</td>
<td align="center" bgcolor="#F0E8FF">
<h3>⏱️ Interval</h3>
15-Minute<br/>Demand
</td>
</tr>
<tr>
<td align="center" bgcolor="#FCE7F3">
<h3>🤖 Baseline</h3>
Linear<br/>Regression
</td>
<td align="center" bgcolor="#E1F5F5">
<h3>📈 Evaluation</h3>
MAPE, MAE<br/>and RMSE
</td>
</tr>
</table>

</div>

| Component | Implementation |
|---|---|
| Data source | NYC Yellow Taxi trip records |
| Geographic features | Pickup latitude and longitude |
| Clustering | MiniBatch K-Means |
| Number of regions | 30 |
| Demand interval | 15 minutes |
| Smoothing | EWMA, alpha = 0.4 |
| Baseline model | Linear Regression |
| Additional models | Ridge, Random Forest, Gradient Boosting, XGBoost |
| Hyperparameter tuning | Optuna |
| Experiment tracking | MLflow and DagsHub |
| Baseline train period | January–February 2016 |
| Baseline test period | March 2016 |

---

## 📊 Project Visualizations

### 1. NYC Taxi Pickup Demand Analysis

This visualization shows the exploratory analysis of taxi pickup demand.

![Taxi Demand Analysis](images/eda-demand-prediction-cell-71.png)

### 2. NYC Pickup Locations Map

A geographical visualization of taxi pickup locations across New York City.

![NYC Pickup Locations](images/plot-map-cell-16.png)

### 3. NYC Regions Visualization

Visualization of the geographical regions created for the taxi demand prediction project.

![NYC Regions](images/breaking-nyc-to-regions-cell-36.png)

## 🔄 5. Complete Project Workflow

The project follows a sequential machine learning pipeline, starting from raw taxi records and ending with model evaluation and geographic visualization.

<div align="center">

<table>
<tr>
<td align="center" bgcolor="#DCEBFF">

### 🔵 STEP 1 — RAW DATA

NYC Yellow Taxi Trip Records

</td>
</tr>
<tr><td align="center">⬇️</td></tr>
<tr>
<td align="center" bgcolor="#FFE8CC">

### 🟠 STEP 2 — DATA PREPARATION

Exploratory Data Analysis<br/>
Cleaning and Outlier Removal

</td>
</tr>
<tr><td align="center">⬇️</td></tr>
<tr>
<td align="center" bgcolor="#DDF5E1">

### 🟢 STEP 3 — GEOGRAPHIC CLUSTERING

MiniBatch K-Means<br/>
30 Geographic Regions

</td>
</tr>
<tr><td align="center">⬇️</td></tr>
<tr>
<td align="center" bgcolor="#EDE2FF">

### 🟣 STEP 4 — DEMAND AGGREGATION

15-Minute Pickup Counts<br/>
For Each Region

</td>
</tr>
<tr><td align="center">⬇️</td></tr>
<tr>
<td align="center" bgcolor="#FCE0EB">

### 🔴 STEP 5 — FEATURE ENGINEERING

EWMA Smoothing<br/>
Lag Features and Calendar Features

</td>
</tr>
<tr><td align="center">⬇️</td></tr>
<tr>
<td align="center" bgcolor="#DDF4F4">

### 🩵 STEP 6 — MODELING

Linear Regression<br/>
Model Selection and Tuning

</td>
</tr>
<tr><td align="center">⬇️</td></tr>
<tr>
<td align="center" bgcolor="#FFF2BF">

### 🟡 STEP 7 — EVALUATION

Forecasting Metrics<br/>
Train-Test Comparison

</td>
</tr>
<tr><td align="center">⬇️</td></tr>
<tr>
<td align="center" bgcolor="#E5E7EB">

### ⚫ STEP 8 — VISUALIZATION

Geographic Demand Analysis<br/>
Map-Based Results

</td>
</tr>
</table>

</div>

### Workflow summary

**Raw Data → Cleaning → Geographic Regions → Demand Time Series → Feature Engineering → Model Training → Evaluation → Visualization**

---

## 🔎 6. Dataset

The project uses NYC Yellow Taxi trip records for three months in 2016:

- January 2016
- February 2016
- March 2016

### Data source

The dataset is available through the official NYC Taxi and Limousine Commission website:

**[NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)**

The source data contains trip-level information, including pickup and drop-off timestamps, pickup and drop-off locations, trip distance, fare information, and other trip attributes, depending on the data version.

### Important fields

| Field | Purpose |
|---|---|
| Pickup datetime | Determines when a taxi pickup occurred |
| Pickup latitude | Geographic latitude of the pickup |
| Pickup longitude | Geographic longitude of the pickup |
| Region | Geographic cluster assigned to the pickup |
| Total pickups | Number of pickups in a time interval |

The geographic and temporal information is used to construct the regional demand dataset.

### Why historical data?

Historical pickup records allow the model to learn recurring patterns and relationships between recent demand, time, and geographic location.

---

## 📈 7. Exploratory Data Analysis

**Notebook:** `EDA-Demand-Prediction.ipynb`

Exploratory Data Analysis (EDA) is the first stage of the project.

Before training any model, the raw data needs to be understood.

### What is explored?

- Dataset structure and data types
- Missing values
- Pickup datetime information
- Geographic coordinate distributions
- Unusual or invalid coordinate values
- Pickup activity across the available period
- Potential data-quality issues

### Why is EDA important?

EDA helps identify problems that may affect later stages of the pipeline.

For example, invalid geographic coordinates can distort K-Means clusters, while incorrect timestamps can lead to inaccurate demand aggregation.

The findings from EDA guide the cleaning, clustering, and feature engineering stages.

---

## 🧹 8. Data Cleaning and Outlier Removal

**Notebook:** `Removing Outliers.ipynb`

Raw taxi trip records may contain invalid coordinates, unusual geographic observations, or records outside the intended analysis area.

These observations need to be examined before geographic clustering.

### Why remove unsuitable observations?

K-Means uses distances between observations and cluster centers. Extreme coordinate values can pull cluster centers away from the main geographic distribution.

Cleaning the data helps the clustering algorithm focus on the intended NYC pickup locations.

### Cleaning workflow

<table>
<tr>
<td bgcolor="#FFF1D6">

**1. Inspect the data**

Identify unusual values and potential geographic outliers.

</td>
</tr>
<tr><td align="center">⬇️</td></tr>
<tr>
<td bgcolor="#FCE7F3">

**2. Filter unsuitable observations**

Remove records that do not meet the project's geographic or data-quality requirements.

</td>
</tr>
<tr><td align="center">⬇️</td></tr>
<tr>
<td bgcolor="#E8F7E9">

**3. Prepare cleaned coordinates**

Use the cleaned pickup locations for geographic clustering.

</td>
</tr>
</table>

The goal is to create a reliable geographic dataset for the next stage.

---

## 🗺️ 9. Geographic Segmentation Using K-Means

**Notebook:** `Breaking_NYC_to_Regions.ipynb`

A major component of this project is dividing New York City into **30 geographic regions**.

Instead of forecasting demand for the entire city as a single unit, the project assigns each pickup location to a geographic cluster.

### Why geographic segmentation?

Demand can vary significantly across different parts of a city.

By creating geographic regions, the project can represent pickup demand separately for different locations.

### Why K-Means?

K-Means groups observations based on their distance from cluster centers.

In this project, the clustering inputs are:

- Pickup latitude
- Pickup longitude

Each cluster represents a geographic grouping of pickup locations.

### Why MiniBatch K-Means?

The project uses `MiniBatchKMeans`, which updates cluster centers using batches of observations.

This approach is useful when working with large datasets because it can reduce the computational cost of fitting standard K-Means on the complete dataset.

### Clustering configuration

| Parameter | Value |
|---|---|
| Algorithm | MiniBatch K-Means |
| Number of clusters | 30 |
| `n_init` | 10 |
| `random_state` | 42 |
| Feature scaler | StandardScaler |
| Incremental scaling chunk size | 100,000 |

### Incremental preprocessing

The project uses `StandardScaler` with `partial_fit` on chunks of 100,000 observations.

The clustering model is also trained incrementally using `MiniBatchKMeans.partial_fit`.

This avoids requiring the complete coordinate dataset to be processed in one batch during fitting.

### How a pickup gets its region

Once the clustering model is trained, pickup coordinates are transformed using the fitted scaler.

The trained K-Means model then predicts the corresponding cluster label.

Conceptually:

```python
# Pickup coordinates
coordinates = [[40.75, -73.98]]

# Apply the fitted scaler
scaled_coordinates = scaler.transform(coordinates)

# Assign a geographic region
region = kmeans.predict(scaled_coordinates)
```

This is a simplified illustration of the process, not a verbatim copy of the notebook.

### Choosing the number of clusters

The notebook investigates several cluster counts, including:

`10, 20, 30, 40, 50, 60, 70, 80, 90`

A diagnostic examines distances between cluster centers and counts centers meeting a specified distance range.

The final model uses **30 clusters**. The final value is explicitly configured in the implementation rather than being automatically selected by the diagnostic.

> **Important:** These are geographic clusters based on coordinates. They should not automatically be interpreted as official NYC neighborhoods or as groups with similar passenger behavior.

---

## ⏱️ 10. Creating Historical Demand Data

**Notebook:** `Creating-Historical-Data.ipynb`

After assigning pickup locations to geographic regions, the next step is to transform individual taxi trips into a structured demand time series.

The project aggregates pickups into **15-minute intervals for each region**.

### From individual trips to demand counts

Imagine that several taxi pickups occur in Region 1 between 09:00 and 09:15.

Instead of modeling every trip separately, the project counts the number of pickups during that interval.

| Time interval | Region | Total pickups |
|---|---|---:|
| 09:00–09:15 | Region 1 | 42 |
| 09:15–09:30 | Region 1 | 37 |
| 09:30–09:45 | Region 1 | 51 |
| 09:45–10:00 | Region 1 | 46 |

These numbers are illustrative examples.

The target variable is:

`total_pickups`

### Why 15-minute intervals?

A 15-minute interval provides a relatively fine-grained view of demand changes.

It allows the project to capture short-term fluctuations without treating every individual pickup as a separate prediction target.

### Why aggregate by region?

The same number of pickups can have different operational implications depending on where they occur.

Regional aggregation makes it possible to study demand separately across geographic areas.

### Output of this stage

A time-series dataset containing pickup counts for each region and time interval.

This dataset becomes the foundation for smoothing and feature engineering.

---

## 📉 11. EWMA Smoothing

After creating the regional demand time series, the project applies **Exponentially Weighted Moving Average (EWMA)** smoothing.

The smoothing parameter used is:

`alpha = 0.4`

### What is EWMA?

EWMA is a smoothing technique that assigns more weight to recent observations while retaining information from earlier observations.

Its recurrence is:

\[
S_t = \alpha X_t + (1-\alpha)S_{t-1}
\]

Where:

- \(X_t\) is the observed pickup count at time \(t\)
- \(S_t\) is the smoothed value
- \(\alpha\) controls the influence of the latest observation

### How does alpha affect smoothing?

- A larger alpha gives more weight to recent observations.
- A smaller alpha produces smoother changes by retaining more influence from previous values.

### Feature created

The project creates the smoothed demand feature:

`avg_pickups`

This feature provides a representation of recent demand that is less sensitive to individual fluctuations.

### Important preprocessing decision

The notebook replaces zero pickup counts with `10` before applying smoothing.

This is an important modeling assumption because a zero pickup count may represent a genuine period with no observed demand.

Replacing zero with 10 changes the observed demand series. This decision should be evaluated carefully before using the resulting values as the forecasting target or as a feature.

---

## 🧩 12. Feature Engineering

Machine learning models require structured input features.

The project creates features that capture recent demand, calendar information, and geographic location.

### 12.1 Lag features

The baseline model uses four lag features:

| Feature | Meaning |
|---|---|
| `lag_1` | Demand from the previous 15-minute interval |
| `lag_2` | Demand from two intervals earlier |
| `lag_3` | Demand from three intervals earlier |
| `lag_4` | Demand from four intervals earlier |

For example, if the current interval is 10:00–10:15:

- `lag_1` represents demand from 09:45–10:00.
- `lag_2` represents demand from 09:30–09:45.
- `lag_3` represents demand from 09:15–09:30.
- `lag_4` represents demand from 09:00–09:15.

### Why lag features?

Recent demand can provide useful information about demand in the next interval.

Lag features allow the model to learn relationships between recent pickup activity and the target.

### 12.2 Calendar features

The baseline pipeline includes:

- `day_of_week`
- `month`

These features allow the model to represent differences associated with the day of the week and the month.

### 12.3 Geographic feature

The `region` feature identifies the geographic cluster associated with the demand observation.

This allows the model to distinguish demand patterns across regions.

### 12.4 Categorical encoding

Categorical variables such as `region` and `day_of_week` are handled using `OneHotEncoder` in the baseline pipeline.

One-hot encoding converts categorical values into numerical indicator columns that can be used by the regression model.

### Feature summary

| Feature | Category | Purpose |
|---|---|---|
| `lag_1` | Historical demand | Most recent demand |
| `lag_2` | Historical demand | Demand two intervals earlier |
| `lag_3` | Historical demand | Demand three intervals earlier |
| `lag_4` | Historical demand | Demand four intervals earlier |
| `avg_pickups` | Smoothed demand | Represents recent demand level |
| `day_of_week` | Calendar | Captures weekly variation |
| `month` | Calendar | Captures month-related variation |
| `region` | Geographic | Identifies the pickup region |

The exact feature set used by each model should be checked in its corresponding notebook.

---

## 🤖 13. Baseline Model: Linear Regression

**Notebook:** `Training-Baseline-Model.ipynb`

The first model is Linear Regression.

The purpose of a baseline is to establish a reference performance before exploring more complex algorithms.

### How Linear Regression works

Linear Regression models the relationship between input features and the target variable.

Its general form is:

\[
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_px_p
\]

Where:

- \(\hat{y}\) is the predicted pickup demand
- \(x_1, x_2, \ldots, x_p\) are input features
- \(\beta_0\) is the intercept
- \(\beta_1, \ldots, \beta_p\) are learned coefficients

The model learns coefficients that minimize the sum of squared residuals during training.

### Why start with Linear Regression?

- It provides a simple baseline.
- It is relatively easy to interpret.
- It trains quickly.
- It helps establish whether more complex models offer improvements.

### Train-test split

The baseline notebook uses a chronological split.

<table>
<tr>
<th>Dataset</th>
<th>Period</th>
<th>Purpose</th>
</tr>
<tr>
<td bgcolor="#DDF5E1"><b>Training</b></td>
<td>January–February 2016</td>
<td>Fit the model</td>
</tr>
<tr>
<td bgcolor="#DCEBFF"><b>Testing</b></td>
<td>March 2016</td>
<td>Evaluate on a later period</td>
</tr>
</table>

### Why chronological splitting?

In forecasting, the model should learn from the past and be evaluated on a later period.

Randomly shuffling time-series observations can allow future information to influence training and produce an unrealistic evaluation.

The chronological split better reflects the intended forecasting setting.

---

## 🧪 14. Model Selection and Hyperparameter Tuning

**Notebook:** `Model-Selection.ipynb`

After establishing the Linear Regression baseline, the project explores additional regression models.

The purpose is to investigate whether alternative models can capture relationships that a simple linear model may not represent adequately.

### Models explored

<table>
<tr>
<th>Model</th>
<th>Approach</th>
</tr>
<tr>
<td bgcolor="#DCEBFF"><b>Linear Regression</b></td>
<td>Models a linear relationship between input features and demand.</td>
</tr>
<tr>
<td bgcolor="#E8E1FF"><b>Ridge Regression</b></td>
<td>Linear regression with L2 regularization.</td>
</tr>
<tr>
<td bgcolor="#DDF5E1"><b>Random Forest</b></td>
<td>Combines predictions from multiple decision trees.</td>
</tr>
<tr>
<td bgcolor="#FFE8CC"><b>Gradient Boosting</b></td>
<td>Builds trees sequentially to reduce prediction errors.</td>
</tr>
<tr>
<td bgcolor="#FCE0EB"><b>XGBoost</b></td>
<td>Uses an optimized gradient-boosted tree framework.</td>
</tr>
</table>

### Hyperparameter tuning with Optuna

The project uses Optuna to explore hyperparameter configurations.

The notebook includes:

- 50 Optuna trials for model-selection experiments
- A separate 100-trial search for Ridge's `alpha` parameter
- MAPE as the optimization objective

Hyperparameter tuning aims to identify configurations that reduce forecasting error.

### Ridge regularization

Ridge Regression adds an L2 penalty to the linear regression objective.

Its objective can be written as:

\[
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
+
\alpha\sum_{j=1}^{p}\beta_j^2
\]

The parameter `alpha` controls the strength of regularization.

A larger penalty discourages large coefficient values.

### Experiment tracking with MLflow and DagsHub

The project uses MLflow and DagsHub to organize model experiments.

Experiment tracking can help record:

- Model configurations
- Hyperparameters
- Evaluation metrics
- Experiment runs
- Comparisons between candidate models

This makes model development easier to organize and reproduce.

> **Model selection note:** The exact best model and its final metrics should be reported from the actual experiment results. A model should not be declared the overall winner without a verified comparison on the same evaluation data.

---

## 📏 15. Model Evaluation

Forecasting models need to be evaluated using metrics that describe the difference between actual and predicted demand.

The project discusses metrics including MAPE, MAE, and RMSE.

### 15.1 Mean Absolute Error (MAE)

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

MAE calculates the average absolute difference between actual and predicted pickup counts.

**Interpretation:** A lower MAE means smaller average absolute prediction errors.

MAE is expressed in the same units as the target variable.

### 15.2 Root Mean Squared Error (RMSE)

\[
RMSE =
\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

RMSE squares the errors before averaging and then takes the square root.

As a result, larger errors receive more weight.

**Interpretation:** A lower RMSE indicates smaller squared-error-based forecasting loss.

### 15.3 Mean Absolute Percentage Error (MAPE)

\[
MAPE =
\frac{100}{n}
\sum_{i=1}^{n}
\left|
\frac{y_i-\hat{y}_i}{y_i}
\right|
\]

MAPE expresses the average absolute error as a percentage of actual values.

**Interpretation:** A lower MAPE indicates lower percentage-based forecasting error, subject to the limitations of the metric.

### Important MAPE limitation

MAPE is undefined when the actual value is zero and can become very large when actual values are close to zero.

This is particularly relevant to demand forecasting, where some regions or time intervals may have very low pickup counts.

Therefore, MAPE should be interpreted alongside metrics such as MAE and RMSE.

---

## 🗺️ 16. Geographic Visualization

**Notebook:** `Plot-Map.ipynb`

The final stage connects geographic locations with the trained models and visualization pipeline.

The notebook loads saved artifacts, including the scaler, K-Means model, prediction model, and encoder.

### Visualization workflow

<table>
<tr>
<td bgcolor="#DCEBFF">

**Step 1 — Load saved artifacts**

Load the previously trained preprocessing, clustering, and prediction components.

</td>
</tr>
<tr><td align="center">⬇️</td></tr>
<tr>
<td bgcolor="#DDF5E1">

**Step 2 — Process coordinates**

Transform pickup latitude and longitude using the saved scaler.

</td>
</tr>
<tr><td align="center">⬇️</td></tr>
<tr>
<td bgcolor="#EDE2FF">

**Step 3 — Assign geographic regions**

Use the trained K-Means model to identify the region for each location.

</td>
</tr>
<tr><td align="center">⬇️</td></tr>
<tr>
<td bgcolor="#FFE8CC">

**Step 4 — Generate predictions**

Prepare the required features and use the saved prediction model.

</td>
</tr>
<tr><td align="center">⬇️</td></tr>
<tr>
<td bgcolor="#FCE0EB">

**Step 5 — Visualize geographic results**

Display geographic locations and the associated results on a map.

</td>
</tr>
</table>

### Why geographic visualization?

A single forecasting metric does not explain where demand occurs.

Map-based visualization can help interpret geographic patterns and provide context for the model's predictions.

It can also support future analysis of regional errors and differences in demand.

> **Reproducibility note:** The reviewed project ZIP did not include the complete `data/` and `models/` folders. The exact saved-model predictions and final map output therefore need to be verified in the complete project environment.

---

## 🛠️ 17. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and aggregation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, clustering, regression, and evaluation |
| MiniBatch K-Means | Geographic segmentation |
| StandardScaler | Scaling geographic coordinates |
| OneHotEncoder | Encoding categorical features |
| Optuna | Hyperparameter optimization |
| XGBoost | Gradient-boosted tree modeling |
| MLflow | Experiment tracking |
| DagsHub | Experiment management |
| Matplotlib and other plotting tools | Data visualization |
| Jupyter Notebook | Interactive development and analysis |

The exact package versions should be recorded from the working environment to ensure reproducibility.

---

## 📁 18. Repository Structure

The notebooks are organized according to the project's end-to-end workflow.

```text
Taxi-Ride-Demand-Prediction-Using-Historical-and-Regional-Data/
│
├── EDA-Demand-Prediction.ipynb
│
├── Removing Outliers.ipynb
│
├── Breaking_NYC_to_Regions.ipynb
│
├── Creating-Historical-Data.ipynb
│
├── Training-Baseline-Model.ipynb
│
├── Model-Selection.ipynb
│
├── Plot-Map.ipynb
│
├── Taxi_Ride_Demand_Prediction_Report.pdf
│
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── scaler
│   ├── kmeans_model
│   ├── prediction_model
│   └── encoder
│
└── requirements.txt
```

**Note:** The `data/`, `models/`, and `requirements.txt` entries show a suggested complete repository layout. They should only be included in the actual repository when the corresponding files exist.

---

## ⚙️ 19. Installation and Setup

### Step 1 — Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Taxi-Ride-Demand-Prediction-Using-Historical-and-Regional-Data
```

Replace the placeholder with the actual repository URL.

### Step 2 — Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### Step 3 — Install dependencies

If the repository contains a verified `requirements.txt`:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies required by the notebooks and record the tested package versions.

### Step 4 — Download the dataset

Download the required NYC Yellow Taxi trip records from the official TLC website:

[NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

Place the data files in the appropriate directory and ensure that the file paths match those used in the notebooks.

### Step 5 — Prepare saved model artifacts

If you want to run the final visualization notebook, make sure the saved scaler, clustering model, prediction model, and encoder are available at the paths expected by the notebook.

---

## ▶️ 20. How to Run the Project

Run the notebooks in the following order.

<table>
<tr>
<th>Order</th>
<th>Notebook</th>
<th>Purpose</th>
</tr>
<tr>
<td>1</td>
<td><code>EDA-Demand-Prediction.ipynb</code></td>
<td>Explore the raw data</td>
</tr>
<tr>
<td>2</td>
<td><code>Removing Outliers.ipynb</code></td>
<td>Clean unsuitable observations</td>
</tr>
<tr>
<td>3</td>
<td><code>Breaking_NYC_to_Regions.ipynb</code></td>
<td>Train geographic clusters</td>
</tr>
<tr>
<td>4</td>
<td><code>Creating-Historical-Data.ipynb</code></td>
<td>Create regional demand time series</td>
</tr>
<tr>
<td>5</td>
<td><code>Training-Baseline-Model.ipynb</code></td>
<td>Train and evaluate the baseline</td>
</tr>
<tr>
<td>6</td>
<td><code>Model-Selection.ipynb</code></td>
<td>Compare models and tune hyperparameters</td>
</tr>
<tr>
<td>7</td>
<td><code>Plot-Map.ipynb</code></td>
<td>Visualize geographic results</td>
</tr>
</table>

### Before running

- Confirm that the raw dataset files are available.
- Check that notebook paths point to the correct directories.
- Ensure that each preprocessing stage produces the files required by the next stage.
- Save trained model artifacts before running the visualization notebook.
- Verify that the same preprocessing is used during training and inference.

---

## 📊 21. Results

### Baseline Linear Regression

The baseline notebook reports the following results:

<table>
<tr>
<th>Metric</th>
<th>Reported Value</th>
</tr>
<tr>
<td bgcolor="#DCEBFF"><b>Training MAPE</b></td>
<td><b>8.78%</b></td>
</tr>
<tr>
<td bgcolor="#DDF5E1"><b>Test MAPE</b></td>
<td><b>7.93%</b></td>
</tr>
</table>

The model is trained on January and February 2016 and evaluated on March 2016.

These values are the results reported by the baseline notebook.

### Interpreting the baseline

The reported test MAPE of 7.93% provides a baseline measure of percentage-based prediction error under the notebook's evaluation setup.

However, MAPE has limitations when actual pickup counts are zero or close to zero. The result should therefore be interpreted alongside the target distribution and other error metrics.

### Model selection results

The model-selection notebook explores multiple regression algorithms and hyperparameter configurations.

The final comparison should be populated using the actual saved experiment results, with all candidate models evaluated on a consistent dataset and metric definition.

| Model | Final evaluation |
|---|---|
| Linear Regression | Baseline result reported above |
| Ridge Regression | Refer to saved experiment results |
| Random Forest | Refer to saved experiment results |
| Gradient Boosting | Refer to saved experiment results |
| XGBoost | Refer to saved experiment results |

No unverified metric values or overall model winner are claimed here.

---

## ⚠️ 22. Limitations

Although the project demonstrates an end-to-end geographic demand forecasting workflow, several limitations should be considered.

### 1. Limited historical period

The analyzed data covers January to March 2016.

Taxi demand patterns may have changed since then, so these results should not automatically be treated as representative of current NYC demand.

### 2. Geographic clusters are approximations

K-Means creates geographic groups based on coordinate distances.

The clusters do not necessarily correspond to official NYC taxi zones, neighborhoods, or borough boundaries.

### 3. Zero-count replacement

The notebook replaces zero pickup counts with 10 before smoothing.

A zero may represent genuine absence of observed pickups. Replacing it changes the demand series and can introduce artificial values.

This assumption should be reviewed and tested against alternative preprocessing strategies.

### 4. Limited external information

The implemented feature set focuses on historical demand, region, and calendar information.

Weather, traffic, major events, and other external demand drivers are potential future extensions, not confirmed features in the current model.

### 5. MAPE limitations

MAPE can be undefined at zero and unstable for small actual values.

MAE, RMSE, WAPE, and other suitable metrics can provide additional information about forecasting performance.

### 6. Model reproducibility

Reproducing the complete workflow requires the original data, dependencies, saved artifacts, and correct file paths.

These should be documented and versioned appropriately.

### 7. Model validation

Hyperparameter tuning and final evaluation should be separated carefully.

Repeatedly selecting models based on the final test period can lead to optimistic performance estimates.

A dedicated validation period or time-series cross-validation is preferable.

---

## 🔮 23. Future Improvements

The project can be extended in several directions.

### 📈 A. Improved time-series validation

- Use rolling-origin evaluation.
- Introduce a dedicated validation period.
- Evaluate multiple forecast horizons.
- Keep the final test period separate from model selection.
- Compare predictions against simple seasonal baselines.

### 🌦️ B. External feature integration

Potential additional features include:

- Weather conditions
- Public holidays
- Special events
- Traffic congestion
- Transit disruptions

These features could help explain demand changes that are not captured by historical pickup counts alone.

### 🗺️ C. Improved geographic modeling

- Compare learned clusters with official NYC taxi zones.
- Evaluate alternative cluster counts.
- Measure forecasting errors separately for each region.
- Investigate demand relationships between neighboring regions.
- Explore spatially aware forecasting methods.

### 🧠 D. Advanced forecasting models

Potential candidates for future experiments include:

- LightGBM
- CatBoost
- SARIMAX
- LSTM
- GRU
- Temporal Fusion Transformer

These models should be evaluated against the existing baseline using the same forecasting protocol.

### 🚀 E. Deployment and dashboard

A future deployment could expose a prediction API and a dashboard for regional demand monitoring.

Potential dashboard components:

- Geographic region selector
- Forecast timestamp
- Predicted pickup demand
- Historical demand chart
- Actual-versus-predicted comparison
- Region-wise error analysis

A deployment would require a validated inference pipeline, saved model artifacts, and a defined interface for generating predictions.

---

## 💡 24. Key Learnings

This project demonstrates practical experience with:

- Exploratory Data Analysis
- Data cleaning and outlier handling
- Geographic clustering
- Incremental data processing
- Time-series aggregation
- EWMA smoothing
- Lag-based feature engineering
- Categorical encoding
- Regression modeling
- Chronological train-test splitting
- Hyperparameter optimization
- Experiment tracking
- Forecast evaluation
- Geographic visualization

It also highlights important real-world machine learning considerations:

- Preventing temporal data leakage
- Handling zero-valued targets carefully
- Choosing appropriate evaluation metrics
- Maintaining consistent preprocessing
- Separating model selection from final testing
- Making experiments reproducible

---

## 📚 25. References

1. **NYC Taxi and Limousine Commission — Trip Record Data**  
   https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

2. **Scikit-learn — MiniBatchKMeans**  
   https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html

3. **Scikit-learn — StandardScaler**  
   https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

4. **Scikit-learn — Linear Regression**  
   https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

5. **Scikit-learn — OneHotEncoder**  
   https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

6. **Optuna Documentation**  
   https://optuna.readthedocs.io/

7. **MLflow Documentation**  
   https://mlflow.org/docs/latest/index.html

8. **XGBoost Documentation**  
   https://xgboost.readthedocs.io/

---

<div align="center">

## 🚕 From Historical Taxi Trips to Geographic Demand Insights

### Data → Regions → Time Series → Features → Forecasts → Insights

**Built with Python, Machine Learning, and Time-Series Analysis**

</div>
