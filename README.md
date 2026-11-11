# Car Fuel Consumption Predictor | Python & Machine Learning

## 📌 Project Overview
An advanced, end-to-end Machine Learning regression project engineered during Semester 7 using Python to model and predict vehicle fuel consumption parameters. The architecture applies multi-output random forest regression models on vehicle engineering datasets to dynamically estimate city, highway, and combined fuel efficiencies based on mechanical, transmission, and environmental emission profiles.

## 🎯 Objectives
- Process and clean vehicle manufacturing datasets containing varied categorical and numerical attributes.
- Implement robust categorical data transformations utilizing separate independent Label Encoding arrays.
- Train a Multi-Output Random Forest Regressor to simultaneously model three independent target metrics.
- Evaluate predictive accuracy thresholds using statistical R-squared ($R^2$) scoring models.
- Build an interactive, real-time command-line interface (CLI) to accept user vehicle dimensions and output precise consumption metrics.

## 🛠️ Tools & Technologies
- **Python (VS Code)** — Primary engineering programming language and runtime execution engine
- **Pandas & NumPy** — Structured data cleansing, vector transformation, and data frame manipulation
- **Scikit-Learn** — Feature preprocessing, dataset splitting, Random Forest fitting, and regression scoring metrics

## 📊 Pipeline Architecture & Engineering
- **Categorical Schema Mapping:** Deploys separate, independent `LabelEncoder` objects across `VEHICLE CLASS`, `TRANSMISSION`, and `FUEL` parameters. This allows for isolated dictionary lookups and prevents category contamination.
- **Feature Matrix Separation:** Sets structural attributes (`ENGINE SIZE`, `CYLINDERS`, `EMISSIONS`, along with the encoded categories) as inputs to simultaneously map a dense 3-dimensional multi-target array (`FUEL CONSUMPTION`, `HWY (L/100 km)`, `COMB (L/100 km)`).
- **Ensemble Tree Modeling:** Fits a robust `RandomForestRegressor` across an `80/20` split, utilizing multiple decision trees to stabilize variance and accurately map non-linear engine data behaviors.
- **Input Error Handling Wrapper:** Encapsulates production steps inside a `try-except` block to intercept unrecognized vehicle text attributes and flag string transformation errors gracefully without crashing the system terminal.

## 🔍 Model Evaluation Metrics
The pipeline runs an intensive statistical diagnostic suite to verify target regression parameters:
1. **Multi-Output Estimation** — Simultaneously maps predictions for City Consumption, Highway (HWY) metrics, and Combined performance indexes.
2. **R-squared ($R^2$) Score** — Evaluates the exact variance score of the model to prove predictive performance boundaries before deployment.

## 📁 Project Files
- `final_fuel_consumption.py` — The core executable Python script handling data loading, model fitting, and user parsing logic.
- `clean_fuel.csv` — The preprocessed source transaction ledger containing engine specifications and performance markers.

## 🚀 How to Run
1. Download the script and data files into a single project directory.
2. Open your terminal window and verify that your system dependencies are active:
   ```bash
   pip install numpy pandas scikit-learn
   ```
3. Execute the master pipeline file to view your accuracy rank and engage with the CLI tool:
   ```bash
   python final_fuel_consumption.py
   ```
