# Smart Energy Consumption & Electricity Bill Prediction System

A comprehensive machine learning-based web application for predicting energy consumption, calculating electricity bills, detecting anomalies, and providing personalized energy-saving recommendations.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Modules Explained](#modules-explained)
7. [Data Flow](#data-flow)
8. [Machine Learning Pipeline](#machine-learning-pipeline)
9. [Web Application](#web-application)
10. [API Reference](#api-reference)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project transforms raw energy consumption data into actionable insights for households. It combines:

- **Energy Prediction**: ML models to forecast consumption
- **Bill Calculation**: Time-of-use and flat-rate tariff calculations
- **Anomaly Detection**: Statistical and ML-based outlier detection
- **Recommendations**: Personalized energy-saving tips with cost savings
- **Web Dashboard**: Interactive Streamlit interface

---

## Project Structure

```
week8_project/
│
├── src/                              # Source code modules
│   ├── data_preprocessing.py         # Data cleaning & normalization
│   ├── feature_engineering.py        # Feature creation (time, lags, rolling)
│   ├── train_model.py                # Model training pipeline
│   ├── predict.py                    # Prediction interface
│   ├── utils.py                      # Helper functions
│   ├── enhanced_bill_calculation.py  # Bill calculation & recommendations
│   └── enhanced_anomaly_detection.py # Anomaly detection algorithms
│
├── app/                              # Web application
│   └── app.py                        # Streamlit main application
│
├── data/                             # Data storage
│   └── sample_energy_data.csv        # Sample dataset
│
├── models/                           # Trained model storage
├── outputs/                          # Output files
│
├── main.py                           # CLI entry point
├── requirements.txt                  # Dependencies
└── README.md                         # Documentation
```

---

## Features

### 1. Energy Consumption Prediction
- Historical data analysis
- Time-series forecasting
- Seasonal pattern recognition

### 2. Bill Calculation
- **Simple Tariff**: Flat-rate billing
- **Time-of-Use**: Peak/off-peak pricing
- **Tax Calculation**: GST inclusion
- **Multi-tariff Support**: Standard, Economy, Premium

### 3. Anomaly Detection
- **Isolation Forest**: ML-based outlier detection
- **Z-Score**: Statistical threshold method
- **IQR Method**: Quartile-based detection
- **Seasonal Analysis**: Time-aware anomaly detection
- **Ensemble Voting**: Combined detection approach

### 4. Recommendations Engine
- Consumption level analysis
- AC optimization tips
- Peak hour shifting
- LED lighting upgrade
- Water heating efficiency
- Per-capita usage analysis

### 5. Web Dashboard (Streamlit)
- **Bill Prediction Tab**: Real-time bill calculation with pie chart
- **Anomaly Detection Tab**: Upload CSV data for anomaly analysis
- **Recommendations Tab**: Personalized energy-saving tips
- **Analytics Tab**: Distribution and time series charts
- **Projections Tab**: Monthly bill projections and scenarios

---

## Installation

### Prerequisites
- Python 3.9+
- pip package manager

### Step 1: Clone/Download
```bash
cd week8_project/
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
pandas>=1.5.0      # Data manipulation
numpy>=1.24.0      # Numerical computing
scikit-learn>=1.3.0 # Machine learning
streamlit>=1.28.0  # Web interface
plotly>=5.15.0     # Visualizations
joblib>=1.3.0      # Model serialization
```

---

## Usage

### Option 1: Run Web Application (Recommended)
```bash
python app/app.py
```
Access at: http://localhost:8501

### Option 2: Run with Streamlit Directly
```bash
python -m streamlit run app/app.py
```

### Option 3: Use Individual Modules
```python
# Bill calculation
from src.enhanced_bill_calculation import calculate_bill_simple

bill = calculate_bill_simple(
    energy_consumption_kwh=100.0,
    tariff_type="residential_standard"
)
print(f"Total Bill: Rs. {bill['total_bill']:.2f}")
```

---

## Modules Explained

### A. Data Preprocessing (`data_preprocessing.py`)

**Purpose**: Clean and prepare raw data for analysis.

**Functions**:
- `load_data(filepath)`: Load CSV data
- `clean_data(df)`: Handle missing values, remove duplicates
- `normalize_features(df, columns, method='standard')`: Scale features using StandardScaler/MinMaxScaler

**Process Flow**:
```
Raw CSV → Load → Clean → Normalize → Processed Data
```

---

### B. Feature Engineering (`feature_engineering.py`)

**Purpose**: Create predictive features from raw data.

**Functions**:
- `create_time_features(df, date_column='Date')`: Extract day, month, hour, weekend flags
- `create_lag_features(df, column, lags=[1, 7, 30])`: Previous period values (1, 7, 30 days)
- `create_rolling_features(df, column, windows=[7, 30])`: Moving averages and standard deviations

**Features Created**:
```
Original: [Date, Energy_Consumption_kWh]
         ↓
Enhanced: [day_of_week, month, hour, is_weekend, 
          consumption_lag_1, consumption_lag_7,
          rolling_mean_7, rolling_std_30]
```

---

### C. Model Training (`train_model.py`)

**Purpose**: Train ML models for consumption prediction.

**Algorithm**: Random Forest Regressor
- Handles non-linear relationships
- Robust to outliers
- Feature importance analysis

**Process**:
1. Split data (80% train, 20% test)
2. Train Random Forest (100 estimators)
3. Evaluate with MAE, RMSE, R²
4. Save model to `models/energy_model.pkl`

**Metrics**:
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- R²: Coefficient of determination

---

### D. Prediction (`predict.py`)

**Purpose**: Interface for making predictions.

**Functions**:
- `load_model(path)`: Load trained model
- `predict_single(model, features)`: Single household prediction
- `predict_batch(model, df, feature_cols)`: Batch predictions

**Usage**:
```python
model = load_model('models/energy_model.pkl')
prediction = predict_single(model, [100, 22, 65, 12, 2, 0])
print(f"Predicted consumption: {prediction:.1f} kWh")
```

---

### E. Bill Calculation (`enhanced_bill_calculation.py`)

**Purpose**: Calculate electricity bills with various tariff structures.

**Tariff Types**:

| Type | Base Rate | Peak Rate | Off-Peak | Fixed Charge | Peak Hours |
|------|-----------|-----------|----------|--------------|------------|
| Standard | Rs. 6.0 | Rs. 7.5 | Rs. 4.0 | Rs. 100 | 6PM-10PM |
| Economy | Rs. 5.5 | Rs. 7.0 | Rs. 3.5 | Rs. 80 | 5PM-11PM |
| Premium | Rs. 6.5 | Rs. 8.0 | Rs. 3.0 | Rs. 150 | 6PM-9PM |

**Functions**:
- `calculate_bill_simple()`: Flat-rate calculation
- `calculate_bill_time_of_use()`: Peak/off-peak pricing
- `generate_energy_recommendations()`: Personalized tips
- `calculate_bill_projection()`: Future bill forecasting
- `compare_bill_scenarios()`: What-if analysis

**Bill Components**:
```
Energy Charge = Consumption × Rate
Fixed Charge = Monthly fee
Subtotal = Energy + Fixed
Tax = Subtotal × 5% (GST)
Total = Subtotal + Tax
```

---

### F. Anomaly Detection (`enhanced_anomaly_detection.py`)

**Purpose**: Identify unusual consumption patterns.

**AnomalyDetector Class**:

#### 1. Isolation Forest
```python
detector = AnomalyDetector(contamination=0.05)
df_result, labels, pct = detector.detect_isolation_forest(df)
```
- Isolates anomalies by random feature selection
- Anomalies require fewer splits to isolate

#### 2. Z-Score Method
```python
df_result, z_scores, pct = detector.detect_zscore(df, threshold=3.0)
```
- Z = (X - μ) / σ
- Flags points beyond threshold (default: 3σ)

#### 3. IQR Method
```python
df_result, anomalies, pct = detector.detect_iqr(df, multiplier=1.5)
```
- IQR = Q3 - Q1
- Bounds: Q1 - 1.5×IQR to Q3 + 1.5×IQR

#### 4. Seasonal Detection
```python
df_result, anomalies, pct = detector.detect_seasonal_anomalies(df, window_days=30)
```
- Rolling mean and std comparison
- Accounts for seasonal patterns

#### 5. Ensemble Detection
```python
df_result, ensemble_flags, pct = detector.ensemble_detection(df, voting_threshold=0.5)
```
- Combines all methods
- Voting-based consensus (50% threshold)

---

### G. Utilities (`utils.py`)

**Helper Functions**:
- `format_currency(amount)`: Format as Rs. X,XXX.XX
- `calculate_savings(current, projected)`: Savings amount and percentage
- `save_results(df, path)`: Export to CSV
- `load_config(path)`: Load YAML configuration

---

## Data Flow

```
┌─────────────────┐
│  Raw Data CSV   │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Preprocessing  │ ← data_preprocessing.py
│  (Clean/Scale)  │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Feature Engineer│ ← feature_engineering.py
│ (Create Features)│
└────────┬────────┘
         ↓
┌─────────────────┐     ┌─────────────────┐
│  Train Model    │────→│  Saved Model    │
│  (Random Forest)│     │  (.pkl file)    │
└─────────────────┘     └─────────────────┘
         ↓
┌─────────────────┐
│    Predict      │ ← predict.py
│  (Consumption)  │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Calculate Bill │ ← enhanced_bill_calculation.py
│  (With Tariff)  │
└────────┬────────┘
         ↓
┌─────────────────┐
│   Anomaly Check │ ← enhanced_anomaly_detection.py
│  (Detect Issues)│
└────────┬────────┘
         ↓
┌─────────────────┐
│ Recommendations │ ← Generate Tips
│ (Save Money)    │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Streamlit App  │ ← app.py
│  (Visualize)    │
└─────────────────┘
```

---

## Machine Learning Pipeline

### Training Phase
```
Historical Data
      ↓
Feature Engineering
      ↓
Train/Test Split (80/20)
      ↓
Random Forest Training
      ↓
Model Evaluation
      ↓
Save Model
```

### Prediction Phase
```
New Input Data
      ↓
Load Trained Model
      ↓
Feature Extraction
      ↓
Model Prediction
      ↓
Consumption Forecast
```

---

## Web Application

### Dashboard Tabs

#### 1. Bill Prediction
- Input monthly consumption
- Select tariff type
- View bill breakdown
- Annual projection
- Pie chart visualization

#### 2. Anomaly Detection
- Upload CSV with consumption data (sample file: `data/sample_energy_data.csv`)
- Statistical summary (mean, median, std, min, max)
- IQR-based anomaly detection
- Anomaly alerts with counts and percentage
- Scatter plot with mean and bounds visualization
- Download anomaly chart as HTML

#### 3. Recommendations
- Consumption level indicator
- Personalized tips:
  - AC optimization
  - Peak hour shifting
  - LED lighting
  - Water heating
- Savings calculator

#### 4. Analytics
- Upload CSV file for analysis
- Consumption distribution histogram
- Time series chart (if date column present)
- Download charts as HTML files

#### 5. Projections
- Monthly bill forecast (1-24 months slider)
- What-if scenarios (current, 10%, 20% reduction)
- Cumulative cost analysis
- Download projection chart as HTML

---

## CSV File Format for Analytics & Anomaly Detection

Required columns for CSV upload:

| Column | Type | Description |
|--------|------|-------------|
| Date | Date | Date of consumption (YYYY-MM-DD) |
| Energy_Consumption_kWh | Number | Energy consumption in kilowatt-hours |
| Household_ID | String | Unique household identifier (optional) |
| Temperature | Number | Temperature in Celsius (optional) |
| Humidity | Number | Humidity percentage (optional) |
| Hour | Number | Hour of day (0-23) (optional) |
| DayOfWeek | Number | Day of week (0-6, 0=Monday) (optional) |
| IsWeekend | Number | Weekend flag (0 or 1) (optional) |

**Sample CSV file location**: `data/sample_energy_data.csv`

**How to upload CSV**:
1. Open the app at http://localhost:8501
2. Go to **Analytics** or **Anomaly Detection** tab
3. In the sidebar, click **"Upload historical consumption data (CSV)"**
4. Select your CSV file
5. Charts and analysis will appear automatically

**Minimum required columns**: At minimum, your CSV needs:
- A consumption column (name containing 'consumption' or 'energy')
- Optionally a date column (name containing 'date')

The app will automatically detect the correct columns even if they have slightly different names.

---

## API Reference

### Enhanced Bill Calculation

```python
from src.enhanced_bill_calculation import (
    calculate_bill_simple,
    calculate_bill_time_of_use,
    generate_energy_recommendations,
    TARIFFS
)

# Simple calculation
bill = calculate_bill_simple(
    energy_consumption_kwh=150.0,
    tariff_type="residential_standard",
    months=1
)
# Returns: {'energy_charge', 'fixed_charge', 'tax_amount', 'total_bill', ...}

# Time-of-use calculation
hourly = {i: 0.5 for i in range(24)}  # 0.5 kWh each hour
bill_tou = calculate_bill_time_of_use(
    hourly_consumption=hourly,
    tariff_type="residential_standard"
)
# Returns: {'peak_consumption', 'offpeak_consumption', 'total_bill', ...}

# Generate recommendations
recommendations = generate_energy_recommendations(
    consumption_data=df,
    user_consumption=120.0,
    household_size=4,
    user_has_ac=True
)
# Returns: List of dicts with 'severity', 'title', 'message', 'action', 'estimated_savings'
```

### Anomaly Detection

```python
from src.enhanced_anomaly_detection import (
    AnomalyDetector,
    generate_anomaly_alerts
)

# Initialize detector
detector = AnomalyDetector(contamination=0.05)

# Detect anomalies
df_result, flags, pct = detector.ensemble_detection(df)

# Generate alerts
alerts = generate_anomaly_alerts(df, flags)
```

---

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

### Issue: Port 8501 already in use
**Solution**: Use different port
```bash
python -m streamlit run app/app.py --server.port 8502
```

### Issue: Model file not found
**Solution**: Train model first
```bash
python -c "from src.train_model import train_model; train_model(df, features, target)"
```

### Issue: Unicode encoding errors
**Solution**: Use ASCII-safe characters (Rs. instead of ₹)

---

## Future Enhancements

1. **Deep Learning Models**: LSTM for time-series forecasting
2. **Real-time Monitoring**: IoT sensor integration
3. **Solar Integration**: Net metering calculations
4. **Mobile App**: React Native/Flutter interface
5. **API Service**: FastAPI backend
6. **Database**: PostgreSQL for historical storage
7. **Authentication**: User login and data privacy

---

**Last Updated**: May 2026
