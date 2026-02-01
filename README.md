[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PrasanaPerumal/Household-Power-Consumption/blob/main/Household_Power_Consumption.ipynb)

# Household Electricity Consumption Forecasting using ARIMA & LSTM

---

## 📌 Project Overview
Accurate forecasting of household electricity consumption is essential for energy management, demand planning, and cost optimization.  
This project implements an **end-to-end time series forecasting pipeline** using a classical **ARIMA baseline** and an optimized **LSTM deep learning model**.  
The project also applies **SHAP-based interpretability** to explain LSTM predictions.

---

## ❓ Problem Statement
Traditional statistical models often fail to capture complex temporal and multivariate patterns in electricity consumption data.  
This project investigates whether a **multivariate LSTM model**, when properly optimized, can outperform a **classical ARIMA baseline** while maintaining interpretability.

---

## 📊 Dataset
- **Name:** Household Electric Power Consumption  
- **Source:** Kaggle / UCI Machine Learning Repository  
- **Original Frequency:** Minute-level  
- **Processed Frequency:** Hourly (resampled)  

### Features Used
- `Global_active_power` *(Target variable)*  
- `Global_reactive_power`  
- `Voltage`  
- `Global_intensity`  
- `Sub_metering_1`  
- `Sub_metering_2`  
- `Sub_metering_3`  

> The raw dataset is not included in this repository due to size and licensing constraints.

---

## 🧠 Methodology

### Step 1 — Data Preparation
- Combined Date and Time into a single datetime index  
- Converted all features to numeric format  
- Handled missing values using forward fill  
- Resampled data to hourly averages  
- Performed stationarity analysis using the Augmented Dickey-Fuller (ADF) test  

### Step 2 — Baseline Model (ARIMA)
- Implemented a univariate ARIMA model on `Global_active_power`  
- Used time-aware train/test split  
- Evaluated using RMSE and MAE  

### Step 3 — Sequence Creation
- Applied sliding-window approach with a lookback of 24 hours  
- Created multivariate sequences for LSTM  
- Scaled features using training data only  

### Step 4 — LSTM Model
- Built an LSTM model using TensorFlow/Keras  
- Applied dropout and early stopping to prevent overfitting  

### Step 5 — Hyperparameter Optimization
- Used Bayesian Optimization (Optuna)  
- Tuned number of units, dropout rate, learning rate, and batch size  
- Saved optimization trial history for reproducibility  

### Step 6 — Model Evaluation
- Compared ARIMA and optimized LSTM on unseen test data  
- Metrics used: RMSE and MAE  

### Step 7 — Model Interpretability
- Applied SHAP (KernelExplainer) to explain LSTM predictions  
- Analyzed feature contributions across time steps  
- Explained at least three representative predictions  

---

## 📈 Results

| Model | RMSE | MAE |
|------|------|-----|
| ARIMA (Baseline) | 0.797300 | 0.640950 |
| LSTM (Optimized) | 0.544975 | 0.396959 |

### Interpretation
- The optimized LSTM model significantly outperformed the ARIMA baseline.  
- Lower RMSE indicates improved handling of peak electricity usage.  
- Lower MAE shows more consistent and reliable predictions.  

---

## 🚀 How to Run (Google Colab)
1. Open the notebook using the **Open in Colab** button above  
2. Upload the dataset file to Colab or mount Google Drive  
3. Run all notebook cells sequentially from Step 1 to Step 7  

---

## ♻️ Reproducibility
- Random seeds are set for NumPy, Python, and TensorFlow  
- Feature scaling is performed after time-based splitting  
- Hyperparameter optimization results are saved  

---

## ⚠️ Limitations
- SHAP explanations for LSTM are approximate and computationally expensive  
- Dataset represents a single household, limiting generalization  
- Hourly resampling may smooth short-term fluctuations  

---

## 🔮 Future Scope
- Incorporate weather and calendar-based features  
- Use attention-based LSTM or Transformer models  
- Extend to multi-step and probabilistic forecasting  

---

## 🎓 Learning Outcomes
- Hands-on experience with time series forecasting  
- Practical understanding of ARIMA and LSTM models  
- Application of Bayesian hyperparameter optimization  
- Use of explainable AI techniques for deep learning  
- End-to-end ML project development  

---

## 👤 Author
**Perumal G**

This project was developed entirely by the author for academic verification and skill demonstration.

---


