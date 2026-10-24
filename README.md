# 🚢 FreightCast AI — 7-Day Logistics & Capacity Decision Engine

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.15%2B-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

> An end-to-end Machine Learning web application designed for freight station managers and supply chain planners to **forecast sustained demand 7 days ahead**, **automate resource booking (ships, warehouses, dock labour)**, and **minimize rush-window spot surcharges**.

---

## 📌 Executive Summary & Business Problem

In maritime freight and logistics operations, scheduling capacity cannot be done at the last minute:
* **The Last-Minute Trap:** Waiting until freight physically arrives forces operators into the **rush/spot market**, where vessel charter fees surge by **25% to 35%**, warehouse space carries premium holding fees, and emergency overtime wages spike.
* **The Operational Fix:** By predicting demand **7 days in advance** (standard booking lead time), stations lock in pre-negotiated contract rates, avoid congestion bottlenecks, and prevent labour roster shortages.

---

## 🧠 Key Data Science & ML Innovations

### 1. Honest EDA: Why Raw Hourly Demand Cannot Be Forecasted
Exploratory Data Analysis on 32,065 hours of telemetry proved that raw single-hour demand has:
* Correlation with operational telemetry: $|r| \le 0.01$ (effectively white noise).
* Autocorrelation across all lags (1 hour to 30 days): $\approx 0$.
* *Conclusion:* Attempting to forecast a single random hour is mathematically unreliable.

### 2. The Engineered Solution: 30-Day Planning Horizon
Freight stations never book shipping vessels for an isolated hour—they provision against **sustained operational volume**:
* We engineer **`planning_demand`** (a 30-day moving average).
* The model predicts this planning level **7 days into the future** ($t + 168\text{ hrs}$).
* This transformed target possesses strong temporal momentum ($r > 0.85$ correlation with the 7-day future), enabling high-confidence machine learning.

---

## 📊 Model Evaluation & Benchmarks (Unseen Test Data)

Validation followed a strict chronological **70% Train / 15% Validation / 15% Test** split using `TimeSeriesSplit` with zero future data leakage.

| Model | Hyperparameters | MAE (Demand Units) | RMSE | Test $R^2$ Score | Selection |
|---|---|---|---|---|---|
| **Ridge Regression** | $\alpha = 100.0$ | **65.13** | **79.26** | **0.5717** | 🏆 **Champion** |
| **Gradient Boosting** | $n=150$, $\text{lr}=0.05$ | 70.31 | 85.67 | 0.4995 | Candidate |
| **Random Forest** | $n=100$, $\text{depth}=10$, $\text{leaf}=3$ | 74.02 | 88.12 | 0.4705 | Candidate |
| **Naive Persistence Baseline** | $y(t+168) = y(t)$ | 114.20 | 142.10 | 0.0000 | Reference Benchmark |

> **Why Ridge Won:** Regularized $L_2$ regression effectively handles collinear rolling lag features without overfitting, offering instant inference and **100% transparent coefficient decomposition**.

---

## ✨ Interactive Dashboard Features (`app.py`)

* ⚡ **What-If Scenario Presets:** 1-click presets for *Normal Baseline*, *Port Congestion Surge*, *Severe Storm / Bad Weather*, and *Supply Chain Bottleneck*.
* 🎛️ **Live Operational Sliders:** Adjust port congestion, weather severity, inventory levels, ETA delays, and transit lead times.
* 📈 **Dynamic Demand Gauge:** Visualizes projected demand against historical Low, Normal, and High capacity bands.
* 🚢 **Automated Resource Allocation:**
  * Number of cargo ships to charter.
  * Square feet of warehouse floor space to reserve.
  * Number of dock labour shifts to roster.
* 💰 **Live Financial Savings Counter:** Calculates total contract cost vs. rush spot market cost, highlighting net dollar savings.
* 🔍 **Factor Attribution (Explainable AI):** Interactive waterfall/bar chart displaying exactly which factors pushed the forecast up or down.

---

## 🏗️ Repository Architecture

```text
├── app.py                  # Main Streamlit interactive web application
├── train_export.py         # Model training, hyperparameter tuning & export pipeline
├── requirements.txt        # Production dependencies for local & cloud environments
├── .streamlit/
│   └── config.toml         # Custom theme (colors, fonts, layout)
├── data/
│   └── dynamic_supply_chain_logistics_dataset.csv   # 32,065 rows of operational telemetry
├── models/
│   ├── ridge_model.pkl     # Tuned Champion Ridge model
│   ├── rf_model.pkl        # Tuned Random Forest model
│   ├── gbr_model.pkl       # Tuned Gradient Boosting model
│   ├── scaler.pkl          # Fitted StandardScaler
│   ├── metadata.json       # Feature stats, assumptions, baseline metrics
│   └── recent_sample.csv   # Historical data preview for the web dashboard
└── README.md               # Project documentation
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Parv2606/-FreightCast-AI.git
cd -FreightCast-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
Your default browser will automatically open at `http://localhost:8501`.

---

## 🌐 Free Cloud Deployment (Streamlit Community Cloud)

1. Fork or push this repository to GitHub.
2. Visit **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **"New app"**.
4. Select repository: `Parv2606/-FreightCast-AI`, branch: `main`, and main file: `app.py`.
5. Click **"Deploy"**!

---

## 📜 Academic Integrity & Dataset License
* **Dataset:** CC0: Public Domain (`dynamic_supply_chain_logistics_dataset.csv`).
* **Frameworks:** Python, Scikit-Learn, Streamlit, Pandas, NumPy, Plotly.
* **Course:** Machine Learning – Innovative Assignment.
