# 🚢 FreightCast AI — 7-Day Logistics & Capacity Decision Engine

An interactive, production-ready machine learning web application for **freight demand forecasting, logistics capacity planning, and rush-cost minimization**.

Built as part of the **Machine Learning Innovative Assignment**.

---

## 📌 Project Overview

Freight operators face a costly operational dilemma:
* **Waiting until the last minute:** Forcing booking into the spot / rush window where vessel charter rates jump by **25–35%** and warehouse/labour fees carry emergency premiums.
* **Booking 7 days in advance:** Locks in planned contract rates and prevents logistics bottleneck disruptions.

**FreightCast AI** bridges machine learning with real-world operations by:
1. Predicting sustained **30-day planning demand 7 days ahead** using operational telemetry (congestion, weather, lead times, inventory).
2. Automatically calculating the **required vessels, warehouse floor space, and labour shifts**.
3. Quantifying **exact dollar savings** from early booking vs. rush-window spot rates.
4. Explaining **which operational factors** (e.g. port congestion, weather severity) are driving demand shifts.

---

## 📁 Repository Structure

```text
standalone_project/
├── app.py                  # Main Streamlit web application
├── train_export.py         # End-to-end model training & artifact exporter
├── requirements.txt        # Python package dependencies
├── .streamlit/
│   └── config.toml         # UI theme and layout configuration
├── data/
│   └── dynamic_supply_chain_logistics_dataset.csv  # Full operational telemetry dataset
├── models/
│   ├── ridge_model.pkl     # Tuned Ridge Regression model (Champion: R² = 0.57)
│   ├── rf_model.pkl        # Tuned Random Forest model
│   ├── gbr_model.pkl       # Tuned Gradient Boosting model
│   ├── scaler.pkl          # Pre-fitted StandardScaler
│   ├── metadata.json       # Feature metadata, baseline metrics, band thresholds
│   └── recent_sample.csv   # Recent historical data sample for dashboard preview
└── README.md               # Deployment and usage instructions
```

---

## 🚀 Quickstart: Running Locally

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
Open your terminal in this directory and run:
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-train & Export Models
The repository comes with pre-trained models in `models/`. To retrain from scratch:
```bash
python train_export.py
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```
The app will automatically open in your default browser at `http://localhost:8501`.

---

## 🌐 How to Deploy to the Web (Free & Public)

### Method 1: Streamlit Community Cloud (Recommended — 3 Minutes)
Streamlit Cloud offers free hosting directly from GitHub.

1. **Push this folder to a GitHub repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit for FreightCast AI"
   git remote add origin https://github.com/<your-username>/freightcast-ai.git
   git branch -M main
   git push -u origin main
   ```
2. **Go to [share.streamlit.io](https://share.streamlit.io/)** and sign in with your GitHub account.
3. Click **"New app"**.
4. Select your repository: `freightcast-ai`.
5. Set **Main file path** to `app.py` (or `standalone_project/app.py`).
6. Click **"Deploy"**!
   * Streamlit will automatically read `requirements.txt`, install dependencies, and give you a live shareable URL (e.g., `https://freightcast-ai.streamlit.app`).

---

### Method 2: Hugging Face Spaces (Free Alternative)
1. Sign in to [Hugging Face](https://huggingface.co/) and navigate to **Spaces** $\to$ **Create new Space**.
2. Name your space (e.g. `freight-forecasting`) and select **Streamlit** as the Space SDK.
3. Clone the repo provided by Hugging Face and copy all files from this folder into it.
4. Push your commit:
   ```bash
   git add .
   git commit -m "Deploy FreightCast AI"
   git push
   ```
5. Your app will automatically build and run with a public link.

---

### Method 3: Render / Railway / Cloud VM
You can also deploy to Render or Railway using a Web Service:
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## 🧠 Machine Learning Details

| Property | Value / Description |
|---|---|
| **Engineered Target** | 30-day moving average demand projected 7 days ahead (`planning_demand(t + 168)`) |
| **Why not raw demand?** | Raw hourly demand autocorrelation is $\approx 0$ ($|r| \le 0.01$). Stations book against sustained volume, which is predictable. |
| **Validation Strategy** | Strict `TimeSeriesSplit` (chronological 70/15/15 split, zero future data leakage). |
| **Champion Model** | **Ridge Regression (tuned)** ($R^2 = 0.5717$, MAE = $65.13$ units). |
| **Interpretability** | Ridge weights provide exact, additive factor contributions for transparent managerial decisions. |

---

## 🛠️ Authors & Attribution
* **Course:** Machine Learning – Innovative Assignment
* **Dataset:** CC0 Public Domain Logistics Telemetry

