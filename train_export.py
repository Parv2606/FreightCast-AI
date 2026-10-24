import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
WINDOW = 720      # 30-day planning window in hours
HORIZON = 168     # 7-day ahead forecast lead time in hours

def train_and_export():
    print("Loading dataset...")
    # Look for dataset in data/ or parent folder
    data_path = os.path.join(os.path.dirname(__file__), "data", "dynamic_supply_chain_logistics_dataset.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "..", "dynamic_supply_chain_logistics_dataset.csv")

    df_raw = pd.read_csv(data_path)
    print(f"Loaded {len(df_raw)} rows.")

    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Preprocessing
    risk_order = {"Low Risk": 0, "Moderate Risk": 1, "High Risk": 2}
    df["risk_level"] = df["risk_classification"].map(risk_order)

    # Feature Engineering
    df["planning_demand"] = df["historical_demand"].rolling(WINDOW).mean()

    for lag in [24, 48, 168, 336, 720]:
        df["planning_lag_" + str(lag)] = df["planning_demand"].shift(lag)

    df["demand_mean_7d"] = df["historical_demand"].rolling(168).mean()
    df["demand_std_7d"] = df["historical_demand"].rolling(168).std()

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Outlier capping on raw features
    OPS_FEATURES = [
        "traffic_congestion_level", "port_congestion_level",
        "warehouse_inventory_level", "shipping_costs", "lead_time_days",
        "weather_condition_severity", "eta_variation_hours",
        "order_fulfillment_status", "supplier_reliability_score",
        "loading_unloading_time", "handling_equipment_availability",
        "customs_clearance_time", "delay_probability"
    ]

    for col in OPS_FEATURES:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    FEATURES = [
        "planning_demand",
        "planning_lag_24", "planning_lag_48", "planning_lag_168",
        "planning_lag_336", "planning_lag_720",
        "demand_mean_7d", "demand_std_7d",
        "hour_of_day", "day_of_week", "month", "risk_level"
    ] + OPS_FEATURES

    df["target"] = df["planning_demand"].shift(-HORIZON)

    model_df = df.dropna(subset=FEATURES + ["target"]).copy()
    model_df = model_df.sort_values("timestamp").reset_index(drop=True)

    n = len(model_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = model_df.iloc[:train_end]
    val = model_df.iloc[train_end:val_end]
    test = model_df.iloc[val_end:]

    X_train = train[FEATURES]
    y_train = train["target"].values
    X_val = val[FEATURES]
    y_val = val["target"].values
    X_test = test[FEATURES]
    y_test = test["target"].values

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Models & GridSearch
    print("Training Ridge (alpha=100.0)...")
    ridge = Ridge(alpha=100.0)
    ridge.fit(X_train_s, y_train)

    print("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train_s, y_train)

    print("Training Gradient Boosting...")
    gbr = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05,
        random_state=RANDOM_STATE
    )
    gbr.fit(X_train_s, y_train)

    # Evaluate on test
    models = {
        "Ridge": ridge,
        "Random Forest": rf,
        "Gradient Boosting": gbr
    }

    test_metrics = {}
    for name, m in models.items():
        preds = m.predict(X_test_s)
        test_metrics[name] = {
            "MAE": round(float(mean_absolute_error(y_test, preds)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 2),
            "R2": round(float(r2_score(y_test, preds)), 4)
        }
        print(f"Test Score - {name}: {test_metrics[name]}")

    # Demand bands (terciles on training set)
    band_edges = [
        float(np.percentile(y_train, 33.33)),
        float(np.percentile(y_train, 66.67))
    ]
    print(f"Band edges (Low < {band_edges[0]:.1f} <= Normal < {band_edges[1]:.1f} <= High)")

    # Feature statistics for interactive UI sliders
    feature_meta = {}
    for col in FEATURES:
        feature_meta[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "recent": float(model_df.iloc[-1][col])
        }

    # Save artifacts
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(ridge, os.path.join(models_dir, "ridge_model.pkl"))
    joblib.dump(rf, os.path.join(models_dir, "rf_model.pkl"))
    joblib.dump(gbr, os.path.join(models_dir, "gbr_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    # Ridge feature weights
    ridge_weights = dict(zip(FEATURES, [float(w) for w in ridge.coef_]))

    meta = {
        "features": FEATURES,
        "ops_features": OPS_FEATURES,
        "test_metrics": test_metrics,
        "band_edges": band_edges,
        "feature_meta": feature_meta,
        "ridge_weights": ridge_weights,
        "assumptions": {
            "demand_per_ship": 250.0,
            "sqft_per_demand": 0.60,
            "demand_per_worker": 120.0,
            "ship_early": 8000.0,
            "warehouse_early": 12.0,
            "labour_early": 1400.0,
            "ship_rush_premium": 0.30,
            "warehouse_rush_premium": 0.15,
            "labour_rush_premium": 0.13
        }
    }

    with open(os.path.join(models_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Save a small sample test dataset for interactive preview in app
    preview_df = model_df.tail(100)[["timestamp", "planning_demand", "target"] + OPS_FEATURES[:5]]
    preview_df.to_csv(os.path.join(models_dir, "recent_sample.csv"), index=False)

    print("All models and metadata exported successfully to:", models_dir)

if __name__ == "__main__":
    train_and_export()

